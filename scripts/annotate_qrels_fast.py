#!/usr/bin/env python3
"""Fast terminal annotation CLI for benchmark qrels labeling.

Presents one candidate at a time. User types 0/1/2/3 for relevance.
Saves each label immediately to adjudicated_qrels.jsonl.

Usage:
    # Start fresh (default annotator)
    python scripts/annotate_qrels_fast.py

    # Resume from a specific query
    python scripts/annotate_qrels_fast.py --query q_0003

    # Named annotator
    python scripts/annotate_qrels_fast.py --annotator sid_2026_06_11

    # Use the base 10-candidate pool (not the expanded one)
    python scripts/annotate_qrels_fast.py --candidates artifacts/field_state/qrels_candidates.jsonl

Relevance scale:
    3 = Highly relevant — modality, species, task, brain region all match; sufficient metadata
    2 = Partially relevant — matches primary goal but missing one constraint
    1 = Weakly relevant — correct domain but wrong species/modality or missing metadata
    0 = Not relevant — off-topic or hard-negative violation

Keys:
    0 1 2 3   rate relevance
    s         skip this candidate
    q         quit and save progress
    ?         show help
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import termios
import tty
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CANDIDATES = ROOT / "artifacts" / "field_state" / "qrels_candidates_full.jsonl"
FALLBACK_CANDIDATES = ROOT / "artifacts" / "field_state" / "qrels_candidates.jsonl"
OUTPUT_PATH = ROOT / "artifacts" / "field_state" / "adjudicated_qrels.jsonl"

RELEVANCE_LABELS = {
    0: "not_relevant",
    1: "weakly_relevant",
    2: "partially_relevant",
    3: "highly_relevant",
}

_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}

_STAR_COLORS = {3: "green", 2: "cyan", 1: "yellow", 0: "red"}


def _c(name: str, text: str) -> str:
    """Wrap text in ANSI color codes if stdout is a terminal."""
    if not sys.stdout.isatty():
        return text
    code = _COLORS.get(name, "")
    return f"{code}{text}{_COLORS['reset']}"


def _bold(text: str) -> str:
    return _c("bold", text)


def _dim(text: str) -> str:
    return _c("dim", text)


def getch() -> str:
    """Read a single character from stdin without waiting for Enter."""
    if not sys.stdin.isatty():
        return sys.stdin.read(1)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def load_candidates(path: Path) -> list[dict]:
    candidates = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def load_labeled_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                ids.add(obj.get("candidate_id", ""))
    return ids


def load_skipped_ids(session_path: Path) -> set[str]:
    """Load IDs skipped in a previous run from the session skip file."""
    skip_file = session_path.parent / f"{session_path.stem}.skips.jsonl"
    if not skip_file.exists():
        return set()
    ids: set[str] = set()
    with open(skip_file) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line.strip('"'))
    return ids


def append_label(path: Path, label: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(label) + "\n")


def build_label(
    candidate: dict,
    relevance: int,
    annotator_id: str,
    hard_negative_violation: bool = False,
    rationale: str = "",
) -> dict:
    return {
        "candidate_id": candidate["id"],
        "query_id": candidate["query_id"],
        "dataset_id": candidate["dataset_id"],
        "relevance": relevance,
        "label": RELEVANCE_LABELS[relevance],
        "rationale": rationale,
        "hard_negative_violation": hard_negative_violation or (relevance == 0),
        "missing_metadata": [],
        "annotator_id": annotator_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "adjudicated": True,
        "adjudication_notes": "",
        "schema_version": "0.3",
    }


def _desc_snippet(desc: str, max_chars: int = 350) -> str:
    if not desc:
        return _dim("(no description)")
    desc = desc.strip().replace("\n", " ").replace("\r", " ")
    while "  " in desc:
        desc = desc.replace("  ", " ")
    if len(desc) > max_chars:
        desc = desc[:max_chars] + "…"
    return desc


def _list_or_str(v: object) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if x]
    if isinstance(v, str) and v:
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if x]
        except (json.JSONDecodeError, ValueError):
            pass
        return [v]
    return []


def display_candidate(
    candidate: dict,
    query_map: dict[str, dict],
    index: int,
    total: int,
    labeled_count: int,
) -> None:
    query_id = candidate["query_id"]
    query = query_map.get(query_id, {})
    query_text = candidate.get("query_text") or query.get("query", "")
    intent = candidate.get("query_intent") or query.get("intent", "")

    title = candidate.get("dataset_title") or "(untitled)"
    source = candidate.get("dataset_source") or ""
    dataset_id = candidate.get("dataset_id") or ""
    desc = _desc_snippet(candidate.get("dataset_description") or "")
    rank = candidate.get("rank", "?")
    score = candidate.get("retrieval_score")

    meta = candidate.get("metadata", {})
    hard_negs = meta.get("query_known_failure_modes") or query.get("known_failure_modes") or []
    expected_species = _list_or_str(query.get("expected_species") or [])
    expected_modalities = _list_or_str(query.get("expected_modalities") or [])
    expected_tasks = _list_or_str(query.get("expected_tasks") or [])
    rec_species = _list_or_str(meta.get("record_species") or candidate.get("dataset_species") or [])
    rec_modalities = _list_or_str(meta.get("record_modalities") or candidate.get("dataset_modalities") or [])
    rec_tasks = _list_or_str(meta.get("record_tasks") or candidate.get("dataset_tasks") or [])

    # ANSI clear — avoids shell invocation, works on any POSIX terminal
    print("\033[2J\033[H", end="", flush=True)
    bar = "─" * 70
    print(bar)
    print(
        f"{_bold('CANDIDATE')} {_c('cyan', str(index))}/{_c('cyan', str(total))}  "
        f"{_dim(f'labeled: {labeled_count}')}  "
        f"query {_bold(query_id)}  rank {rank}"
        + (f"  score {score:.3f}" if score else "")
    )
    print(bar)
    print(f"\n{_bold('QUERY')}  [{_c('magenta', intent)}]")
    print(f"  {query_text}")
    if expected_modalities:
        print(f"  {_dim('Expected modalities:')} {', '.join(expected_modalities)}")
    if expected_species:
        print(f"  {_dim('Expected species:')} {', '.join(expected_species)}")
    if expected_tasks:
        print(f"  {_dim('Expected tasks:')} {', '.join(expected_tasks)}")
    if hard_negs:
        print(f"  {_c('red', '⚠ Hard negatives:')} {' | '.join(hard_negs)}")

    print(f"\n{_bold('DATASET')}  [{_c('blue', source)}:{dataset_id.split(':')[-1]}]")
    print(f"  {_c('bold', title)}")
    if rec_modalities:
        marker = _c("green", "✓") if set(rec_modalities) & set(expected_modalities) else _c("yellow", "?")
        print(f"  modalities: {marker} {', '.join(rec_modalities)}")
    if rec_species:
        marker = _c("green", "✓") if set(rec_species) & set(expected_species) else _c("yellow", "?")
        print(f"  species:    {marker} {', '.join(rec_species)}")
    if rec_tasks:
        marker = _c("green", "✓") if set(rec_tasks) & set(expected_tasks) else _c("yellow", "?")
        print(f"  tasks:      {marker} {', '.join(rec_tasks)}")
    print(f"\n  {desc}")

    print(f"\n{bar}")
    print(
        f"  {_c('green', '[3]')} highly relevant  "
        f"{_c('cyan', '[2]')} partially  "
        f"{_c('yellow', '[1]')} weakly  "
        f"{_c('red', '[0]')} not relevant  "
        f"{_dim('[s]')} skip  "
        f"{_dim('[q]')} quit"
    )
    print(f"{bar}\n  > ", end="", flush=True)


def compute_ndcg(labels: list[dict]) -> float | None:
    """Compute NDCG@10 over all labeled pairs grouped by query."""
    from collections import defaultdict

    by_query: dict[str, list[int]] = defaultdict(list)
    for label in labels:
        by_query[label["query_id"]].append(label["relevance"])

    if not by_query:
        return None

    ndcgs = []
    for query_id, scores in by_query.items():
        dcg = sum(
            (2**s - 1) / math.log2(i + 2) for i, s in enumerate(scores[:10])
        )
        ideal = sorted(scores, reverse=True)[:10]
        idcg = sum(
            (2**s - 1) / math.log2(i + 2) for i, s in enumerate(ideal)
        )
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(ndcgs) / len(ndcgs)


def print_help() -> None:
    print("\n" + "=" * 60)
    print("RELEVANCE SCALE:")
    print("  3 = Highly relevant — all constraints match, sufficient metadata")
    print("  2 = Partially relevant — matches goal, missing one constraint")
    print("  1 = Weakly relevant — correct domain, wrong species/modality or missing data")
    print("  0 = Not relevant — off-topic or explicit hard-negative violation")
    print("\nKEYS:")
    print("  0 1 2 3  rate relevance")
    print("  s        skip (come back later)")
    print("  q        quit and show summary")
    print("  ?        show this help")
    print("=" * 60 + "\n")
    input("Press Enter to continue...")


def run_annotation(
    candidates: list[dict],
    query_map: dict[str, dict],
    labeled_ids: set[str],
    annotator_id: str,
    output_path: Path,
    start_query: str | None = None,
    resume: bool = True,
) -> int:
    """Run the annotation loop. Returns number of labels saved this session."""
    queue = [c for c in candidates if not (resume and c["id"] in labeled_ids)]

    if start_query:
        # Put start_query candidates first
        priority = [c for c in queue if c["query_id"] == start_query]
        rest = [c for c in queue if c["query_id"] != start_query]
        queue = priority + rest

    if not queue:
        print("Nothing left to annotate (all candidates labeled).")
        return 0

    total = len(queue)
    session_labels: list[dict] = []
    index = 0

    for index, candidate in enumerate(queue, start=1):
        labeled_count = len(labeled_ids) + len(session_labels)
        display_candidate(candidate, query_map, index, total, labeled_count)

        while True:
            key = getch()

            if key in ("0", "1", "2", "3"):
                relevance = int(key)
                label = build_label(candidate, relevance, annotator_id)
                append_label(output_path, label)
                labeled_ids.add(candidate["id"])
                session_labels.append(label)
                break

            elif key == "s":
                # skip — don't label, don't add to labeled set
                print(_dim("\n  [skipped]"))
                import time
                time.sleep(0.4)
                break

            elif key in ("q", "\x03"):  # q or Ctrl-C
                print(f"\n\n{_bold('Session ended.')} Labeled {len(session_labels)} pairs this session.")
                if len(session_labels) >= 5:
                    ndcg = compute_ndcg(session_labels)
                    if ndcg is not None:
                        print(f"Session NDCG@10 (preliminary): {ndcg:.4f} ({len(session_labels)} pairs)")
                return len(session_labels)

            elif key == "?":
                print_help()
                display_candidate(candidate, query_map, index, total, len(labeled_ids) + len(session_labels))

    # Completed queue
    print(f"\n\n{_bold('All candidates annotated!')} Labeled {len(session_labels)} pairs this session.")
    if len(session_labels) >= 5:
        ndcg = compute_ndcg(session_labels)
        if ndcg is not None:
            print(f"Session NDCG@10 (preliminary): {ndcg:.4f} ({len(session_labels)} pairs)")
    return len(session_labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast terminal annotation CLI for benchmark qrels.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES if DEFAULT_CANDIDATES.exists() else FALLBACK_CANDIDATES,
        metavar="FILE",
        help="JSONL file of candidates to annotate (default: qrels_candidates_full.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        metavar="FILE",
        help="JSONL file to write adjudicated qrels (default: adjudicated_qrels.jsonl)",
    )
    parser.add_argument("--annotator", default="annotator_01", metavar="ID")
    parser.add_argument("--query", default=None, metavar="Q_ID", help="Start with this query first")
    parser.add_argument("--no-resume", action="store_true", help="Show all candidates, even already-labeled ones")
    args = parser.parse_args()

    if not args.candidates.exists():
        print(f"ERROR: candidates file not found: {args.candidates}", file=sys.stderr)
        print("Run scripts/eval/expand_candidate_pool.py first.", file=sys.stderr)
        sys.exit(1)

    candidates = load_candidates(args.candidates)
    labeled_ids = load_labeled_ids(args.output)

    # Build query map for fast lookup during display
    queries_path = ROOT / "artifacts" / "benchmark_queries.jsonl"
    query_map: dict[str, dict] = {}
    if queries_path.exists():
        with open(queries_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    q = json.loads(line)
                    query_map[q["query_id"]] = q

    already_labeled = len(labeled_ids)
    remaining = len([c for c in candidates if c["id"] not in labeled_ids])
    print(f"Loaded {len(candidates)} candidates, {already_labeled} already labeled, {remaining} remaining.")

    if not sys.stdin.isatty():
        print("WARNING: stdin is not a terminal. Annotation requires interactive input.", file=sys.stderr)
        sys.exit(1)

    print(f"Annotator: {args.annotator}")
    print(f"Output: {args.output}")
    print(f"Press '?' during annotation to see the relevance scale.\n")
    input("Press Enter to start...")

    saved = run_annotation(
        candidates,
        query_map,
        labeled_ids,
        args.annotator,
        args.output,
        start_query=args.query,
        resume=not args.no_resume,
    )

    total_labeled = already_labeled + saved
    print(f"\nTotal labeled pairs in {args.output.name}: {total_labeled}")
    if total_labeled >= 5:
        print(f"Run: python scripts/eval/report_benchmark_metrics.py")


if __name__ == "__main__":
    main()
