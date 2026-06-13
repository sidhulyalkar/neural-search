#!/usr/bin/env python3
"""Fast terminal annotation CLI for benchmark qrels labeling.

Presents one candidate at a time. User types 0/1/2/3 for relevance.
Saves each label immediately to adjudicated_qrels.jsonl.

Usage:
    # Start fresh (pooled candidates, default annotator)
    python scripts/annotate_qrels_fast.py

    # Resume where you left off
    python scripts/annotate_qrels_fast.py --resume

    # Annotate only one query
    python scripts/annotate_qrels_fast.py --query q_0003

    # Limit candidates shown (good for quick sessions)
    python scripts/annotate_qrels_fast.py --limit 20

    # Shuffle order (reduces anchoring bias)
    python scripts/annotate_qrels_fast.py --shuffle

    # Hide retrieval system info (blind annotation, reduces system bias)
    python scripts/annotate_qrels_fast.py --systems-hidden

    # Named annotator
    python scripts/annotate_qrels_fast.py --annotator sid_2026_06_11

    # Use a specific candidate pool
    python scripts/annotate_qrels_fast.py --candidates artifacts/field_state/qrels_candidates_pooled.jsonl

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
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POOLED_CANDIDATES = ROOT / "artifacts" / "field_state" / "qrels_candidates_pooled.jsonl"
LEGACY_CANDIDATES = ROOT / "artifacts" / "field_state" / "qrels_candidates_full.jsonl"
FALLBACK_CANDIDATES = ROOT / "artifacts" / "field_state" / "qrels_candidates.jsonl"
OUTPUT_PATH = ROOT / "artifacts" / "field_state" / "adjudicated_qrels.jsonl"

# Default: use pooled candidates if available, fall back to legacy, then base
DEFAULT_CANDIDATES = (
    POOLED_CANDIDATES if POOLED_CANDIDATES.exists()
    else (LEGACY_CANDIDATES if LEGACY_CANDIDATES.exists() else FALLBACK_CANDIDATES)
)

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
        "timestamp": datetime.now(UTC).isoformat(),
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
    systems_hidden: bool = False,
) -> None:
    query_id = candidate["query_id"]
    query = query_map.get(query_id, {})
    query_text = candidate.get("query_text") or query.get("query", "")
    intent = candidate.get("query_intent") or query.get("intent", "")

    title = candidate.get("dataset_title") or "(untitled)"
    source = candidate.get("dataset_source") or ""
    source_url = candidate.get("dataset_source_url") or ""
    dataset_id = candidate.get("dataset_id") or ""
    desc = _desc_snippet(candidate.get("dataset_description") or "")
    retrieval_sources = candidate.get("retrieval_sources") or []
    ranks_by_system = candidate.get("ranks_by_system") or {}
    usefulness_score = candidate.get("usefulness_score")
    affordance_matches = candidate.get("affordance_matches") or []
    is_hard_neg_warn = candidate.get("hard_negative_warning", False)

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

    # Header line — hide system info if --systems-hidden
    header = (
        f"{_bold('CANDIDATE')} {_c('cyan', str(index))}/{_c('cyan', str(total))}  "
        f"{_dim(f'labeled: {labeled_count}')}  "
        f"query {_bold(query_id)}"
    )
    if not systems_hidden and retrieval_sources:
        n = len(retrieval_sources)
        color = "green" if n >= 3 else ("cyan" if n == 2 else "dim")
        header += f"  {_c(color, f'[{n} systems]')}"
    print(header)
    print(bar)

    # Query section
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

    # Hard negative warning for this candidate
    if is_hard_neg_warn:
        print(f"  {_c('red', '⚠ WARNING: this candidate may be a hard negative')}")

    # Dataset section
    print(f"\n{_bold('DATASET')}  [{_c('blue', source)}:{dataset_id.split(':')[-1]}]")
    print(f"  {_c('bold', title)}")
    if source_url:
        print(f"  {_dim('URL:')} {source_url}")
    if rec_modalities:
        marker = _c("green", "✓") if set(rec_modalities) & set(expected_modalities) else _c("yellow", "?")
        print(f"  modalities: {marker} {', '.join(rec_modalities)}")
    if rec_species:
        marker = _c("green", "✓") if set(rec_species) & set(expected_species) else _c("yellow", "?")
        print(f"  species:    {marker} {', '.join(rec_species)}")
    if rec_tasks:
        marker = _c("green", "✓") if set(rec_tasks) & set(expected_tasks) else _c("yellow", "?")
        print(f"  tasks:      {marker} {', '.join(rec_tasks)}")
    if affordance_matches:
        print(f"  {_c('green', 'affordances:')} {', '.join(affordance_matches)}")
    print(f"\n  {desc}")

    # Retrieval provenance (hidden if --systems-hidden)
    if not systems_hidden and retrieval_sources:
        sys_parts = []
        for sys_name in sorted(retrieval_sources):
            rank_str = f"@{ranks_by_system[sys_name]}" if sys_name in ranks_by_system else ""
            sys_parts.append(f"{sys_name}{rank_str}")
        score_str = f"  usefulness={usefulness_score:.3f}" if usefulness_score else ""
        print(f"  {_dim('systems:')} {', '.join(sys_parts)}{score_str}")

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
    for _query_id, scores in by_query.items():
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
    shuffle: bool = False,
    limit: int | None = None,
    systems_hidden: bool = False,
) -> int:
    """Run the annotation loop. Returns number of labels saved this session."""
    import random
    import time

    queue = [c for c in candidates if not (resume and c["id"] in labeled_ids)]

    if start_query:
        priority = [c for c in queue if c["query_id"] == start_query]
        rest = [c for c in queue if c["query_id"] != start_query]
        queue = priority + rest

    if shuffle:
        random.shuffle(queue)

    if limit is not None:
        queue = queue[:limit]

    if not queue:
        print("Nothing left to annotate (all candidates labeled).")
        return 0

    total = len(queue)
    session_labels: list[dict] = []

    for index, candidate in enumerate(queue, start=1):
        labeled_count = len(labeled_ids) + len(session_labels)
        display_candidate(candidate, query_map, index, total, labeled_count, systems_hidden=systems_hidden)

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
                print(_dim("\n  [skipped]"))
                time.sleep(0.4)
                break

            elif key in ("q", "\x03"):  # q or Ctrl-C
                _print_session_summary(session_labels)
                return len(session_labels)

            elif key == "?":
                print_help()
                display_candidate(
                    candidate, query_map, index, total,
                    len(labeled_ids) + len(session_labels),
                    systems_hidden=systems_hidden,
                )

    _print_session_summary(session_labels)
    return len(session_labels)


def _print_session_summary(session_labels: list[dict]) -> None:
    from collections import Counter

    n = len(session_labels)
    if n == 0:
        print("\nNo labels saved this session.")
        return

    print(f"\n\n{_bold('Session summary.')} {n} pairs labeled.")

    by_query: dict[str, Counter] = {}
    by_intent: dict[str, list[int]] = {}
    for label in session_labels:
        qid = label["query_id"]
        if qid not in by_query:
            by_query[qid] = Counter()
        by_query[qid][label["relevance"]] += 1

        intent = label.get("intent", "unknown")
        by_intent.setdefault(intent, []).append(label["relevance"])

    print(f"\n{'Query':<12} {'N':>4}  {'rel dist (0/1/2/3)'}")
    for qid in sorted(by_query):
        cnt = by_query[qid]
        total = sum(cnt.values())
        dist = "/".join(str(cnt.get(i, 0)) for i in range(4))
        print(f"  {qid:<10} {total:>4}  {dist}")

    if n >= 5:
        ndcg = compute_ndcg(session_labels)
        if ndcg is not None:
            print(f"\nSession NDCG@10 (preliminary, {n} pairs): {ndcg:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast terminal annotation CLI for benchmark qrels.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES,
        metavar="FILE",
        help="JSONL file of candidates to annotate (default: qrels_candidates_pooled.jsonl if present)",
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
    parser.add_argument("--resume", action="store_true", help="Skip already-labeled candidates")
    parser.add_argument("--no-resume", action="store_true", help="Show all candidates, even already-labeled ones")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle candidate order to reduce anchoring bias")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Cap number of candidates shown this session")
    parser.add_argument("--systems-hidden", action="store_true", help="Hide retrieval system provenance (blind annotation)")
    args = parser.parse_args()

    if not args.candidates.exists():
        print(f"ERROR: candidates file not found: {args.candidates}", file=sys.stderr)
        print("Run: python scripts/eval/build_pooled_qrels_candidates.py", file=sys.stderr)
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
    remaining = sum(1 for c in candidates if c["id"] not in labeled_ids)
    pool_name = args.candidates.name
    print(f"Pool: {pool_name} — {len(candidates)} candidates, {already_labeled} labeled, {remaining} remaining.")

    if not sys.stdin.isatty():
        print("WARNING: stdin is not a terminal. Annotation requires interactive input.", file=sys.stderr)
        sys.exit(1)

    resume = not args.no_resume  # default to resuming
    if args.limit:
        print(f"Session limit: {args.limit} candidates")
    if args.shuffle:
        print("Shuffle: ON (order randomized, reduces anchoring bias)")
    if args.systems_hidden:
        print("Systems hidden: ON (blind annotation mode)")

    print(f"Annotator: {args.annotator}")
    print(f"Output: {args.output}")
    print("Press '?' during annotation to see the relevance scale.\n")
    input("Press Enter to start...")

    saved = run_annotation(
        candidates,
        query_map,
        labeled_ids,
        args.annotator,
        args.output,
        start_query=args.query,
        resume=resume,
        shuffle=args.shuffle,
        limit=args.limit,
        systems_hidden=args.systems_hidden,
    )

    total_labeled = already_labeled + saved
    print(f"\nTotal labeled pairs in {args.output.name}: {total_labeled}")
    if total_labeled >= 5:
        print("Run: python scripts/eval/report_benchmark_metrics.py")


if __name__ == "__main__":
    main()
