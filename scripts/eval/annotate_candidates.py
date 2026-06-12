#!/usr/bin/env python3
"""Interactive CLI for annotating pooled retrieval candidates with graded relevance labels.

This tool presents each (query, dataset) pair from the pooled candidates file
and asks the annotator to assign a 0-3 relevance label and optional rationale.
Progress is saved after each annotation so you can stop and resume at any time.

Label scale:
  0 = Not useful: topic, modality, or species mismatch; dataset is unusable for the query goal.
  1 = Weakly useful: superficially related but missing key metadata or weak task match.
  2 = Useful: matches the main scientific goal with enough metadata for plausible reuse.
  3 = Highly useful: strong match across goal, modality, species, task, metadata, and provenance.

Usage:
    # Annotate from a pool file (requires running run_retrieval_baselines.py first)
    python scripts/eval/annotate_candidates.py \
        --pool reports/eval/benchmark_pool.jsonl \
        --queries artifacts/benchmark_queries.jsonl \
        --corpus data/corpus/normalized/combined_corpus.jsonl \
        --out artifacts/qrels.jsonl

    # Resume from an existing qrels file (skips already annotated pairs)
    python scripts/eval/annotate_candidates.py \
        --pool reports/eval/benchmark_pool.jsonl \
        --queries artifacts/benchmark_queries.jsonl \
        --corpus data/corpus/normalized/combined_corpus.jsonl \
        --out artifacts/qrels.jsonl \
        --resume

    # Show statistics only (don't annotate)
    python scripts/eval/annotate_candidates.py \
        --pool reports/eval/benchmark_pool.jsonl \
        --out artifacts/qrels.jsonl \
        --stats-only
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_queries(path: Path) -> dict[str, dict]:
    queries: dict[str, dict] = {}
    if not path.exists():
        return queries
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                queries[str(q["query_id"])] = q
    return queries


def load_corpus(path: Path) -> dict[str, dict]:
    """Load corpus records keyed by stable {source}:{source_id} ID."""
    corpus: dict[str, dict] = {}
    if not path.exists():
        return corpus
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            source = str(record.get("source", "unknown"))
            source_id = str(
                record.get("source_id") or record.get("dataset_id") or record.get("id") or "unknown"
            )
            stable_id = f"{source}:{source_id}"
            corpus[stable_id] = record
    return corpus


def load_pool(path: Path) -> list[dict]:
    pool: list[dict] = []
    if not path.exists():
        return pool
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pool.append(json.loads(line))
    return pool


def load_existing_qrels(path: Path) -> set[tuple[str, str]]:
    """Return set of (query_id, record_id) pairs already annotated."""
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                done.add((str(row["query_id"]), str(row["record_id"])))
    return done


def append_qrel(
    path: Path,
    query_id: str,
    record_id: str,
    label: int,
    rationale: str,
    annotator_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps({
                "query_id": query_id,
                "record_id": record_id,
                "label": label,
                "rationale": rationale,
                "annotator_id": annotator_id,
            }) + "\n"
        )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _trunc(text: str | None, n: int = 200) -> str:
    if not text:
        return "(empty)"
    text = str(text).replace("\n", " ").strip()
    return text[:n] + "..." if len(text) > n else text


def format_list(items: list | None) -> str:
    if not items:
        return "(none)"
    return ", ".join(str(i) for i in items[:10])


def display_pair(query: dict, record: dict, record_id: str, pair_num: int, total: int) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  [{pair_num}/{total}]  Query: {query['query_id']} | Intent: {query.get('intent', '?')}")
    print(sep)
    print(f"  QUERY: {query['query']}")
    goal = query.get("scientific_goal", "")
    if goal:
        print(f"  GOAL:  {goal}")
    failure = query.get("known_failure_modes", [])
    if failure:
        modes = failure if isinstance(failure, list) else [failure]
        print(f"  HARD NEGATIVES: {', '.join(modes[:3])}")
    print()

    title = record.get("title") or "(no title)"
    print(f"  DATASET: {_trunc(title, 100)}")
    print(f"  ID:      {record_id}")
    print(f"  Source:  {record.get('source', '?')}")
    print(f"  Species: {format_list(record.get('species'))}")
    print(f"  Modality:{format_list(record.get('modalities'))}")
    print(f"  Tasks:   {format_list(record.get('tasks'))}")
    print(f"  Regions: {format_list(record.get('brain_regions'))}")
    print(f"  License: {record.get('license') or '(unknown)'}")
    print(f"  DOI:     {record.get('doi') or record.get('url') or '(none)'}")
    desc = record.get("description")
    if desc:
        wrapped = textwrap.fill(_trunc(desc, 300), width=70, initial_indent="  ", subsequent_indent="  ")
        print(f"\n  Description:\n{wrapped}")
    print()


def display_label_guide() -> None:
    print("  Labels:")
    print("    0 = Not useful (topic/modality/species mismatch, unusable)")
    print("    1 = Weakly useful (related but missing key metadata or task match)")
    print("    2 = Useful (matches goal, enough for plausible reuse)")
    print("    3 = Highly useful (strong match across goal, modality, species, task)")
    print("  Commands: 0/1/2/3 = label | s = skip | q = quit and save | ? = show guide")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(qrels_path: Path, pool: list[dict]) -> None:
    existing = load_existing_qrels(qrels_path)
    label_counts: dict[int, int] = defaultdict(int)
    query_counts: dict[str, int] = defaultdict(int)

    if qrels_path.exists():
        with qrels_path.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    label_counts[int(row["label"])] += 1
                    query_counts[str(row["query_id"])] += 1

    total_pool = len(pool)
    annotated = len(existing)
    remaining = total_pool - annotated

    print(f"\nAnnotation progress:")
    print(f"  Pool size:       {total_pool}")
    print(f"  Annotated:       {annotated} ({100*annotated/max(total_pool,1):.1f}%)")
    print(f"  Remaining:       {remaining}")
    print(f"\nLabel distribution:")
    for label in range(4):
        count = label_counts[label]
        bar = "█" * int(count * 40 / max(sum(label_counts.values()), 1))
        print(f"  {label}: {bar} {count}")
    print(f"\nQueries annotated: {len(query_counts)}")
    for qid, count in sorted(query_counts.items()):
        print(f"  {qid}: {count} pairs")


# ---------------------------------------------------------------------------
# Main annotation loop
# ---------------------------------------------------------------------------

def annotate(
    pool: list[dict],
    queries: dict[str, dict],
    corpus: dict[str, dict],
    qrels_path: Path,
    annotator_id: str,
    resume: bool,
) -> None:
    existing = load_existing_qrels(qrels_path) if resume else set()
    pairs = [(p["query_id"], p["record_id"]) for p in pool]
    todo = [(qid, rid) for qid, rid in pairs if (qid, rid) not in existing]

    if not todo:
        print("Nothing left to annotate! All pool pairs are already labeled.")
        return

    print(f"\nStarting annotation: {len(todo)} pairs remaining ({len(existing)} already done).")
    print("Labels will be saved after each annotation. Press Ctrl-C or type 'q' to stop.\n")
    display_label_guide()

    pair_num = len(existing)
    total = len(pairs)

    for qid, rid in todo:
        query = queries.get(qid, {"query_id": qid, "query": "(query not found)"})
        record = corpus.get(rid, {"title": "(record not in corpus)", "source": rid.split(":")[0]})
        pair_num += 1

        while True:
            display_pair(query, record, rid, pair_num, total)
            display_label_guide()

            try:
                raw = input("  Label [0/1/2/3/s/q/?]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\nSaving and quitting.")
                return

            if raw == "?":
                display_label_guide()
                continue
            if raw == "q":
                print("Quitting and saving progress.")
                return
            if raw == "s":
                print("  Skipped.")
                break
            if raw in ("0", "1", "2", "3"):
                label = int(raw)
                try:
                    rationale = input("  Rationale (optional, press Enter to skip): ").strip()
                except (KeyboardInterrupt, EOFError):
                    rationale = ""
                append_qrel(qrels_path, qid, rid, label, rationale, annotator_id)
                print(f"  Saved: label={label}")
                break
            print("  Invalid input. Enter 0, 1, 2, 3, s (skip), or q (quit).")

    print(f"\nAnnotation session complete.")
    print_stats(qrels_path, pool)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_qrels(qrels_path: Path) -> None:
    if not qrels_path.exists():
        print("No qrels file found.")
        return
    rows = []
    with qrels_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"\nqrels validation: {len(rows)} annotations")
    label_counts: dict[int, int] = defaultdict(int)
    query_counts: dict[str, int] = defaultdict(int)
    issues: list[str] = []
    for row in rows:
        label = row.get("label")
        if label not in (0, 1, 2, 3):
            issues.append(f"  Invalid label {label} for {row.get('query_id')}:{row.get('record_id')}")
        label_counts[int(label)] += 1
        query_counts[str(row["query_id"])] += 1
    print(f"  Queries covered: {len(query_counts)}")
    print(f"  Label distribution: {dict(sorted(label_counts.items()))}")
    if issues:
        print("  Issues:")
        for issue in issues:
            print(issue)
    else:
        print("  All labels valid (0-3).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive annotation CLI for retrieval candidates.")
    parser.add_argument("--pool", type=Path, default=Path("reports/eval/benchmark_pool.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("artifacts/benchmark_queries.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/normalized/combined_corpus.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/qrels.jsonl"))
    parser.add_argument("--annotator-id", default="ann_001")
    parser.add_argument("--resume", action="store_true", help="Skip already-annotated pairs")
    parser.add_argument("--stats-only", action="store_true", help="Print stats and exit")
    parser.add_argument("--validate", action="store_true", help="Validate existing qrels and exit")
    parser.add_argument("--limit-per-query", type=int, default=None,
                        help="Max pairs to annotate per query (use for focused sessions, e.g. 20)")
    parser.add_argument("--consensus-only", action="store_true",
                        help="Annotate only pairs appearing in 2+ retrieval variants")
    args = parser.parse_args(argv)

    if args.validate:
        validate_qrels(args.out)
        return 0

    pool = load_pool(args.pool)
    if not pool:
        print(f"Pool file not found or empty: {args.pool}")
        print("Run scripts/eval/run_retrieval_baselines.py and scripts/eval/build_benchmark_pool.py first.")
        return 1

    if args.stats_only:
        print_stats(args.out, pool)
        return 0

    queries = load_queries(args.queries)
    if not queries:
        print(f"Warning: No queries loaded from {args.queries}")

    print(f"Loading corpus from {args.corpus} ...", flush=True)
    corpus = load_corpus(args.corpus)
    print(f"  {len(corpus)} records loaded.")

    # Apply filters based on flags
    if args.consensus_only:
        pool = [p for p in pool if len(p.get("pooled_from", [])) >= 2]
        print(f"Consensus-only filter: {len(pool)} pairs remain.")

    if args.limit_per_query is not None:
        from collections import defaultdict
        per_query: dict[str, list[dict]] = defaultdict(list)
        for p in pool:
            per_query[p["query_id"]].append(p)
        pool = []
        for qid in sorted(per_query.keys()):
            pool.extend(per_query[qid][:args.limit_per_query])
        print(f"Limit-per-query={args.limit_per_query}: {len(pool)} pairs remain.")

    annotate(pool, queries, corpus, args.out, args.annotator_id, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
