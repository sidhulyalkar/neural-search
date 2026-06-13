#!/usr/bin/env python3
"""Pool candidate records from retrieval run files for annotation.

Prioritizes candidates appearing in more variants (consensus) and earlier ranks.
This ordering allows annotators to start with the most informative pairs.

Usage:
    python scripts/eval/build_benchmark_pool.py \
        --runs-dir reports/eval/runs \
        --out reports/eval/benchmark_pool.jsonl \
        --depth 50

Output JSONL schema:
    {"query_id": "q_0001", "record_id": "dandi:000003",
     "pooled_from": ["bm25", "usefulness"], "min_rank": 3,
     "priority": 2, "status": "needs_annotation"}
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pool retrieval candidates for annotation.")
    parser.add_argument("--runs-dir", type=Path, default=Path("reports/eval/runs"))
    parser.add_argument("--out", type=Path, default=Path("reports/eval/benchmark_pool.jsonl"))
    parser.add_argument("--depth", type=int, default=50,
                        help="Max rank to include from each run file")
    args = parser.parse_args(argv)

    # pool[query_id][record_id] = {variant: min_rank}
    pool: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))

    run_files = sorted(args.runs_dir.glob("*.jsonl")) if args.runs_dir.exists() else []
    for run_file in run_files:
        variant = run_file.stem
        with run_file.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rank = int(row.get("rank", 10 ** 9))
                if rank <= args.depth:
                    qid = str(row["query_id"])
                    rid = str(row["record_id"])
                    existing_rank = pool[qid][rid].get(variant, rank)
                    pool[qid][rid][variant] = min(existing_rank, rank)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0

    with args.out.open("w", encoding="utf-8") as handle:
        for query_id, records in sorted(pool.items()):
            # Build list with priority score for sorting
            entries = []
            for record_id, variant_ranks in records.items():
                n_variants = len(variant_ranks)
                min_rank = min(variant_ranks.values())
                # Priority: more variants = higher priority; lower rank = higher priority
                # Sort key: (-n_variants, min_rank) ascending
                entries.append((n_variants, min_rank, record_id, variant_ranks))

            # Sort: consensus pairs first (more variants), then by rank
            entries.sort(key=lambda x: (-x[0], x[1]))

            for n_variants, min_rank, record_id, variant_ranks in entries:
                handle.write(
                    json.dumps({
                        "query_id": query_id,
                        "record_id": record_id,
                        "pooled_from": sorted(variant_ranks.keys()),
                        "min_rank": min_rank,
                        "priority": n_variants,
                        "status": "needs_annotation",
                    }) + "\n"
                )
                total_pairs += 1

    result = {"run_files": len(run_files), "queries": len(pool), "total_pairs": total_pairs}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
