"""Build a multi-strategy candidate pool that includes concept-rerank results.

Extends build_benchmark_pool.py by also running the concept-memory reranker
when the field-state artifacts are present, and adding those results as a
`concept_rerank` strategy alongside BM25 and usefulness.

Output JSONL schema (same as build_benchmark_pool.py):
    {
      "query_id": "q_0001",
      "record_id": "dandi:000003",
      "pooled_from": ["bm25", "concept_rerank", "usefulness"],
      "min_rank": 3,
      "priority": 3,
      "status": "needs_annotation"
    }

Usage:
    python scripts/eval/sample_candidate_pool.py \\
        --queries artifacts/benchmark_queries.jsonl \\
        --runs-dir reports/eval/runs \\
        --out reports/eval/benchmark_pool_with_concept.jsonl \\
        --depth 50 \\
        [--root .]  # root for field-state artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from scripts.eval.benchmark_schema import BenchmarkQueryV1


def _load_queries(path: Path) -> list[BenchmarkQueryV1]:
    if not path.exists():
        return []
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            q = BenchmarkQueryV1.model_validate(json.loads(line))
            queries.append(q)
        except Exception as e:
            print(f"  [WARN] Skipping invalid query: {e}", file=sys.stderr)
    return queries


def _run_concept_rerank(
    queries: list[BenchmarkQueryV1],
    depth: int,
    root: Path | None,
) -> dict[str, list[tuple[int, str]]]:
    """Return {query_id: [(rank, record_id), ...]} for concept-rerank strategy.

    Returns an empty dict if field-state artifacts are not present.
    """
    try:
        from neural_search.field_state.concept_memory.reranker import (
            rerank_from_artifacts,
        )
    except ImportError:
        print("  [INFO] concept_memory not available — skipping concept_rerank strategy")  # noqa: E501
        return {}

    results: dict[str, list[tuple[int, str]]] = {}
    for q in queries:
        try:
            reranked = rerank_from_artifacts(
                query=q.query_text,
                limit=depth,
                root=root,
            )
        except Exception as e:
            print(
                f"  [WARN] concept_rerank failed for {q.query_id}: {e}",
                file=sys.stderr,
            )
            results[q.query_id] = []
            continue

        ranked = [(rank + 1, r.dataset_id) for rank, r in enumerate(reranked)]
        results[q.query_id] = ranked

    return results


def _load_existing_runs(
    runs_dir: Path, depth: int
) -> dict[str, dict[str, dict[str, int]]]:
    """pool[query_id][record_id][variant] = min_rank"""
    pool: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    run_files = sorted(runs_dir.glob("*.jsonl")) if runs_dir.exists() else []
    for run_file in run_files:
        variant = run_file.stem
        with run_file.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rank = int(row.get("rank", 10**9))
                if rank <= depth:
                    qid = str(row["query_id"])
                    rid = str(row["record_id"])
                    existing = pool[qid][rid].get(variant, rank)
                    pool[qid][rid][variant] = min(existing, rank)
    return pool


def build_pool(  # noqa: C901
    queries_path: Path,
    runs_dir: Path,
    out_path: Path,
    depth: int = 50,
    root: Path | None = None,
    skip_concept: bool = False,
) -> dict:
    """Build the multi-strategy pool and write to out_path.

    Returns a summary dict.
    """
    queries = _load_queries(queries_path)
    if not queries:
        print("[WARN] No queries loaded — pool will be empty")

    # Load existing run files
    pool = _load_existing_runs(runs_dir, depth)

    # Add concept-rerank strategy
    concept_results: dict[str, list[tuple[int, str]]] = {}
    if not skip_concept:
        concept_results = _run_concept_rerank(queries, depth, root)

    for q in queries:
        if q.query_id not in concept_results:
            continue
        for rank, record_id in concept_results[q.query_id]:
            existing = pool[q.query_id][record_id].get("concept_rerank", rank)
            pool[q.query_id][record_id]["concept_rerank"] = min(existing, rank)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0

    tmp = out_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for query_id, records in sorted(pool.items()):
            entries = []
            for record_id, variant_ranks in records.items():
                n_variants = len(variant_ranks)
                min_rank = min(variant_ranks.values())
                entries.append((n_variants, min_rank, record_id, variant_ranks))

            entries.sort(key=lambda x: (-x[0], x[1]))

            for n_variants, min_rank, record_id, variant_ranks in entries:
                fh.write(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "record_id": record_id,
                            "pooled_from": sorted(variant_ranks.keys()),
                            "min_rank": min_rank,
                            "priority": n_variants,
                            "status": "needs_annotation",
                        }
                    )
                    + "\n"
                )
                total_pairs += 1

    tmp.replace(out_path)

    strategies: set[str] = set()
    for records in pool.values():
        for v_ranks in records.values():
            strategies.update(v_ranks.keys())

    summary = {
        "queries_loaded": len(queries),
        "queries_in_pool": len(pool),
        "total_pairs": total_pairs,
        "strategies": sorted(strategies),
        "out": str(out_path),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build multi-strategy candidate pool (concept-rerank aware)"
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("artifacts/benchmark_queries.jsonl"),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("reports/eval/runs"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/eval/benchmark_pool.jsonl"),
    )
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for field-state artifacts (default: CWD)",
    )
    parser.add_argument(
        "--skip-concept",
        action="store_true",
        help="Skip concept-rerank strategy (fall back to run files only)",
    )
    args = parser.parse_args(argv)

    summary = build_pool(
        queries_path=args.queries,
        runs_dir=args.runs_dir,
        out_path=args.out,
        depth=args.depth,
        root=args.root,
        skip_concept=args.skip_concept,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
