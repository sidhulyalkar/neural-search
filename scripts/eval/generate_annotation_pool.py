"""Generate an annotation pool from neural-search results for human qrel labelling.

Runs neural-search on benchmark queries, then creates a balanced pool of
(query, dataset) pairs for annotation with annotate_candidates.py.

The pool is stratified across:
  - Top-5 results per query (likely relevant, easy to annotate)
  - Results at rank 6-15 (borderline — most valuable for NDCG calibration)
  - 1-2 low-scoring results per query (negative examples)

Output format is JSONL compatible with annotate_candidates.py --pool.

Usage
-----
    # Generate pool (loads full corpus, ~30s)
    python scripts/eval/generate_annotation_pool.py \
        --queries data/eval/benchmark_queries_v2.yaml \
        --out reports/eval/annotation_pool.jsonl

    # Dry-run: show pool stats without running search
    python scripts/eval/generate_annotation_pool.py --dry-run

    # Then annotate (interactive CLI):
    python scripts/eval/annotate_candidates.py \
        --pool reports/eval/annotation_pool.jsonl \
        --queries reports/eval/annotation_queries.jsonl \
        --out artifacts/qrels_human.jsonl

"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

QUERIES_PATH = ROOT / "data" / "eval" / "benchmark_queries_v2.yaml"
OUT_POOL = ROOT / "reports" / "eval" / "annotation_pool.jsonl"
OUT_QUERIES = ROOT / "reports" / "eval" / "annotation_queries.jsonl"
TOP_K_SEARCH = 20
SAMPLE_TOP = 5       # always include top-5 per query
SAMPLE_BORDER = 8    # include ranks 6–13 (borderline)
SAMPLE_NEG = 2       # include 2 low-rank negatives


def _load_corpus():
    from neural_search.ingestion.demo_seed import build_combined_corpus
    t0 = time.time()
    corpus = build_combined_corpus()
    log.info("Corpus: %d records in %.1fs", len(corpus), time.time() - t0)
    return corpus


def _search_one(query_text: str, corpus: list, retrieval_config: dict | None) -> list[dict]:
    from neural_search.search.core import search_datasets
    resp = search_datasets(
        query=query_text,
        datasets=corpus,
        limit=TOP_K_SEARCH,
        retrieval_config=retrieval_config,
    )
    return [
        {
            "dataset_id": str(r.dataset_id),
            "source": r.source,
            "score": round(r.score, 2),
            "why_matched": r.why_matched[:3],
        }
        for r in resp.results
    ]


def build_pool(
    queries: list[dict],
    corpus: list,
    retrieval_config: dict | None,
) -> tuple[list[dict], list[dict]]:
    """Return (pool_rows, query_rows) for writing."""
    pool_rows: list[dict] = []
    query_rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for i, q in enumerate(queries):
        qid = q["id"]
        text = q["query"]
        log.info("[%d/%d] %s", i + 1, len(queries), qid)

        query_rows.append({"query_id": qid, "query_text": text, **{
            k: v for k, v in q.items() if k not in ("id", "query")
        }})

        try:
            results = _search_one(text, corpus, retrieval_config)
        except Exception as e:
            log.warning("Search failed %s: %s", qid, e)
            continue

        n = len(results)

        def _add(
            rank: int,
            reason: str,
            query_id: str = qid,
            result_rows: list[dict] = results,
            max_rank: int = n,
        ) -> None:
            if rank >= max_rank:
                return
            r = result_rows[rank]
            key = (query_id, r["dataset_id"])
            if key in seen:
                return
            seen.add(key)
            pool_rows.append({
                "query_id": query_id,
                "dataset_id": r["dataset_id"],
                "source": r.get("source", ""),
                "rank": rank + 1,
                "score": r["score"],
                "why_matched": r.get("why_matched", []),
                "pool_reason": reason,
            })

        for rank in range(min(SAMPLE_TOP, n)):
            _add(rank, "top")
        for rank in range(SAMPLE_TOP, min(SAMPLE_TOP + SAMPLE_BORDER, n)):
            _add(rank, "borderline")
        for rank in range(max(n - SAMPLE_NEG, SAMPLE_TOP + SAMPLE_BORDER), n):
            _add(rank, "negative")

    return pool_rows, query_rows


def write_pool(pool_rows: list[dict], pool_path: Path) -> None:
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with pool_path.open("w") as f:
        for row in pool_rows:
            f.write(json.dumps(row) + "\n")
    log.info("Pool: %d pairs → %s", len(pool_rows), pool_path)


def write_queries(query_rows: list[dict], queries_path: Path) -> None:
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    with queries_path.open("w") as f:
        for row in query_rows:
            f.write(json.dumps(row) + "\n")
    log.info("Queries: %d → %s", len(query_rows), queries_path)


def print_pool_stats(pool_rows: list[dict]) -> None:
    by_reason: dict[str, int] = {}
    for row in pool_rows:
        r = row["pool_reason"]
        by_reason[r] = by_reason.get(r, 0) + 1
    unique_queries = len({r["query_id"] for r in pool_rows})
    print("\n=== Annotation Pool Summary ===")
    print(f"  Total pairs:   {len(pool_rows)}")
    print(f"  Unique queries: {unique_queries}")
    for reason, count in sorted(by_reason.items()):
        print(f"  {reason:12s}: {count}")
    print(f"\nEstimated annotation time: ~{len(pool_rows) * 30 // 60} minutes")
    print("  (assuming 30s/pair)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--out", type=Path, default=OUT_POOL)
    parser.add_argument("--out-queries", type=Path, default=OUT_QUERIES)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit to first N queries (for testing)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.queries.exists():
        print(f"Queries file not found: {args.queries}", file=sys.stderr)
        return 1

    raw = yaml.safe_load(args.queries.read_text())
    queries = raw.get("benchmark_queries", [])
    if args.max_queries:
        queries = queries[:args.max_queries]
    log.info("Loaded %d queries", len(queries))

    if args.dry_run:
        for q in queries[:5]:
            print(f"  {q['id']}: {q['query'][:70]}")
        print(f"  ... ({len(queries)} total)")
        print(f"\nEstimated pool size: ~{len(queries) * (SAMPLE_TOP + SAMPLE_BORDER // 2 + SAMPLE_NEG)} pairs")
        print("Dry run — not running search.")
        return 0

    retrieval_config: dict | None = None
    if args.config and args.config.exists():
        retrieval_config = yaml.safe_load(args.config.read_text())

    corpus = _load_corpus()
    pool_rows, query_rows = build_pool(queries, corpus, retrieval_config)

    write_pool(pool_rows, args.out)
    write_queries(query_rows, args.out_queries)
    print_pool_stats(pool_rows)

    print("\nNext step — annotate:")
    print("  python scripts/eval/annotate_candidates.py \\")
    print(f"    --pool {args.out} \\")
    print(f"    --queries {args.out_queries} \\")
    print("    --out artifacts/qrels_human.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
