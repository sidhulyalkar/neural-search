"""Run neural-search on all benchmark queries and write a TREC-style run file.

This script produces the neural-search run file needed to compute NDCG/MRR
against silver qrels and to compare with the DANDI baseline and SPECTER2.

Output format (JSONL, one row per result):
  {"query_id": "q001", "dataset_id": "...", "rank": 1, "score": 95.2, "system": "neural_search"}

Usage
-----
    # Full run (loads live corpus — takes ~30s)
    python scripts/eval/run_neural_search_baseline.py \
        --queries data/eval/benchmark_queries_v2.yaml \
        --out reports/eval/runs/neural_search.jsonl

    # Quick smoke test (dry-run shows queries without searching)
    python scripts/eval/run_neural_search_baseline.py --dry-run --queries data/eval/benchmark_queries_v2.yaml

    # Use a specific retrieval config override
    python scripts/eval/run_neural_search_baseline.py \
        --queries data/eval/benchmark_queries_v2.yaml \
        --config data/config/retrieval.yaml \
        --out reports/eval/runs/neural_search.jsonl

After running, compare with baseline:
    python scripts/eval/compute_bootstrap_ci.py \
        --qrels artifacts/qrels_silver.jsonl \
        --runs reports/eval/runs/neural_search.jsonl \
        --allow-silver
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
OUT_DIR = ROOT / "reports" / "eval" / "runs"
TOP_K = 20
SYSTEM_NAME = "neural_search"


def _load_corpus():
    from neural_search.ingestion.demo_seed import build_combined_corpus
    t0 = time.time()
    corpus = build_combined_corpus()
    log.info("Corpus loaded: %d records in %.1fs", len(corpus), time.time() - t0)
    return corpus


def run_search(
    queries: list[dict],
    corpus: list,
    top_k: int,
    retrieval_config: dict | None,
) -> dict[str, list[dict]]:
    """Run neural-search on all queries; return {qid: [{rank, dataset_id, score}]}."""
    from neural_search.search.core import search_datasets

    results: dict[str, list[dict]] = {}
    for i, q in enumerate(queries):
        qid = q["id"]
        text = q["query"]
        log.info("[%d/%d] %s: %s", i + 1, len(queries), qid, text[:70])
        t0 = time.time()
        try:
            resp = search_datasets(
                query=text,
                datasets=corpus,
                limit=top_k,
                retrieval_config=retrieval_config,
            )
            ranked = [
                {
                    "query_id": qid,
                    "dataset_id": str(r.dataset_id),
                    "rank": rank,
                    "score": round(r.score, 4),
                    "system": SYSTEM_NAME,
                }
                for rank, r in enumerate(resp.results, 1)
            ]
        except Exception as e:
            log.warning("Search failed for %s: %s", qid, e)
            ranked = []
        results[qid] = ranked
        elapsed = time.time() - t0
        log.debug("  %s: %d results in %.2fs", qid, len(ranked), elapsed)

    return results


def write_run(results: dict[str, list[dict]], out_path: Path) -> int:
    """Write results to JSONL; return total rows written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for ranked in results.values():
            for row in ranked:
                f.write(json.dumps(row) + "\n")
                n += 1
    log.info("Run written: %d rows → %s", n, out_path)
    return n


def print_summary(results: dict[str, list[dict]]) -> None:
    total = sum(len(v) for v in results.values())
    empty = sum(1 for v in results.values() if not v)
    print("\n=== Neural-Search Run Summary ===")
    print(f"  Queries:          {len(results)}")
    print(f"  Total results:    {total}")
    print(f"  Empty queries:    {empty}")
    print(f"  Avg results/q:    {total / max(len(results), 1):.1f}")
    if results:
        samples = list(results.items())[:3]
        print("\n  Sample top-1 results:")
        for qid, ranked in samples:
            top = ranked[0]["dataset_id"] if ranked else "(none)"
            score = ranked[0]["score"] if ranked else 0
            print(f"    {qid}: {top} ({score:.1f})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH,
                        help="YAML benchmark queries file")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "neural_search.jsonl",
                        help="Output JSONL run file")
    parser.add_argument("--config", type=Path, default=None,
                        help="Retrieval config YAML (default: data/config/retrieval.yaml)")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="Results per query (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print queries without running search")
    args = parser.parse_args(argv)

    if not args.queries.exists():
        print(f"Queries file not found: {args.queries}", file=sys.stderr)
        return 1

    raw = yaml.safe_load(args.queries.read_text())
    queries = raw.get("benchmark_queries", [])
    log.info("Loaded %d queries from %s", len(queries), args.queries)

    if args.dry_run:
        for q in queries[:5]:
            print(f"  {q['id']}: {q['query']}")
        print(f"  ... ({len(queries)} total queries)")
        print("Dry run — not running search.")
        return 0

    retrieval_config: dict | None = None
    if args.config and args.config.exists():
        retrieval_config = yaml.safe_load(args.config.read_text())
        log.info("Using retrieval config: %s", args.config)

    corpus = _load_corpus()
    results = run_search(queries, corpus, args.top_k, retrieval_config)
    write_run(results, args.out)
    print_summary(results)

    print(f"\nRun file: {args.out}")
    print(f"Compare: python scripts/eval/compute_bootstrap_ci.py "
          f"--qrels artifacts/qrels_silver.jsonl --runs {args.out} --allow-silver")
    return 0


if __name__ == "__main__":
    sys.exit(main())
