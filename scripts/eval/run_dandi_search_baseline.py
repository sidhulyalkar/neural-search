"""Compare neural-search vs DANDI native search on benchmark queries.

Queries DANDI's public search API for each benchmark query, maps results
to dataset IDs, and compares rank positions against neural-search results.
This produces a head-to-head NDCG@10 comparison for the paper.

Usage
-----
    python scripts/eval/run_dandi_search_baseline.py \
        --queries data/eval/benchmark_queries.yaml \
        --out reports/eval/runs/dandi_baseline.jsonl

    # After running, compare with qrels:
    python scripts/eval/compute_bootstrap_ci.py \
        --qrels artifacts/qrels_silver.jsonl \
        --runs reports/eval/runs/dandi_baseline.jsonl \
        --allow-silver
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, parse, request

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DANDI_SEARCH_URL = "https://api.dandiarchive.org/api/dandisets/"
QUERIES_PATH = ROOT / "data" / "eval" / "benchmark_queries.yaml"
OUT_DIR = ROOT / "reports" / "eval" / "runs"
TOP_K = 20
RATE_LIMIT_DELAY = 0.5  # seconds between requests


def dandi_search(query_text: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Query DANDI search API and return ordered results."""
    params = parse.urlencode({
        "search": query_text,
        "page_size": top_k,
        "ordering": "-created",
    })
    url = f"{DANDI_SEARCH_URL}?{params}"
    req = request.Request(url, headers={"Accept": "application/json", "User-Agent": "neural-search-eval/1.0"})
    try:
        with request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data.get("results", [])
    except error.URLError as e:
        log.warning("DANDI API request failed: %s", e)
        return []
    except Exception as e:
        log.warning("DANDI API error: %s", e)
        return []


def dandiset_to_dataset_id(dandiset: dict[str, Any]) -> str:
    """Convert DANDI API result to our corpus dataset_id format."""
    identifier = dandiset.get("identifier", "")
    if identifier:
        # Normalize: DANDI:000026 → dataset:dandi:000026
        num = identifier.lstrip("DANDI:").lstrip("0") or "0"
        return f"dataset:dandi:{num.zfill(6)}"
    return f"dataset:dandi:unknown_{dandiset.get('id', '')}"


def run_dandi_baseline(queries: list[dict], top_k: int) -> dict[str, list[tuple[str, float]]]:
    """Run DANDI search for all queries, return {qid: [(dataset_id, rank_score)]}."""
    results: dict[str, list[tuple[str, float]]] = {}
    for i, q in enumerate(queries):
        qid = q["id"]
        text = q["query"]
        log.info("[%d/%d] %s: %s", i + 1, len(queries), qid, text[:60])
        hits = dandi_search(text, top_k)
        ranked = []
        for rank, hit in enumerate(hits, 1):
            did = dandiset_to_dataset_id(hit)
            score = 1.0 / rank  # reciprocal rank as proxy score
            ranked.append((did, score))
        results[qid] = ranked
        if i < len(queries) - 1:
            time.sleep(RATE_LIMIT_DELAY)
    return results


def write_run(results: dict[str, list[tuple[str, float]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for qid, ranked in results.items():
            for rank, (did, score) in enumerate(ranked, 1):
                f.write(json.dumps({
                    "query_id": qid,
                    "dataset_id": did,
                    "rank": rank,
                    "score": round(score, 6),
                    "system": "dandi_native_search",
                }) + "\n")
    log.info("DANDI baseline run written → %s", out_path)


def coverage_report(
    dandi_results: dict[str, list[tuple[str, float]]],
    neural_run_path: Path | None,
) -> None:
    """Print dataset overlap and rank position comparison."""
    print("\n=== DANDI Baseline Coverage ===")
    total_dandi = 0
    for qid, ranked in dandi_results.items():
        dandi_ids = {did for did, _ in ranked}
        total_dandi += len(dandi_ids)
        print(f"  {qid}: {len(ranked)} DANDI results")

    print(f"\nTotal DANDI results: {total_dandi} across {len(dandi_results)} queries")
    print(f"Avg results/query: {total_dandi / max(len(dandi_results), 1):.1f}")

    if neural_run_path and neural_run_path.exists():
        neural: dict[str, list[str]] = {}
        with neural_run_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                neural.setdefault(rec["query_id"], []).append(rec["dataset_id"])

        overlaps = []
        for qid in dandi_results:
            dandi_ids = {did for did, _ in dandi_results[qid]}
            neural_ids = set(neural.get(qid, [])[:20])
            overlap = len(dandi_ids & neural_ids)
            overlaps.append(overlap)

        avg_overlap = sum(overlaps) / max(len(overlaps), 1)
        print(f"\nOverlap (DANDI ∩ neural-search top-20): {avg_overlap:.1f} per query")
        print("Low overlap = neural-search surfaces different (possibly better) datasets")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "dandi_baseline.jsonl")
    parser.add_argument("--neural-run", type=Path, default=None,
                        help="Neural-search run file for overlap comparison")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print queries without calling the DANDI API")
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
        print("Dry run — not querying DANDI API.")
        return 0

    log.info("Querying DANDI search API (rate-limited to %.1fs between requests)…", RATE_LIMIT_DELAY)
    results = run_dandi_baseline(queries, args.top_k)
    write_run(results, args.out)
    coverage_report(results, args.neural_run)

    print(f"\nRun file: {args.out}")
    print(f"Compare: python scripts/eval/compute_bootstrap_ci.py "
          f"--qrels artifacts/qrels_silver.jsonl --runs {args.out} --allow-silver")
    return 0


if __name__ == "__main__":
    sys.exit(main())
