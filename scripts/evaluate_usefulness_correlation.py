#!/usr/bin/env python3
"""Evaluate usefulness_score correlation with benchmark relevance labels.

Runs real_corpus benchmark queries through search_datasets(), then measures
Spearman correlation between usefulness_score.total_score and binary relevance
(result matches expected_tasks or expected_modalities_any from the benchmark).

Usage:
    python scripts/evaluate_usefulness_correlation.py
    python scripts/evaluate_usefulness_correlation.py --n-queries 10
    python scripts/evaluate_usefulness_correlation.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from scipy.stats import spearmanr

BENCHMARK_PATH = Path("data/eval/benchmark_queries_real_corpus.yaml")
REPORT_PATH = Path("reports/usefulness_correlation_v09.json")


def _result_is_relevant(result_dump: dict, query: dict) -> bool | None:
    """Return True if result matches query signals, False if hard-negative, None if unknown."""
    matched_text = " ".join(
        result_dump.get("why_matched", []) + result_dump.get("matched_terms", [])
    ).lower()

    for hn_mod in query.get("hard_negative_modalities", []):
        if hn_mod.lower() in matched_text:
            return False

    for hn_spec in query.get("hard_negative_species", []):
        if hn_spec.lower() in matched_text:
            return False

    expected_signals = query.get("expected_tasks", []) + query.get("expected_modalities_any", [])
    if not expected_signals:
        return None

    for term in expected_signals:
        if term.lower() in matched_text:
            return True

    return False


def load_benchmark_queries(n: int) -> list[dict]:
    data = yaml.safe_load(BENCHMARK_PATH.read_text())
    queries = data.get("benchmark_queries", [])
    return queries[:n]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate usefulness score correlation")
    parser.add_argument("--n-queries", type=int, default=20,
                        help="Number of benchmark queries to run (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print query list without running search")
    args = parser.parse_args(argv)

    queries = load_benchmark_queries(args.n_queries)

    if args.dry_run:
        print(f"DRY RUN — would run {len(queries)} queries:")
        for q in queries:
            print(f"  {q['id']}: {q['query'][:70]}")
        return 0

    from neural_search.search import search_datasets

    sys_scores: list[float] = []
    binary_labels: list[int] = []
    query_breakdown: list[dict] = []

    for q in queries:
        print(f"Running {q['id']}: {q['query'][:60]}...", flush=True)
        response = search_datasets(q["query"])

        relevant_scores: list[float] = []
        irrelevant_scores: list[float] = []

        for result in response.results:
            if result.usefulness_score is None:
                continue
            relevance = _result_is_relevant(result.model_dump(), q)
            if relevance is None:
                continue
            score = result.usefulness_score["total_score"]
            sys_scores.append(score)
            binary_labels.append(int(relevance))
            (relevant_scores if relevance else irrelevant_scores).append(score)

        query_breakdown.append({
            "query_id": q["id"],
            "mean_relevant": round(sum(relevant_scores) / len(relevant_scores), 4) if relevant_scores else None,
            "mean_irrelevant": round(sum(irrelevant_scores) / len(irrelevant_scores), 4) if irrelevant_scores else None,
            "n_relevant": len(relevant_scores),
            "n_irrelevant": len(irrelevant_scores),
        })

    n = len(sys_scores)
    print(f"\nTotal (score, label) pairs collected: {n}")

    if n < 5:
        print("Too few pairs for correlation. Try --n-queries 30 or check the corpus has matches.")
        return 1

    corr, pval = spearmanr(sys_scores, binary_labels)
    rel_all = [s for s, r in zip(sys_scores, binary_labels) if r == 1]
    irrel_all = [s for s, r in zip(sys_scores, binary_labels) if r == 0]

    print(f"\nSpearman r = {corr:.4f}  (p = {pval:.4f})")
    if rel_all:
        print(f"Mean usefulness score [relevant]:   {sum(rel_all)/len(rel_all):.4f}  (n={len(rel_all)})")
    if irrel_all:
        print(f"Mean usefulness score [irrelevant]: {sum(irrel_all)/len(irrel_all):.4f}  (n={len(irrel_all)})")

    report = {
        "n_queries_run": len(queries),
        "n_pairs": n,
        "spearman_r": round(corr, 4),
        "spearman_p": round(pval, 4),
        "mean_score_relevant": round(sum(rel_all) / len(rel_all), 4) if rel_all else None,
        "mean_score_irrelevant": round(sum(irrel_all) / len(irrel_all), 4) if irrel_all else None,
        "query_breakdown": query_breakdown,
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
