#!/usr/bin/env python3
"""Ablate s9 graph proximity: compare real graph vs neutral prior (0.3).

Runs benchmark queries through the scorer twice — once with the real graph,
once with graph=None (forcing 0.3 neutral prior) — and reports:
  - % of query-candidate pairs where graph_score (score_breakdown) differs
  - NDCG@10 with s9=0.3 vs real s9

Signal used: result.score_breakdown["graph_score"] from graph_context_score()
(not usefulness_score.graph_proximity, which requires query to be a graph node).

Performance note: the full real corpus (~873 records) is slow to score because
graph.model_dump() is called per-candidate in score_usefulness (pre-existing
issue). Use --fast-corpus to build a minimal corpus directly from normalized
JSONL (skips extract_dataset_labels), or --demo-only for the 26-record demo.

Usage:
    python scripts/ablate_graph_proximity.py --fast-corpus
    python scripts/ablate_graph_proximity.py --demo-only
    python scripts/ablate_graph_proximity.py --n-queries 10 --dry-run
    python scripts/ablate_graph_proximity.py --n-queries 20  # slow (~30min)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

BENCHMARK_PATH = Path("data/eval/benchmark_queries_real_corpus.yaml")
GRAPH_PATH = Path("data/graph/neural_search_graph.real_corpus.json")
REPORT_PATH = Path("reports/graph_ablation.json")
REAL_DANDI_PATH = Path("data/corpus/normalized/real_dandi.jsonl")


def _dcg(gains: list[float]) -> float:
    import math
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg(ranked_gains: list[float], ideal_gains: list[float], k: int = 10) -> float:
    dcg = _dcg(ranked_gains[:k])
    idcg = _dcg(sorted(ideal_gains, reverse=True)[:k])
    return dcg / idcg if idcg > 0 else 0.0


def _build_fast_corpus(jsonl_path: Path) -> list[dict]:
    """Build minimal search records from normalized JSONL without extract_dataset_labels.

    Skips the slow extraction step; builds flat dataset dicts that the search
    loop (via record.get('dataset', record)) can use directly.
    """
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            dataset_id = data.get("dataset_id") or data.get("source_id", "")
            if not dataset_id:
                continue

            def extract_labels(items: list) -> list[str]:
                result = []
                for item in items:
                    if isinstance(item, dict):
                        label = item.get("label") or item.get("id")
                        if label:
                            result.append(label)
                    elif isinstance(item, str):
                        result.append(item)
                return result

            dataset = {
                "id": dataset_id,
                "source": data.get("source", "unknown"),
                "source_id": data.get("source_id", dataset_id),
                "title": data.get("title", ""),
                "description": data.get("description", ""),
                "species": extract_labels(data.get("species", [])),
                "modalities": extract_labels(data.get("modalities", [])),
                "brain_regions": extract_labels(data.get("brain_regions", [])),
                "tasks": extract_labels(data.get("tasks", [])),
                "behaviors": extract_labels(data.get("behavioral_events", [])),
                "data_standards": extract_labels(data.get("data_standards", [])),
                "analysis_goals": extract_labels(data.get("analysis_goals", [])),
                "linked_paper_ids": data.get("linked_papers", []),
                "has_raw_data": data.get("usability_flags", {}).get("has_raw_data", True),
            }
            records.append({"dataset": dataset, "assets": [], "papers": [], "extraction": {}})
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Use only the 26-record demo seed corpus (fast but IDs not in real graph)",
    )
    parser.add_argument(
        "--fast-corpus",
        action="store_true",
        help="Build minimal corpus from normalized JSONL without extract_dataset_labels (~1s vs ~150s)",
    )
    parser.add_argument(
        "--n-records",
        type=int,
        default=None,
        help="Limit corpus to first N records (useful for fast ablation runs with --fast-corpus)",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        print(f"DRY RUN — would ablate {args.n_queries} queries against graph at {GRAPH_PATH}")
        return 0

    if not GRAPH_PATH.exists():
        print(f"ERROR: graph not found at {GRAPH_PATH}")
        return 1

    if not BENCHMARK_PATH.exists():
        print(f"ERROR: benchmark not found at {BENCHMARK_PATH}")
        return 1

    data = yaml.safe_load(BENCHMARK_PATH.read_text())
    queries = data.get("benchmark_queries", [])[:args.n_queries]

    from neural_search.search.core import search_datasets

    print("Loading corpus (once)…", flush=True)
    if args.demo_only:
        from neural_search.ingestion.demo_seed import build_demo_seed
        corpus = list(build_demo_seed())
        print(f"Demo corpus loaded: {len(corpus)} records (note: DEMO_* IDs not in real graph)", flush=True)
    elif args.fast_corpus:
        if not REAL_DANDI_PATH.exists():
            print(f"ERROR: fast corpus path not found at {REAL_DANDI_PATH}")
            return 1
        corpus = _build_fast_corpus(REAL_DANDI_PATH)
        if args.n_records:
            corpus = corpus[:args.n_records]
        print(f"Fast corpus loaded: {len(corpus)} real DANDI records (no extraction overhead)", flush=True)
    else:
        from neural_search.ingestion.demo_seed import build_combined_corpus
        corpus = build_combined_corpus()
        print(f"Corpus loaded: {len(corpus)} records", flush=True)

    retrieval_with_graph = {
        "graph": {"enabled": True, "path": str(GRAPH_PATH)},
    }
    retrieval_no_graph = {
        "graph": {"enabled": False},
    }

    pairs_changed = 0
    total_pairs = 0
    ndcg_with_graph: list[float] = []
    ndcg_without_graph: list[float] = []

    for q in queries:
        query_text = q["query"]

        resp_with = search_datasets(query_text, limit=10, retrieval_config=retrieval_with_graph, datasets=corpus)
        resp_without = search_datasets(query_text, limit=10, retrieval_config=retrieval_no_graph, datasets=corpus)

        expected_ids = set(q.get("expected_dataset_ids", []))
        gains_with, gains_without = [], []
        for r_with, r_without in zip(resp_with.results, resp_without.results, strict=False):
            # Use graph_score from score_breakdown — directly reflects graph_context_score
            # which IS affected by the T1-2 node-ID lookup fix (adds node: prefix).
            gs_with = r_with.score_breakdown.get("graph_score", 0.0)
            gs_without = r_without.score_breakdown.get("graph_score", 0.0)
            total_pairs += 1
            if abs(gs_with - gs_without) > 1e-6:
                pairs_changed += 1

            # Relevance: use canonical expected_dataset_ids from benchmark (not text heuristics)
            gain = 2.0 if (expected_ids and str(r_with.dataset_id) in expected_ids) else 0.0
            gains_with.append(gain)
            gain2 = 2.0 if (expected_ids and str(r_without.dataset_id) in expected_ids) else 0.0
            gains_without.append(gain2)

        # NDCG@10: each run self-normalized against its own ideal ordering
        ndcg_with_graph.append(_ndcg(gains_with, gains_with))
        ndcg_without_graph.append(_ndcg(gains_without, gains_without))

    pct_changed = 100 * pairs_changed / total_pairs if total_pairs > 0 else 0.0
    mean_ndcg_with = sum(ndcg_with_graph) / len(ndcg_with_graph) if ndcg_with_graph else 0.0
    mean_ndcg_without = sum(ndcg_without_graph) / len(ndcg_without_graph) if ndcg_without_graph else 0.0

    report = {
        "n_queries": len(queries),
        "total_pairs": total_pairs,
        "pairs_changed": pairs_changed,
        "pct_pairs_changed": round(pct_changed, 1),
        "mean_ndcg_with_graph": round(mean_ndcg_with, 4),
        "mean_ndcg_without_graph": round(mean_ndcg_without, 4),
        "ndcg_delta": round(mean_ndcg_with - mean_ndcg_without, 4),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if pct_changed < 10.0:
        print(f"\nWARNING: Only {pct_changed:.1f}% of pairs changed — s9 fix may not be working.")
        print("Investigate: are node IDs resolving in the real graph?")
    else:
        print(f"\ns9 fix confirmed: {pct_changed:.1f}% of pairs changed rank.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
