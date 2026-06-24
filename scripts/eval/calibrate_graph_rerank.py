#!/usr/bin/env python3
"""Calibrate graph reranking weights against canonical qrels.

This script does not run dense retrieval. It starts from an existing
``hybrid_rrf`` run, extracts graph feature components for those candidates,
sweeps conservative graph-weight profiles, and reports whether any graph boost
improves qrels-backed NDCG/MRR over the unmodified RRF ranking.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from neural_search.graph.search_features import (  # noqa: E402
    DEFAULT_GRAPH_SEARCH_WEIGHTS,
    compute_graph_features_for_result,
    load_graph_if_exists,
)
from scripts.eval.compute_ndcg_from_qrels import (  # noqa: E402
    K_NDCG,
    K_RECALL,
    _load_qrels,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from scripts.eval.run_ablation_ladder import _graph_context_dict  # noqa: E402

DEFAULT_QUERIES = Path("data/eval/benchmark_queries_canonical.yaml")
DEFAULT_QRELS = Path("data/qrels/qrels.canonical.trec")
DEFAULT_RUN = Path("reports/eval/runs/hybrid_rrf.jsonl")
DEFAULT_GRAPH = Path("data/graph/neural_search_graph.real_corpus.json")
DEFAULT_OUT = Path("reports/eval/graph_weight_calibration.json")
DEFAULT_MD = Path("reports/eval/graph_weight_calibration.md")

GLOBAL_WEIGHTS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]

WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "default": dict(DEFAULT_GRAPH_SEARCH_WEIGHTS),
    "no_degree": {**DEFAULT_GRAPH_SEARCH_WEIGHTS, "degree": 0.0},
    "query_match_only": {
        **{key: 0.0 for key in DEFAULT_GRAPH_SEARCH_WEIGHTS},
        "requirement_match": 0.01,
        "analysis_requirement_coverage": 0.015,
        "task_match": 0.03,
        "modality_match": 0.02,
        "species_match": 0.02,
        "taxon_match": 0.01,
        "brain_region_match": 0.02,
    },
    "relationship_only": {
        **{key: 0.0 for key in DEFAULT_GRAPH_SEARCH_WEIGHTS},
        "relationship_edge": 0.012,
        "reanalysis_edge": 0.018,
    },
    "conservative_relationship": {
        **DEFAULT_GRAPH_SEARCH_WEIGHTS,
        "linked_paper": 0.01,
        "analysis_affordance": 0.01,
        "degree": 0.0,
        "relationship_edge": 0.004,
        "reanalysis_edge": 0.006,
    },
}


def load_queries(path: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = payload.get("benchmark_queries", []) if isinstance(payload, dict) else payload
    return {str(row.get("id")): row for row in rows if isinstance(row, dict) and row.get("id")}


def load_scored_run(path: Path) -> dict[str, list[tuple[str, float]]]:
    run: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qid = str(rec["query_id"])
        rank = int(rec.get("rank", 0))
        rid = str(rec.get("record_id") or rec.get("dataset_id") or rec.get("doc_id"))
        score = float(rec.get("score", 1.0 / max(rank, 1)))
        run[qid].append((rank, rid, score))
    return {
        qid: [(rid, score) for rank, rid, score in sorted(rows)]
        for qid, rows in run.items()
    }


def evaluate_rankings(
    qrels: dict[str, dict[str, int]],
    rankings: dict[str, list[str]],
) -> dict[str, float]:
    judged = sorted(set(qrels) & set(rankings))
    if not judged:
        return {"ndcg@10": 0.0, "mrr": 0.0, "recall@50": 0.0, "judged_queries": 0}
    ndcgs: list[float] = []
    mrrs: list[float] = []
    recalls: list[float] = []
    for qid in judged:
        ranked = [(i + 1, rid) for i, rid in enumerate(rankings[qid])]
        ndcgs.append(ndcg_at_k(qrels[qid], ranked, K_NDCG))
        mrrs.append(mrr(qrels[qid], ranked))
        recalls.append(recall_at_k(qrels[qid], ranked, K_RECALL))
    return {
        "ndcg@10": sum(ndcgs) / len(ndcgs),
        "mrr": sum(mrrs) / len(mrrs),
        "recall@50": sum(recalls) / len(recalls),
        "judged_queries": len(judged),
    }


def graph_components(graph: Any, record_id: str, query: dict[str, Any]) -> dict[str, float]:
    features = compute_graph_features_for_result(graph, record_id, _graph_context_dict(query))
    matched = features.get("matched_query_context", {})
    req = features.get("requirement_matches", {})
    return {
        "linked_paper": float(min(len(features.get("linked_papers", [])), 3)),
        "analysis_affordance": float(min(len(features.get("analysis_affordances", [])), 3)),
        "requirement_match": float(min(sum(len(values) for values in req.values()), 5)),
        "analysis_requirement_coverage": float(min(sum(1 for values in req.values() if values), 4)),
        "task_match": float(len(matched.get("tasks", []))),
        "modality_match": float(len(matched.get("modalities", []))),
        "species_match": float(len(matched.get("species", []))),
        "taxon_match": float(len(matched.get("taxon_groups", []))),
        "brain_region_match": float(len(matched.get("brain_regions", []))),
        "degree": float(min(int(features.get("graph_degree", 0)), 10)),
        "relationship_edge": float(min(len(features.get("relationship_edges", [])), 5)),
        "reanalysis_edge": float(min(len(features.get("reanalysis_edges", [])), 5)),
    }


def component_score(components: dict[str, float], weights: dict[str, float]) -> float:
    return min(sum(components.get(key, 0.0) * weight for key, weight in weights.items()), 0.25)


def calibrate(
    *,
    qrels: dict[str, dict[str, int]],
    queries: dict[str, dict[str, Any]],
    run: dict[str, list[tuple[str, float]]],
    graph: Any,
    profiles: dict[str, dict[str, float]] | None = None,
    global_weights: list[float] | None = None,
) -> dict[str, Any]:
    profiles = profiles or WEIGHT_PROFILES
    global_weights = global_weights or GLOBAL_WEIGHTS
    base_rankings = {qid: [rid for rid, _ in rows] for qid, rows in run.items()}
    baseline = evaluate_rankings(qrels, base_rankings)

    component_cache: dict[tuple[str, str], dict[str, float]] = {}
    for qid, rows in run.items():
        query = queries.get(qid, {"id": qid, "query": ""})
        for rid, _ in rows:
            component_cache[(qid, rid)] = graph_components(graph, rid, query)

    rows: list[dict[str, Any]] = []
    for profile_name, weights in profiles.items():
        graph_scores = {
            key: component_score(components, weights)
            for key, components in component_cache.items()
        }
        for global_weight in global_weights:
            rankings: dict[str, list[str]] = {}
            for qid, candidates in run.items():
                rescored = [
                    (rid, base_score + (global_weight * graph_scores.get((qid, rid), 0.0)))
                    for rid, base_score in candidates
                ]
                rescored.sort(key=lambda item: (-item[1], item[0]))
                rankings[qid] = [rid for rid, _ in rescored]
            metrics = evaluate_rankings(qrels, rankings)
            rows.append(
                {
                    "profile": profile_name,
                    "global_weight": global_weight,
                    **metrics,
                    "ndcg_delta_vs_rrf": metrics["ndcg@10"] - baseline["ndcg@10"],
                    "mrr_delta_vs_rrf": metrics["mrr"] - baseline["mrr"],
                    "recall_delta_vs_rrf": metrics["recall@50"] - baseline["recall@50"],
                }
            )

    rows.sort(
        key=lambda row: (
            -float(row["ndcg@10"]),
            -float(row["mrr"]),
            -float(row["recall@50"]),
            str(row["profile"]),
            float(row["global_weight"]),
        )
    )
    best = rows[0] if rows else {}
    balanced_rows = [
        row
        for row in rows
        if row["ndcg_delta_vs_rrf"] > 0 and row["mrr_delta_vs_rrf"] >= 0
    ]
    best_balanced = balanced_rows[0] if balanced_rows else {}
    recommendation = (
        "enable_calibrated_graph_boost"
        if best_balanced
        else "keep_hybrid_rrf_as_quality_baseline"
    )
    return {
        "baseline_system": "hybrid_rrf",
        "baseline_metrics": baseline,
        "best": best,
        "best_balanced": best_balanced,
        "recommendation": recommendation,
        "candidates": rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    baseline = report["baseline_metrics"]
    lines = [
        "# Graph Weight Calibration",
        "",
        f"Baseline: `hybrid_rrf` NDCG@10={baseline['ndcg@10']:.4f}, "
        f"MRR={baseline['mrr']:.4f}, Recall@50={baseline['recall@50']:.4f}",
        f"Recommendation: `{report['recommendation']}`",
        (
            "Balanced choice: "
            f"`{report['best_balanced'].get('profile', 'none')}` at "
            f"{report['best_balanced'].get('global_weight', 0.0):.3f}"
            if report.get("best_balanced")
            else "Balanced choice: none found"
        ),
        "",
        "| Rank | Profile | Global weight | NDCG@10 | ΔNDCG | MRR | ΔMRR | Recall@50 | ΔRecall |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(report["candidates"][:15], start=1):
        lines.append(
            f"| {rank} | `{row['profile']}` | {row['global_weight']:.3f} "
            f"| {row['ndcg@10']:.4f} | {row['ndcg_delta_vs_rrf']:+.4f} "
            f"| {row['mrr']:.4f} | {row['mrr_delta_vs_rrf']:+.4f} "
            f"| {row['recall@50']:.4f} | {row['recall_delta_vs_rrf']:+.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args(argv)

    graph = load_graph_if_exists(args.graph)
    if graph is None:
        raise SystemExit(f"Graph not found: {args.graph}")
    report = calibrate(
        qrels=_load_qrels(args.qrels),
        queries=load_queries(args.queries),
        run=load_scored_run(args.run),
        graph=graph,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.md)
    print(f"Graph calibration -> {args.out}")
    print(f"Markdown -> {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
