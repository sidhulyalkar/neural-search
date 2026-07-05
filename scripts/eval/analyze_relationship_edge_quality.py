#!/usr/bin/env python3
"""Analyze whether graph relationship edges help or hurt reranking."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from neural_search.graph.search_features import (  # noqa: E402
    compute_graph_features_for_result,
    load_graph_if_exists,
)
from scripts.eval.compute_ndcg_from_qrels import _load_qrels, _load_run  # noqa: E402

DEFAULT_QRELS = Path("data/qrels/qrels.canonical.trec")
DEFAULT_GRAPH = Path("data/graph/neural_search_graph.real_corpus.json")
DEFAULT_BASE_RUN = Path("reports/eval/runs/hybrid_rrf.jsonl")
DEFAULT_GRAPH_RUN = Path("reports/eval/runs/hybrid_graph.jsonl")
DEFAULT_OUT = Path("reports/eval/relationship_edge_quality.json")
DEFAULT_MD = Path("reports/eval/relationship_edge_quality.md")


def rank_map(run: dict[str, list[tuple[int, str]]]) -> dict[str, dict[str, int]]:
    return {qid: {rid: rank for rank, rid in rows} for qid, rows in run.items()}


def edge_types_for_result(graph: Any, record_id: str) -> list[str]:
    features = compute_graph_features_for_result(graph, record_id)
    edge_types = [
        str(edge.get("edge_type"))
        for edge in [
            *features.get("relationship_edges", []),
            *features.get("reanalysis_edges", []),
        ]
        if edge.get("edge_type")
    ]
    return sorted(dict.fromkeys(edge_types)) or ["no_relationship_edge"]


def analyze_edge_quality(
    *,
    qrels: dict[str, dict[str, int]],
    base_run: dict[str, list[tuple[int, str]]],
    graph_run: dict[str, list[tuple[int, str]]],
    graph: Any,
    top_k: int = 10,
) -> dict[str, Any]:
    base_ranks = rank_map(base_run)
    edge_stats: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = {
        "helpful_promotions": [],
        "harmful_promotions": [],
    }
    total_promotions = 0
    judged_promotions = 0

    for qid, graph_rows in graph_run.items():
        q_qrels = qrels.get(qid, {})
        for graph_rank, rid in graph_rows[:top_k]:
            base_rank = base_ranks.get(qid, {}).get(rid, 10**6)
            if graph_rank >= base_rank:
                continue
            total_promotions += 1
            grade = q_qrels.get(rid)
            if grade is not None:
                judged_promotions += 1
            edge_types = edge_types_for_result(graph, rid)
            for edge_type in edge_types:
                edge_stats[edge_type]["promotions"] += 1
                if grade is not None:
                    edge_stats[edge_type]["judged_promotions"] += 1
                    edge_stats[edge_type]["grade_sum"] += grade
                    if grade >= 2:
                        edge_stats[edge_type]["helpful_promotions"] += 1
                    else:
                        edge_stats[edge_type]["harmful_promotions"] += 1
            if grade is None:
                continue
            bucket = "helpful_promotions" if grade >= 2 else "harmful_promotions"
            if len(examples[bucket]) < 25:
                examples[bucket].append(
                    {
                        "query_id": qid,
                        "record_id": rid,
                        "relevance": grade,
                        "base_rank": None if base_rank == 10**6 else base_rank,
                        "graph_rank": graph_rank,
                        "edge_types": edge_types,
                    }
                )

    edge_rows: list[dict[str, Any]] = []
    for edge_type, stats in edge_stats.items():
        judged = int(stats["judged_promotions"])
        helpful = int(stats["helpful_promotions"])
        harmful = int(stats["harmful_promotions"])
        edge_rows.append(
            {
                "edge_type": edge_type,
                "promotions": int(stats["promotions"]),
                "judged_promotions": judged,
                "helpful_promotions": helpful,
                "harmful_promotions": harmful,
                "helpful_rate": helpful / judged if judged else 0.0,
                "mean_relevance": stats["grade_sum"] / judged if judged else 0.0,
            }
        )
    edge_rows.sort(
        key=lambda row: (
            -int(row["judged_promotions"]),
            float(row["helpful_rate"]),
            str(row["edge_type"]),
        )
    )
    return {
        "top_k": top_k,
        "total_promotions_into_top_k": total_promotions,
        "judged_promotions_into_top_k": judged_promotions,
        "edge_quality": edge_rows,
        "examples": examples,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Relationship Edge Quality",
        "",
        f"- Top-K analyzed: {report['top_k']}",
        f"- Promotions into top-K: {report['total_promotions_into_top_k']}",
        f"- Judged promotions into top-K: {report['judged_promotions_into_top_k']}",
        "",
        "| Edge type | Promotions | Judged | Helpful | Harmful | Helpful rate | Mean relevance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["edge_quality"]:
        lines.append(
            f"| `{row['edge_type']}` | {row['promotions']} | {row['judged_promotions']} "
            f"| {row['helpful_promotions']} | {row['harmful_promotions']} "
            f"| {row['helpful_rate']:.3f} | {row['mean_relevance']:.3f} |"
        )
    for title, key in [
        ("Helpful Promotion Examples", "helpful_promotions"),
        ("Harmful Promotion Examples", "harmful_promotions"),
    ]:
        lines.extend([
            "",
            f"## {title}",
            "",
            "| Query | Record | Relevance | Base rank | Graph rank | Edge types |",
            "|---|---|---:|---:|---:|---|",
        ])
        for ex in report["examples"][key]:
            edge_types = ", ".join(f"`{edge}`" for edge in ex["edge_types"])
            lines.append(
                f"| `{ex['query_id']}` | `{ex['record_id']}` | {ex['relevance']} "
                f"| {ex['base_rank'] or 'not top-100'} | {ex['graph_rank']} | {edge_types} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--base-run", type=Path, default=DEFAULT_BASE_RUN)
    parser.add_argument("--graph-run", type=Path, default=DEFAULT_GRAPH_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    graph = load_graph_if_exists(args.graph)
    if graph is None:
        raise SystemExit(f"Graph not found: {args.graph}")
    report = analyze_edge_quality(
        qrels=_load_qrels(args.qrels),
        base_run=_load_run(args.base_run),
        graph_run=_load_run(args.graph_run),
        graph=graph,
        top_k=args.top_k,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.md)
    print(f"Relationship edge quality -> {args.out}")
    print(f"Markdown -> {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

