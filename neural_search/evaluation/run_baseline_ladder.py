"""Baseline ladder evaluation for comparing retrieval system configurations.

Runs benchmark queries with incrementally complex retrieval configurations
to measure the contribution of each component.

Usage:
    python -m neural_search.evaluation.run_baseline_ladder --suite demo_v02
    python -m neural_search.evaluation.run_baseline_ladder --suite demo_v02 --modes bm25,dense_only,full
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neural_search.evaluation.run_benchmark import (
    SUITE_CHOICES,
    EvaluationReport,
    benchmark_path_for_suite,
    output_dir_for_suite,
    run_full_benchmark,
)
from neural_search.search.core import load_retrieval_config

# Baseline ladder modes from simplest to most complex
LADDER_MODES = (
    "keyword",
    "bm25",
    "field_weighted_bm25",
    "dense_only",
    "bm25_dense_rrf",
    "plus_ontology",
    "plus_graph",
    "full_system",
)

LADDER_DESCRIPTIONS = {
    "keyword": "Exact keyword matching on text fields",
    "bm25": "Standard BM25 on concatenated text",
    "field_weighted_bm25": "BM25 with domain-specific field weights",
    "dense_only": "Dense embedding retrieval only",
    "bm25_dense_rrf": "BM25 + dense with Reciprocal Rank Fusion",
    "plus_ontology": "RRF + ontology matching signals",
    "plus_graph": "RRF + ontology + graph features",
    "full_system": "All scoring components enabled",
}


def _build_ladder_config(mode: str) -> dict[str, Any]:
    """Build retrieval config for a given ladder mode."""
    base = deepcopy(load_retrieval_config())
    weights = base.get("weights", {})

    # Zero out all weights first for controlled experiments
    for key in weights:
        weights[key] = 0.0

    if mode == "keyword":
        # Simple keyword matching through semantic with minimal processing
        weights["semantic"] = 1.0
        base["use_embeddings"] = False
        base["use_ontology"] = False
        base["use_graph"] = False
        base["penalties"] = {"modality_mismatch": 0.0, "exclusion_violation": 0.0}

    elif mode == "bm25":
        # Standard BM25 scoring
        weights["semantic"] = 1.0
        base["use_embeddings"] = False
        base["use_ontology"] = False
        base["use_graph"] = False
        base["field_weights"] = {
            "title": 1.0,
            "description": 1.0,
            "tasks": 1.0,
            "modalities": 1.0,
        }
        base["penalties"] = {"modality_mismatch": 0.0, "exclusion_violation": 0.0}

    elif mode == "field_weighted_bm25":
        # Field-weighted BM25 with domain-specific weights
        weights["semantic"] = 1.0
        base["use_embeddings"] = False
        base["use_ontology"] = False
        base["use_graph"] = False
        # Higher weights for structured fields
        base["field_weights"] = {
            "title": 1.5,
            "description": 1.0,
            "tasks": 2.0,
            "modalities": 2.0,
            "brain_regions": 1.5,
            "behavioral_events": 1.5,
        }
        base["penalties"] = {"modality_mismatch": 0.0, "exclusion_violation": 0.0}

    elif mode == "dense_only":
        # Dense embedding retrieval only
        weights["semantic"] = 1.0
        base["use_embeddings"] = True
        base["use_ontology"] = False
        base["use_graph"] = False
        base["penalties"] = {"modality_mismatch": 0.0, "exclusion_violation": 0.0}

    elif mode == "bm25_dense_rrf":
        # Hybrid BM25 + dense with RRF
        weights["semantic"] = 0.6
        weights["ontology"] = 0.0
        weights["behavior"] = 0.0
        weights["modality"] = 0.0
        base["use_embeddings"] = True
        base["use_ontology"] = False
        base["use_graph"] = False
        base["use_rrf"] = True
        base["penalties"] = {"modality_mismatch": 0.0, "exclusion_violation": 0.0}

    elif mode == "plus_ontology":
        # Add ontology matching to hybrid
        weights["semantic"] = 0.4
        weights["ontology"] = 0.25
        weights["behavior"] = 0.15
        weights["modality"] = 0.2
        base["use_embeddings"] = True
        base["use_ontology"] = True
        base["use_graph"] = False
        base["use_rrf"] = True
        base["penalties"] = {"modality_mismatch": 0.1, "exclusion_violation": 0.0}

    elif mode == "plus_graph":
        # Add graph features
        weights["semantic"] = 0.35
        weights["ontology"] = 0.20
        weights["behavior"] = 0.15
        weights["modality"] = 0.15
        if "graph" in base:
            for key in base.get("graph", {}).get("weights", {}):
                base["graph"]["weights"][key] = 0.15
        base["use_embeddings"] = True
        base["use_ontology"] = True
        base["use_graph"] = True
        base["use_rrf"] = True
        base["penalties"] = {"modality_mismatch": 0.1, "exclusion_violation": 0.0}

    elif mode == "full_system":
        # Restore default full system weights
        full_base = load_retrieval_config()
        return full_base

    else:
        raise ValueError(f"Unknown ladder mode: {mode}")

    base["weights"] = weights
    return base


@dataclass
class LadderResult:
    """Evaluation results for a single ladder mode."""

    mode: str
    description: str
    config: dict[str, Any]
    report: EvaluationReport
    latency_ms: float
    num_candidates: int


@dataclass
class BaselineLadderReport:
    """Complete baseline ladder comparison report."""

    generated_at: str
    suite: str
    modes: list[str]
    results: list[LadderResult]
    comparison: list[dict[str, Any]] = field(default_factory=list)
    delta_analysis: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    total_hard_negative_violations: int = 0


def run_baseline_ladder(
    suite: str,
    modes: list[str] | None = None,
) -> BaselineLadderReport:
    """Run benchmark with each ladder mode and collect results."""

    if modes is None:
        modes = list(LADDER_MODES)

    benchmark_path = benchmark_path_for_suite(suite)
    results: list[LadderResult] = []

    for mode in modes:
        config = _build_ladder_config(mode)
        start_time = time.perf_counter()
        report = run_full_benchmark(benchmark_path, suite=suite, retrieval_config=config)
        latency_ms = (time.perf_counter() - start_time) * 1000 / max(1, report.total_queries)

        # Count hard-negative violations
        violations = 0
        candidates = 0
        for q in report.queries:
            violations += len(q.hard_negative_violations)
            candidates += q.num_results

        results.append(
            LadderResult(
                mode=mode,
                description=LADDER_DESCRIPTIONS.get(mode, mode),
                config=config,
                report=report,
                latency_ms=round(latency_ms, 2),
                num_candidates=candidates,
            )
        )

    # Build comparison table
    comparison = []
    metric_keys = [
        "mean_precision_at_5",
        "mean_recall_at_10",
        "mean_mrr",
        "mean_ndcg_at_10",
    ]
    for metric in metric_keys:
        row = {"metric": metric}
        for result in results:
            row[result.mode] = getattr(result.report, metric, 0.0)
        comparison.append(row)

    # Add latency row
    latency_row = {"metric": "latency_ms"}
    for result in results:
        latency_row[result.mode] = result.latency_ms
    comparison.append(latency_row)

    # Add violation row
    violation_row = {"metric": "hard_negative_violations"}
    total_violations = 0
    for result in results:
        v_count = sum(len(q.hard_negative_violations) for q in result.report.queries)
        violation_row[result.mode] = v_count
        total_violations += v_count
    comparison.append(violation_row)

    # Compute delta analysis (improvement over previous mode)
    delta_analysis = []
    for i, result in enumerate(results):
        delta_row = {"mode": result.mode}
        if i == 0:
            delta_row["delta_precision"] = 0.0
            delta_row["delta_mrr"] = 0.0
            delta_row["delta_ndcg"] = 0.0
        else:
            prev = results[i - 1]
            delta_row["delta_precision"] = (
                result.report.mean_precision_at_5 - prev.report.mean_precision_at_5
            )
            delta_row["delta_mrr"] = result.report.mean_mrr - prev.report.mean_mrr
            delta_row["delta_ndcg"] = (
                result.report.mean_ndcg_at_10 - prev.report.mean_ndcg_at_10
            )
        delta_analysis.append(delta_row)

    # Summary
    summary: dict[str, Any] = {
        "best_precision_mode": max(results, key=lambda r: r.report.mean_precision_at_5).mode,
        "best_mrr_mode": max(results, key=lambda r: r.report.mean_mrr).mode,
        "best_ndcg_mode": max(results, key=lambda r: r.report.mean_ndcg_at_10).mode,
        "full_vs_baseline_delta": {
            "precision": (
                results[-1].report.mean_precision_at_5 - results[0].report.mean_precision_at_5
                if len(results) >= 2
                else 0.0
            ),
            "mrr": (
                results[-1].report.mean_mrr - results[0].report.mean_mrr
                if len(results) >= 2
                else 0.0
            ),
            "ndcg": (
                results[-1].report.mean_ndcg_at_10 - results[0].report.mean_ndcg_at_10
                if len(results) >= 2
                else 0.0
            ),
        },
        "biggest_delta": max(delta_analysis, key=lambda d: d.get("delta_precision", 0))
        if delta_analysis
        else {},
    }

    return BaselineLadderReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        suite=suite,
        modes=modes,
        results=results,
        comparison=comparison,
        delta_analysis=delta_analysis,
        summary=summary,
        total_hard_negative_violations=total_violations,
    )


def generate_ladder_markdown(report: BaselineLadderReport) -> str:
    """Generate Markdown ladder report."""

    lines = [
        "# Neural Search Baseline Ladder Report",
        "",
        f"Generated: {report.generated_at}",
        f"Suite: {report.suite}",
        f"Modes: {', '.join(report.modes)}",
        "",
        "## Overview",
        "",
        "This report compares retrieval quality across incrementally complex system configurations.",
        "",
    ]

    # Comparison table
    lines.extend([
        "## Metric Comparison",
        "",
        "| Metric |" + " | ".join(report.modes) + " |",
        "| --- |" + " | ".join(["---"] * len(report.modes)) + " |",
    ])

    for row in report.comparison:
        metric = row["metric"]
        cells = [metric]
        for mode in report.modes:
            value = row.get(mode, 0.0)
            if metric in ("latency_ms", "hard_negative_violations"):
                cells.append(f"{value:.1f}" if isinstance(value, float) else str(value))
            else:
                cells.append(f"{value:.1%}" if value <= 1.0 else f"{value:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Delta analysis
    lines.extend([
        "## Incremental Improvement (Delta)",
        "",
        "| Mode | P@5 Delta | MRR Delta | NDCG Delta |",
        "| --- | --- | --- | --- |",
    ])

    for delta in report.delta_analysis:
        lines.append(
            f"| {delta['mode']} | {delta.get('delta_precision', 0):+.1%} | "
            f"{delta.get('delta_mrr', 0):+.3f} | {delta.get('delta_ndcg', 0):+.3f} |"
        )
    lines.append("")

    # Summary
    if report.summary:
        lines.extend([
            "## Summary",
            "",
            f"- **Best Precision@5**: {report.summary.get('best_precision_mode', 'N/A')}",
            f"- **Best MRR**: {report.summary.get('best_mrr_mode', 'N/A')}",
            f"- **Best NDCG@10**: {report.summary.get('best_ndcg_mode', 'N/A')}",
            "",
        ])

        full_vs_base = report.summary.get("full_vs_baseline_delta", {})
        lines.extend([
            "### Full System vs Baseline Improvement",
            "",
            f"- Precision@5: {full_vs_base.get('precision', 0):+.1%}",
            f"- MRR: {full_vs_base.get('mrr', 0):+.3f}",
            f"- NDCG@10: {full_vs_base.get('ndcg', 0):+.3f}",
            "",
        ])

    # Hard-negative violations
    lines.extend([
        "## Hard-Negative Violations",
        "",
        f"Total violations across all modes: {report.total_hard_negative_violations}",
        "",
    ])

    # Per-mode details
    lines.extend(["## Per-Mode Details", ""])
    for result in report.results:
        r = result.report
        lines.extend([
            f"### {result.mode}",
            "",
            f"*{result.description}*",
            "",
            f"- Precision@5: {r.mean_precision_at_5:.1%}",
            f"- Recall@10: {r.mean_recall_at_10:.1%}",
            f"- MRR: {r.mean_mrr:.3f}",
            f"- NDCG@10: {r.mean_ndcg_at_10:.3f}",
            f"- Latency: {result.latency_ms:.1f} ms/query",
            f"- Candidates: {result.num_candidates}",
            "",
        ])

    return "\n".join(lines)


def generate_ladder_json(report: BaselineLadderReport) -> str:
    """Generate JSON ladder report."""

    def serialize_result(result: LadderResult) -> dict[str, Any]:
        return {
            "mode": result.mode,
            "description": result.description,
            "config": result.config,
            "report": asdict(result.report),
            "latency_ms": result.latency_ms,
            "num_candidates": result.num_candidates,
        }

    payload = {
        "generated_at": report.generated_at,
        "suite": report.suite,
        "modes": report.modes,
        "results": [serialize_result(r) for r in report.results],
        "comparison": report.comparison,
        "delta_analysis": report.delta_analysis,
        "summary": report.summary,
        "total_hard_negative_violations": report.total_hard_negative_violations,
    }
    return json.dumps(payload, indent=2, default=str)


def write_ladder_reports(
    report: BaselineLadderReport,
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Write ladder reports to files."""

    out = output_dir or output_dir_for_suite(report.suite)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "baseline_ladder_results.md"
    json_path = out / "baseline_ladder_results.json"

    md_path.write_text(generate_ladder_markdown(report), encoding="utf-8")
    json_path.write_text(generate_ladder_json(report), encoding="utf-8")

    return {"markdown": str(md_path), "json": str(json_path)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        prog="python -m neural_search.evaluation.run_baseline_ladder",
        description="Run baseline ladder evaluation comparing retrieval configurations.",
    )
    parser.add_argument(
        "--suite",
        choices=[s for s in SUITE_CHOICES if s != "all"],
        default="demo_v02",
        help="Benchmark suite to run.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="all",
        help="Comma-separated ladder modes or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output JSON to stdout instead of writing files.",
    )
    args = parser.parse_args(argv)

    # Parse modes
    if args.modes == "all":
        modes = list(LADDER_MODES)
    else:
        modes = [m.strip() for m in args.modes.split(",")]
        for mode in modes:
            if mode not in LADDER_MODES:
                print(f"Unknown mode: {mode}")
                print(f"Available modes: {', '.join(LADDER_MODES)}")
                return 1

    print(f"Running baseline ladder for suite={args.suite}")
    print(f"Modes: {', '.join(modes)}")
    print()

    report = run_baseline_ladder(args.suite, modes)

    if args.json_only:
        print(generate_ladder_json(report))
        return 0

    paths = write_ladder_reports(report, args.output_dir)

    # Print summary
    print("=" * 70)
    print("BASELINE LADDER RESULTS")
    print("=" * 70)
    print()

    # Print comparison table
    headers = ["Metric"] + modes
    col_widths = [max(len(h), 18) for h in headers]
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=False))
    print(header_row)
    print("-" * len(header_row))

    for row in report.comparison:
        cells = [row["metric"]]
        for mode in modes:
            value = row.get(mode, 0.0)
            if row["metric"] in ("latency_ms", "hard_negative_violations"):
                cells.append(f"{value:.1f}" if isinstance(value, float) else str(value))
            else:
                cells.append(f"{value:.1%}" if value <= 1.0 else f"{value:.3f}")
        print(" | ".join(c.ljust(w) for c, w in zip(cells, col_widths, strict=False)))
    print()

    # Summary
    print("SUMMARY")
    print("-" * 50)
    print(f"  Best Precision@5: {report.summary.get('best_precision_mode', 'N/A')}")
    print(f"  Best MRR: {report.summary.get('best_mrr_mode', 'N/A')}")
    print(f"  Best NDCG@10: {report.summary.get('best_ndcg_mode', 'N/A')}")
    print()

    full_vs_base = report.summary.get("full_vs_baseline_delta", {})
    print("Full System vs Baseline Improvement:")
    print(f"  Precision@5: {full_vs_base.get('precision', 0):+.1%}")
    print(f"  MRR: {full_vs_base.get('mrr', 0):+.3f}")
    print(f"  NDCG@10: {full_vs_base.get('ndcg', 0):+.3f}")
    print()

    print("=" * 70)
    print("Reports written to:")
    print(f"  Markdown: {paths['markdown']}")
    print(f"  JSON: {paths['json']}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
