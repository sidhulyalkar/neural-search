"""Ablation evaluation runner for comparing retrieval head configurations.

Runs benchmark queries with different scoring weight configurations to identify
which retrieval heads contribute most to performance.

Usage:
    python -m neural_search.evaluation.run_ablation --suite demo_v02 --modes all
    python -m neural_search.evaluation.run_ablation --suite adversarial --modes lexical_only,ontology_only,hybrid_default
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
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

ABLATION_MODES = (
    "lexical_only",
    "ontology_only",
    "semantic_only",
    "ontology_plus_semantic",
    "hybrid_default",
    "hybrid_no_provenance",
    "hybrid_no_analysis_fit",
    "hybrid_no_negative_constraints",
)

# Component-level ablation: disable one scoring component at a time
COMPONENT_ABLATION_MODES = (
    "no_ontology",
    "no_behavior",
    "no_modality",
    "no_semantic",
    "no_affordance",
    "no_metadata",
    "no_readiness",
    "no_paper_confidence",
    "no_graph_context",
    "no_hard_negative",
)


def _build_config(mode: str) -> dict[str, Any]:
    """Build retrieval config for a given ablation mode."""

    base = deepcopy(load_retrieval_config())
    weights = base.get("weights", {})

    if mode == "lexical_only":
        for key in weights:
            weights[key] = 0.0
        weights["semantic"] = 1.0  # semantic includes keyword/lexical
        base["penalties"] = {"modality_mismatch": 0.0, "missing_required_field": 0.0}

    elif mode == "ontology_only":
        for key in weights:
            weights[key] = 0.0
        weights["ontology"] = 0.4
        weights["behavior"] = 0.3
        weights["modality"] = 0.3
        base["penalties"] = {"modality_mismatch": 0.1, "missing_required_field": 0.0}

    elif mode == "semantic_only":
        for key in weights:
            weights[key] = 0.0
        weights["semantic"] = 1.0
        base["penalties"] = {"modality_mismatch": 0.0, "missing_required_field": 0.0}

    elif mode == "ontology_plus_semantic":
        for key in weights:
            weights[key] = 0.0
        weights["ontology"] = 0.25
        weights["behavior"] = 0.20
        weights["modality"] = 0.15
        weights["semantic"] = 0.40
        base["penalties"] = {"modality_mismatch": 0.1, "missing_required_field": 0.0}

    elif mode == "hybrid_default":
        # Use default weights from config
        pass

    elif mode == "hybrid_no_provenance":
        weights["paper_confidence"] = 0.0

    elif mode == "hybrid_no_analysis_fit":
        # Analysis fit is scored through metadata match; zero out readiness
        weights["readiness"] = 0.0

    elif mode == "hybrid_no_negative_constraints":
        base["penalties"]["modality_mismatch"] = 0.0
        base["penalties"]["exclusion_violation"] = 0.0

    # Component-level ablation modes
    elif mode == "no_ontology":
        weights["ontology"] = 0.0

    elif mode == "no_behavior":
        weights["behavior"] = 0.0

    elif mode == "no_modality":
        weights["modality"] = 0.0

    elif mode == "no_semantic":
        weights["semantic"] = 0.0

    elif mode == "no_affordance":
        weights["affordance"] = 0.0

    elif mode == "no_metadata":
        weights["metadata"] = 0.0

    elif mode == "no_readiness":
        weights["readiness"] = 0.0

    elif mode == "no_paper_confidence":
        weights["paper_confidence"] = 0.0

    elif mode == "no_graph_context":
        # If graph weights exist, zero them out
        if "graph" in base:
            for key in base.get("graph", {}).get("weights", {}):
                base["graph"]["weights"][key] = 0.0

    elif mode == "no_hard_negative":
        base["penalties"]["modality_mismatch"] = 0.0
        base["penalties"]["exclusion_violation"] = 0.0

    else:
        raise ValueError(f"Unknown ablation mode: {mode}")

    base["weights"] = weights
    return base


@dataclass
class AblationResult:
    """Evaluation results for a single ablation mode."""

    mode: str
    config: dict[str, Any]
    report: EvaluationReport


@dataclass
class ComponentImpact:
    """Impact of ablating a single component."""

    component: str
    baseline_precision: float
    ablated_precision: float
    delta_precision: float
    baseline_mrr: float
    ablated_mrr: float
    delta_mrr: float
    is_critical: bool  # Impact > 10%
    recommendation: str


@dataclass
class AblationReport:
    """Complete ablation comparison report."""

    generated_at: str
    suite: str
    modes: list[str]
    results: list[AblationResult]
    comparison: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    component_impacts: list[ComponentImpact] = field(default_factory=list)


def compute_component_impact(
    baseline_result: AblationResult,
    ablated_result: AblationResult,
    critical_threshold: float = 0.10,
) -> ComponentImpact:
    """Compute the impact of ablating a component.

    Args:
        baseline_result: Results with all components enabled
        ablated_result: Results with one component disabled
        critical_threshold: Delta threshold to mark as critical (default 10%)

    Returns:
        ComponentImpact with delta metrics and recommendation
    """
    baseline_p5 = baseline_result.report.mean_precision_at_5
    ablated_p5 = ablated_result.report.mean_precision_at_5
    delta_p5 = ablated_p5 - baseline_p5

    baseline_mrr = baseline_result.report.mean_mrr
    ablated_mrr = ablated_result.report.mean_mrr
    delta_mrr = ablated_mrr - baseline_mrr

    # Component name from ablation mode
    component = ablated_result.mode.replace("no_", "")

    # Determine criticality and recommendation
    is_critical = abs(delta_p5) >= critical_threshold or abs(delta_mrr) >= critical_threshold

    if delta_p5 < -critical_threshold:
        recommendation = "KEEP - critical for search quality"
    elif delta_p5 < -0.05:
        recommendation = "KEEP - contributes meaningfully"
    elif delta_p5 > 0.05:
        recommendation = "REMOVE - may be hurting performance"
    else:
        recommendation = "OPTIONAL - marginal impact"

    return ComponentImpact(
        component=component,
        baseline_precision=baseline_p5,
        ablated_precision=ablated_p5,
        delta_precision=delta_p5,
        baseline_mrr=baseline_mrr,
        ablated_mrr=ablated_mrr,
        delta_mrr=delta_mrr,
        is_critical=is_critical,
        recommendation=recommendation,
    )


def run_component_ablation(
    suite: str,
    components: list[str] | None = None,
) -> AblationReport:
    """Run component-level ablation study.

    Compares each component ablation against the baseline (hybrid_default)
    to measure individual component contribution.

    Args:
        suite: Benchmark suite to run
        components: List of components to ablate (default: all)

    Returns:
        AblationReport with component impact analysis
    """
    if components is None:
        components = list(COMPONENT_ABLATION_MODES)

    # Run baseline first
    baseline_modes = ["hybrid_default"] + list(components)
    report = run_ablation(suite, baseline_modes)

    # Find baseline result
    baseline_result = next(
        (r for r in report.results if r.mode == "hybrid_default"),
        None
    )

    if baseline_result is None:
        return report

    # Compute impact for each component
    impacts = []
    for result in report.results:
        if result.mode.startswith("no_"):
            impact = compute_component_impact(baseline_result, result)
            impacts.append(impact)

    # Sort by impact magnitude (most impactful first)
    impacts.sort(key=lambda x: abs(x.delta_precision), reverse=True)

    report.component_impacts = impacts

    # Add component summary to summary
    report.summary["component_ranking"] = [
        {
            "component": imp.component,
            "delta_precision": f"{imp.delta_precision:+.1%}",
            "is_critical": imp.is_critical,
            "recommendation": imp.recommendation,
        }
        for imp in impacts
    ]

    return report


def run_ablation(
    suite: str,
    modes: list[str] | None = None,
) -> AblationReport:
    """Run benchmark with each ablation mode and collect results."""

    if modes is None:
        modes = list(ABLATION_MODES)

    benchmark_path = benchmark_path_for_suite(suite)
    results: list[AblationResult] = []

    for mode in modes:
        config = _build_config(mode)
        report = run_full_benchmark(benchmark_path, suite=suite, retrieval_config=config)
        results.append(AblationResult(mode=mode, config=config, report=report))

    # Build comparison table
    comparison = []
    metric_keys = [
        "mean_precision_at_5",
        "mean_label_recall_at_10",
        "mean_mrr",
        "mean_ndcg_at_10",
    ]
    for metric in metric_keys:
        row = {"metric": metric}
        for result in results:
            row[result.mode] = getattr(result.report, metric, 0.0)
        comparison.append(row)

    # Compute summary: which mode performed best on each metric
    summary: dict[str, Any] = {"best_by_metric": {}}
    for metric in metric_keys:
        best_mode = max(results, key=lambda r: getattr(r.report, metric, 0.0))
        summary["best_by_metric"][metric] = {
            "mode": best_mode.mode,
            "value": getattr(best_mode.report, metric, 0.0),
        }

    # Identify failure mode differences
    failure_analysis: list[dict[str, Any]] = []
    for result in results:
        failed_queries = [q for q in result.report.queries if q.why_failed]
        failure_analysis.append({
            "mode": result.mode,
            "failed_count": len(failed_queries),
            "failed_queries": [q.query_id for q in failed_queries[:5]],
        })
    summary["failure_analysis"] = failure_analysis

    return AblationReport(
        generated_at=datetime.now(UTC).isoformat(),
        suite=suite,
        modes=modes,
        results=results,
        comparison=comparison,
        summary=summary,
    )


def generate_ablation_markdown(report: AblationReport) -> str:
    """Generate Markdown ablation report."""

    lines = [
        "# Neural Search Ablation Report",
        "",
        f"Generated: {report.generated_at}",
        f"Suite: {report.suite}",
        f"Modes: {', '.join(report.modes)}",
        "",
    ]

    # Component impact analysis (if available)
    if report.component_impacts:
        lines.extend([
            "## Component Impact Analysis",
            "",
            "Impact of ablating each scoring component (compared to baseline):",
            "",
            "| Component | P@5 Delta | MRR Delta | Critical | Recommendation |",
            "| --- | --- | --- | --- | --- |",
        ])

        for impact in report.component_impacts:
            critical_marker = "⚠️" if impact.is_critical else ""
            lines.append(
                f"| {impact.component} | {impact.delta_precision:+.1%} | "
                f"{impact.delta_mrr:+.3f} | {critical_marker} | {impact.recommendation} |"
            )
        lines.append("")

        # Summary sections by impact
        critical_components = [c for c in report.component_impacts if c.is_critical and c.delta_precision < 0]
        marginal_components = [c for c in report.component_impacts if not c.is_critical]
        negative_components = [c for c in report.component_impacts if c.delta_precision > 0]

        if critical_components:
            lines.extend([
                "### Critical Components (>10% impact)",
                "",
            ])
            for c in critical_components:
                lines.append(f"- **{c.component}**: {c.delta_precision:+.1%} precision")
            lines.append("")

        if marginal_components:
            lines.extend([
                "### Marginal Components (<5% impact)",
                "",
            ])
            for c in marginal_components:
                lines.append(f"- {c.component}: {c.delta_precision:+.1%} precision")
            lines.append("")

        if negative_components:
            lines.extend([
                "### Potentially Harmful Components (negative impact)",
                "",
            ])
            for c in negative_components:
                lines.append(f"- {c.component}: {c.delta_precision:+.1%} (consider removing)")
            lines.append("")

    # Original comparison table
    lines.extend([
        "## Metric Comparison",
        "",
    ])

    # Build comparison table
    if report.comparison:
        headers = ["Metric"] + report.modes
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in report.comparison:
            cells = [row["metric"]]
            for mode in report.modes:
                value = row.get(mode, 0.0)
                cells.append(f"{value:.1%}" if value <= 1.0 else f"{value:.3f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Best by metric
    if report.summary.get("best_by_metric"):
        lines.extend(["## Best Mode by Metric", ""])
        for metric, info in report.summary["best_by_metric"].items():
            lines.append(f"- **{metric}**: {info['mode']} ({info['value']:.1%})")
        lines.append("")

    # Failure analysis
    if report.summary.get("failure_analysis"):
        lines.extend(["## Failure Analysis", ""])
        lines.append("| Mode | Failed Queries | Examples |")
        lines.append("| --- | --- | --- |")
        for entry in report.summary["failure_analysis"]:
            examples = ", ".join(entry["failed_queries"][:3])
            lines.append(f"| {entry['mode']} | {entry['failed_count']} | {examples} |")
        lines.append("")

    # Per-mode detailed results
    lines.extend(["## Per-Mode Details", ""])
    for result in report.results:
        r = result.report
        lines.extend([
            f"### {result.mode}",
            "",
            f"- Precision@5: {r.mean_precision_at_5:.1%}",
            f"- Label Recall@10: {r.mean_label_recall_at_10:.1%}",
            f"- MRR: {r.mean_mrr:.3f}",
            f"- NDCG@10: {r.mean_ndcg_at_10:.3f}",
            "",
        ])

    return "\n".join(lines)


def generate_ablation_json(report: AblationReport) -> str:
    """Generate JSON ablation report."""

    def serialize_result(result: AblationResult) -> dict[str, Any]:
        return {
            "mode": result.mode,
            "config": result.config,
            "report": asdict(result.report),
        }

    payload = {
        "generated_at": report.generated_at,
        "suite": report.suite,
        "modes": report.modes,
        "results": [serialize_result(r) for r in report.results],
        "comparison": report.comparison,
        "summary": report.summary,
    }
    return json.dumps(payload, indent=2, default=str)


def write_ablation_reports(
    report: AblationReport,
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Write ablation reports to files."""

    out = output_dir or output_dir_for_suite(report.suite)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "ablation_report.md"
    json_path = out / "ablation_report.json"

    md_path.write_text(generate_ablation_markdown(report), encoding="utf-8")
    json_path.write_text(generate_ablation_json(report), encoding="utf-8")

    return {"markdown": str(md_path), "json": str(json_path)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        prog="python -m neural_search.evaluation.run_ablation",
        description="Run ablation evaluation comparing retrieval configurations.",
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
        help="Comma-separated ablation modes or 'all'.",
    )
    parser.add_argument(
        "--component-ablation",
        action="store_true",
        help="Run component-level ablation (disable one component at a time).",
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

    # Component ablation mode
    if args.component_ablation:
        print(f"Running component ablation for suite={args.suite}")
        print(f"Components: {', '.join(COMPONENT_ABLATION_MODES)}")
        print()
        report = run_component_ablation(args.suite)

        if args.json_only:
            print(generate_ablation_json(report))
            return 0

        paths = write_ablation_reports(report, args.output_dir)

        # Print component impact summary
        print("=" * 70)
        print("COMPONENT ABLATION RESULTS")
        print("=" * 70)
        print()

        if report.component_impacts:
            print("Component Impact (vs baseline):")
            print("-" * 50)
            print(f"{'Component':<20} {'P@5 Delta':<12} {'MRR Delta':<12} {'Recommendation':<25}")
            print("-" * 70)
            for impact in report.component_impacts:
                print(
                    f"{impact.component:<20} {impact.delta_precision:+.1%}          "
                    f"{impact.delta_mrr:+.3f}        {impact.recommendation}"
                )
            print()

        print("=" * 70)
        print("Reports written to:")
        print(f"  Markdown: {paths['markdown']}")
        print(f"  JSON: {paths['json']}")
        print("=" * 70)
        return 0

    # Parse modes
    all_modes = list(ABLATION_MODES) + list(COMPONENT_ABLATION_MODES)
    if args.modes == "all":
        modes = list(ABLATION_MODES)
    else:
        modes = [m.strip() for m in args.modes.split(",")]
        for mode in modes:
            if mode not in all_modes:
                print(f"Unknown mode: {mode}")
                print(f"Available modes: {', '.join(all_modes)}")
                return 1

    print(f"Running ablation for suite={args.suite} with modes={modes}")
    print()

    report = run_ablation(args.suite, modes)

    if args.json_only:
        print(generate_ablation_json(report))
        return 0

    paths = write_ablation_reports(report, args.output_dir)

    print("=" * 70)
    print("NEURAL SEARCH ABLATION REPORT")
    print("=" * 70)
    print()
    print("COMPARISON SUMMARY")
    print("-" * 50)
    print()

    # Print comparison table
    if report.comparison:
        headers = ["Metric"] + report.modes
        col_widths = [max(len(h), 20) for h in headers]
        header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=False))
        print(header_row)
        print("-" * len(header_row))
        for row in report.comparison:
            cells = [row["metric"]]
            for mode in report.modes:
                value = row.get(mode, 0.0)
                cells.append(f"{value:.1%}" if value <= 1.0 else f"{value:.3f}")
            print(" | ".join(c.ljust(w) for c, w in zip(cells, col_widths, strict=False)))
    print()

    if report.summary.get("best_by_metric"):
        print("BEST BY METRIC")
        print("-" * 50)
        for metric, info in report.summary["best_by_metric"].items():
            print(f"  {metric}: {info['mode']} ({info['value']:.1%})")
        print()

    print("=" * 70)
    print("Reports written to:")
    print(f"  Markdown: {paths['markdown']}")
    print(f"  JSON: {paths['json']}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
