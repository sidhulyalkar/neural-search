"""Unified experiment report generator.

Combines results from all evaluation experiments into a single report.

Usage:
    python -m neural_search.evaluation.unified_report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def load_json_report(path: Path) -> dict[str, Any] | None:
    """Load a JSON report if it exists."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def generate_unified_report(
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate unified report from available experiment outputs."""
    out = reports_dir or REPORTS_DIR

    # Try to load each report
    baseline_ladder = load_json_report(out / "baseline_ladder_results.json")
    hard_negative = load_json_report(out / "hard_negative_report.json")
    affordance = load_json_report(out / "affordance_validation_report.json")
    pairing = load_json_report(out / "cross_dataset_pairing_report.json")
    robustness = load_json_report(out / "metadata_robustness_report.json")

    # Build unified summary
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports_found": [],
        "reports_missing": [],
    }

    # Check which reports exist
    report_names = {
        "baseline_ladder": baseline_ladder,
        "hard_negative": hard_negative,
        "affordance_validation": affordance,
        "cross_dataset_pairing": pairing,
        "metadata_robustness": robustness,
    }

    for name, data in report_names.items():
        if data is not None:
            summary["reports_found"].append(name)
        else:
            summary["reports_missing"].append(name)

    # Extract key metrics from each report
    metrics: dict[str, Any] = {}

    if baseline_ladder:
        summary_data = baseline_ladder.get("summary", {})
        metrics["baseline_ladder"] = {
            "best_precision_mode": summary_data.get("best_precision_mode"),
            "best_mrr_mode": summary_data.get("best_mrr_mode"),
            "full_vs_baseline_delta": summary_data.get("full_vs_baseline_delta", {}),
        }

    if hard_negative:
        metrics["hard_negative"] = {
            "total_queries": hard_negative.get("total_queries", 0),
            "total_violations": hard_negative.get("total_violations", 0),
            "compliance_rate": hard_negative.get("compliance_rate", 1.0),
        }

    if affordance:
        metrics["affordance_validation"] = {
            "total_datasets": affordance.get("total_datasets", 0),
            "validation_rate": affordance.get("validation_rate", 0.0),
            "invalid_rate": affordance.get("invalid_rate", 0.0),
        }

    if pairing:
        summary_data = pairing.get("summary", {})
        metrics["cross_dataset_pairing"] = {
            "total_pairs": summary_data.get("total_pairs", 0),
            "mean_compatibility": summary_data.get("mean_compatibility", 0.0),
        }

    if robustness:
        metrics["metadata_robustness"] = {
            "most_impactful": robustness.get("most_impactful"),
            "least_impactful": robustness.get("least_impactful"),
            "critical_perturbations": robustness.get("summary", {}).get(
                "critical_perturbations", []
            ),
        }

    # Determine overall status
    all_green = True
    status_messages = []

    if hard_negative and hard_negative.get("total_violations", 0) > 0:
        all_green = False
        status_messages.append("Hard-negative violations detected")

    if affordance and affordance.get("invalid_rate", 0) > 0.3:
        all_green = False
        status_messages.append("High affordance invalidation rate")

    if robustness:
        critical = robustness.get("summary", {}).get("critical_perturbations", [])
        if critical:
            status_messages.append(f"Critical robustness gaps: {', '.join(critical)}")

    summary["overall_status"] = "PASS" if all_green else "NEEDS ATTENTION"
    summary["status_messages"] = status_messages
    summary["metrics"] = metrics

    return summary


def generate_unified_markdown(summary: dict[str, Any]) -> str:
    """Generate unified Markdown report."""
    lines = [
        "# Neural Search Unified Experiment Report",
        "",
        f"Generated: {summary.get('generated_at', 'unknown')}",
        "",
        "## Status",
        "",
        f"**Overall: {summary.get('overall_status', 'UNKNOWN')}**",
        "",
    ]

    if summary.get("status_messages"):
        lines.append("Issues:")
        for msg in summary["status_messages"]:
            lines.append(f"- {msg}")
        lines.append("")

    # Reports found/missing
    lines.extend([
        "## Report Availability",
        "",
        f"- Reports found: {', '.join(summary.get('reports_found', []))}",
        f"- Reports missing: {', '.join(summary.get('reports_missing', [])) or 'none'}",
        "",
    ])

    # Metrics summary
    metrics = summary.get("metrics", {})

    if metrics.get("baseline_ladder"):
        bl = metrics["baseline_ladder"]
        lines.extend([
            "## Baseline Ladder",
            "",
            f"- Best Precision Mode: {bl.get('best_precision_mode', 'N/A')}",
            f"- Best MRR Mode: {bl.get('best_mrr_mode', 'N/A')}",
        ])
        delta = bl.get("full_vs_baseline_delta", {})
        if delta:
            lines.append(
                f"- Full vs Baseline: P@5 {delta.get('precision', 0):+.1%}, "
                f"MRR {delta.get('mrr', 0):+.3f}"
            )
        lines.append("")

    if metrics.get("hard_negative"):
        hn = metrics["hard_negative"]
        lines.extend([
            "## Hard-Negative Benchmark",
            "",
            f"- Total Queries: {hn.get('total_queries', 0)}",
            f"- Total Violations: {hn.get('total_violations', 0)}",
            f"- Compliance Rate: {hn.get('compliance_rate', 0):.1%}",
            "",
        ])

    if metrics.get("affordance_validation"):
        av = metrics["affordance_validation"]
        lines.extend([
            "## Affordance Validation",
            "",
            f"- Total Datasets: {av.get('total_datasets', 0)}",
            f"- Validation Rate: {av.get('validation_rate', 0):.1%}",
            f"- Invalid Rate: {av.get('invalid_rate', 0):.1%}",
            "",
        ])

    if metrics.get("cross_dataset_pairing"):
        cp = metrics["cross_dataset_pairing"]
        lines.extend([
            "## Cross-Dataset Pairing",
            "",
            f"- Total Pairs: {cp.get('total_pairs', 0)}",
            f"- Mean Compatibility: {cp.get('mean_compatibility', 0):.1%}",
            "",
        ])

    if metrics.get("metadata_robustness"):
        mr = metrics["metadata_robustness"]
        lines.extend([
            "## Metadata Robustness",
            "",
            f"- Most Impactful: {mr.get('most_impactful', 'N/A')}",
            f"- Least Impactful: {mr.get('least_impactful', 'N/A')}",
        ])
        if mr.get("critical_perturbations"):
            lines.append(f"- Critical: {', '.join(mr['critical_perturbations'])}")
        lines.append("")

    # TODOs for incomplete experiments
    if summary.get("reports_missing"):
        lines.extend([
            "## TODOs",
            "",
            "The following experiments need to be run:",
            "",
        ])
        todo_commands = {
            "baseline_ladder": "python -m neural_search.evaluation.run_baseline_ladder",
            "hard_negative": "python -m neural_search.evaluation.run_hard_negative_benchmark",
            "affordance_validation": "python -m neural_search.evaluation.affordance_validation",
            "cross_dataset_pairing": "python -m neural_search.evaluation.cross_dataset_pairing",
            "metadata_robustness": "python -m neural_search.evaluation.metadata_robustness",
        }
        for missing in summary["reports_missing"]:
            cmd = todo_commands.get(missing, f"Run {missing} experiment")
            lines.append(f"- [ ] `{cmd}`")
        lines.append("")

    return "\n".join(lines)


def generate_unified_json(summary: dict[str, Any]) -> str:
    """Generate unified JSON report."""
    return json.dumps(summary, indent=2, default=str)


def write_unified_reports(
    summary: dict[str, Any],
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Write unified reports to files."""
    out = output_dir or REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "neural_search_experiment_report.md"
    json_path = out / "neural_search_experiment_report.json"

    md_path.write_text(generate_unified_markdown(summary), encoding="utf-8")
    json_path.write_text(generate_unified_json(summary), encoding="utf-8")

    return {"markdown": str(md_path), "json": str(json_path)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.evaluation.unified_report",
        description="Generate unified experiment report.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Directory containing experiment reports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for unified report.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output JSON to stdout instead of writing files.",
    )
    args = parser.parse_args(argv)

    print("Generating unified experiment report...")
    print()

    summary = generate_unified_report(args.reports_dir)

    if args.json_only:
        print(generate_unified_json(summary))
        return 0

    paths = write_unified_reports(summary, args.output_dir)

    # Print summary
    print("=" * 70)
    print("UNIFIED EXPERIMENT REPORT")
    print("=" * 70)
    print()
    print(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
    print()
    print(f"Reports Found: {', '.join(summary.get('reports_found', []))}")
    print(f"Reports Missing: {', '.join(summary.get('reports_missing', [])) or 'none'}")
    print()

    if summary.get("status_messages"):
        print("Issues:")
        for msg in summary["status_messages"]:
            print(f"  - {msg}")
        print()

    print("=" * 70)
    print("Reports written to:")
    print(f"  Markdown: {paths['markdown']}")
    print(f"  JSON: {paths['json']}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
