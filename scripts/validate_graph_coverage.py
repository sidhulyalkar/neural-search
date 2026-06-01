#!/usr/bin/env python3
"""Validate knowledge graph coverage against quality gates.

Usage:
    python scripts/validate_graph_coverage.py --graph data/graph/neural_search_graph.json
    python scripts/validate_graph_coverage.py --graph data/graph/neural_search_graph.json --profile release
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neural_search.graph.quality import validate_graph_coverage, GraphQualityReport
from neural_search.graph.schema import read_graph_json, read_graph_jsonl


def format_report(report: GraphQualityReport, verbose: bool = False) -> str:
    """Format a quality report for human reading."""

    lines = []
    lines.append("=" * 60)
    lines.append("GRAPH COVERAGE VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    status = "PASSED" if report.passed else "FAILED"
    lines.append(f"Status: {status}")
    lines.append(f"Nodes: {report.node_count}")
    lines.append(f"Edges: {report.edge_count}")
    lines.append(f"Errors: {report.error_count}")
    lines.append(f"Warnings: {report.warning_count}")
    lines.append("")

    # Node type counts
    lines.append("Node Types:")
    for node_type, count in sorted(report.node_type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {node_type}: {count}")
    lines.append("")

    # Edge type counts
    lines.append("Edge Types:")
    for edge_type, count in sorted(report.edge_type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {edge_type}: {count}")
    lines.append("")

    # Coverage thresholds
    if report.coverage_thresholds:
        lines.append("Coverage Thresholds:")
        for ct in report.coverage_thresholds:
            status_mark = "[OK]" if ct.passed else "[FAIL]"
            lines.append(
                f"  {status_mark} {ct.metric_name}: {ct.actual:.1%} (required: {ct.threshold:.1%})"
            )
        lines.append("")

    # Issues
    if report.issues:
        lines.append("Issues:")
        error_issues = [i for i in report.issues if i.severity == "error"]
        warning_issues = [i for i in report.issues if i.severity == "warning"]

        if error_issues:
            lines.append("  ERRORS:")
            for issue in error_issues[:20]:  # Limit to first 20
                lines.append(f"    [{issue.code}] {issue.message}")
            if len(error_issues) > 20:
                lines.append(f"    ... and {len(error_issues) - 20} more errors")

        if warning_issues and verbose:
            lines.append("  WARNINGS:")
            for issue in warning_issues[:30]:  # Limit to first 30
                lines.append(f"    [{issue.code}] {issue.message}")
            if len(warning_issues) > 30:
                lines.append(f"    ... and {len(warning_issues) - 30} more warnings")
        elif warning_issues:
            lines.append(f"  WARNINGS: {len(warning_issues)} (use --verbose to see details)")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate graph coverage against quality gates"
    )
    parser.add_argument(
        "--graph",
        required=True,
        help="Path to graph JSON or JSONL file",
    )
    parser.add_argument(
        "--profile",
        default="ci",
        choices=["ci", "local", "release"],
        help="Quality gate profile to use",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed warnings",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Exit with error if warnings are present",
    )

    args = parser.parse_args(argv)
    graph_path = Path(args.graph)

    if not graph_path.exists():
        print(f"Error: Graph file not found: {graph_path}", file=sys.stderr)
        return 1

    # Load graph
    print(f"Loading graph from {graph_path}...")
    if graph_path.suffix == ".jsonl":
        graph = read_graph_jsonl(graph_path)
    else:
        graph = read_graph_json(graph_path)

    # Validate
    print(f"Validating with profile: {args.profile}")
    report = validate_graph_coverage(graph, profile=args.profile)

    # Output
    if args.json:
        output = {
            "passed": report.passed,
            "node_count": report.node_count,
            "edge_count": report.edge_count,
            "error_count": report.error_count,
            "warning_count": report.warning_count,
            "node_type_counts": report.node_type_counts,
            "edge_type_counts": report.edge_type_counts,
            "coverage_thresholds": [
                {
                    "metric": ct.metric_name,
                    "threshold": ct.threshold,
                    "actual": ct.actual,
                    "passed": ct.passed,
                }
                for ct in report.coverage_thresholds
            ],
            "issues": [
                {
                    "code": i.code,
                    "severity": i.severity,
                    "message": i.message,
                    "node_type": i.node_type,
                    "edge_type": i.edge_type,
                }
                for i in report.issues[:100]  # Limit to 100 issues in JSON
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_report(report, verbose=args.verbose))

    # Exit code
    if not report.passed:
        return 1
    if args.fail_on_warnings and report.warning_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
