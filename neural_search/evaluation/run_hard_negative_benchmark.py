"""Hard-negative adversarial benchmark evaluation.

Tests constraint satisfaction for queries with explicit exclusions.
Tracks violations where excluded items appear in search results.

Usage:
    python -m neural_search.evaluation.run_hard_negative_benchmark
    python -m neural_search.evaluation.run_hard_negative_benchmark --config benchmarks/hard_negative_queries.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.ontology import normalize_text
from neural_search.search import search_datasets

BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
DEFAULT_CONFIG = BENCHMARKS_DIR / "hard_negative_queries.yaml"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


@dataclass
class Exclusion:
    """A single exclusion constraint."""

    field: str
    values: list[str]


@dataclass
class HardNegativeQuery:
    """A query with explicit exclusions."""

    id: str
    query: str
    exclusions: list[Exclusion]
    expected_species: list[str] = field(default_factory=list)
    expected_modalities: list[str] = field(default_factory=list)
    expected_tasks: list[str] = field(default_factory=list)
    expected_regions: list[str] = field(default_factory=list)
    expected_standards: list[str] = field(default_factory=list)
    expected_affordances: list[str] = field(default_factory=list)
    expected_behaviors: list[str] = field(default_factory=list)
    expected_subject_states: list[str] = field(default_factory=list)


@dataclass
class Violation:
    """A single hard-negative violation."""

    query_id: str
    result_id: str
    result_rank: int
    excluded_field: str
    offending_value: str
    exclusion_values: list[str]
    evidence_source: str
    explanation: str
    severity: str


@dataclass
class QueryViolationResult:
    """Violation results for a single query."""

    query_id: str
    query: str
    num_results: int
    num_violations: int
    violations: list[Violation]
    compliant_results: int
    compliance_rate: float


@dataclass
class HardNegativeReport:
    """Complete hard-negative benchmark report."""

    generated_at: str
    config_path: str
    total_queries: int
    queries_with_violations: int
    total_violations: int
    compliance_rate: float
    query_results: list[QueryViolationResult]
    violations_by_field: dict[str, int]
    violations_by_severity: dict[str, int]
    summary: dict[str, Any]


def load_hard_negative_config(path: Path) -> tuple[list[HardNegativeQuery], dict[str, str]]:
    """Load hard-negative queries from YAML config."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    queries = []
    for item in data.get("hard_negative_queries", []):
        exclusions = []
        for exc in item.get("exclusions", []):
            exclusions.append(
                Exclusion(
                    field=exc.get("field", ""),
                    values=exc.get("values", []),
                )
            )

        queries.append(
            HardNegativeQuery(
                id=item.get("id", ""),
                query=item.get("query", ""),
                exclusions=exclusions,
                expected_species=item.get("expected_species", []),
                expected_modalities=item.get("expected_modalities", []),
                expected_tasks=item.get("expected_tasks", []),
                expected_regions=item.get("expected_regions", []),
                expected_standards=item.get("expected_standards", []),
                expected_affordances=item.get("expected_affordances", []),
                expected_behaviors=item.get("expected_behaviors", []),
                expected_subject_states=item.get("expected_subject_states", []),
            )
        )

    severity_map = data.get("violation_severity", {})

    return queries, severity_map


def _normalize_values(values: list[str]) -> set[str]:
    """Normalize values for comparison."""
    return {normalize_text(v) for v in values}


def _extract_field_values(result: dict[str, Any], field_name: str) -> set[str]:
    """Extract normalized values for a field from a search result."""
    values: set[str] = set()

    # Direct fields
    direct_value = result.get(field_name)
    if direct_value:
        if isinstance(direct_value, list):
            values.update(normalize_text(str(v)) for v in direct_value)
        else:
            values.add(normalize_text(str(direct_value)))

    # From dataset_card_preview
    preview = result.get("dataset_card_preview", {})
    if isinstance(preview, dict):
        preview_value = preview.get(field_name)
        if preview_value:
            if isinstance(preview_value, list):
                values.update(normalize_text(str(v)) for v in preview_value)
            else:
                values.add(normalize_text(str(preview_value)))

    # Map field names to preview keys
    field_map = {
        "species": "species",
        "modality": "modalities",
        "brain_region": "brain_regions",
        "task": "tasks",
        "data_standard": "data_standards",
        "subject_state": "subject_states",
    }

    mapped_key = field_map.get(field_name)
    if mapped_key and mapped_key in preview:
        mapped_values = preview[mapped_key]
        if isinstance(mapped_values, list):
            values.update(normalize_text(str(v)) for v in mapped_values)
        else:
            values.add(normalize_text(str(mapped_values)))

    # From why_matched reasons
    for reason in result.get("why_matched", []):
        reason_lower = reason.lower()
        if field_name in reason_lower or field_map.get(field_name, "") in reason_lower:
            # Extract value from reason like "Species matched: mouse"
            if ":" in reason:
                parts = reason.split(":", 1)
                if len(parts) == 2:
                    values.add(normalize_text(parts[1].strip()))

    return values


def check_violations(
    query: HardNegativeQuery,
    results: list[dict[str, Any]],
    severity_map: dict[str, str],
) -> list[Violation]:
    """Check search results for constraint violations."""
    violations: list[Violation] = []

    for rank, result in enumerate(results, start=1):
        result_id = str(result.get("dataset_id", result.get("id", f"result_{rank}")))

        for exclusion in query.exclusions:
            excluded_normalized = _normalize_values(exclusion.values)
            result_values = _extract_field_values(result, exclusion.field)

            # Check for overlap
            overlap = result_values & excluded_normalized
            if overlap:
                for offending in overlap:
                    violations.append(
                        Violation(
                            query_id=query.id,
                            result_id=result_id,
                            result_rank=rank,
                            excluded_field=exclusion.field,
                            offending_value=offending,
                            exclusion_values=exclusion.values,
                            evidence_source=f"Extracted from result {exclusion.field} field",
                            explanation=f"Result contains excluded {exclusion.field} value '{offending}'",
                            severity=severity_map.get(exclusion.field, "major"),
                        )
                    )

    return violations


def run_hard_negative_benchmark(
    config_path: Path | None = None,
    datasets: list[dict[str, Any]] | None = None,
) -> HardNegativeReport:
    """Run hard-negative benchmark and generate report."""
    config = config_path or DEFAULT_CONFIG
    queries, severity_map = load_hard_negative_config(config)

    if datasets is None:
        datasets = build_demo_seed()

    query_results: list[QueryViolationResult] = []
    total_violations = 0
    violations_by_field: dict[str, int] = {}
    violations_by_severity: dict[str, int] = {}

    for query in queries:
        # Run search
        results = search_datasets(query.query, datasets, top_k=10)

        # Check for violations
        violations = check_violations(query, results, severity_map)

        # Aggregate statistics
        for v in violations:
            violations_by_field[v.excluded_field] = violations_by_field.get(v.excluded_field, 0) + 1
            violations_by_severity[v.severity] = violations_by_severity.get(v.severity, 0) + 1

        total_violations += len(violations)
        compliant_results = len(results) - len({v.result_rank for v in violations})

        query_results.append(
            QueryViolationResult(
                query_id=query.id,
                query=query.query,
                num_results=len(results),
                num_violations=len(violations),
                violations=violations,
                compliant_results=compliant_results,
                compliance_rate=compliant_results / len(results) if results else 1.0,
            )
        )

    queries_with_violations = sum(1 for qr in query_results if qr.num_violations > 0)
    total_results = sum(qr.num_results for qr in query_results)
    overall_compliance = (total_results - total_violations) / total_results if total_results > 0 else 1.0

    summary = {
        "total_queries": len(queries),
        "queries_with_zero_violations": len(queries) - queries_with_violations,
        "worst_query": max(query_results, key=lambda qr: qr.num_violations).query_id if query_results else None,
        "most_violated_field": max(violations_by_field, key=violations_by_field.get) if violations_by_field else None,
        "recommendation": "All constraints satisfied" if total_violations == 0 else f"Fix {total_violations} violations",
    }

    return HardNegativeReport(
        generated_at=datetime.now(UTC).isoformat(),
        config_path=str(config),
        total_queries=len(queries),
        queries_with_violations=queries_with_violations,
        total_violations=total_violations,
        compliance_rate=overall_compliance,
        query_results=query_results,
        violations_by_field=violations_by_field,
        violations_by_severity=violations_by_severity,
        summary=summary,
    )


def generate_hard_negative_markdown(report: HardNegativeReport) -> str:
    """Generate Markdown hard-negative report."""
    lines = [
        "# Hard-Negative Adversarial Benchmark Report",
        "",
        f"Generated: {report.generated_at}",
        f"Config: {report.config_path}",
        "",
        "## Summary",
        "",
        f"- **Total Queries**: {report.total_queries}",
        f"- **Queries with Violations**: {report.queries_with_violations}",
        f"- **Total Violations**: {report.total_violations}",
        f"- **Overall Compliance Rate**: {report.compliance_rate:.1%}",
        "",
    ]

    if report.total_violations == 0:
        lines.extend([
            "**Result: PASS - All hard-negative constraints satisfied.**",
            "",
        ])
    else:
        lines.extend([
            f"**Result: FAIL - {report.total_violations} violations detected.**",
            "",
        ])

    # Violations by field
    if report.violations_by_field:
        lines.extend([
            "## Violations by Field",
            "",
            "| Field | Count |",
            "| --- | --- |",
        ])
        for field_name, count in sorted(report.violations_by_field.items(), key=lambda x: -x[1]):
            lines.append(f"| {field_name} | {count} |")
        lines.append("")

    # Violations by severity
    if report.violations_by_severity:
        lines.extend([
            "## Violations by Severity",
            "",
            "| Severity | Count |",
            "| --- | --- |",
        ])
        for severity, count in sorted(report.violations_by_severity.items(), key=lambda x: -x[1]):
            lines.append(f"| {severity} | {count} |")
        lines.append("")

    # Per-query results
    lines.extend([
        "## Per-Query Results",
        "",
        "| Query ID | Query | Results | Violations | Compliance |",
        "| --- | --- | --- | --- | --- |",
    ])

    for qr in report.query_results:
        status = "PASS" if qr.num_violations == 0 else "FAIL"
        query_short = qr.query[:40] + "..." if len(qr.query) > 40 else qr.query
        lines.append(
            f"| {qr.query_id} | {query_short} | {qr.num_results} | {qr.num_violations} | {status} |"
        )
    lines.append("")

    # Detailed violations
    if report.total_violations > 0:
        lines.extend([
            "## Violation Details",
            "",
        ])

        for qr in report.query_results:
            if qr.violations:
                lines.extend([
                    f"### {qr.query_id}",
                    "",
                    f"Query: *{qr.query}*",
                    "",
                    "| Rank | Result ID | Field | Offending Value | Severity |",
                    "| --- | --- | --- | --- | --- |",
                ])
                for v in qr.violations:
                    lines.append(
                        f"| {v.result_rank} | {v.result_id} | {v.excluded_field} | {v.offending_value} | {v.severity} |"
                    )
                lines.append("")

    # Recommendations
    if report.summary:
        lines.extend([
            "## Recommendations",
            "",
            f"- {report.summary.get('recommendation', 'No specific recommendations')}",
        ])
        if report.summary.get("most_violated_field"):
            lines.append(f"- Focus on improving constraint handling for: {report.summary['most_violated_field']}")
        if report.summary.get("worst_query"):
            lines.append(f"- Review constraint parsing for query: {report.summary['worst_query']}")
        lines.append("")

    return "\n".join(lines)


def generate_hard_negative_json(report: HardNegativeReport) -> str:
    """Generate JSON hard-negative report."""

    def serialize_violation(v: Violation) -> dict[str, Any]:
        return asdict(v)

    def serialize_query_result(qr: QueryViolationResult) -> dict[str, Any]:
        return {
            "query_id": qr.query_id,
            "query": qr.query,
            "num_results": qr.num_results,
            "num_violations": qr.num_violations,
            "violations": [serialize_violation(v) for v in qr.violations],
            "compliant_results": qr.compliant_results,
            "compliance_rate": qr.compliance_rate,
        }

    payload = {
        "generated_at": report.generated_at,
        "config_path": report.config_path,
        "total_queries": report.total_queries,
        "queries_with_violations": report.queries_with_violations,
        "total_violations": report.total_violations,
        "compliance_rate": report.compliance_rate,
        "query_results": [serialize_query_result(qr) for qr in report.query_results],
        "violations_by_field": report.violations_by_field,
        "violations_by_severity": report.violations_by_severity,
        "summary": report.summary,
    }
    return json.dumps(payload, indent=2, default=str)


def write_hard_negative_reports(
    report: HardNegativeReport,
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Write hard-negative reports to files."""
    out = output_dir or REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "hard_negative_report.md"
    json_path = out / "hard_negative_report.json"

    md_path.write_text(generate_hard_negative_markdown(report), encoding="utf-8")
    json_path.write_text(generate_hard_negative_json(report), encoding="utf-8")

    return {"markdown": str(md_path), "json": str(json_path)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.evaluation.run_hard_negative_benchmark",
        description="Run hard-negative adversarial benchmark.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to hard-negative queries YAML config.",
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

    config = args.config or DEFAULT_CONFIG
    print(f"Running hard-negative benchmark from: {config}")
    print()

    report = run_hard_negative_benchmark(config)

    if args.json_only:
        print(generate_hard_negative_json(report))
        return 0

    paths = write_hard_negative_reports(report, args.output_dir)

    # Print summary
    print("=" * 70)
    print("HARD-NEGATIVE BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print(f"Total Queries: {report.total_queries}")
    print(f"Queries with Violations: {report.queries_with_violations}")
    print(f"Total Violations: {report.total_violations}")
    print(f"Overall Compliance: {report.compliance_rate:.1%}")
    print()

    if report.total_violations == 0:
        print("RESULT: PASS - All hard-negative constraints satisfied")
    else:
        print(f"RESULT: FAIL - {report.total_violations} violations detected")
        print()
        print("Violations by Field:")
        for field_name, count in report.violations_by_field.items():
            print(f"  {field_name}: {count}")

    print()
    print("=" * 70)
    print("Reports written to:")
    print(f"  Markdown: {paths['markdown']}")
    print(f"  JSON: {paths['json']}")
    print("=" * 70)

    return 0 if report.total_violations == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
