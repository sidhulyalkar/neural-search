"""Affordance validation rubric evaluation.

Validates whether datasets actually support predicted analysis affordances
by checking required and optional field presence.

Usage:
    python -m neural_search.evaluation.affordance_validation
    python -m neural_search.evaluation.affordance_validation --rubric config/affordance_rubric.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from neural_search.analysis_affordances import (
    AFFORDANCE_IDS,
    detect_analysis_affordances,
)
from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.normalized import load_normalized_records
from neural_search.schemas import NormalizedDatasetRecord

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DEFAULT_RUBRIC = CONFIG_DIR / "affordance_rubric.yaml"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


@dataclass
class AffordanceRequirement:
    """Requirements for a single affordance."""

    affordance_id: str
    description: str
    required_fields: list[str]
    optional_fields: list[str]
    confidence_threshold: float
    evidence_fields: list[str]
    failure_reasons: list[str]
    minimum_trials: int = 0
    minimum_sessions: int = 0
    minimum_conditions: int = 0
    minimum_duration_seconds: int = 0


@dataclass
class FieldMapping:
    """Mapping from abstract field to schema fields."""

    field_name: str
    schema_fields: list[str]
    usability_flags: list[str]


@dataclass
class AffordanceValidationResult:
    """Validation result for a single dataset-affordance pair."""

    dataset_id: str
    affordance_id: str
    predicted_support: str
    predicted_confidence: float
    required_present: list[str]
    required_missing: list[str]
    optional_present: list[str]
    optional_missing: list[str]
    validation_status: str  # "valid", "invalid", "uncertain"
    validation_confidence: float
    failure_reasons: list[str]
    notes: str


@dataclass
class DatasetValidationSummary:
    """Validation summary for a single dataset."""

    dataset_id: str
    dataset_title: str
    total_affordances: int
    supported_affordances: int
    validated_affordances: int
    invalid_affordances: int
    uncertain_affordances: int
    results: list[AffordanceValidationResult]


@dataclass
class AffordanceValidationReport:
    """Complete affordance validation report."""

    generated_at: str
    rubric_path: str
    total_datasets: int
    total_validations: int
    validation_rate: float
    invalid_rate: float
    dataset_summaries: list[DatasetValidationSummary]
    affordance_statistics: dict[str, dict[str, int]]
    summary: dict[str, Any]


def load_rubric(path: Path) -> tuple[dict[str, AffordanceRequirement], dict[str, FieldMapping]]:
    """Load affordance rubric from YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Rubric file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    requirements = {}
    for aff_id, config in data.get("affordances", {}).items():
        requirements[aff_id] = AffordanceRequirement(
            affordance_id=aff_id,
            description=config.get("description", ""),
            required_fields=config.get("required_fields", []),
            optional_fields=config.get("optional_fields", []),
            confidence_threshold=config.get("confidence_threshold", 0.7),
            evidence_fields=config.get("evidence_fields", []),
            failure_reasons=config.get("failure_reasons", []),
            minimum_trials=config.get("minimum_trials", 0),
            minimum_sessions=config.get("minimum_sessions", 0),
            minimum_conditions=config.get("minimum_conditions", 0),
            minimum_duration_seconds=config.get("minimum_duration_seconds", 0),
        )

    mappings = {}
    for field_name, config in data.get("field_mappings", {}).items():
        mappings[field_name] = FieldMapping(
            field_name=field_name,
            schema_fields=config.get("schema_fields", []),
            usability_flags=config.get("usability_flags", []),
        )

    return requirements, mappings


def _check_field_present(
    record: NormalizedDatasetRecord,
    field_name: str,
    mappings: dict[str, FieldMapping],
) -> bool:
    """Check if an abstract field is present in the dataset."""
    # Direct usability flag check
    mapping = mappings.get(field_name)
    if mapping:
        for flag in mapping.usability_flags:
            flag_value = getattr(record.usability_flags, flag, None)
            if flag_value is True:
                return True

    # Check for field keywords in various places
    text = " ".join(
        [
            record.title or "",
            record.description or "",
            " ".join(str(label.label) for label in record.tasks),
            " ".join(str(label.label) for label in record.modalities),
            " ".join(str(label.label) for label in record.behavioral_events),
        ]
    ).lower()

    # Check direct field names
    field_keywords = [field_name.lower(), field_name.replace("_", " ")]
    if mapping:
        field_keywords.extend(f.lower() for f in mapping.schema_fields)

    return any(kw in text for kw in field_keywords)


def validate_affordance(
    record: NormalizedDatasetRecord,
    predicted_affordance: Any,
    requirement: AffordanceRequirement | None,
    mappings: dict[str, FieldMapping],
) -> AffordanceValidationResult:
    """Validate a single predicted affordance against requirements."""
    dataset_id = record.source_id or record.id or "unknown"
    affordance_id = predicted_affordance.analysis_id

    if requirement is None:
        # No rubric entry for this affordance
        return AffordanceValidationResult(
            dataset_id=dataset_id,
            affordance_id=affordance_id,
            predicted_support=predicted_affordance.support_level,
            predicted_confidence=predicted_affordance.confidence,
            required_present=[],
            required_missing=[],
            optional_present=[],
            optional_missing=[],
            validation_status="uncertain",
            validation_confidence=0.5,
            failure_reasons=["No validation rubric defined for this affordance"],
            notes="Rubric not available",
        )

    # Check required fields
    required_present = []
    required_missing = []
    for field_name in requirement.required_fields:
        if _check_field_present(record, field_name, mappings):
            required_present.append(field_name)
        else:
            required_missing.append(field_name)

    # Check optional fields
    optional_present = []
    optional_missing = []
    for field_name in requirement.optional_fields:
        if _check_field_present(record, field_name, mappings):
            optional_present.append(field_name)
        else:
            optional_missing.append(field_name)

    # Determine validation status
    failure_reasons = []

    if required_missing:
        for missing in required_missing:
            reasons = [r for r in requirement.failure_reasons if missing.lower() in r.lower()]
            failure_reasons.extend(reasons or [f"Missing required field: {missing}"])

    # Calculate validation confidence
    required_score = len(required_present) / len(requirement.required_fields) if requirement.required_fields else 1.0
    optional_score = len(optional_present) / len(requirement.optional_fields) if requirement.optional_fields else 1.0
    validation_confidence = 0.7 * required_score + 0.3 * optional_score

    # Determine status
    if required_score == 1.0:
        validation_status = "valid"
    elif required_score >= 0.5:
        validation_status = "uncertain"
    else:
        validation_status = "invalid"

    # Check if prediction aligns with validation
    notes = ""
    if predicted_affordance.support_level in ("high", "medium") and validation_status == "invalid":
        notes = "Potential false positive: predicted support but validation failed"
    elif predicted_affordance.support_level == "unsupported" and validation_status == "valid":
        notes = "Potential false negative: validation passed but predicted unsupported"

    return AffordanceValidationResult(
        dataset_id=dataset_id,
        affordance_id=affordance_id,
        predicted_support=predicted_affordance.support_level,
        predicted_confidence=predicted_affordance.confidence,
        required_present=required_present,
        required_missing=required_missing,
        optional_present=optional_present,
        optional_missing=optional_missing,
        validation_status=validation_status,
        validation_confidence=validation_confidence,
        failure_reasons=failure_reasons,
        notes=notes,
    )


def validate_dataset(
    record: NormalizedDatasetRecord,
    requirements: dict[str, AffordanceRequirement],
    mappings: dict[str, FieldMapping],
) -> DatasetValidationSummary:
    """Validate all affordances for a single dataset."""
    # Get predicted affordances
    affordances = detect_analysis_affordances(record)

    results = []
    for aff in affordances:
        requirement = requirements.get(aff.analysis_id)
        result = validate_affordance(record, aff, requirement, mappings)
        results.append(result)

    # Count statistics
    supported = sum(1 for r in results if r.predicted_support in ("high", "medium"))
    validated = sum(1 for r in results if r.validation_status == "valid")
    invalid = sum(1 for r in results if r.validation_status == "invalid")
    uncertain = sum(1 for r in results if r.validation_status == "uncertain")

    return DatasetValidationSummary(
        dataset_id=record.source_id or record.id or "unknown",
        dataset_title=record.title or "Untitled",
        total_affordances=len(results),
        supported_affordances=supported,
        validated_affordances=validated,
        invalid_affordances=invalid,
        uncertain_affordances=uncertain,
        results=results,
    )


def run_affordance_validation(
    rubric_path: Path | None = None,
    datasets: list[NormalizedDatasetRecord] | None = None,
) -> AffordanceValidationReport:
    """Run affordance validation across all datasets."""
    rubric = rubric_path or DEFAULT_RUBRIC
    requirements, mappings = load_rubric(rubric)

    if datasets is None:
        # Load from demo seed and convert to NormalizedDatasetRecord
        seed_data = build_demo_seed()
        datasets = []
        for item in seed_data:
            ds = item.get("dataset", item)
            # Create minimal NormalizedDatasetRecord
            from neural_search.schemas import UsabilityFlags

            record = NormalizedDatasetRecord(
                id=ds.get("id", ""),
                source_id=ds.get("source_id", ds.get("id", "")),
                title=ds.get("title", ""),
                description=ds.get("description", ""),
                source=ds.get("source", ""),
                usability_flags=UsabilityFlags(
                    has_neural_data=ds.get("has_neural_data", False),
                    has_behavior=ds.get("has_behavior", False),
                    has_trials=ds.get("has_trials", False),
                ),
            )
            datasets.append(record)

    summaries = [validate_dataset(ds, requirements, mappings) for ds in datasets]

    # Aggregate statistics
    total_validations = sum(s.total_affordances for s in summaries)
    total_valid = sum(s.validated_affordances for s in summaries)
    total_invalid = sum(s.invalid_affordances for s in summaries)

    # Per-affordance statistics
    affordance_stats: dict[str, dict[str, int]] = {}
    for summary in summaries:
        for result in summary.results:
            if result.affordance_id not in affordance_stats:
                affordance_stats[result.affordance_id] = {
                    "total": 0,
                    "valid": 0,
                    "invalid": 0,
                    "uncertain": 0,
                    "predicted_supported": 0,
                }
            affordance_stats[result.affordance_id]["total"] += 1
            affordance_stats[result.affordance_id][result.validation_status] += 1
            if result.predicted_support in ("high", "medium"):
                affordance_stats[result.affordance_id]["predicted_supported"] += 1

    summary = {
        "most_validated_affordance": max(
            affordance_stats.items(),
            key=lambda x: x[1]["valid"],
            default=("none", {}),
        )[0] if affordance_stats else None,
        "most_invalid_affordance": max(
            affordance_stats.items(),
            key=lambda x: x[1]["invalid"],
            default=("none", {}),
        )[0] if affordance_stats else None,
        "recommendation": (
            "Validation rate is good"
            if total_validations > 0 and total_valid / total_validations > 0.5
            else "Review affordance detection rules"
        ),
    }

    return AffordanceValidationReport(
        generated_at=datetime.now(UTC).isoformat(),
        rubric_path=str(rubric),
        total_datasets=len(summaries),
        total_validations=total_validations,
        validation_rate=total_valid / total_validations if total_validations > 0 else 0.0,
        invalid_rate=total_invalid / total_validations if total_validations > 0 else 0.0,
        dataset_summaries=summaries,
        affordance_statistics=affordance_stats,
        summary=summary,
    )


def generate_validation_markdown(report: AffordanceValidationReport) -> str:
    """Generate Markdown validation report."""
    lines = [
        "# Affordance Validation Report",
        "",
        f"Generated: {report.generated_at}",
        f"Rubric: {report.rubric_path}",
        "",
        "## Summary",
        "",
        f"- **Total Datasets**: {report.total_datasets}",
        f"- **Total Validations**: {report.total_validations}",
        f"- **Validation Rate**: {report.validation_rate:.1%}",
        f"- **Invalid Rate**: {report.invalid_rate:.1%}",
        "",
    ]

    # Per-affordance statistics
    if report.affordance_statistics:
        lines.extend([
            "## Affordance Statistics",
            "",
            "| Affordance | Total | Valid | Invalid | Uncertain | Predicted Supported |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for aff_id, stats in sorted(report.affordance_statistics.items()):
            lines.append(
                f"| {aff_id} | {stats['total']} | {stats['valid']} | "
                f"{stats['invalid']} | {stats['uncertain']} | {stats['predicted_supported']} |"
            )
        lines.append("")

    # Per-dataset summaries
    lines.extend([
        "## Dataset Summaries",
        "",
        "| Dataset | Title | Supported | Valid | Invalid |",
        "| --- | --- | --- | --- | --- |",
    ])

    for summary in report.dataset_summaries:
        title_short = (summary.dataset_title[:30] + "...") if len(summary.dataset_title) > 30 else summary.dataset_title
        lines.append(
            f"| {summary.dataset_id} | {title_short} | "
            f"{summary.supported_affordances} | {summary.validated_affordances} | {summary.invalid_affordances} |"
        )
    lines.append("")

    # Potential issues
    potential_issues = []
    for summary in report.dataset_summaries:
        for result in summary.results:
            if result.notes:
                potential_issues.append(f"- **{summary.dataset_id}/{result.affordance_id}**: {result.notes}")

    if potential_issues:
        lines.extend([
            "## Potential Issues",
            "",
            *potential_issues[:20],  # Limit to 20
            "",
        ])

    # Recommendations
    if report.summary:
        lines.extend([
            "## Recommendations",
            "",
            f"- {report.summary.get('recommendation', 'No specific recommendations')}",
        ])
        if report.summary.get("most_invalid_affordance"):
            lines.append(f"- Review detection rules for: {report.summary['most_invalid_affordance']}")
        lines.append("")

    return "\n".join(lines)


def generate_validation_json(report: AffordanceValidationReport) -> str:
    """Generate JSON validation report."""
    from dataclasses import asdict

    def serialize_result(r: AffordanceValidationResult) -> dict[str, Any]:
        return asdict(r)

    def serialize_summary(s: DatasetValidationSummary) -> dict[str, Any]:
        return {
            "dataset_id": s.dataset_id,
            "dataset_title": s.dataset_title,
            "total_affordances": s.total_affordances,
            "supported_affordances": s.supported_affordances,
            "validated_affordances": s.validated_affordances,
            "invalid_affordances": s.invalid_affordances,
            "uncertain_affordances": s.uncertain_affordances,
            "results": [serialize_result(r) for r in s.results],
        }

    payload = {
        "generated_at": report.generated_at,
        "rubric_path": report.rubric_path,
        "total_datasets": report.total_datasets,
        "total_validations": report.total_validations,
        "validation_rate": report.validation_rate,
        "invalid_rate": report.invalid_rate,
        "dataset_summaries": [serialize_summary(s) for s in report.dataset_summaries],
        "affordance_statistics": report.affordance_statistics,
        "summary": report.summary,
    }
    return json.dumps(payload, indent=2, default=str)


def write_validation_reports(
    report: AffordanceValidationReport,
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Write validation reports to files."""
    out = output_dir or REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "affordance_validation_report.md"
    json_path = out / "affordance_validation_report.json"

    md_path.write_text(generate_validation_markdown(report), encoding="utf-8")
    json_path.write_text(generate_validation_json(report), encoding="utf-8")

    return {"markdown": str(md_path), "json": str(json_path)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.evaluation.affordance_validation",
        description="Run affordance validation against rubric.",
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=None,
        help="Path to affordance rubric YAML.",
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

    rubric = args.rubric or DEFAULT_RUBRIC
    print(f"Running affordance validation with rubric: {rubric}")
    print()

    report = run_affordance_validation(rubric)

    if args.json_only:
        print(generate_validation_json(report))
        return 0

    paths = write_validation_reports(report, args.output_dir)

    # Print summary
    print("=" * 70)
    print("AFFORDANCE VALIDATION RESULTS")
    print("=" * 70)
    print()
    print(f"Total Datasets: {report.total_datasets}")
    print(f"Total Validations: {report.total_validations}")
    print(f"Validation Rate: {report.validation_rate:.1%}")
    print(f"Invalid Rate: {report.invalid_rate:.1%}")
    print()

    print("Affordance Statistics:")
    for aff_id, stats in sorted(report.affordance_statistics.items()):
        print(f"  {aff_id}: {stats['valid']}/{stats['total']} valid")

    print()
    print("=" * 70)
    print("Reports written to:")
    print(f"  Markdown: {paths['markdown']}")
    print(f"  JSON: {paths['json']}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
