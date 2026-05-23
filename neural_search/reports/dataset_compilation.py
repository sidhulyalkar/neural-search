"""Dataset compilation report generator.

Generates comprehensive reports on dataset coverage, metadata quality,
and analysis readiness across all seed and ingested data sources.

Usage:
    python -m neural_search.reports.dataset_compilation
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_search.cards import generate_dataset_card_json
from neural_search.ingestion.curated import (
    summarize_curated_sources,
)
from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.qa import qa_counts, top_demo_ready

REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "reports"
METADATA_FIELDS = [
    "species",
    "modalities",
    "brain_regions",
    "tasks",
    "behaviors",
    "data_standards",
    "license",
    "url",
    "description",
]


def _count_by_field(records: list[dict], field: str) -> dict[str, int]:
    """Count occurrences of each value in a list field across records."""
    counter: Counter[str] = Counter()
    for record in records:
        dataset = record.get("dataset", record)
        values = dataset.get(field, [])
        if isinstance(values, list):
            counter.update(values)
        elif values:
            counter[str(values)] += 1
    return dict(counter.most_common())


def _count_boolean(records: list[dict], field: str) -> dict[str, int]:
    """Count True/False occurrences of a boolean field."""
    true_count = sum(
        1 for r in records if r.get("dataset", r).get(field, False)
    )
    return {"true": true_count, "false": len(records) - true_count}


def _count_with_linked_papers(records: list[dict]) -> int:
    """Count datasets that have at least one linked paper."""
    count = 0
    for record in records:
        papers = record.get("papers", [])
        if papers:
            count += 1
            continue
        dataset = record.get("dataset", record)
        linked_ids = dataset.get("metadata_json", {}).get("linked_paper_ids", [])
        if linked_ids:
            count += 1
    return count


def _compute_analysis_readiness(records: list[dict]) -> list[dict[str, Any]]:
    """Compute analysis readiness scores for all datasets."""
    scores = []
    for record in records:
        dataset = record.get("dataset", record)
        extraction = record.get("extraction")
        papers = record.get("papers", [])

        if extraction is None:
            continue

        try:
            card = generate_dataset_card_json(dataset, extraction, papers)
            scores.append({
                "source_id": dataset.get("source_id", "unknown"),
                "title": dataset.get("title", "Untitled"),
                "source": dataset.get("source", "unknown"),
                "score": card.analysis_readiness.score,
                "strengths": card.analysis_readiness.strengths,
                "limitations": card.analysis_readiness.limitations,
            })
        except Exception:
            pass

    return sorted(scores, key=lambda x: x["score"], reverse=True)


def _missing_metadata_summary(records: list[dict]) -> dict[str, Any]:
    """Summarize missing metadata fields across datasets."""
    missing_counts: Counter[str] = Counter()
    datasets_with_missing: list[dict[str, Any]] = []

    for record in records:
        dataset = record.get("dataset", record)
        missing = []

        for field in METADATA_FIELDS:
            value = dataset.get(field)
            if value is None or value == [] or value == "":
                missing.append(field)
                missing_counts[field] += 1

        if missing:
            datasets_with_missing.append({
                "source_id": dataset.get("source_id", "unknown"),
                "missing_fields": missing,
            })

    return {
        "fields_missing_count": dict(missing_counts.most_common()),
        "datasets_with_incomplete_metadata": len(datasets_with_missing),
        "datasets_by_missing_field_count": dict(
            Counter(len(d["missing_fields"]) for d in datasets_with_missing).most_common()
        ),
        "incomplete_datasets": datasets_with_missing[:20],
    }


def compile_dataset_report() -> dict[str, Any]:
    """Compile comprehensive dataset statistics from all sources.

    Returns:
        Dict containing all report sections and statistics.
    """
    records = build_demo_seed()
    curated_summary = summarize_curated_sources()

    by_source = _count_by_field(records, "source")
    by_task = _count_by_field(records, "tasks")
    by_modality = _count_by_field(records, "modalities")
    by_species = _count_by_field(records, "species")
    by_brain_region = _count_by_field(records, "brain_regions")
    by_data_standard = _count_by_field(records, "data_standards")

    has_behavior = _count_boolean(records, "has_behavior")
    has_trials = _count_boolean(records, "has_trials")

    with_papers = _count_with_linked_papers(records)

    readiness_scores = _compute_analysis_readiness(records)
    top_20_readiness = readiness_scores[:20]

    missing_summary = _missing_metadata_summary(records)
    review_counts = qa_counts(records)
    demo_ready = top_demo_ready(records, limit=10)

    unique_papers = len({
        p.get("id", p.get("doi", i))
        for r in records
        for i, p in enumerate(r.get("papers", []))
    })

    avg_readiness = (
        sum(s["score"] for s in readiness_scores) / len(readiness_scores)
        if readiness_scores else 0
    )

    return {
        "report_generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "total_datasets": len(records),
            "total_papers_linked": unique_papers,
            "total_dataset_cards": len(readiness_scores),
            "average_readiness_score": round(avg_readiness, 1),
            "datasets_with_behavior": has_behavior["true"],
            "datasets_with_trials": has_trials["true"],
            "datasets_with_linked_papers": with_papers,
            "number_reviewed": review_counts.get("reviewed", 0),
            "number_trusted": review_counts.get("trusted", 0),
            "number_rejected": review_counts.get("rejected", 0),
        },
        "curated_sources": curated_summary,
        "by_source": by_source,
        "by_task": by_task,
        "by_modality": by_modality,
        "by_species": by_species,
        "by_brain_region": by_brain_region,
        "by_data_standard": by_data_standard,
        "has_behavior": has_behavior,
        "has_trials": has_trials,
        "top_20_analysis_readiness": top_20_readiness,
        "top_datasets_ready_for_demo": demo_ready,
        "qa_review_counts": review_counts,
        "common_missing_metadata": missing_summary["fields_missing_count"],
        "missing_metadata": missing_summary,
    }


def generate_markdown_report(report: dict[str, Any]) -> str:
    """Generate a Markdown-formatted report from compiled statistics."""
    lines = [
        "# Neural Search Dataset Compilation Report",
        "",
        f"Generated: {report['report_generated_at']}",
        "",
        "## Summary",
        "",
        f"- **Total datasets**: {report['summary']['total_datasets']}",
        f"- **Total linked papers**: {report['summary']['total_papers_linked']}",
        f"- **Dataset cards generated**: {report['summary']['total_dataset_cards']}",
        f"- **Average analysis readiness**: {report['summary']['average_readiness_score']}/100",
        f"- **Datasets with behavior**: {report['summary']['datasets_with_behavior']}",
        f"- **Datasets with trial structure**: {report['summary']['datasets_with_trials']}",
        f"- **Datasets with linked papers**: {report['summary']['datasets_with_linked_papers']}",
        f"- **Reviewed**: {report['summary']['number_reviewed']}",
        f"- **Trusted**: {report['summary']['number_trusted']}",
        f"- **Rejected**: {report['summary']['number_rejected']}",
        "",
        "## Curated Sources (Pending Ingestion)",
        "",
        f"- Total curated entries: {report['curated_sources']['total']}",
        f"- Datasets: {report['curated_sources']['datasets']}",
        f"- Papers: {report['curated_sources']['papers']}",
        "",
        "### By Source Type",
        "",
    ]

    for source_type, count in report["curated_sources"]["by_source_type"].items():
        lines.append(f"- {source_type}: {count}")

    lines.extend([
        "",
        "### By Priority",
        "",
    ])

    for priority, count in report["curated_sources"]["by_priority"].items():
        lines.append(f"- {priority}: {count}")

    lines.extend([
        "",
        "## Datasets by Source",
        "",
    ])
    for source, count in report["by_source"].items():
        lines.append(f"- {source}: {count}")

    lines.extend([
        "",
        "## Datasets by Task",
        "",
    ])
    for task, count in report["by_task"].items():
        lines.append(f"- {task}: {count}")

    lines.extend([
        "",
        "## Datasets by Modality",
        "",
    ])
    for modality, count in report["by_modality"].items():
        lines.append(f"- {modality}: {count}")

    lines.extend([
        "",
        "## Datasets by Species",
        "",
    ])
    for species, count in report["by_species"].items():
        lines.append(f"- {species}: {count}")

    lines.extend([
        "",
        "## Datasets by Brain Region",
        "",
    ])
    for region, count in report["by_brain_region"].items():
        lines.append(f"- {region}: {count}")

    lines.extend([
        "",
        "## Datasets by Data Standard",
        "",
    ])
    for standard, count in report["by_data_standard"].items():
        lines.append(f"- {standard}: {count}")

    lines.extend([
        "",
        "## Behavioral Coverage",
        "",
        f"- Datasets with behavior: {report['has_behavior']['true']} "
        f"({_percent(report['has_behavior']['true'], report['summary']['total_datasets'])})",
        f"- Datasets with trial/event structure: {report['has_trials']['true']} "
        f"({_percent(report['has_trials']['true'], report['summary']['total_datasets'])})",
        "",
        "## Top 20 Analysis-Ready Datasets",
        "",
        "| Rank | Source ID | Title | Score | Source |",
        "|------|-----------|-------|-------|--------|",
    ])

    for i, ds in enumerate(report["top_20_analysis_readiness"], 1):
        title = ds["title"][:40] + "..." if len(ds["title"]) > 40 else ds["title"]
        lines.append(f"| {i} | {ds['source_id']} | {title} | {ds['score']} | {ds['source']} |")

    lines.extend([
        "",
        "## Top Datasets Ready for Demo",
        "",
        "| Rank | Source ID | Title | QA Status | Score |",
        "|------|-----------|-------|-----------|-------|",
    ])

    if report["top_datasets_ready_for_demo"]:
        for i, ds in enumerate(report["top_datasets_ready_for_demo"], 1):
            title = ds["title"][:40] + "..." if len(ds["title"]) > 40 else ds["title"]
            lines.append(
                f"| {i} | {ds['source_id']} | {title} | {ds['qa_status']} | {ds['score']} |"
            )
    else:
        lines.append("| - | - | No reviewed or trusted datasets yet | - | - |")

    lines.extend([
        "",
        "## Missing Metadata Summary",
        "",
        f"- Datasets with incomplete metadata: {report['missing_metadata']['datasets_with_incomplete_metadata']}",
        "",
        "### Fields Most Often Missing",
        "",
    ])

    for field, count in report["missing_metadata"]["fields_missing_count"].items():
        lines.append(f"- {field}: {count} datasets")

    lines.extend([
        "",
        "### Distribution by Missing Field Count",
        "",
    ])

    for count, num_datasets in sorted(report["missing_metadata"]["datasets_by_missing_field_count"].items()):
        lines.append(f"- {count} fields missing: {num_datasets} datasets")

    lines.append("")
    return "\n".join(lines)


def _percent(part: int, total: int) -> str:
    """Format a percentage string."""
    if total == 0:
        return "0%"
    return f"{round(100 * part / total)}%"


def generate_json_report(report: dict[str, Any], indent: int = 2) -> str:
    """Generate a JSON-formatted report from compiled statistics."""
    return json.dumps(report, indent=indent, default=str)


def write_reports(output_dir: Path | None = None) -> dict[str, str]:
    """Compile report and write to Markdown and JSON files.

    Args:
        output_dir: Directory for output files. Defaults to data/reports/.

    Returns:
        Dict with paths to generated files.
    """
    out = output_dir or REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    report = compile_dataset_report()

    md_path = out / "dataset_compilation_report.md"
    json_path = out / "dataset_compilation_report.json"

    md_content = generate_markdown_report(report)
    json_content = generate_json_report(report)

    md_path.write_text(md_content, encoding="utf-8")
    json_path.write_text(json_content, encoding="utf-8")

    return {
        "markdown": str(md_path),
        "json": str(json_path),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for dataset compilation report."""
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.reports.dataset_compilation",
        description="Generate dataset compilation report for Neural Search.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports. Defaults to data/reports/.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output JSON to stdout instead of writing files.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Output only the summary section.",
    )
    args = parser.parse_args(argv)

    report = compile_dataset_report()

    if args.summary_only:
        print(json.dumps(report["summary"], indent=2))
        return 0

    if args.json_only:
        print(generate_json_report(report))
        return 0

    paths = write_reports(args.output_dir)

    print("=" * 60)
    print("NEURAL SEARCH DATASET COMPILATION REPORT")
    print("=" * 60)
    print()
    print("SUMMARY")
    print("-" * 40)
    for key, value in report["summary"].items():
        label = key.replace("_", " ").title()
        print(f"  {label}: {value}")
    print()
    print("DATASETS BY SOURCE")
    print("-" * 40)
    for source, count in report["by_source"].items():
        print(f"  {source}: {count}")
    print()
    print("DATASETS BY TASK")
    print("-" * 40)
    for task, count in list(report["by_task"].items())[:10]:
        print(f"  {task}: {count}")
    if len(report["by_task"]) > 10:
        print(f"  ... and {len(report['by_task']) - 10} more tasks")
    print()
    print("DATASETS BY MODALITY")
    print("-" * 40)
    for modality, count in list(report["by_modality"].items())[:10]:
        print(f"  {modality}: {count}")
    if len(report["by_modality"]) > 10:
        print(f"  ... and {len(report['by_modality']) - 10} more modalities")
    print()
    print("DATASETS BY SPECIES")
    print("-" * 40)
    for species, count in report["by_species"].items():
        print(f"  {species}: {count}")
    print()
    print("BEHAVIORAL COVERAGE")
    print("-" * 40)
    print(f"  With behavior data: {report['has_behavior']['true']}")
    print(f"  With trial/event structure: {report['has_trials']['true']}")
    print()
    print("TOP 10 ANALYSIS-READY DATASETS")
    print("-" * 40)
    for i, ds in enumerate(report["top_20_analysis_readiness"][:10], 1):
        print(f"  {i}. [{ds['score']}] {ds['source_id']}: {ds['title'][:50]}")
    print()
    print("CORPUS QA")
    print("-" * 40)
    print(f"  Reviewed: {report['summary']['number_reviewed']}")
    print(f"  Trusted: {report['summary']['number_trusted']}")
    print(f"  Rejected: {report['summary']['number_rejected']}")
    print()
    print("TOP DATASETS READY FOR DEMO")
    print("-" * 40)
    ready = report["top_datasets_ready_for_demo"][:10]
    if ready:
        for i, ds in enumerate(ready, 1):
            print(f"  {i}. [{ds['score']}] {ds['source_id']} ({ds['qa_status']}): {ds['title'][:45]}")
    else:
        print("  No reviewed or trusted datasets yet.")
    print()
    print("MISSING METADATA")
    print("-" * 40)
    print(f"  Datasets with incomplete metadata: {report['missing_metadata']['datasets_with_incomplete_metadata']}")
    for field, count in list(report["missing_metadata"]["fields_missing_count"].items())[:5]:
        print(f"  Missing {field}: {count}")
    print()
    print("CURATED SOURCES (pending ingestion)")
    print("-" * 40)
    print(f"  Total: {report['curated_sources']['total']}")
    print(f"  High priority: {report['curated_sources']['by_priority'].get('high', 0)}")
    print()
    print("=" * 60)
    print("Reports written to:")
    print(f"  Markdown: {paths['markdown']}")
    print(f"  JSON: {paths['json']}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
