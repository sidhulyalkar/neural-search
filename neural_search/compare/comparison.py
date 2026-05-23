"""Dataset comparison logic for side-by-side analysis."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from neural_search.cards import generate_dataset_card_json
from neural_search.extraction import extract_dataset_labels
from neural_search.notebooks.templates import available_templates_for_dataset
from neural_search.recipes import match_recipes_for_tasks


class DatasetComparisonItem(BaseModel):
    """Comparison data for a single dataset."""

    dataset_id: str
    title: str
    source: str
    source_id: str
    url: str | None = None
    doi: str | None = None
    license: str | None = None

    # Scientific labels
    task_labels: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    species: list[str] = Field(default_factory=list)
    brain_regions: list[str] = Field(default_factory=list)
    behavior_labels: list[str] = Field(default_factory=list)
    data_standards: list[str] = Field(default_factory=list)

    # Experimental structure
    has_trials: bool = False
    has_events: bool = False
    has_behavior: bool = False
    subject_count: int | None = None
    session_count: int | None = None

    # Linked papers
    linked_paper_count: int = 0
    linked_papers: list[dict[str, Any]] = Field(default_factory=list)

    # Analysis readiness
    analysis_readiness_score: int = 0
    strengths: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    missing_metadata: list[str] = Field(default_factory=list)

    # Available resources
    available_notebook_templates: list[str] = Field(default_factory=list)
    suggested_analyses: list[str] = Field(default_factory=list)
    matched_recipes: list[dict[str, Any]] = Field(default_factory=list)

    # QA status
    qa_status: str = "auto_generated"


class FieldComparison(BaseModel):
    """Comparison of a single field across datasets."""

    field_name: str
    field_label: str
    values: dict[str, Any]  # dataset_id -> value
    all_same: bool = False
    union_values: list[Any] = Field(default_factory=list)
    intersection_values: list[Any] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    """Complete comparison result for multiple datasets."""

    dataset_ids: list[str]
    datasets: list[DatasetComparisonItem]
    field_comparisons: list[FieldComparison] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _extract_comparison_item(
    dataset: dict[str, Any],
    papers: list[dict[str, Any]],
    extraction: dict[str, Any] | None = None,
    card: Any | None = None,
) -> DatasetComparisonItem:
    """Extract comparison data from a dataset record."""
    # Generate extraction if not provided
    if extraction is None:
        extraction = extract_dataset_labels(
            title=dataset.get("title", ""),
            description=dataset.get("description", ""),
            file_paths=[],
            source_metadata=dataset,
            linked_paper_abstracts=[p.get("abstract", "") for p in papers],
        )

    # Generate card if not provided
    if card is None:
        card = generate_dataset_card_json(dataset, extraction, papers)

    # Get available templates
    templates = available_templates_for_dataset(dataset)

    # Get matched recipes
    task_ids = dataset.get("tasks", [])
    recipes = match_recipes_for_tasks(task_ids) if task_ids else []

    # Extract subject/session counts from metadata
    metadata = dataset.get("metadata_json", {})
    subject_count = metadata.get("subject_count") or metadata.get("num_subjects")
    session_count = metadata.get("session_count") or metadata.get("num_sessions")

    return DatasetComparisonItem(
        dataset_id=dataset.get("id", dataset.get("source_id", "")),
        title=dataset.get("title", "Untitled"),
        source=dataset.get("source", "other"),
        source_id=dataset.get("source_id", ""),
        url=dataset.get("url"),
        doi=dataset.get("doi"),
        license=dataset.get("license"),
        task_labels=dataset.get("tasks", []),
        modalities=dataset.get("modalities", []),
        species=dataset.get("species", []),
        brain_regions=dataset.get("brain_regions", []),
        behavior_labels=dataset.get("behaviors", []),
        data_standards=dataset.get("data_standards", []),
        has_trials=dataset.get("has_trials", False),
        has_events=metadata.get("has_events", False),
        has_behavior=dataset.get("has_behavior", False),
        subject_count=subject_count,
        session_count=session_count,
        linked_paper_count=len(papers),
        linked_papers=[
            {
                "title": p.get("title", ""),
                "doi": p.get("doi"),
                "year": p.get("publication_year"),
            }
            for p in papers[:5]  # Limit to 5 papers
        ],
        analysis_readiness_score=card.analysis_readiness.score,
        strengths=card.analysis_readiness.strengths,
        limitations=card.analysis_readiness.limitations,
        missing_metadata=card.missing_fields,
        available_notebook_templates=[t["id"] for t in templates if t.get("available", True)],
        suggested_analyses=card.suggested_analyses[:5],
        matched_recipes=[
            {"id": r["id"], "title": r["title"], "match_score": r.get("match_score", 0)}
            for r in recipes[:3]
        ],
        qa_status=dataset.get("qa_status", "auto_generated"),
    )


def _compare_list_field(
    field_name: str,
    field_label: str,
    datasets: list[DatasetComparisonItem],
) -> FieldComparison:
    """Compare a list field across datasets."""
    values: dict[str, list[str]] = {}
    all_sets: list[set[str]] = []

    for ds in datasets:
        field_value = getattr(ds, field_name, [])
        values[ds.dataset_id] = field_value
        all_sets.append(set(field_value))

    # Compute union and intersection
    if all_sets:
        union = set.union(*all_sets) if all_sets else set()
        intersection = set.intersection(*all_sets) if all_sets else set()
    else:
        union = set()
        intersection = set()

    return FieldComparison(
        field_name=field_name,
        field_label=field_label,
        values=values,
        all_same=len(union) == len(intersection) and len(union) > 0,
        union_values=sorted(union),
        intersection_values=sorted(intersection),
    )


def _compare_scalar_field(
    field_name: str,
    field_label: str,
    datasets: list[DatasetComparisonItem],
) -> FieldComparison:
    """Compare a scalar field across datasets."""
    values: dict[str, Any] = {}
    unique_values: set[Any] = set()

    for ds in datasets:
        field_value = getattr(ds, field_name, None)
        values[ds.dataset_id] = field_value
        if field_value is not None:
            unique_values.add(field_value)

    return FieldComparison(
        field_name=field_name,
        field_label=field_label,
        values=values,
        all_same=len(unique_values) <= 1,
        union_values=sorted(unique_values) if unique_values else [],
        intersection_values=sorted(unique_values) if len(unique_values) == 1 else [],
    )


def compare_datasets(
    records: list[dict[str, Any]],
) -> ComparisonResult:
    """
    Compare multiple datasets side-by-side.

    Args:
        records: List of dataset records, each containing:
            - dataset: dict with dataset metadata
            - papers: list of linked papers
            - extraction: optional extraction result
            - card: optional generated card

    Returns:
        ComparisonResult with comparison data for all datasets
    """
    if len(records) < 2:
        raise ValueError("At least 2 datasets required for comparison")
    if len(records) > 5:
        raise ValueError("Maximum 5 datasets can be compared at once")

    # Extract comparison items for each dataset
    datasets: list[DatasetComparisonItem] = []
    for record in records:
        item = _extract_comparison_item(
            dataset=record["dataset"],
            papers=record.get("papers", []),
            extraction=record.get("extraction"),
            card=record.get("card"),
        )
        datasets.append(item)

    # Build field comparisons
    field_comparisons: list[FieldComparison] = []

    # Source comparison
    field_comparisons.append(_compare_scalar_field("source", "Source Archive", datasets))

    # Scientific labels (list fields)
    list_fields = [
        ("task_labels", "Task Labels"),
        ("modalities", "Modalities"),
        ("species", "Species"),
        ("brain_regions", "Brain Regions"),
        ("behavior_labels", "Behavior Labels"),
        ("data_standards", "Data Standards"),
    ]
    for field_name, field_label in list_fields:
        field_comparisons.append(_compare_list_field(field_name, field_label, datasets))

    # Experimental structure (scalar/boolean fields)
    field_comparisons.append(_compare_scalar_field("has_trials", "Trial Structure", datasets))
    field_comparisons.append(_compare_scalar_field("has_events", "Event Structure", datasets))
    field_comparisons.append(_compare_scalar_field("has_behavior", "Behavioral Data", datasets))

    # Linked papers
    field_comparisons.append(
        _compare_scalar_field("linked_paper_count", "Linked Papers", datasets)
    )

    # Analysis readiness
    field_comparisons.append(
        _compare_scalar_field("analysis_readiness_score", "Analysis Readiness", datasets)
    )

    # Missing metadata comparison
    field_comparisons.append(
        _compare_list_field("missing_metadata", "Missing Metadata", datasets)
    )

    # Available templates
    field_comparisons.append(
        _compare_list_field("available_notebook_templates", "Notebook Templates", datasets)
    )

    # Suggested analyses
    field_comparisons.append(
        _compare_list_field("suggested_analyses", "Suggested Analyses", datasets)
    )

    # Build summary
    summary = _build_comparison_summary(datasets, field_comparisons)

    return ComparisonResult(
        dataset_ids=[ds.dataset_id for ds in datasets],
        datasets=datasets,
        field_comparisons=field_comparisons,
        summary=summary,
    )


def _build_comparison_summary(
    datasets: list[DatasetComparisonItem],
    field_comparisons: list[FieldComparison],
) -> dict[str, Any]:
    """Build a summary of the comparison."""
    # Find common ground
    common_fields = [fc for fc in field_comparisons if fc.all_same and fc.intersection_values]

    # Find differences
    different_fields = [
        fc for fc in field_comparisons if not fc.all_same and len(fc.union_values) > 1
    ]

    # Readiness ranking
    readiness_ranking = sorted(
        [(ds.dataset_id, ds.title, ds.analysis_readiness_score) for ds in datasets],
        key=lambda x: x[2],
        reverse=True,
    )

    # Find the dataset with most templates
    template_counts = [
        (ds.dataset_id, ds.title, len(ds.available_notebook_templates)) for ds in datasets
    ]
    most_templates = max(template_counts, key=lambda x: x[2])

    # Find shared task coverage
    task_fc = next((fc for fc in field_comparisons if fc.field_name == "task_labels"), None)
    shared_tasks = task_fc.intersection_values if task_fc else []

    # Find shared modalities
    modality_fc = next((fc for fc in field_comparisons if fc.field_name == "modalities"), None)
    shared_modalities = modality_fc.intersection_values if modality_fc else []

    return {
        "dataset_count": len(datasets),
        "common_fields": [fc.field_label for fc in common_fields],
        "different_fields": [fc.field_label for fc in different_fields],
        "readiness_ranking": [
            {"dataset_id": r[0], "title": r[1], "score": r[2]} for r in readiness_ranking
        ],
        "highest_readiness": {
            "dataset_id": readiness_ranking[0][0],
            "title": readiness_ranking[0][1],
            "score": readiness_ranking[0][2],
        }
        if readiness_ranking
        else None,
        "most_notebook_templates": {
            "dataset_id": most_templates[0],
            "title": most_templates[1],
            "count": most_templates[2],
        },
        "shared_tasks": shared_tasks,
        "shared_modalities": shared_modalities,
        "all_tasks": task_fc.union_values if task_fc else [],
        "all_modalities": modality_fc.union_values if modality_fc else [],
    }


def generate_comparison_markdown(result: ComparisonResult) -> str:
    """Generate a Markdown report from a comparison result."""
    lines: list[str] = []

    # Header
    lines.append("# Dataset Comparison Report")
    lines.append("")
    lines.append(f"Generated: {result.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    # Summary section
    lines.append("## Summary")
    lines.append("")
    lines.append(f"Comparing **{len(result.datasets)}** datasets:")
    lines.append("")
    for ds in result.datasets:
        lines.append(f"- **{ds.title}** ({ds.source.upper()}, {ds.source_id})")
    lines.append("")

    # Key insights
    summary = result.summary
    if summary.get("shared_tasks"):
        lines.append(f"**Shared tasks:** {', '.join(summary['shared_tasks'])}")
        lines.append("")
    if summary.get("shared_modalities"):
        lines.append(f"**Shared modalities:** {', '.join(summary['shared_modalities'])}")
        lines.append("")

    # Readiness ranking
    if summary.get("readiness_ranking"):
        lines.append("### Analysis Readiness Ranking")
        lines.append("")
        for i, item in enumerate(summary["readiness_ranking"], 1):
            lines.append(f"{i}. **{item['title']}** - {item['score']}/100")
        lines.append("")

    # Comparison table
    lines.append("## Detailed Comparison")
    lines.append("")

    # Build the header row
    header = ["Field"] + [ds.title[:30] for ds in result.datasets]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    # Add rows for each field comparison
    for fc in result.field_comparisons:
        row = [fc.field_label]
        for ds in result.datasets:
            value = fc.values.get(ds.dataset_id)
            if isinstance(value, list):
                cell = ", ".join(str(v) for v in value[:3])
                if len(value) > 3:
                    cell += f" (+{len(value) - 3})"
            elif isinstance(value, bool):
                cell = "Yes" if value else "No"
            elif value is None:
                cell = "-"
            else:
                cell = str(value)
            row.append(cell[:40])  # Truncate long cells
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Common ground section
    if summary.get("common_fields"):
        lines.append("## Common Ground")
        lines.append("")
        lines.append("These datasets share the following characteristics:")
        lines.append("")
        for field in summary["common_fields"]:
            lines.append(f"- {field}")
        lines.append("")

    # Differences section
    if summary.get("different_fields"):
        lines.append("## Key Differences")
        lines.append("")
        lines.append("These fields differ across datasets:")
        lines.append("")
        for field in summary["different_fields"]:
            lines.append(f"- {field}")
        lines.append("")

    # Individual dataset details
    lines.append("## Dataset Details")
    lines.append("")

    for ds in result.datasets:
        lines.append(f"### {ds.title}")
        lines.append("")
        lines.append(f"- **Source:** {ds.source.upper()}")
        lines.append(f"- **ID:** {ds.source_id}")
        if ds.url:
            lines.append(f"- **URL:** {ds.url}")
        if ds.doi:
            lines.append(f"- **DOI:** {ds.doi}")
        lines.append(f"- **Analysis Readiness:** {ds.analysis_readiness_score}/100")
        lines.append(f"- **QA Status:** {ds.qa_status}")
        lines.append("")

        if ds.strengths:
            lines.append("**Strengths:**")
            for s in ds.strengths:
                lines.append(f"- {s}")
            lines.append("")

        if ds.limitations:
            lines.append("**Limitations:**")
            for lim in ds.limitations:
                lines.append(f"- {lim}")
            lines.append("")

        if ds.missing_metadata:
            lines.append("**Missing Metadata:**")
            for m in ds.missing_metadata:
                lines.append(f"- {m}")
            lines.append("")

        if ds.suggested_analyses:
            lines.append("**Suggested Analyses:**")
            for a in ds.suggested_analyses:
                lines.append(f"- {a}")
            lines.append("")

        if ds.linked_papers:
            lines.append("**Linked Papers:**")
            for p in ds.linked_papers:
                year = f" ({p['year']})" if p.get("year") else ""
                lines.append(f"- {p['title']}{year}")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Generated by Neural Search Dataset Comparison Tool*")

    return "\n".join(lines)
