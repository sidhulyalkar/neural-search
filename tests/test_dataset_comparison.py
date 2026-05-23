"""Tests for dataset comparison functionality."""

import pytest

from neural_search.compare import (
    compare_datasets,
    generate_comparison_markdown,
    ComparisonResult,
    DatasetComparisonItem,
)
from neural_search.ingestion.demo_seed import build_demo_seed


@pytest.fixture
def demo_records():
    """Get demo dataset records for testing."""
    return build_demo_seed()


def test_compare_two_datasets_returns_valid_result(demo_records):
    """Test that comparing two datasets returns a valid ComparisonResult."""
    # Get first two records
    records = demo_records[:2]

    result = compare_datasets(records)

    assert isinstance(result, ComparisonResult)
    assert len(result.dataset_ids) == 2
    assert len(result.datasets) == 2
    assert len(result.field_comparisons) > 0
    assert result.summary is not None
    assert result.generated_at is not None


def test_compare_requires_at_least_two_datasets(demo_records):
    """Test that comparison requires at least 2 datasets."""
    with pytest.raises(ValueError, match="At least 2"):
        compare_datasets([demo_records[0]])


def test_compare_rejects_more_than_five_datasets(demo_records):
    """Test that comparison rejects more than 5 datasets."""
    # Create 6 records by duplicating
    records = demo_records[:5] + [demo_records[0]]

    with pytest.raises(ValueError, match="Maximum 5"):
        compare_datasets(records)


def test_comparison_includes_expected_fields(demo_records):
    """Test that comparison includes all expected field comparisons."""
    records = demo_records[:2]
    result = compare_datasets(records)

    field_names = [fc.field_name for fc in result.field_comparisons]

    # Check for expected fields
    expected_fields = [
        "source",
        "task_labels",
        "modalities",
        "species",
        "brain_regions",
        "behavior_labels",
        "has_trials",
        "analysis_readiness_score",
        "missing_metadata",
        "available_notebook_templates",
        "suggested_analyses",
    ]

    for expected in expected_fields:
        assert expected in field_names, f"Missing field: {expected}"


def test_comparison_item_has_required_attributes(demo_records):
    """Test that DatasetComparisonItem has all required attributes."""
    records = demo_records[:2]
    result = compare_datasets(records)

    for item in result.datasets:
        assert isinstance(item, DatasetComparisonItem)
        assert item.dataset_id
        assert item.title
        assert item.source
        assert isinstance(item.task_labels, list)
        assert isinstance(item.modalities, list)
        assert isinstance(item.species, list)
        assert isinstance(item.brain_regions, list)
        assert isinstance(item.analysis_readiness_score, int)
        assert isinstance(item.strengths, list)
        assert isinstance(item.limitations, list)


def test_field_comparison_tracks_union_and_intersection(demo_records):
    """Test that field comparisons track union and intersection values."""
    records = demo_records[:2]
    result = compare_datasets(records)

    # Find task_labels comparison
    task_fc = next(
        (fc for fc in result.field_comparisons if fc.field_name == "task_labels"),
        None,
    )
    assert task_fc is not None
    assert isinstance(task_fc.union_values, list)
    assert isinstance(task_fc.intersection_values, list)
    assert isinstance(task_fc.all_same, bool)


def test_summary_includes_readiness_ranking(demo_records):
    """Test that summary includes readiness ranking."""
    records = demo_records[:3]
    result = compare_datasets(records)

    assert "readiness_ranking" in result.summary
    ranking = result.summary["readiness_ranking"]
    assert len(ranking) == 3

    # Verify ranking is sorted by score descending
    scores = [item["score"] for item in ranking]
    assert scores == sorted(scores, reverse=True)


def test_summary_identifies_shared_tasks_and_modalities(demo_records):
    """Test that summary identifies shared tasks and modalities."""
    records = demo_records[:2]
    result = compare_datasets(records)

    assert "shared_tasks" in result.summary
    assert "shared_modalities" in result.summary
    assert "all_tasks" in result.summary
    assert "all_modalities" in result.summary

    # All tasks should include shared tasks
    shared = set(result.summary["shared_tasks"])
    all_tasks = set(result.summary["all_tasks"])
    assert shared.issubset(all_tasks)


def test_generate_comparison_markdown_returns_string(demo_records):
    """Test that Markdown generation returns a non-empty string."""
    records = demo_records[:2]
    result = compare_datasets(records)

    markdown = generate_comparison_markdown(result)

    assert isinstance(markdown, str)
    assert len(markdown) > 0
    assert "# Dataset Comparison Report" in markdown


def test_comparison_markdown_includes_all_datasets(demo_records):
    """Test that Markdown includes all compared datasets."""
    records = demo_records[:3]
    result = compare_datasets(records)

    markdown = generate_comparison_markdown(result)

    for item in result.datasets:
        assert item.title in markdown


def test_comparison_markdown_includes_table(demo_records):
    """Test that Markdown includes a comparison table."""
    records = demo_records[:2]
    result = compare_datasets(records)

    markdown = generate_comparison_markdown(result)

    # Check for table markers
    assert "| Field |" in markdown
    assert "| --- |" in markdown


def test_comparison_markdown_includes_summary_sections(demo_records):
    """Test that Markdown includes expected summary sections."""
    records = demo_records[:2]
    result = compare_datasets(records)

    markdown = generate_comparison_markdown(result)

    assert "## Summary" in markdown
    assert "## Detailed Comparison" in markdown
    assert "## Dataset Details" in markdown


def test_comparison_with_different_sources(demo_records):
    """Test comparison works with datasets from different sources."""
    # Find datasets with different sources
    sources_seen = set()
    selected_records = []
    for record in demo_records:
        source = record["dataset"].get("source")
        if source not in sources_seen and len(selected_records) < 2:
            sources_seen.add(source)
            selected_records.append(record)

    if len(selected_records) >= 2:
        result = compare_datasets(selected_records)

        source_fc = next(
            (fc for fc in result.field_comparisons if fc.field_name == "source"),
            None,
        )
        assert source_fc is not None
        # If sources differ, all_same should be False
        if len(sources_seen) > 1:
            assert source_fc.all_same is False


def test_comparison_handles_missing_metadata_gracefully(demo_records):
    """Test that comparison handles datasets with missing metadata."""
    records = demo_records[:2]

    # Ensure we can compare even with sparse metadata
    result = compare_datasets(records)

    # Should not raise and should have valid structure
    assert result.datasets is not None
    assert all(isinstance(d.missing_metadata, list) for d in result.datasets)
