"""Validation tests for DuckDB coverage store metrics.

These tests run against the REAL coverage DB (not fixtures) and assert that
current metric values meet known baselines. They act as regression guards.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_search.coverage.duckdb_store import CoverageStore


@pytest.fixture(scope="module")
def store():
    s = CoverageStore()
    yield s
    s.close()


class TestCoverageBaselines:
    """Regression assertions against known coverage baselines."""

    def test_brain_region_coverage_above_45_pct(self, store):
        summary = store.coverage_summary()
        pct = summary["dimension_coverage"]["brain_regions"]["pct"]
        assert pct >= 45.0, f"Region coverage {pct:.1f}% below 45% baseline"

    def test_modality_coverage_above_80_pct(self, store):
        summary = store.coverage_summary()
        pct = summary["dimension_coverage"]["modalities"]["pct"]
        assert pct >= 80.0, f"Modality coverage {pct:.1f}% below 80% baseline"

    def test_species_coverage_above_70_pct(self, store):
        summary = store.coverage_summary()
        pct = summary["dimension_coverage"]["species"]["pct"]
        assert pct >= 70.0, f"Species coverage {pct:.1f}% below 70% baseline"

    def test_task_coverage_above_23_pct_baseline(self, store):
        summary = store.coverage_summary()
        pct = summary["dimension_coverage"]["tasks"]["pct"]
        assert pct >= 23.0, f"Task coverage {pct:.1f}% dropped below 23% baseline"

    def test_total_datasets_above_7000(self, store):
        summary = store.coverage_summary()
        assert summary["total_datasets"] >= 7000

    def test_total_entries_above_25000(self, store):
        summary = store.coverage_summary()
        assert summary["total_entries"] >= 25000

    def test_region_dataset_counts_returns_many_regions(self, store):
        counts = store.region_dataset_counts(min_confidence=0.0)
        assert len(counts) >= 100

    def test_uncovered_regions_excludes_heavily_covered(self, store):
        # uncovered_regions() returns a DuckDBPyRelation; rows are (id, label, ...)
        rows = store.uncovered_regions().fetchall()
        uncovered_ids = {r[0] for r in rows}
        # Hippocampus has 800+ datasets — must not appear in uncovered
        assert "hippocampus" not in uncovered_ids

    def test_gap_matrix_has_region_modality_combos(self, store):
        rows = store.gap_matrix(row_dim="brain_regions", col_dim="modalities").fetchall()
        assert len(rows) > 0
        # Each row is (dim_a_value, dim_b_value, n_datasets)
        for row in rows[:5]:
            assert row[2] >= 0

    def test_source_coverage_rates_has_dandi(self, store):
        rows = store.source_coverage_rates().fetchall()
        sources = [r[0] for r in rows]
        assert "dandi" in sources


class TestRegionDatasetQueries:
    """Integration tests for region-specific dataset queries."""

    def test_hippocampus_has_many_datasets(self, store):
        result = store.datasets_for_region("hippocampus", limit=5)
        assert len(result) > 0
        assert result[0]["dataset_id"] is not None

    def test_datasets_for_region_respects_limit(self, store):
        result = store.datasets_for_region("visual_cortex", limit=3)
        assert len(result) <= 3

    def test_datasets_for_region_has_required_fields(self, store):
        result = store.datasets_for_region("hippocampus", limit=1)
        assert len(result) > 0
        ds = result[0]
        for key in ("dataset_id", "source", "title", "access_tier", "confidence"):
            assert key in ds, f"Missing field: {key}"
