"""Tests for Sprint 2 DuckDB coverage store."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from neural_search.coverage.duckdb_store import CoverageStore


def _sample_records() -> list[dict]:
    return [
        {
            "dataset_id": "dataset:dandi:000001",
            "source": "dandi",
            "source_id": "000001",
            "title": "Mouse hippocampus CA1 LFP during spatial navigation",
            "species": ["mus_musculus"],
            "brain_regions": ["ca1", "hippocampus"],
            "modalities": ["lfp", "extracellular_ephys"],
            "tasks": ["spatial_navigation"],
            "behavioral_events": ["reward"],
            "usability_flags": {"has_raw_data": True, "has_behavior": True},
        },
        {
            "dataset_id": "dataset:dandi:000002",
            "source": "dandi",
            "source_id": "000002",
            "title": "Rat prefrontal cortex mPFC during reversal learning",
            "species": ["rattus_norvegicus"],
            "brain_regions": ["mPFC", "OFC"],
            "modalities": ["extracellular_ephys"],
            "tasks": ["reversal_learning"],
            "behavioral_events": ["lick", "reward"],
            "usability_flags": {"has_raw_data": True, "has_behavior": True},
        },
        {
            "dataset_id": "dataset:openneuro:ds001",
            "source": "openneuro",
            "source_id": "ds001",
            "title": "Human fMRI visual cortex retinotopy",
            "species": ["homo_sapiens"],
            "brain_regions": ["v1", "v2", "visual_cortex"],
            "modalities": ["fmri"],
            "tasks": ["visual_perception"],
            "behavioral_events": [],
            "usability_flags": {"has_processed_data": True, "has_standard_format": True},
        },
        {
            "dataset_id": "dataset:neurovault:001",
            "source": "neurovault",
            "source_id": "001",
            "title": "Whole-brain fMRI contrast maps",
            "species": ["homo_sapiens"],
            "brain_regions": [],
            "modalities": ["fmri"],
            "tasks": [],
            "behavioral_events": [],
            "usability_flags": {},
        },
    ]


@pytest.fixture
def store(tmp_path: Path) -> CoverageStore:
    db = tmp_path / "test_ledger.duckdb"
    s = CoverageStore(db)
    records = _sample_records()
    # Write records to a temp JSONL for load_dataset_mappings
    jsonl = tmp_path / "corpus.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in records))
    s.build(jsonl, snapshot_id="test", replace=True)
    return s


class TestCoverageStoreBuild:
    def test_datasets_loaded(self, store: CoverageStore) -> None:
        n = store.sql("SELECT COUNT(*) FROM datasets").fetchone()[0]
        assert n == 4

    def test_entries_created(self, store: CoverageStore) -> None:
        n = store.sql("SELECT COUNT(*) FROM coverage_entries").fetchone()[0]
        assert n > 0

    def test_ontology_regions_loaded(self, store: CoverageStore) -> None:
        n = store.sql("SELECT COUNT(*) FROM ontology_regions").fetchone()[0]
        assert n >= 106

    def test_ontology_species_loaded(self, store: CoverageStore) -> None:
        n = store.sql("SELECT COUNT(*) FROM ontology_species").fetchone()[0]
        assert n >= 5

    def test_uberon_enrichment_on_brain_region_entries(self, store: CoverageStore) -> None:
        rows = store.sql(
            "SELECT value_id, uberon_id FROM coverage_entries "
            "WHERE dimension='brain_regions' AND uberon_id IS NOT NULL"
        ).fetchall()
        assert len(rows) > 0
        ids = {r[0] for r in rows}
        assert "ca1" in ids

    def test_allen_ccf_enrichment(self, store: CoverageStore) -> None:
        rows = store.sql(
            "SELECT value_id, allen_ccf_mouse_id FROM coverage_entries "
            "WHERE dimension='brain_regions' AND allen_ccf_mouse_id IS NOT NULL"
        ).fetchall()
        assert any(r[0] == "ca1" and r[1] == "382" for r in rows)

    def test_ncbitaxon_enrichment_on_species_entries(self, store: CoverageStore) -> None:
        # Species value_ids are canonicalized by canonical_species_id() —
        # "mus_musculus" normalises to "mouse"; check the NCBITaxon column instead.
        rows = store.sql(
            "SELECT value_id, ncbitaxon_id FROM coverage_entries "
            "WHERE dimension='species' AND ncbitaxon_id IS NOT NULL"
        ).fetchall()
        assert len(rows) > 0
        ncbi_ids = {r[1] for r in rows}
        assert "NCBITaxon:10090" in ncbi_ids  # mouse

    def test_incremental_build_skips_existing(self, store: CoverageStore, tmp_path: Path) -> None:
        n_before = store.sql("SELECT COUNT(*) FROM datasets").fetchone()[0]
        # Build again with same data — should add 0 new datasets
        jsonl = tmp_path / "corpus.jsonl"
        jsonl.write_text("\n".join(json.dumps(r) for r in _sample_records()))
        stats = store.build(jsonl, snapshot_id="test")
        assert stats["new_datasets"] == 0
        n_after = store.sql("SELECT COUNT(*) FROM datasets").fetchone()[0]
        assert n_after == n_before


class TestGapQueries:
    def test_gap_matrix_returns_rows(self, store: CoverageStore) -> None:
        rows = store.gap_matrix("brain_regions", "modalities").fetchall()
        assert len(rows) > 0

    def test_gap_matrix_species_filter(self, store: CoverageStore) -> None:
        mouse_rows = store.gap_matrix(
            "brain_regions", "modalities", species_filter="mus_musculus"
        ).fetchall()
        all_rows = store.gap_matrix("brain_regions", "modalities").fetchall()
        assert len(mouse_rows) <= len(all_rows)

    def test_uncovered_regions_excludes_covered(self, store: CoverageStore) -> None:
        uncovered = {r[0] for r in store.uncovered_regions().fetchall()}
        assert "ca1" not in uncovered
        assert "hippocampus" not in uncovered

    def test_uncovered_regions_includes_unrepresented(self, store: CoverageStore) -> None:
        uncovered = {r[0] for r in store.uncovered_regions().fetchall()}
        # Most regions should be uncovered in this small sample
        assert len(uncovered) > 50

    def test_dark_pairs_returns_zero_count_pairs(self, store: CoverageStore) -> None:
        rows = store.dark_pairs("brain_regions", "modalities", top_n=10).fetchall()
        for row in rows:
            n_observed = row[2]
            assert n_observed == 0

    def test_source_coverage_rates(self, store: CoverageStore) -> None:
        rows = store.source_coverage_rates().fetchall()
        sources = {r[0] for r in rows}
        assert {"dandi", "openneuro", "neurovault"} <= sources

    def test_atlas_coverage_allen_ccf(self, store: CoverageStore) -> None:
        rows = store.atlas_coverage("allen_ccf_mouse_id").fetchall()
        assert len(rows) > 0
        ids = {r[0] for r in rows}
        assert "382" in ids  # CA1

    def test_atlas_coverage_uberon(self, store: CoverageStore) -> None:
        rows = store.atlas_coverage("uberon_id").fetchall()
        uberon_ids = {r[0] for r in rows}
        assert "UBERON:0003881" in uberon_ids  # CA1

    def test_coverage_summary_shape(self, store: CoverageStore) -> None:
        summary = store.coverage_summary()
        assert summary["total_datasets"] == 4
        assert "dimension_coverage" in summary
        dims = set(summary["dimension_coverage"].keys())
        assert {"brain_regions", "modalities", "species"} <= dims

    def test_raw_sql(self, store: CoverageStore) -> None:
        result = store.sql(
            "SELECT COUNT(*) FROM ontology_regions WHERE uberon_id IS NOT NULL"
        ).fetchone()[0]
        assert result >= 106


class TestContextManager:
    def test_context_manager_closes_connection(self, tmp_path: Path) -> None:
        db = tmp_path / "ctx_test.duckdb"
        with CoverageStore(db) as store:
            store.upsert_ontology()  # populate ontology tables
            n = store.sql("SELECT COUNT(*) FROM ontology_regions").fetchone()[0]
            assert n >= 106
        # After close, attempting to use the store should raise
        with pytest.raises(duckdb.ConnectionException):
            store.sql("SELECT 1")


def test_region_dataset_counts_returns_list(tmp_path):
    """region_dataset_counts returns a list of dicts with region_id, region_label, n_datasets."""
    from neural_search.coverage.duckdb_store import CoverageStore
    store = CoverageStore(tmp_path / "test.duckdb")
    test_corpus = tmp_path / "corpus.jsonl"
    test_corpus.write_text(
        '{"id": "dandi:000001", "source": "dandi", "source_id": "000001", '
        '"title": "Test dataset", "brain_regions": ["visual_cortex"], '
        '"modalities": ["ephys"], "species": ["mouse"], "tasks": [], '
        '"has_behavior": false, "has_raw_data": true}\n'
    )
    store.build(corpus_path=test_corpus)
    counts = store.region_dataset_counts()
    assert isinstance(counts, list)
    assert len(counts) > 0, "visual_cortex should appear above min_confidence threshold"
    for item in counts:
        assert "region_id" in item
        assert "region_label" in item
        assert "n_datasets" in item
        assert isinstance(item["n_datasets"], int)


def test_datasets_for_region_returns_list(tmp_path):
    """datasets_for_region returns datasets tagged with that region."""
    from neural_search.coverage.duckdb_store import CoverageStore
    store = CoverageStore(tmp_path / "test.duckdb")
    test_corpus = tmp_path / "corpus.jsonl"
    test_corpus.write_text(
        '{"id": "dandi:000001", "source": "dandi", "source_id": "000001", '
        '"title": "Visual cortex recording", "brain_regions": ["visual_cortex"], '
        '"modalities": ["ephys"], "species": ["mouse"], "tasks": [], '
        '"has_behavior": false, "has_raw_data": true}\n'
    )
    store.build(corpus_path=test_corpus)
    results = store.datasets_for_region("visual_cortex")
    assert isinstance(results, list)
    assert len(results) > 0, "visual_cortex dataset should be found above min_confidence threshold"
    for r in results:
        assert "dataset_id" in r
        assert "source" in r
        assert "title" in r
        assert "access_tier" in r
        assert "confidence" in r
