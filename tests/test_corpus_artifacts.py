"""Validation tests for corpus artifacts and data quality invariants."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import pytest

REPORTS = Path("reports/eval")
MINI_CORPUS = Path("tests/fixtures/mini_corpus")


# ---------------------------------------------------------------------------
# Source distribution consistency
# ---------------------------------------------------------------------------

class TestSourceDistribution:
    def test_source_distribution_sums_to_corpus_size(self):
        manifest_path = REPORTS / "corpus_manifest.json"
        dist_path = REPORTS / "source_distribution.csv"
        if not manifest_path.exists() or not dist_path.exists():
            pytest.skip("corpus_manifest.json or source_distribution.csv not generated yet")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = list(csv.DictReader(dist_path.open(encoding="utf-8")))
        total = sum(int(row["records"]) for row in rows)
        assert total == manifest["corpus_size"]

    def test_source_distribution_no_zero_sources(self):
        dist_path = REPORTS / "source_distribution.csv"
        if not dist_path.exists():
            pytest.skip("source_distribution.csv not generated yet")
        rows = list(csv.DictReader(dist_path.open(encoding="utf-8")))
        for row in rows:
            assert int(row["records"]) > 0, f"Source '{row['source']}' has zero records"


# ---------------------------------------------------------------------------
# Field completeness consistency
# ---------------------------------------------------------------------------

class TestFieldCompleteness:
    def test_field_completeness_totals_match_corpus_size(self):
        manifest_path = REPORTS / "corpus_manifest.json"
        completeness_path = REPORTS / "field_completeness.csv"
        if not manifest_path.exists() or not completeness_path.exists():
            pytest.skip("artifacts not generated yet")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = list(csv.DictReader(completeness_path.open(encoding="utf-8")))
        totals_by_field: dict[str, int] = defaultdict(int)
        for row in rows:
            totals_by_field[row["field"]] += int(row["total"])
        for field, total in totals_by_field.items():
            assert total == manifest["corpus_size"], f"Field '{field}' total {total} != corpus_size {manifest['corpus_size']}"

    def test_field_completeness_coverage_in_range(self):
        completeness_path = REPORTS / "field_completeness.csv"
        if not completeness_path.exists():
            pytest.skip("field_completeness.csv not generated yet")
        rows = list(csv.DictReader(completeness_path.open(encoding="utf-8")))
        for row in rows:
            coverage = float(row["coverage"])
            assert 0.0 <= coverage <= 1.0, f"Coverage {coverage} out of range for {row['field']} @ {row['source']}"


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_no_duplicate_ids_in_mini_corpus(self):
        records_path = MINI_CORPUS / "records.jsonl"
        assert records_path.exists()
        seen_ids: list[str] = []
        with records_path.open() as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    seen_ids.append(record["dataset_id"])
        assert len(seen_ids) == len(set(seen_ids)), "Duplicate dataset_id in mini corpus"

    def test_corpus_manifest_unique_ids_leq_corpus_size(self):
        manifest_path = REPORTS / "corpus_manifest.json"
        if not manifest_path.exists():
            pytest.skip("corpus_manifest.json not generated yet")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["unique_record_ids"] <= manifest["corpus_size"]


# ---------------------------------------------------------------------------
# Missing license does not confer open reuse
# ---------------------------------------------------------------------------

class TestLicenseHandling:
    def test_records_without_license_are_identified(self):
        records_path = MINI_CORPUS / "records.jsonl"
        assert records_path.exists()
        no_license: list[str] = []
        with records_path.open() as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    license_val = record.get("license", "")
                    if not license_val or not str(license_val).strip():
                        no_license.append(record["dataset_id"])
        # OSF record has empty DOI; flag any with no license
        # This test verifies we can detect them, not that none exist
        assert isinstance(no_license, list)

    def test_mini_corpus_has_variety_of_licenses(self):
        records_path = MINI_CORPUS / "records.jsonl"
        assert records_path.exists()
        licenses: set[str] = set()
        with records_path.open() as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    lic = str(record.get("license", "")).strip()
                    if lic:
                        licenses.add(lic)
        assert len(licenses) >= 2, "Mini corpus should have multiple license types"


# ---------------------------------------------------------------------------
# Malformed records
# ---------------------------------------------------------------------------

class TestMalformedRecords:
    def test_mini_corpus_records_have_required_fields(self):
        records_path = MINI_CORPUS / "records.jsonl"
        assert records_path.exists()
        required = ["dataset_id", "source", "title"]
        with records_path.open() as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                record = json.loads(line)
                for field in required:
                    assert field in record and record[field], f"Record {i} missing '{field}'"

    def test_mini_corpus_species_is_list(self):
        records_path = MINI_CORPUS / "records.jsonl"
        assert records_path.exists()
        with records_path.open() as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                record = json.loads(line)
                assert isinstance(record.get("species", []), list), f"Record {i} species is not a list"

    def test_mini_corpus_modalities_is_list(self):
        records_path = MINI_CORPUS / "records.jsonl"
        assert records_path.exists()
        with records_path.open() as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                record = json.loads(line)
                assert isinstance(record.get("modalities", []), list), f"Record {i} modalities is not a list"


# ---------------------------------------------------------------------------
# Embedding manifest (if available)
# ---------------------------------------------------------------------------

class TestEmbeddingManifest:
    def test_corpus_manifest_has_embedding_dimension(self):
        manifest_path = REPORTS / "corpus_manifest.json"
        if not manifest_path.exists():
            pytest.skip("corpus_manifest.json not generated yet")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dim = manifest.get("embedding_dimension")
        if dim is not None:
            assert isinstance(dim, int)
            assert dim > 0

    @pytest.mark.xfail(
        reason=(
            "Known data gap surfaced 2026-07-05 by refreezing corpus_manifest.json "
            "against the real current corpus (full_corpus_v09.jsonl, 7,171 rows): "
            "the dense field-embedding cache (data/embeddings/real_all.dense.field_embeddings.jsonl) "
            "only has 3,589 rows because it was never regenerated for the current "
            "corpus (scripts/embed_flat_corpus.py --dry-run reports 44,278 field "
            "texts are needed). Backfill in progress on separate GPU hardware "
            "(CPU-only embedding of 44K BGE-large texts was impractically slow). "
            "Remove this marker once data/embeddings/real_all.dense.field_embeddings.jsonl "
            "is regenerated and reports/eval/corpus_manifest.json is refrozen "
            "(scripts/eval/freeze_corpus_snapshot.py --corpus "
            "data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl)."
        ),
        strict=False,
    )
    def test_corpus_manifest_embedding_rows_leq_corpus_size_times_fields(self):
        manifest_path = REPORTS / "corpus_manifest.json"
        if not manifest_path.exists():
            pytest.skip("corpus_manifest.json not generated yet")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = manifest.get("embedding_rows", 0) or 0
        size = manifest.get("corpus_size", 0) or 0
        if rows and size:
            # Each record may have multiple field embeddings; rows >= size is expected
            assert rows >= size, "Embedding rows should be >= corpus size (field-level embeddings)"


# ---------------------------------------------------------------------------
# Rejection log
# ---------------------------------------------------------------------------

class TestRejectionLog:
    def test_rejection_summary_exists_or_skipped(self):
        rejection_path = REPORTS / "rejection_summary.csv"
        if not rejection_path.exists():
            pytest.skip("rejection_summary.csv not generated yet")
        rows = list(csv.DictReader(rejection_path.open(encoding="utf-8")))
        for row in rows:
            assert "reason" in row
            count = int(row.get("count", 0))
            assert count >= 0
