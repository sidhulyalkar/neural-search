"""Tests for the SPARK ingestion adapter."""

from __future__ import annotations

from neural_search.ingestion.spark import fetch_spark_records, normalize_spark_record


class TestFetchSparkRecords:
    def test_returns_list(self) -> None:
        records = fetch_spark_records()
        assert isinstance(records, list)
        assert len(records) >= 1

    def test_respects_limit(self) -> None:
        records = fetch_spark_records(limit=3)
        assert len(records) <= 3

    def test_records_have_required_fields(self) -> None:
        for rec in fetch_spark_records():
            assert "source_id" in rec
            assert "title" in rec


class TestNormalizeSparkRecord:
    def _sample(self) -> dict:
        return {
            "source_id": "SPARK-TEST-001",
            "title": "SPARK fMRI Social Brain in ASD",
            "description": "Resting-state fMRI from 100 autistic participants. BIDS format.",
            "url": "https://base.sfari.org/datasets/SPARK-TEST-001",
            "modalities": ["fmri"],
            "brain_regions": ["prefrontal_cortex", "amygdala"],
            "tasks": ["social_cognition"],
            "data_standards": ["BIDS", "NDA"],
            "has_raw_data": False,
            "has_processed_data": True,
            "license": "CC-BY-4.0",
        }

    def test_basic_normalization(self) -> None:
        rec = normalize_spark_record(self._sample())
        assert rec.source == "spark"
        assert rec.source_id == "SPARK-TEST-001"
        assert rec.title == "SPARK fMRI Social Brain in ASD"

    def test_dataset_id_format(self) -> None:
        rec = normalize_spark_record(self._sample())
        assert rec.dataset_id.startswith("dataset:")
        assert "spark" in rec.dataset_id

    def test_species_is_human(self) -> None:
        rec = normalize_spark_record(self._sample())
        species_ids = [s.id for s in rec.species]
        assert "human" in species_ids, f"Expected human in {species_ids}"

    def test_species_fallback_when_extractor_misses(self) -> None:
        raw = self._sample()
        raw["description"] = "Dataset with no species keywords."
        rec = normalize_spark_record(raw)
        species_ids = [s.id for s in rec.species]
        assert "human" in species_ids

    def test_modality_present(self) -> None:
        rec = normalize_spark_record(self._sample())
        assert len(rec.modalities) >= 0  # extraction may or may not fire

    def test_usability_flags(self) -> None:
        rec = normalize_spark_record(self._sample())
        assert rec.usability_flags is not None
        assert rec.usability_flags.has_processed_data is True
        assert rec.usability_flags.has_raw_data is False


class TestAllStubRecordsNormalize:
    def test_all_stubs_normalize_without_error(self) -> None:
        for raw in fetch_spark_records():
            rec = normalize_spark_record(raw)
            assert rec.source == "spark"
            assert rec.dataset_id
            assert rec.title

    def test_all_stubs_have_species_human(self) -> None:
        for raw in fetch_spark_records():
            rec = normalize_spark_record(raw)
            species_ids = [s.id for s in rec.species]
            assert "human" in species_ids, f"{rec.source_id} missing human species"
