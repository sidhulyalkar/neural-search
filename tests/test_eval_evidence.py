"""Tests for evidence data models."""
from __future__ import annotations
import pytest
from neural_search.eval.evidence import (
    QuerySpec,
    DatasetEvidence,
    PairEvidence,
    LFVote,
    dataset_evidence_from_record,
    compute_metadata_completeness,
)


class TestQuerySpec:
    def test_default_lists_are_empty(self):
        q = QuerySpec(query_id="q1", query_text="test", intent="META_ANALYSIS", scientific_goal="x")
        assert q.required_modalities == []
        assert q.hard_negatives == []

    def test_to_dict_roundtrip(self):
        q = QuerySpec(
            query_id="q1", query_text="test", intent="META_ANALYSIS",
            scientific_goal="x", required_modalities=["fmri"], hard_negatives=["resting state"]
        )
        d = q.to_dict()
        assert d["query_id"] == "q1"
        assert d["required_modalities"] == ["fmri"]


class TestDatasetEvidence:
    def test_from_record_basic(self):
        record = {
            "source": "dandi", "source_id": "000004", "title": "A dataset",
            "description": "Human ephys", "species": ["human"],
            "modalities": ["extracellular_ephys"], "brain_regions": [],
            "tasks": [], "license": "CC-BY-4.0", "url": "https://example.com",
            "has_raw_data": True, "has_processed_data": False,
            "has_behavior": False, "has_trials": False,
            "data_standards": ["NWB"], "metadata_json": {},
        }
        ev = dataset_evidence_from_record(record)
        assert ev.record_id == "dandi:000004"
        assert ev.raw_data_available is True
        assert "raw" in ev.data_levels

    def test_metadata_completeness_all_present(self):
        record = {
            "source": "dandi", "source_id": "1", "title": "T",
            "description": "D", "species": ["human"], "modalities": ["fmri"],
            "brain_regions": ["prefrontal"], "tasks": ["reward"],
            "license": "CC-BY-4.0", "url": "https://x.com",
            "has_raw_data": True, "has_processed_data": False,
            "has_behavior": True, "has_trials": True,
            "data_standards": [], "metadata_json": {},
        }
        score = compute_metadata_completeness(record)
        assert score == 1.0

    def test_metadata_completeness_minimal(self):
        record = {
            "source": "dandi", "source_id": "2", "title": "T",
            "description": None, "species": [], "modalities": [],
            "brain_regions": [], "tasks": [], "license": None, "url": None,
            "has_raw_data": False, "has_processed_data": False,
            "has_behavior": False, "has_trials": False,
            "data_standards": [], "metadata_json": {},
        }
        score = compute_metadata_completeness(record)
        assert score < 0.3


class TestLFVote:
    def test_abstain_default_false(self):
        v = LFVote(lf_name="lf_test", label=2, confidence=0.8, rationale="ok")
        assert v.abstain is False

    def test_to_dict(self):
        v = LFVote(lf_name="lf_test", label=0, confidence=0.95, rationale="hard neg", abstain=False)
        d = v.to_dict()
        assert d["lf_name"] == "lf_test"
        assert d["label"] == 0
