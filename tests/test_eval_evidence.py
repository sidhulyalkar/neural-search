"""Tests for evidence data models."""
from __future__ import annotations

from neural_search.eval.evidence import (
    LFVote,
    PairEvidence,
    QuerySpec,
    compute_metadata_completeness,
    dataset_evidence_from_record,
)
from neural_search.eval.query_decomposition import decompose_query


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


class TestQueryDecomposition:
    def test_extracts_fmri_modality(self):
        record = {
            "query_id": "q_0001",
            "intent": "META_ANALYSIS",
            "query": "human fMRI reward prediction error reinforcement learning task",
            "scientific_goal": "Find datasets for meta-analysis.",
            "required_evidence": ["species", "modality"],
            "nice_to_have": [],
            "known_failure_modes": ["resting-state fMRI with reward words in description"],
        }
        spec = decompose_query(record)
        assert "fmri" in spec.required_modalities
        assert "human" in spec.required_species

    def test_extracts_hard_negatives(self):
        record = {
            "query_id": "q_0002",
            "intent": "MODEL_VALIDATION",
            "query": "mouse visual cortex calcium imaging",
            "scientific_goal": "Find mouse calcium datasets.",
            "required_evidence": [],
            "nice_to_have": [],
            "known_failure_modes": [
                "mouse electrophysiology visual cortex — modality mismatch",
                "human visual fMRI — species mismatch",
            ],
        }
        spec = decompose_query(record)
        assert len(spec.hard_negatives) == 2
        assert "calcium_imaging" in spec.required_modalities
        assert "mouse" in spec.required_species

    def test_neuropixels_query(self):
        record = {
            "query_id": "q_0003",
            "intent": "PIPELINE_REUSE",
            "query": "extracellular electrophysiology spike sorting neuropixels single unit",
            "scientific_goal": "Find ephys datasets.",
            "required_evidence": ["modality"],
            "nice_to_have": [],
            "known_failure_modes": [],
        }
        spec = decompose_query(record)
        assert any(m in spec.required_modalities for m in ["neuropixels", "extracellular_ephys"])


class TestPairEvidenceConstruction:
    def test_pair_evidence_links_query_and_dataset(self):
        from neural_search.eval.evidence import dataset_evidence_from_record
        q_record = {
            "query_id": "q_0001",
            "intent": "META_ANALYSIS",
            "query": "human fMRI reward",
            "scientific_goal": "meta-analysis",
            "required_evidence": [],
            "nice_to_have": [],
            "known_failure_modes": ["resting state"],
        }
        d_record = {
            "source": "dandi", "source_id": "000004", "title": "Human ephys",
            "description": None, "species": ["human"], "modalities": ["extracellular_ephys"],
            "brain_regions": [], "tasks": [], "license": None, "url": None,
            "has_raw_data": True, "has_processed_data": False,
            "has_behavior": False, "has_trials": False,
            "data_standards": [], "metadata_json": {},
        }
        spec = decompose_query(q_record)
        evidence = dataset_evidence_from_record(d_record)
        pair = PairEvidence(
            query_id="q_0001", record_id="dandi:000004",
            query=spec, dataset=evidence,
            pooled_from=["usefulness"], min_rank=3, priority="high",
        )
        d = pair.to_dict()
        assert d["query_id"] == "q_0001"
        assert d["dataset"]["record_id"] == "dandi:000004"
        assert d["query"]["required_species"] == ["human"]
