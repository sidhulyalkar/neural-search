"""Tests for the reanalysis-candidate edge builder."""

from __future__ import annotations

from neural_search.graph.method_registry_builder import load_method_registry
from neural_search.graph.reanalysis_candidates import (
    build_reanalysis_candidate_edges,
    dataset_node_id,
    extract_str_list,
)

EPHYS_RECORD = {
    "source": "dandi",
    "source_id": "000123",
    "dataset_id": "dataset:dandi:000123",
    "modalities": ["neuropixels"],
    "tasks": [],
    "species": ["mouse"],
    "brain_regions": ["hippocampus"],
    "linked_papers": [],
}

EEG_RECORD_WITH_PAPERS = {
    "source": "openneuro",
    "source_id": "ds555",
    "dataset_id": "dataset:openneuro:ds555",
    "modalities": ["eeg"],
    "tasks": [],
    "species": ["human"],
    "brain_regions": [],
    "linked_papers": ["paper:openalex:W123"],
}

NO_MATCH_RECORD = {
    "source": "curated",
    "source_id": "xyz",
    "dataset_id": "dataset:curated:xyz",
    "modalities": ["something_unrecognized"],
    "tasks": [],
    "species": [],
    "brain_regions": [],
    "linked_papers": [],
}


def test_dataset_node_id_uses_source_and_source_id():
    assert dataset_node_id(EPHYS_RECORD) == "node:dataset:dandi:000123"


def test_dataset_node_id_falls_back_to_dataset_id_when_source_id_missing():
    record = {"source": "dandi", "dataset_id": "dataset:dandi:000999"}
    assert dataset_node_id(record) == "node:dataset:dandi:000999"


def test_extract_str_list_handles_strings_and_dicts():
    assert extract_str_list(["mouse", {"label": "rat"}, {"id": "human"}]) == [
        "mouse",
        "rat",
        "human",
    ]
    assert extract_str_list(None) == []
    assert extract_str_list([]) == []


def test_ephys_record_produces_candidate_edges():
    edges = build_reanalysis_candidate_edges([EPHYS_RECORD])
    assert edges
    for edge in edges:
        assert edge.source_node_id == "node:dataset:dandi:000123"
        assert edge.target_node_id.startswith("method:")
        assert edge.edge_type == "dataset_old_dataset_new_method_candidate"
        assert edge.properties["requires_human_review"] is True
        assert edge.properties["has_linked_papers"] is False
        assert edge.properties["evidence_tier"] == "heuristic_candidate"


def test_has_linked_papers_flag_reflects_record():
    edges = build_reanalysis_candidate_edges([EEG_RECORD_WITH_PAPERS])
    assert edges
    assert all(edge.properties["has_linked_papers"] is True for edge in edges)


def test_no_match_record_produces_no_edges():
    edges = build_reanalysis_candidate_edges([NO_MATCH_RECORD])
    assert edges == []


def test_confidence_matches_registry_link_confidence():
    registry = load_method_registry()
    links_by_family = {link.analysis_family: link for link in registry.links}
    edges = build_reanalysis_candidate_edges([EPHYS_RECORD])
    for edge in edges:
        family = edge.properties["analysis_family"]
        assert edge.confidence == links_by_family[family].confidence


def test_unlinked_analysis_family_produces_no_edges_for_that_family():
    """Honest-gap behavior: a data form's analysis_family with no
    method_registry.yaml entry must not silently produce edges."""

    registry = load_method_registry()
    linked_families = {link.analysis_family for link in registry.links}
    # intracellular_ephys's analysis_families (cellular_physiology,
    # excitability_analysis) are both confirmed unlinked in method_registry.yaml.
    record = {
        "source": "dandi",
        "source_id": "000456",
        "dataset_id": "dataset:dandi:000456",
        "modalities": ["patch_clamp"],
        "tasks": [],
        "species": ["mouse"],
        "brain_regions": [],
        "linked_papers": [],
    }
    edges = build_reanalysis_candidate_edges([record])
    produced_families = {e.properties["analysis_family"] for e in edges}
    assert not produced_families & ({"cellular_physiology", "excitability_analysis"} - linked_families)


def test_no_duplicate_edge_ids_across_records():
    edges = build_reanalysis_candidate_edges([EPHYS_RECORD, EPHYS_RECORD])
    edge_ids = [e.edge_id for e in edges]
    assert len(edge_ids) == len(set(edge_ids))
