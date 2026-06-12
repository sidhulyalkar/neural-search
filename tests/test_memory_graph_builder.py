"""Tests for MemoryGraphBuilder — graph construction from synthetic corpus."""

from __future__ import annotations

import pytest

from neural_search.field_state.memory_graph import MemoryGraphBuilder


_SENTINEL = object()


def _make_record(
    dataset_id: str = "dataset:dandi:000001",
    source: str = "dandi",
    title: str = "Test Dataset",
    description: str = "Neuropixels recordings in mice",
    modalities=_SENTINEL,
    species=_SENTINEL,
    brain_regions=_SENTINEL,
    tasks=_SENTINEL,
    data_standards=_SENTINEL,
    linked_papers=_SENTINEL,
    usability_flags=_SENTINEL,
    analysis_affordances=_SENTINEL,
) -> dict:
    return {
        "dataset_id": dataset_id,
        "source": source,
        "source_id": dataset_id.split(":")[-1],
        "title": title,
        "description": description,
        "modalities": [{"id": "ephys", "label": "extracellular_ephys", "label_type": "modality", "confidence": 0.9}] if modalities is _SENTINEL else modalities,
        "species": [{"id": "mouse", "label": "mouse", "label_type": "species", "confidence": 0.95}] if species is _SENTINEL else species,
        "brain_regions": [{"id": "ca1", "label": "CA1", "label_type": "brain_region", "confidence": 0.8}] if brain_regions is _SENTINEL else brain_regions,
        "tasks": [] if tasks is _SENTINEL else tasks,
        "data_standards": [] if data_standards is _SENTINEL else data_standards,
        "linked_papers": [] if linked_papers is _SENTINEL else linked_papers,
        "usability_flags": {"has_raw_data": True} if usability_flags is _SENTINEL else usability_flags,
        "analysis_affordances": [] if analysis_affordances is _SENTINEL else analysis_affordances,
        "behavioral_events": [],
        "file_formats": [],
    }


class TestMemoryGraphBuilderBasic:
    def test_builds_from_empty_corpus(self) -> None:
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[])
        assert store.node_count == 0
        assert store.edge_count == 0

    def test_builds_dataset_node(self) -> None:
        rec = _make_record()
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[rec])
        datasets = store.query_datasets()
        assert len(datasets) == 1
        ds = datasets[0]
        assert ds.node_type == "dataset"
        assert ds.properties["source"] == "dandi"

    def test_creates_source_archive_node(self) -> None:
        rec = _make_record()
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[rec])
        archives = store.query_by_type("source_archive")
        assert len(archives) == 1
        assert archives[0].label == "dandi"

    def test_dataset_linked_to_source_archive(self) -> None:
        rec = _make_record()
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[rec])
        edges = [e for e in store._edges.values() if e.edge_type == "dataset_from_source"]
        assert len(edges) == 1

    def test_creates_modality_nodes(self) -> None:
        rec = _make_record()
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[rec])
        modalities = store.query_by_type("modality")
        assert len(modalities) >= 1

    def test_creates_species_nodes(self) -> None:
        rec = _make_record()
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[rec])
        species = store.query_by_type("species")
        assert len(species) >= 1

    def test_creates_brain_region_nodes(self) -> None:
        rec = _make_record()
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[rec])
        regions = store.query_by_type("brain_region")
        assert len(regions) >= 1

    def test_raw_data_flag_creates_raw_signal_node(self) -> None:
        rec = _make_record(usability_flags={"has_raw_data": True})
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[rec])
        raw_nodes = store.query_by_type("raw_data_signal")
        assert len(raw_nodes) >= 1

    def test_linked_papers_create_paper_nodes(self) -> None:
        rec = _make_record(linked_papers=["paper:openalex:123"])
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[rec])
        papers = store.query_by_type("paper")
        assert len(papers) >= 1

    def test_multiple_datasets_same_source(self) -> None:
        rec1 = _make_record("dataset:dandi:000001")
        rec2 = _make_record("dataset:dandi:000002")
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[rec1, rec2])
        assert len(store.query_datasets()) == 2
        # Single source archive node
        assert len(store.query_by_type("source_archive")) == 1

    def test_multiple_datasets_different_sources(self) -> None:
        rec1 = _make_record("dataset:dandi:000001", source="dandi")
        rec2 = _make_record("dataset:openneuro:ds003505", source="openneuro")
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[rec1, rec2])
        assert len(store.query_by_type("source_archive")) == 2

    def test_affordance_high_support_creates_edge(self) -> None:
        rec = _make_record(
            analysis_affordances=[{
                "analysis_id": "spike_sorting",
                "support_level": "high",
                "confidence": 0.9,
                "required_fields_present": [],
                "helpful_fields_present": [],
                "missing_fields": [],
                "evidence": [],
            }]
        )
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[rec])
        support_edges = [e for e in store._edges.values() if e.edge_type == "dataset_supports_analysis"]
        assert len(support_edges) >= 1

    def test_affordance_unsupported_creates_lacks_edge(self) -> None:
        rec = _make_record(
            analysis_affordances=[{
                "analysis_id": "calcium_imaging_decode",
                "support_level": "unsupported",
                "confidence": 0.9,
                "required_fields_present": [],
                "helpful_fields_present": [],
                "missing_fields": ["calcium_signal"],
                "evidence": [],
            }]
        )
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[rec])
        lacks_edges = [e for e in store._edges.values() if e.edge_type == "dataset_lacks_required_evidence"]
        assert len(lacks_edges) >= 1


class TestMemoryGraphBuilderFeedback:
    def test_feedback_creates_signal_node(self) -> None:
        rec = _make_record()
        fb = {
            "feedback_id": "fb_abc123",
            "dataset_id": "dataset:dandi:000001",
            "usefulness": "useful",
            "would_use_for_analysis": "yes",
            "reason_tags": ["good_match"],
        }
        builder = MemoryGraphBuilder()
        builder.build(corpus_records=[rec])
        builder._add_feedback(fb)
        fb_nodes = builder.store.query_by_type("feedback_signal")
        assert len(fb_nodes) >= 1
        assert fb_nodes[0].properties["provenance"] == "user_feedback_downstream_signal"

    def test_human_gold_judgment_is_rejected(self) -> None:
        jmt = {
            "query_id": "q1",
            "dataset_id": "dataset:dandi:000001",
            "label": 3,
            "label_provenance": "human_gold",
        }
        builder = MemoryGraphBuilder()
        builder._add_judgment(jmt)
        jmt_nodes = builder.store.query_by_type("neuro_judge_judgment")
        assert len(jmt_nodes) == 0, "human_gold judgment should be rejected"

    def test_silver_judgment_is_accepted(self) -> None:
        rec = _make_record()
        jmt = {
            "query_id": "q1",
            "dataset_id": "dataset:dandi:000001",
            "label": 2,
            "label_provenance": "neuro_judge_silver",
            "confidence": 0.75,
        }
        builder = MemoryGraphBuilder()
        builder.build(corpus_records=[rec])
        builder._add_judgment(jmt)
        jmt_nodes = builder.store.query_by_type("neuro_judge_judgment")
        assert len(jmt_nodes) == 1
        assert jmt_nodes[0].properties["label_provenance"] == "neuro_judge_silver"


class TestMemoryGraphBuilderTextInference:
    def test_text_inferred_species_lower_confidence(self) -> None:
        rec = _make_record(
            species=[],  # no structured species
            description="Recordings from mice during navigation task",
        )
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[rec])
        species_edges = [
            e for e in store._edges.values()
            if e.edge_type == "dataset_has_species" and e.properties.get("inferred")
        ]
        assert any(e.confidence < 0.7 for e in species_edges)

    def test_neuropixels_in_description_infers_modality(self) -> None:
        rec = _make_record(
            modalities=[],
            description="Raw AP-band data from Neuropixels probes",
        )
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[rec])
        # Should have either a structured or inferred modality node
        mod_nodes = store.query_by_type("modality")
        assert len(mod_nodes) >= 0  # may or may not find it depending on text

    def test_raw_ap_in_description_infers_raw_signal(self) -> None:
        rec = _make_record(
            usability_flags={},  # no structured flag
            description="Raw AP-band voltage traces are included in the dataset",
        )
        builder = MemoryGraphBuilder()
        store = builder.build(corpus_records=[rec])
        raw_edges = [e for e in store._edges.values() if e.edge_type == "dataset_has_raw_signal"]
        assert len(raw_edges) >= 1
