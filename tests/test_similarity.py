"""Tests for neural_search.graph.similarity."""

from __future__ import annotations

from datetime import UTC, datetime

from neural_search.graph import build_dataset_subgraph, dataset_node_id, merge_graphs
from neural_search.graph.similarity import (
    SIMILARITY_DERIVATION_METHOD,
    add_similarity_edges_to_graph,
    build_similarity_edges,
    compute_dataset_similarity,
    find_similar_datasets,
)
from neural_search.normalized import (
    make_dataset_id,
    make_evidence_label_id,
    make_paper_id,
)
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord, UsabilityFlags


def _label(label_type: str, label: str, confidence: float = 0.9) -> EvidenceLabel:
    return EvidenceLabel(
        id=make_evidence_label_id(label_type, label),
        label=label,
        label_type=label_type,
        confidence=confidence,
        evidence_text=label,
        source_field=label_type,
        source_value=label,
        extractor_name="test",
        extractor_version="v0.5.0",
    )


def _dataset(source_id: str, **overrides) -> NormalizedDatasetRecord:
    payload = {
        "dataset_id": make_dataset_id("dandi", source_id),
        "source": "dandi",
        "source_id": source_id,
        "title": f"Dataset {source_id}",
        "description": "Test dataset.",
        "species": [_label("species", "mouse")],
        "modalities": [_label("modality", "Neuropixels")],
        "brain_regions": [_label("brain_region", "hippocampus")],
        "tasks": [_label("task", "reversal_learning")],
        "linked_papers": [make_paper_id("openalex", "W123")],
        "usability_flags": UsabilityFlags(has_neural_data=True),
        "created_at": "2026-05-24T00:00:00+00:00",
        "extractor_version": "v0.5.0",
    }
    payload.update(overrides)
    return NormalizedDatasetRecord(**payload)


def _two_similar_dataset_graph():
    ds_a = _dataset("000001")
    ds_b = _dataset("000002")
    graph = merge_graphs([build_dataset_subgraph(ds_a), build_dataset_subgraph(ds_b)])
    return graph, dataset_node_id(ds_a), dataset_node_id(ds_b)


class TestComputeDatasetSimilarity:
    def test_shared_concepts_yield_positive_score(self):
        graph, id_a, id_b = _two_similar_dataset_graph()
        sim = compute_dataset_similarity(graph, id_a, id_b)
        assert sim.similarity_score > 0
        assert sim.shared_tasks == ["reversal_learning"]
        assert sim.shared_regions == ["Hippocampus"] or sim.shared_regions == ["hippocampus"]

    def test_no_shared_concepts_yields_zero_score(self):
        graph, id_a, id_b = _two_similar_dataset_graph()
        # Overwrite one dataset's neighbors with no overlap by building a
        # fresh disjoint pair instead of mutating the shared-concept graph.
        ds_c = _dataset(
            "000003",
            modalities=[_label("modality", "EEG")],
            brain_regions=[_label("brain_region", "amygdala")],
            tasks=[_label("task", "fear_conditioning")],
        )
        graph2 = merge_graphs([build_dataset_subgraph(_dataset("000004")), build_dataset_subgraph(ds_c)])
        id_x = dataset_node_id(_dataset("000004"))
        id_y = dataset_node_id(ds_c)
        sim = compute_dataset_similarity(graph2, id_x, id_y)
        assert sim.similarity_score == 0.0


class TestFindSimilarDatasets:
    def test_returns_similar_dataset_above_threshold(self):
        graph, id_a, id_b = _two_similar_dataset_graph()
        results = find_similar_datasets(graph, id_a, min_similarity=0.1)
        assert any(r.target_id == id_b for r in results)


class TestBuildSimilarityEdgesLifecycleMetadata:
    def test_edge_created_for_similar_pair(self):
        graph, id_a, id_b = _two_similar_dataset_graph()
        edges = build_similarity_edges(graph, min_similarity=0.1, min_shared_concepts=1)
        assert len(edges) == 1
        edge = edges[0]
        assert edge.edge_type == "dataset_similar_to_dataset"
        assert edge.directed is False
        assert {edge.source_node_id, edge.target_node_id} == {id_a, id_b}

    def test_edge_carries_lifecycle_metadata(self):
        graph, _id_a, _id_b = _two_similar_dataset_graph()
        edges = build_similarity_edges(graph, min_similarity=0.1, min_shared_concepts=1)
        props = edges[0].properties

        assert props["derivation_method"] == SIMILARITY_DERIVATION_METHOD
        assert props["review_status"] == "unreviewed"
        assert props["calibration_bin"] in {"low", "medium", "high"}
        refresh_due = datetime.fromisoformat(props["refresh_due"])
        assert refresh_due > datetime.now(UTC)

    def test_calibration_bin_matches_score_thresholds(self):
        graph, _id_a, _id_b = _two_similar_dataset_graph()
        edges = build_similarity_edges(graph, min_similarity=0.1, min_shared_concepts=1)
        score = edges[0].confidence
        bin_ = edges[0].properties["calibration_bin"]
        if score >= 0.75:
            assert bin_ == "high"
        elif score >= 0.55:
            assert bin_ == "medium"
        else:
            assert bin_ == "low"

    def test_add_similarity_edges_to_graph_inserts_edges(self):
        graph, id_a, id_b = _two_similar_dataset_graph()
        n_added = add_similarity_edges_to_graph(graph, min_similarity=0.1, min_shared_concepts=1)
        assert n_added == 1
        similarity_edges = [
            e for e in graph.edges.values() if e.edge_type == "dataset_similar_to_dataset"
        ]
        assert len(similarity_edges) == 1
        assert similarity_edges[0].properties["review_status"] == "unreviewed"

    def test_below_threshold_pairs_produce_no_edge(self):
        # Partial overlap (shared task only) keeps the score below 1.0 so a
        # near-1.0 threshold genuinely excludes the pair.
        ds_a = _dataset("000005")
        ds_b = _dataset(
            "000006",
            modalities=[_label("modality", "EEG")],
            brain_regions=[_label("brain_region", "amygdala")],
        )
        graph = merge_graphs([build_dataset_subgraph(ds_a), build_dataset_subgraph(ds_b)])
        edges = build_similarity_edges(graph, min_similarity=0.99, min_shared_concepts=1)
        assert edges == []
