"""Tests for the literature-contradiction-backed reinterpretation candidate builder."""

from __future__ import annotations

import json

from neural_search.graph.reinterpretation_candidate_builder import (
    build_reinterpretation_candidate_edges,
    load_paper_contradictions,
)
from neural_search.graph.schema import GraphNode, KnowledgeGraph
from neural_search.kg.schemas.evidence_tier import EvidenceTier


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_load_paper_contradictions_filters_to_contradicts_edges(tmp_path):
    path = tmp_path / "finding_edges.jsonl"
    _write_jsonl(
        path,
        [
            {
                "edge_type": "contradicts",
                "paper_id_a": "paper:openalex:W1",
                "paper_id_b": "paper:openalex:W2",
                "confidence": 0.6,
            },
            {
                "edge_type": "supports",
                "paper_id_a": "paper:openalex:W1",
                "paper_id_b": "paper:openalex:W3",
                "confidence": 0.9,
            },
        ],
    )
    contradictions = load_paper_contradictions(path)
    assert len(contradictions) == 1
    assert contradictions[0]["paper_id_a"] == "paper:openalex:W1"


def test_load_paper_contradictions_keeps_strongest_per_pair(tmp_path):
    path = tmp_path / "finding_edges.jsonl"
    _write_jsonl(
        path,
        [
            {
                "edge_type": "contradicts",
                "paper_id_a": "paper:openalex:W1",
                "paper_id_b": "paper:openalex:W2",
                "confidence": 0.4,
            },
            {
                "edge_type": "contradicts",
                "paper_id_a": "paper:openalex:W2",
                "paper_id_b": "paper:openalex:W1",
                "confidence": 0.8,
            },
        ],
    )
    contradictions = load_paper_contradictions(path)
    assert len(contradictions) == 1
    assert contradictions[0]["confidence"] == 0.8


def _dataset_node(record_id: str) -> GraphNode:
    source, source_id = record_id.split(":", 1)
    return GraphNode(
        node_id=f"node:dataset:{source}:{source_id}", node_type="dataset", label=record_id
    )


def test_build_reinterpretation_edges_requires_both_sides_matched():
    graph = KnowledgeGraph(nodes=[_dataset_node("dandi:000001")], edges=[])
    matches = {"dandi:000001": {"paper_openalex_id": "W1", "confidence": 0.9}}
    contradictions = [
        {
            "paper_id_a": "paper:openalex:W1",
            "paper_id_b": "paper:openalex:W2",  # W2 has no matched dataset
            "confidence": 0.6,
            "contradiction_subtype": "opposite_direction",
        }
    ]
    edges = build_reinterpretation_candidate_edges(graph, matches, contradictions)
    assert edges == []


def test_build_reinterpretation_edges_emits_both_directions():
    graph = KnowledgeGraph(
        nodes=[_dataset_node("dandi:000001"), _dataset_node("dandi:000002")], edges=[]
    )
    matches = {
        "dandi:000001": {"paper_openalex_id": "W1", "confidence": 0.9},
        "dandi:000002": {"paper_openalex_id": "W2", "confidence": 0.8},
    }
    contradictions = [
        {
            "paper_id_a": "paper:openalex:W1",
            "paper_id_b": "paper:openalex:W2",
            "confidence": 0.6,
            "contradiction_subtype": "opposite_direction",
            "direction_a": "increase",
            "direction_b": "decrease",
            "shared_regions": ["hippocampus"],
            "shared_tasks": [],
        }
    ]
    edges = build_reinterpretation_candidate_edges(graph, matches, contradictions)

    assert len(edges) == 2
    node_a = "node:dataset:dandi:000001"
    node_b = "node:dataset:dandi:000002"
    pairs = {(e.source_node_id, e.target_node_id) for e in edges}
    assert pairs == {(node_a, node_b), (node_b, node_a)}

    for edge in edges:
        assert edge.edge_type == "dataset_reinterpretation_candidate"
        assert edge.properties["evidence_tier"] == EvidenceTier.EVIDENCE_BACKED_BRIDGE.value
        assert edge.properties["requires_human_review"] is True
        assert edge.properties["shared_regions"] == ["hippocampus"]
        assert edge.confidence == 0.6 * 0.9 * 0.8

    a_to_b = next(e for e in edges if e.source_node_id == node_a)
    assert a_to_b.properties["direction_self"] == "increase"
    assert a_to_b.properties["direction_other"] == "decrease"


def test_build_reinterpretation_edges_skips_missing_graph_nodes():
    """dataset_paper_matches may reference a dataset that isn't actually in
    this particular graph (e.g. a fixture/subset graph) -- must not crash or
    fabricate a node."""

    graph = KnowledgeGraph(nodes=[_dataset_node("dandi:000001")], edges=[])
    matches = {
        "dandi:000001": {"paper_openalex_id": "W1", "confidence": 0.9},
        "dandi:000002": {"paper_openalex_id": "W2", "confidence": 0.8},
    }
    contradictions = [
        {
            "paper_id_a": "paper:openalex:W1",
            "paper_id_b": "paper:openalex:W2",
            "confidence": 0.6,
            "contradiction_subtype": "opposite_direction",
        }
    ]
    edges = build_reinterpretation_candidate_edges(graph, matches, contradictions)
    assert edges == []
