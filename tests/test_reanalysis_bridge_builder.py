"""Tests for the evidence-backed dataset_reanalysis_bridge_dataset builder."""

from __future__ import annotations

import json

from neural_search.graph.reanalysis_bridge_builder import (
    build_reanalysis_bridge_edges,
    load_dataset_method_evidence,
    load_dataset_paper_matches,
    load_paper_method_mentions,
)
from neural_search.graph.schema import GraphNode, KnowledgeGraph, validate_graph


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_load_dataset_paper_matches_filters_to_real_matches(tmp_path):
    path = tmp_path / "paper_dataset_links.jsonl"
    _write_jsonl(
        path,
        [
            {"dataset_record_id": "dandi:000001", "paper_openalex_id": "W1", "match_method": "doi_exact", "confidence": 0.95},
            {"dataset_record_id": "dandi:000002", "paper_openalex_id": "", "match_method": "not_found", "confidence": 0.0},
            {"dataset_record_id": "dandi:000003", "paper_openalex_id": "W2", "match_method": "title_fuzzy_local", "confidence": 0.8},
        ],
    )
    matches = load_dataset_paper_matches(path)
    assert set(matches) == {"dandi:000001", "dandi:000003"}
    assert matches["dandi:000001"]["confidence"] == 0.95


def test_load_paper_method_mentions_reads_only_paper_uses_method_edges(tmp_path):
    path = tmp_path / "ner_kg.jsonl"
    _write_jsonl(
        path,
        [
            {"record_type": "metadata", "metadata": {}},
            {
                "record_type": "edge",
                "edge": {
                    "edge_type": "paper_uses_method",
                    "source_node_id": "paper:openalex:W1",
                    "target_node_id": "method:coherence",
                    "confidence": 0.72,
                },
            },
            {
                "record_type": "edge",
                "edge": {
                    "edge_type": "paper_mentions_region",
                    "source_node_id": "paper:openalex:W1",
                    "target_node_id": "ontology_region:hippocampus",
                    "confidence": 0.7,
                },
            },
        ],
    )
    mentions = load_paper_method_mentions(path)
    assert mentions == {"W1": {"coherence": 0.72}}


def test_load_dataset_method_evidence_joins_both_sources(tmp_path):
    links_path = tmp_path / "paper_dataset_links.jsonl"
    ner_path = tmp_path / "ner_kg.jsonl"
    _write_jsonl(
        links_path,
        [{"dataset_record_id": "dandi:000001", "paper_openalex_id": "W1", "match_method": "doi_exact", "confidence": 0.9}],
    )
    _write_jsonl(
        ner_path,
        [
            {
                "record_type": "edge",
                "edge": {
                    "edge_type": "paper_uses_method",
                    "source_node_id": "paper:openalex:W1",
                    "target_node_id": "method:coherence",
                    "confidence": 0.72,
                },
            }
        ],
    )
    evidence = load_dataset_method_evidence(links_path, ner_path)
    assert evidence == {
        "dandi:000001": {
            "coherence": {
                "paper_openalex_id": "W1",
                "paper_confidence": 0.9,
                "method_confidence": 0.72,
            }
        }
    }


def test_dataset_with_no_paper_match_produces_no_evidence(tmp_path):
    links_path = tmp_path / "paper_dataset_links.jsonl"
    ner_path = tmp_path / "ner_kg.jsonl"
    _write_jsonl(links_path, [])
    _write_jsonl(ner_path, [])
    assert load_dataset_method_evidence(links_path, ner_path) == {}


def _dataset_node(node_id: str) -> GraphNode:
    return GraphNode(node_id=node_id, node_type="dataset", label=node_id)


def _similarity_edge(a: str, b: str, cross_type: str = "same_region_cross_modality"):
    from neural_search.graph.schema import GraphEdge

    return GraphEdge(
        edge_id=f"edge:sim:{a}:{b}",
        source_node_id=a,
        target_node_id=b,
        edge_type="dataset_similar_to_dataset",
        directed=False,
        confidence=0.5,
        properties={"cross_type": cross_type},
    )


def test_build_reanalysis_bridge_edges_creates_candidate_to_precedent_edge():
    precedent = "node:dataset:dandi:000166"
    candidate = "node:dataset:dandi:000005"
    graph = KnowledgeGraph(
        nodes=[_dataset_node(precedent), _dataset_node(candidate)],
        edges=[_similarity_edge(candidate, precedent)],
    )
    evidence = {
        "dandi:000166": {
            "coherence": {"paper_openalex_id": "W1", "paper_confidence": 0.95, "method_confidence": 0.72}
        }
    }
    edges = build_reanalysis_bridge_edges(graph, evidence)
    assert len(edges) == 1
    edge = edges[0]
    assert edge.source_node_id == candidate
    assert edge.target_node_id == precedent
    assert edge.edge_type == "dataset_reanalysis_bridge_dataset"
    assert edge.properties["method"] == "Coherence"
    assert edge.properties["relationship_type"] == "method_reanalysis_bridge"
    assert edge.properties["requires_human_review"] is True
    assert "explanation" in edge.properties
    assert edge.confidence == 0.95 * 0.72 * 0.5
    assert edge.properties["evidence_tier"] == "evidence_backed_bridge"


def test_no_edge_when_candidate_already_has_same_method_evidence():
    precedent = "node:dataset:dandi:000166"
    candidate = "node:dataset:dandi:000005"
    graph = KnowledgeGraph(
        nodes=[_dataset_node(precedent), _dataset_node(candidate)],
        edges=[_similarity_edge(candidate, precedent)],
    )
    evidence = {
        "dandi:000166": {"coherence": {"paper_openalex_id": "W1", "paper_confidence": 0.95, "method_confidence": 0.72}},
        "dandi:000005": {"coherence": {"paper_openalex_id": "W2", "paper_confidence": 0.9, "method_confidence": 0.72}},
    }
    edges = build_reanalysis_bridge_edges(graph, evidence)
    assert edges == []


def test_no_edge_without_a_similarity_edge():
    precedent = "node:dataset:dandi:000166"
    unrelated = "node:dataset:dandi:999999"
    graph = KnowledgeGraph(nodes=[_dataset_node(precedent), _dataset_node(unrelated)], edges=[])
    evidence = {
        "dandi:000166": {"coherence": {"paper_openalex_id": "W1", "paper_confidence": 0.95, "method_confidence": 0.72}}
    }
    edges = build_reanalysis_bridge_edges(graph, evidence)
    assert edges == []


def test_result_validates_and_has_no_dangling_edges():
    precedent = "node:dataset:dandi:000166"
    candidate = "node:dataset:dandi:000005"
    graph = KnowledgeGraph(
        nodes=[_dataset_node(precedent), _dataset_node(candidate)],
        edges=[_similarity_edge(candidate, precedent)],
    )
    evidence = {
        "dandi:000166": {"coherence": {"paper_openalex_id": "W1", "paper_confidence": 0.95, "method_confidence": 0.72}}
    }
    edges = build_reanalysis_bridge_edges(graph, evidence)
    full_graph = KnowledgeGraph(nodes=graph.nodes, edges={**graph.edges, **{e.edge_id: e for e in edges}})
    validate_graph(full_graph, strict=True)
