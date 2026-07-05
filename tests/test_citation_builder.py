"""Tests for the graph-scoped citation edge builder
(neural_search.ingestion.citation_builder.build_citation_edges_for_graph),
which fixes two real bugs in the pre-existing build_citation_edges():
node ID prefix mismatch with paper_node_builder.py's real paper nodes, and
re-scanning raw local files instead of the actual graph being built."""

from __future__ import annotations

import json

import neural_search.ingestion.citation_builder as citation_builder
from neural_search.graph.schema import KnowledgeGraph, KnowledgeGraphNode


def _write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _graph_with_openalex_papers(*bare_ids: str) -> KnowledgeGraph:
    nodes = {}
    for bare_id in bare_ids:
        node_id = f"node:paper:openalex:{bare_id}"
        nodes[node_id] = KnowledgeGraphNode(
            node_id=node_id, node_type="paper", label=bare_id, properties={"source": "openalex"}
        )
    return KnowledgeGraph(nodes=nodes, edges={})


def test_emits_edge_only_when_both_papers_are_real_graph_nodes(tmp_path, monkeypatch):
    citation_path = tmp_path / "citation_edges.jsonl"
    _write_jsonl(
        citation_path,
        [
            {"citing_paper_id": "paper:openalex:W1", "cited_paper_id": "paper:openalex:W2", "citing_year": 2021},
            {"citing_paper_id": "paper:openalex:W1", "cited_paper_id": "paper:openalex:W999", "citing_year": 2021},
        ],
    )
    monkeypatch.setattr(citation_builder, "CITATION_JSONL", citation_path)
    graph = _graph_with_openalex_papers("W1", "W2")

    edges = citation_builder.build_citation_edges_for_graph(graph)

    assert len(edges) == 1
    edge = edges[0]
    assert edge.source_node_id == "node:paper:openalex:W1"
    assert edge.target_node_id == "node:paper:openalex:W2"
    assert edge.edge_type == "paper_cites_paper"


def test_uses_node_prefixed_id_scheme_matching_paper_node_builder(tmp_path, monkeypatch):
    citation_path = tmp_path / "citation_edges.jsonl"
    _write_jsonl(
        citation_path,
        [{"citing_paper_id": "paper:openalex:W1", "cited_paper_id": "paper:openalex:W2"}],
    )
    monkeypatch.setattr(citation_builder, "CITATION_JSONL", citation_path)
    graph = _graph_with_openalex_papers("W1", "W2")

    edges = citation_builder.build_citation_edges_for_graph(graph)

    # The old build_citation_edges() used a bare "paper:openalex:W1" id with
    # no "node:" prefix -- that would never match a real graph node.
    assert edges[0].source_node_id.startswith("node:paper:openalex:")


def test_ignores_non_openalex_paper_nodes(tmp_path, monkeypatch):
    citation_path = tmp_path / "citation_edges.jsonl"
    _write_jsonl(
        citation_path,
        [{"citing_paper_id": "paper:openalex:W1", "cited_paper_id": "paper:openalex:W2"}],
    )
    monkeypatch.setattr(citation_builder, "CITATION_JSONL", citation_path)
    # W2 exists but is crossref-sourced, not openalex -- citation_edges.jsonl
    # only ever has OpenAlex-format ids on both sides, so this must not match.
    graph = KnowledgeGraph(
        nodes={
            "node:paper:openalex:W1": KnowledgeGraphNode(
                node_id="node:paper:openalex:W1", node_type="paper", label="W1", properties={"source": "openalex"}
            ),
            "node:paper:crossref:W2": KnowledgeGraphNode(
                node_id="node:paper:crossref:W2", node_type="paper", label="W2", properties={"source": "crossref"}
            ),
        },
        edges={},
    )

    edges = citation_builder.build_citation_edges_for_graph(graph)

    assert edges == []


def test_returns_empty_when_citation_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(citation_builder, "CITATION_JSONL", tmp_path / "does_not_exist.jsonl")
    graph = _graph_with_openalex_papers("W1", "W2")

    edges = citation_builder.build_citation_edges_for_graph(graph)

    assert edges == []


def test_returns_empty_when_no_openalex_papers_in_graph(tmp_path, monkeypatch):
    citation_path = tmp_path / "citation_edges.jsonl"
    _write_jsonl(
        citation_path,
        [{"citing_paper_id": "paper:openalex:W1", "cited_paper_id": "paper:openalex:W2"}],
    )
    monkeypatch.setattr(citation_builder, "CITATION_JSONL", citation_path)
    graph = KnowledgeGraph(nodes={}, edges={})

    edges = citation_builder.build_citation_edges_for_graph(graph)

    assert edges == []


def test_deduplicates_repeated_pairs(tmp_path, monkeypatch):
    citation_path = tmp_path / "citation_edges.jsonl"
    _write_jsonl(
        citation_path,
        [
            {"citing_paper_id": "paper:openalex:W1", "cited_paper_id": "paper:openalex:W2"},
            {"citing_paper_id": "paper:openalex:W1", "cited_paper_id": "paper:openalex:W2"},
        ],
    )
    monkeypatch.setattr(citation_builder, "CITATION_JSONL", citation_path)
    graph = _graph_with_openalex_papers("W1", "W2")

    edges = citation_builder.build_citation_edges_for_graph(graph)

    assert len(edges) == 1


def test_build_citation_kg_for_graph_wraps_in_knowledge_graph(tmp_path, monkeypatch):
    citation_path = tmp_path / "citation_edges.jsonl"
    _write_jsonl(
        citation_path,
        [{"citing_paper_id": "paper:openalex:W1", "cited_paper_id": "paper:openalex:W2"}],
    )
    monkeypatch.setattr(citation_builder, "CITATION_JSONL", citation_path)
    graph = _graph_with_openalex_papers("W1", "W2")

    layer_kg = citation_builder.build_citation_kg_for_graph(graph)

    assert isinstance(layer_kg, KnowledgeGraph)
    assert len(layer_kg.edges) == 1
    assert len(layer_kg.nodes) == 0
