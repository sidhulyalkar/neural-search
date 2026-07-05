"""Tests for neural_search.graph.schema.resolve_dangling_edges.

This utility exists because individual KG builder modules use inconsistent,
independently-authored node-id conventions and routinely reference concepts
no single layer creates (see reports/architecture_connectivity_audit_2026-07-01.md).
"""

from __future__ import annotations

from neural_search.graph.schema import (
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    resolve_dangling_edges,
    validate_graph,
)


def _edge(source: str, target: str, edge_id: str | None = None) -> GraphEdge:
    return GraphEdge(
        edge_id=edge_id or f"edge:{source}:{target}",
        source_node_id=source,
        target_node_id=target,
        edge_type="related_to",
    )


def test_no_dangling_edges_returns_same_graph_unchanged():
    node = GraphNode(node_id="disorder:epilepsy", node_type="disorder", label="Epilepsy")
    edge = _edge("disorder:epilepsy", "disorder:epilepsy")
    graph = KnowledgeGraph(nodes=[node], edges=[edge])
    resolved, stub_count = resolve_dangling_edges(graph)
    assert stub_count == 0
    assert len(resolved.nodes) == 1


def test_creates_stub_for_dangling_target_hand_rolled_id():
    node = GraphNode(node_id="disorder:epilepsy", node_type="disorder", label="Epilepsy")
    edge = _edge("disorder:epilepsy", "oscillation:theta")
    graph = KnowledgeGraph(nodes=[node], edges=[edge])
    resolved, stub_count = resolve_dangling_edges(graph)
    assert stub_count == 1
    assert "oscillation:theta" in resolved.nodes
    stub = resolved.nodes["oscillation:theta"]
    assert stub.node_type == "oscillation"
    assert stub.properties["stub"] is True
    assert stub.properties["source"] == "auto_generated_placeholder"
    assert stub.confidence == 0.3


def test_creates_stub_for_dangling_endpoint_canonical_make_node_id_format():
    node = GraphNode(node_id="node:method:fft", node_type="method", label="FFT")
    edge = _edge("node:method:fft", "node:analysis_affordance:time_frequency")
    graph = KnowledgeGraph(nodes=[node], edges=[edge])
    resolved, stub_count = resolve_dangling_edges(graph)
    assert stub_count == 1
    stub = resolved.nodes["node:analysis_affordance:time_frequency"]
    assert stub.node_type == "analysis_affordance"


def test_resolved_graph_has_zero_dangling_edges_under_strict_validation():
    node = GraphNode(node_id="paradigm:reversal_learning", node_type="paradigm", label="RL")
    edges = [
        _edge("paradigm:reversal_learning", "circuit:orbitofrontal", "e1"),
        _edge("paradigm:reversal_learning", "topic:decision_making", "e2"),
    ]
    graph = KnowledgeGraph(nodes=[node], edges=edges)
    resolved, stub_count = resolve_dangling_edges(graph)
    assert stub_count == 2
    validate_graph(resolved, strict=True)


def test_deduplicates_stub_nodes_referenced_by_multiple_edges():
    node = GraphNode(node_id="disorder:a", node_type="disorder", label="A")
    edges = [
        _edge("disorder:a", "topic:shared", "e1"),
        _edge("disorder:a", "topic:shared", "e2"),
    ]
    graph = KnowledgeGraph(nodes=[node], edges=edges)
    resolved, stub_count = resolve_dangling_edges(graph)
    assert stub_count == 1
    assert len([n for n in resolved.nodes if n == "topic:shared"]) == 1
