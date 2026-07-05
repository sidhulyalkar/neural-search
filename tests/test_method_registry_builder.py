"""Tests for the method-registry KG edge builder."""

from __future__ import annotations

from neural_search.graph.builder import build_graph_from_records
from neural_search.graph.method_registry_builder import (
    build_method_registry_edges,
    build_method_registry_subgraph,
    load_method_registry,
)
from neural_search.graph.schema import (
    SUPPORTED_EDGE_TYPES,
    make_node_id,
    validate_graph,
)


def test_build_method_registry_edges_matches_registry_size():
    registry = load_method_registry()
    edges = build_method_registry_edges(registry)
    expected = sum(len(link.taxonomy_method_ids) for link in registry.links)
    assert len(edges) == expected


def test_all_edges_are_method_supports_analysis():
    registry = load_method_registry()
    edges = build_method_registry_edges(registry)
    assert edges
    assert "method_supports_analysis" in SUPPORTED_EDGE_TYPES
    for edge in edges:
        assert edge.edge_type == "method_supports_analysis"


def test_edge_ids_match_upstream_node_id_conventions():
    """Regression guard: methods_builder mints method:<id> (not make_node_id),
    while graph.builder mints analysis_affordance nodes via make_node_id.
    If either upstream format ever changes, this must fail loudly instead of
    silently producing dangling edges."""

    registry = load_method_registry()
    edges = build_method_registry_edges(registry)
    for edge in edges:
        assert edge.source_node_id.startswith("method:")
        assert not edge.source_node_id.startswith("node:")
        family = _family_from_target(edge.target_node_id)
        assert edge.target_node_id == make_node_id("analysis_affordance", family)


def _family_from_target(target_node_id: str) -> str:
    # node:analysis_affordance:<family> -> <family>
    return target_node_id.split(":", 2)[2]


def test_build_method_registry_subgraph_validates():
    kg = build_method_registry_subgraph()
    validate_graph(kg)
    assert len(kg.nodes) == 0
    assert len(kg.edges) > 0


def test_merged_graph_has_no_dangling_method_supports_analysis_edges():
    graph = build_graph_from_records()
    method_edges = [
        edge for edge in graph.edges.values() if edge.edge_type == "method_supports_analysis"
    ]
    assert method_edges
    for edge in method_edges:
        assert edge.source_node_id in graph.nodes, f"dangling source: {edge.source_node_id}"
        assert edge.target_node_id in graph.nodes, f"dangling target: {edge.target_node_id}"


def test_merged_graph_includes_taxonomy_method_nodes():
    graph = build_graph_from_records()
    method_nodes = [n for n in graph.nodes.values() if n.node_type == "method"]
    assert len(method_nodes) >= 25
