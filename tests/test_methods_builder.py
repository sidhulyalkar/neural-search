"""Tests for the methods-taxonomy KG builder.

Regression coverage for a bug found 2026-07-01: build_methods_edges emitted
method_computes/method_used_for_topic/method_assumes/method_related_to_method
edges whose target nodes were never created by build_methods_nodes, producing
207 dangling edges once this builder was first merged into a real graph
output (scripts/build_real_corpus_graph.py).
"""

from __future__ import annotations

from neural_search.graph.schema import validate_graph
from neural_search.ingestion.methods_builder import (
    _load_taxonomy,
    build_methods_kg,
    known_method_ids,
)


def test_build_methods_kg_validates():
    kg = build_methods_kg()
    validate_graph(kg)


def test_build_methods_kg_has_no_dangling_edges():
    kg = build_methods_kg()
    dangling = [
        edge
        for edge in kg.edges.values()
        if edge.source_node_id not in kg.nodes or edge.target_node_id not in kg.nodes
    ]
    assert dangling == []


def test_known_method_ids_covers_fft():
    taxonomy = _load_taxonomy()
    ids = known_method_ids(taxonomy)
    assert "fft" in ids
    assert "dcm" in ids


def test_related_methods_referencing_unknown_ids_are_skipped_not_dangling():
    """ccm/state_space_models/kalman_filter are referenced as related_methods
    but have no taxonomy entry of their own — must be skipped, not linked to
    a nonexistent node."""

    kg = build_methods_kg()
    method_related_edges = [e for e in kg.edges.values() if e.edge_type == "method_related_to_method"]
    for edge in method_related_edges:
        assert edge.target_node_id in kg.nodes
