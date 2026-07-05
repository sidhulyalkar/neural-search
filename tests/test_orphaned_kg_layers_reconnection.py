"""Tests for the 8 KG builders reconnected into scripts/build_real_corpus_graph.py
(disorder, allen_connectivity, paradigm, oscillation, species_homology,
hcp_connectivity on 2026-07-01; scholarpedia, concept on 2026-07-02) — see
reports/architecture_connectivity_audit_2026-07-01.md for why these were
previously orphaned.

Seven of the eight are individually far from self-contained (every referenced
concept node — regions, circuits, topics, oscillation bands, paradigms, methods
— is usually created by a *different* builder or none at all), so most of
these tests check them only after resolve_dangling_edges is applied, matching
how they are actually used in scripts/build_real_corpus_graph.py. scholarpedia
is the exception — see test_scholarpedia_is_self_contained_unlike_the_other_six.
"""

from __future__ import annotations

from neural_search.graph.schema import (
    KnowledgeGraph,
    resolve_dangling_edges,
    validate_graph,
)
from neural_search.ingestion.allen_connectivity_builder import (
    build_allen_connectivity_kg,
)
from neural_search.ingestion.concept_builder import build_concept_kg
from neural_search.ingestion.disorder_builder import build_disorder_kg
from neural_search.ingestion.hcp_connectivity import build_hcp_kg
from neural_search.ingestion.oscillation_builder import build_oscillation_kg
from neural_search.ingestion.paradigm_builder import build_paradigm_kg
from neural_search.ingestion.scholarpedia_builder import build_scholarpedia_kg
from neural_search.ingestion.species_homology_builder import build_homology_kg

BUILDERS = {
    "disorder": build_disorder_kg,
    "allen_connectivity": build_allen_connectivity_kg,
    "paradigm": build_paradigm_kg,
    "oscillation": build_oscillation_kg,
    "species_homology": build_homology_kg,
    "hcp_connectivity": build_hcp_kg,
    "scholarpedia": build_scholarpedia_kg,
    "concept": build_concept_kg,
}


def test_scholarpedia_is_self_contained_unlike_the_other_six():
    """Unlike the other six (which reference concepts no single builder
    creates), scholarpedia_builder's concept/domain/alias graph is fully
    self-contained — confirmed 2026-07-02 before reconnecting it, the same
    diligence applied to the other six."""

    kg = build_scholarpedia_kg()
    dangling = [
        e for e in kg.edges.values() if e.source_node_id not in kg.nodes or e.target_node_id not in kg.nodes
    ]
    assert dangling == []


def test_each_builder_produces_edges():
    for name, builder_fn in BUILDERS.items():
        kg = builder_fn()
        assert len(kg.edges) > 0, f"{name} produced zero edges"


def test_merged_layers_have_zero_dangling_edges_after_resolution():
    nodes: dict = {}
    edges: dict = {}
    for builder_fn in BUILDERS.values():
        kg = builder_fn()
        nodes.update(kg.nodes)
        edges.update(kg.edges)
    merged = KnowledgeGraph(nodes=nodes, edges=edges)
    resolved, stub_count = resolve_dangling_edges(merged)
    validate_graph(resolved, strict=True)
    assert stub_count > 0  # confirms the fixture actually exercises the gap


def test_hcp_circuit_edges_point_at_real_region_nodes_not_edge_ids():
    """Regression test for a bug found 2026-07-01: circuit annotation edges
    set source_node_id to another edge's edge_id (a meta-edge the schema
    doesn't support), which could never resolve. Fixed to link both the
    source and target region to the circuit instead."""

    kg = build_hcp_kg()
    circuit_edges = [e for e in kg.edges.values() if e.edge_type == "circuit_studied_by_method"]
    assert circuit_edges, "fixture data should include at least one circuit annotation"
    for edge in circuit_edges:
        assert edge.source_node_id.startswith("ontology_region:")
        assert not edge.source_node_id.startswith("edge:")
        assert edge.target_node_id.startswith("circuit:")


def test_hcp_structural_connections_are_bidirectional_and_well_formed():
    kg = build_hcp_kg()
    struct_edges = [
        e for e in kg.edges.values() if e.edge_type == "region_structurally_connected"
    ]
    assert struct_edges
    for edge in struct_edges:
        assert edge.directed is False
        assert 0.0 <= edge.confidence <= 1.0
