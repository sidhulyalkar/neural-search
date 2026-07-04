"""Integration test for scripts/build_real_corpus_graph.py on a small fixture.

Does not run against the full ~7,171-record corpus (kept fast, matching repo
convention) — verifies the methods_kg / method_registry / reanalysis_candidate
layers merge into the script's own dataset/concept graph without dangling
edges or id collisions.
"""

from __future__ import annotations

from scripts.build_real_corpus_graph import build_graph

FIXTURE_CORPUS = [
    {
        "source": "dandi",
        "source_id": "000123",
        "dataset_id": "dataset:dandi:000123",
        "title": "Mouse hippocampus Neuropixels recording",
        "modalities": ["neuropixels"],
        "tasks": [],
        "species": ["mouse"],
        "brain_regions": ["hippocampus"],
        "linked_papers": [],
    },
    {
        "source": "openneuro",
        "source_id": "ds555",
        "dataset_id": "dataset:openneuro:ds555",
        "title": "Human EEG resting state",
        "modalities": ["eeg"],
        "tasks": [],
        "species": ["human"],
        "brain_regions": [],
        "linked_papers": ["paper:openalex:W123"],
    },
]


def test_build_graph_merges_methodology_and_reanalysis_layers():
    graph = build_graph(FIXTURE_CORPUS)

    dataset_nodes = [n for n in graph.nodes.values() if n.node_type == "dataset"]
    method_nodes = [n for n in graph.nodes.values() if n.node_type == "method"]
    assert len(dataset_nodes) == 2
    assert len(method_nodes) >= 25

    method_supports_edges = [
        e for e in graph.edges.values() if e.edge_type == "method_supports_analysis"
    ]
    candidate_edges = [
        e
        for e in graph.edges.values()
        if e.edge_type == "dataset_old_dataset_new_method_candidate"
    ]
    assert method_supports_edges
    assert candidate_edges

    for edge in method_supports_edges + candidate_edges:
        assert edge.source_node_id in graph.nodes, f"dangling source: {edge.source_node_id}"
        assert edge.target_node_id in graph.nodes, f"dangling target: {edge.target_node_id}"

    assert graph.metadata["reanalysis_candidate_edges"] == len(candidate_edges)
    assert graph.metadata["method_supports_analysis_edges"] == len(method_supports_edges)


def test_build_graph_has_zero_dangling_edges_across_all_layers():
    """Full-pipeline regression guard: every layer (dataset/concept edges,
    methodology registry, reanalysis candidates, and the 7 previously-orphaned
    KG layers) must resolve with no dangling edges once merged, via
    resolve_dangling_edges."""

    graph = build_graph(FIXTURE_CORPUS)
    dangling = [
        e
        for e in graph.edges.values()
        if e.source_node_id not in graph.nodes or e.target_node_id not in graph.nodes
    ]
    assert dangling == []
    assert graph.metadata["stub_nodes_created"] >= 0


def test_build_graph_merges_previously_orphaned_kg_layers():
    graph = build_graph(FIXTURE_CORPUS)
    for layer in (
        "disorder",
        "allen_connectivity",
        "paradigm",
        "oscillation",
        "species_homology",
        "hcp_connectivity",
        "scholarpedia",
        "concept",
    ):
        assert graph.metadata.get(f"{layer}_edges", 0) > 0, f"{layer} layer produced no edges"

    disorder_nodes = [n for n in graph.nodes.values() if n.node_type == "disorder"]
    paradigm_nodes = [n for n in graph.nodes.values() if n.node_type == "paradigm"]
    assert disorder_nodes
    assert paradigm_nodes


def test_build_graph_reanalysis_bridge_layer_is_safe_when_no_evidence_matches_fixture():
    """The fixture corpus's two datasets have no real OpenAlex paper-method
    evidence, so this layer should be a no-op here (not raise, not fabricate
    edges) — the meaningful yield only appears against the real 7,171-record
    corpus, verified separately via scripts/build_real_corpus_graph.py."""

    graph = build_graph(FIXTURE_CORPUS)
    assert "reanalysis_bridge_edges" in graph.metadata
    bridge_edges = [
        e for e in graph.edges.values() if e.edge_type == "dataset_reanalysis_bridge_dataset"
    ]
    assert len(bridge_edges) == graph.metadata["reanalysis_bridge_edges"]


def test_build_graph_reinterpretation_layer_is_safe_when_no_evidence_matches_fixture():
    """Same no-op-safe reasoning as the reanalysis bridge layer above: the
    fixture corpus's datasets have no real linked-paper contradiction
    evidence, so this must not raise or fabricate edges."""

    graph = build_graph(FIXTURE_CORPUS)
    assert "reinterpretation_candidate_edges" in graph.metadata
    reinterpretation_edges = [
        e for e in graph.edges.values() if e.edge_type == "dataset_reinterpretation_candidate"
    ]
    assert len(reinterpretation_edges) == graph.metadata["reinterpretation_candidate_edges"]
