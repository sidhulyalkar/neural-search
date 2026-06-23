"""Tests for neural_search.literature.relationship_kg_builder."""

from __future__ import annotations

import json
from pathlib import Path

from neural_search.graph.schema import KnowledgeGraph, make_node_id, validate_graph
from neural_search.literature.kg_builder import add_findings_to_graph
from neural_search.literature.relationship_kg_builder import (
    add_finding_relationships_to_graph,
    add_region_cooccurrence_to_graph,
)

SUPPORTS_EDGE = {
    "edge_type": "supports",
    "finding_id_a": "p1:f0",
    "finding_id_b": "p2:f0",
    "paper_id_a": "p1",
    "paper_id_b": "p2",
    "shared_regions": ["hippocampus"],
    "shared_tasks": ["navigation"],
    "direction_a": "increase",
    "direction_b": "increase",
    "n_supporting_papers": 1,
    "confidence": 0.7,
}

CONTRADICTS_EDGE = {
    "edge_type": "contradicts",
    "finding_id_a": "p1:f0",
    "finding_id_b": "p3:f0",
    "paper_id_a": "p1",
    "paper_id_b": "p3",
    "shared_regions": ["amygdala"],
    "shared_tasks": [],
    "direction_a": "increase",
    "direction_b": "decrease",
    "n_supporting_papers": 1,
    "confidence": 0.6,
}

REGION_COOCCURRENCE_EDGE = {
    "edge_type": "region_co_occurs_with",
    "region_a": "amygdala",
    "region_b": "hippocampus",
    "n_findings": 3,
    "finding_ids": ["p1:f0", "p2:f0", "p3:f0"],
    "confidence": 0.55,
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


class TestAddFindingRelationshipsToGraph:
    def test_supports_edge_creates_finding_nodes_and_edge(self, tmp_path):
        path = tmp_path / "finding_edges.jsonl"
        _write_jsonl(path, [SUPPORTS_EDGE])

        graph = KnowledgeGraph()
        stats = add_finding_relationships_to_graph(graph, path)

        assert stats["finding_nodes_added"] == 2
        assert stats["edges_added"] == 1

        node_a = make_node_id("finding", "p1:f0")
        node_b = make_node_id("finding", "p2:f0")
        assert node_a in graph.nodes
        assert node_b in graph.nodes

        edges = [e for e in graph.edges.values() if e.edge_type == "finding_supports_finding"]
        assert len(edges) == 1
        assert edges[0].directed is False
        assert edges[0].properties["shared_regions"] == ["hippocampus"]

    def test_contradicts_edge_creates_correct_edge_type(self, tmp_path):
        path = tmp_path / "finding_edges.jsonl"
        _write_jsonl(path, [CONTRADICTS_EDGE])

        graph = KnowledgeGraph()
        add_finding_relationships_to_graph(graph, path)

        edges = [e for e in graph.edges.values() if e.edge_type == "finding_contradicts_finding"]
        assert len(edges) == 1

    def test_graph_validates_after_relationship_ingestion(self, tmp_path):
        path = tmp_path / "finding_edges.jsonl"
        _write_jsonl(path, [SUPPORTS_EDGE, CONTRADICTS_EDGE])

        graph = KnowledgeGraph()
        add_finding_relationships_to_graph(graph, path)
        validate_graph(graph)  # raises on invalid graph

    def test_skips_record_missing_finding_ids(self, tmp_path):
        path = tmp_path / "finding_edges.jsonl"
        _write_jsonl(path, [{"edge_type": "supports", "finding_id_a": "p1:f0"}])

        graph = KnowledgeGraph()
        stats = add_finding_relationships_to_graph(graph, path)
        assert stats["edges_added"] == 0
        assert stats["edges_skipped"] == 1

    def test_skips_unknown_edge_type(self, tmp_path):
        path = tmp_path / "finding_edges.jsonl"
        bad = {**SUPPORTS_EDGE, "edge_type": "co_occurs_in"}
        _write_jsonl(path, [bad])

        graph = KnowledgeGraph()
        stats = add_finding_relationships_to_graph(graph, path)
        assert stats["edges_added"] == 0
        assert stats["edges_skipped"] == 1

    def test_missing_file_returns_empty_stats(self, tmp_path):
        graph = KnowledgeGraph()
        stats = add_finding_relationships_to_graph(graph, tmp_path / "missing.jsonl")
        assert stats == {"finding_nodes_added": 0, "edges_added": 0, "edges_skipped": 0}

    def test_attaches_to_existing_rich_finding_nodes(self, tmp_path):
        """Relationship edges must connect to the same nodes add_findings_to_graph creates."""
        findings_path = tmp_path / "findings.jsonl"
        _write_jsonl(
            findings_path,
            [
                {
                    "paper_id": "paper:test:p1",
                    "finding_id": "p1:f0",
                    "finding_text": "Theta power increased in hippocampus during navigation.",
                    "result_direction": "increase",
                    "regions": ["hippocampus"],
                    "tasks": ["navigation"],
                    "modalities": [],
                    "species": [],
                    "confidence": 0.9,
                },
                {
                    "paper_id": "paper:test:p2",
                    "finding_id": "p2:f0",
                    "finding_text": "Theta power increased in hippocampus during navigation.",
                    "result_direction": "increase",
                    "regions": ["hippocampus"],
                    "tasks": ["navigation"],
                    "modalities": [],
                    "species": [],
                    "confidence": 0.9,
                },
            ],
        )
        edges_path = tmp_path / "finding_edges.jsonl"
        _write_jsonl(edges_path, [SUPPORTS_EDGE])

        graph = KnowledgeGraph()
        add_findings_to_graph(graph, findings_path)
        node_a = make_node_id("finding", "p1:f0")
        assert graph.nodes[node_a].properties.get("placeholder") is not True

        add_finding_relationships_to_graph(graph, edges_path)

        # The rich finding node (with real finding_text) must remain — not be
        # overwritten by a placeholder from the relationship builder.
        assert graph.nodes[node_a].label != "p1:f0"
        edges = [e for e in graph.edges.values() if e.edge_type == "finding_supports_finding"]
        assert len(edges) == 1
        assert node_a in (edges[0].source_node_id, edges[0].target_node_id)
        validate_graph(graph)


class TestAddRegionCooccurrenceToGraph:
    def test_creates_region_nodes_and_edge(self, tmp_path):
        path = tmp_path / "region_cooccurrence.jsonl"
        _write_jsonl(path, [REGION_COOCCURRENCE_EDGE])

        graph = KnowledgeGraph()
        stats = add_region_cooccurrence_to_graph(graph, path)

        assert stats["region_nodes_added"] == 2
        assert stats["edges_added"] == 1

        node_a = make_node_id("brain_region", "amygdala")
        node_b = make_node_id("brain_region", "hippocampus")
        assert node_a in graph.nodes
        assert node_b in graph.nodes

        edges = [
            e for e in graph.edges.values() if e.edge_type == "region_co_occurs_with_region"
        ]
        assert len(edges) == 1
        assert edges[0].directed is False
        assert edges[0].properties["n_findings"] == 3

    def test_graph_validates_after_region_ingestion(self, tmp_path):
        path = tmp_path / "region_cooccurrence.jsonl"
        _write_jsonl(path, [REGION_COOCCURRENCE_EDGE])

        graph = KnowledgeGraph()
        add_region_cooccurrence_to_graph(graph, path)
        validate_graph(graph)

    def test_skips_record_missing_regions(self, tmp_path):
        path = tmp_path / "region_cooccurrence.jsonl"
        _write_jsonl(path, [{"edge_type": "region_co_occurs_with", "region_a": "amygdala"}])

        graph = KnowledgeGraph()
        stats = add_region_cooccurrence_to_graph(graph, path)
        assert stats["edges_added"] == 0
        assert stats["edges_skipped"] == 1

    def test_attaches_to_existing_brain_region_node(self, tmp_path):
        """Region co-occurrence edges must connect to nodes add_findings_to_graph creates."""
        findings_path = tmp_path / "findings.jsonl"
        _write_jsonl(
            findings_path,
            [
                {
                    "paper_id": "paper:test:p1",
                    "finding_id": "p1:f0",
                    "finding_text": "Amygdala and hippocampus co-activated.",
                    "result_direction": "increase",
                    "regions": ["Amygdala", "Hippocampus"],
                    "tasks": [],
                    "modalities": [],
                    "species": [],
                    "confidence": 0.9,
                },
            ],
        )
        region_path = tmp_path / "region_cooccurrence.jsonl"
        _write_jsonl(region_path, [REGION_COOCCURRENCE_EDGE])

        graph = KnowledgeGraph()
        add_findings_to_graph(graph, findings_path)
        add_region_cooccurrence_to_graph(graph, region_path)

        node_a = make_node_id("brain_region", "amygdala")
        assert graph.nodes[node_a].properties.get("placeholder") is not True
        validate_graph(graph)
