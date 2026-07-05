"""Tests for the reanalysis-insight-synthesizer's ranking logic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from neural_search.graph.schema import GraphEdge, GraphNode, KnowledgeGraph

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "generate_reanalysis_insight_report.py"
_spec = importlib.util.spec_from_file_location("generate_reanalysis_insight_report", SCRIPT_PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


def _dataset_node(node_id: str, label: str) -> GraphNode:
    return GraphNode(node_id=node_id, node_type="dataset", label=label)


def _bridge_edge(candidate: str, precedent: str, method: str, confidence: float) -> GraphEdge:
    return GraphEdge(
        edge_id=f"edge:bridge:{candidate}:{precedent}:{method}",
        source_node_id=candidate,
        target_node_id=precedent,
        edge_type="dataset_reanalysis_bridge_dataset",
        confidence=confidence,
        properties={"method": method, "explanation": "test"},
    )


def test_top_bridge_opportunities_caps_rows_per_precedent():
    precedent = "node:dataset:dandi:000001"
    nodes = [_dataset_node(precedent, "Precedent")]
    edges = []
    for i in range(5):
        candidate = f"node:dataset:dandi:{i:06d}"
        nodes.append(_dataset_node(candidate, f"Candidate {i}"))
        edges.append(_bridge_edge(candidate, precedent, "Coherence", 0.9 - i * 0.01))

    graph = KnowledgeGraph(nodes=nodes, edges=edges)
    rows = _module._top_bridge_opportunities(graph, top_n=10)

    # Only MAX_ROWS_PER_PRECEDENT rows survive even though 5 candidates qualify.
    assert len(rows) == _module.MAX_ROWS_PER_PRECEDENT
    # Highest-confidence candidates win.
    assert rows[0]["candidate_label"] == "Candidate 0"
    assert rows[1]["candidate_label"] == "Candidate 1"


def test_top_bridge_opportunities_ranks_by_confidence_across_precedents():
    nodes = []
    edges = []
    for i in range(3):
        precedent = f"node:dataset:dandi:precedent{i}"
        candidate = f"node:dataset:dandi:candidate{i}"
        nodes.append(_dataset_node(precedent, f"Precedent {i}"))
        nodes.append(_dataset_node(candidate, f"Candidate {i}"))
        edges.append(_bridge_edge(candidate, precedent, "Coherence", 0.5 + i * 0.1))

    graph = KnowledgeGraph(nodes=nodes, edges=edges)
    rows = _module._top_bridge_opportunities(graph, top_n=2)

    assert len(rows) == 2
    assert rows[0]["confidence"] > rows[1]["confidence"]
    assert rows[0]["candidate_label"] == "Candidate 2"  # highest confidence (0.7)
