from __future__ import annotations

import json
from pathlib import Path

from neural_search.graph.schema import KnowledgeGraph, validate_graph
from neural_search.literature.claim_kg_builder import add_claims_to_graph


def _write_claims(path: Path, claims: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(c) for c in claims), encoding="utf-8")


SAMPLE_CLAIM = {
    "claim_id": "node:claim:hippocampus_increase_abc12345",
    "statement": "Theta oscillations increase during spatial navigation in mouse hippocampus",
    "direction": "increase",
    "regions": ["hippocampus"],
    "species": ["mouse"],
    "consensus_confidence": 0.87,
    "n_supporting_findings": 5,
    "n_contradicting_findings": 0,
    "magnitude_summary": "r=0.7",
    "timescale": "millisecond",
    "evidence_strength": "direct",
    "status": "active",
    "supporting_papers": ["paper:openalex:W123"],
    "supporting_datasets": ["dandi:000026"],
    "contradicted_by": [],
    "synthesis_model": "claude-haiku-4-5-20251001",
    "synthesis_prompt_version": "synthesis_v1",
    "synthesized_at": "2026-06-21T00:00:00+00:00",
}


def test_add_claims_creates_claim_node(tmp_path):
    claims_path = tmp_path / "claims.jsonl"
    _write_claims(claims_path, [SAMPLE_CLAIM])

    graph = KnowledgeGraph()
    stats = add_claims_to_graph(graph, claims_path)

    assert stats["claims_added"] == 1
    assert SAMPLE_CLAIM["claim_id"] in graph.nodes


def test_add_claims_creates_supported_by_dataset_edge(tmp_path):
    claims_path = tmp_path / "claims.jsonl"
    _write_claims(claims_path, [SAMPLE_CLAIM])

    graph = KnowledgeGraph()
    add_claims_to_graph(graph, claims_path)

    edge_ids = list(graph.edges.keys())
    dataset_edges = [e for e in edge_ids if "claim_supported_by_dataset" in e]
    assert len(dataset_edges) == 1


def test_add_claims_creates_supported_by_paper_edge(tmp_path):
    claims_path = tmp_path / "claims.jsonl"
    _write_claims(claims_path, [SAMPLE_CLAIM])

    graph = KnowledgeGraph()
    add_claims_to_graph(graph, claims_path)

    edge_ids = list(graph.edges.keys())
    paper_edges = [e for e in edge_ids if "claim_supported_by_paper" in e]
    assert len(paper_edges) == 1


def test_add_claims_graph_validates(tmp_path):
    claims_path = tmp_path / "claims.jsonl"
    _write_claims(claims_path, [SAMPLE_CLAIM])

    graph = KnowledgeGraph()
    add_claims_to_graph(graph, claims_path)
    validate_graph(graph)  # raises on invalid graph


def test_add_claims_skips_missing_claim_id(tmp_path):
    claims_path = tmp_path / "claims.jsonl"
    bad_claim = {"statement": "no id here"}
    _write_claims(claims_path, [bad_claim])

    graph = KnowledgeGraph()
    stats = add_claims_to_graph(graph, claims_path)
    assert stats["claims_added"] == 0
