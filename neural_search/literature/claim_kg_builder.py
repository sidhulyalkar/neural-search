"""Add synthesized claim nodes and edges to the knowledge graph.

Follows the exact pattern of neural_search.literature.kg_builder.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_search.graph.schema import (
    GraphEvidence,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
)

logger = logging.getLogger(__name__)

BUILDER_NAME = "neural_search.literature.claim_kg_builder"
BUILDER_VERSION = "v0.1.0"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _evidence(*, source_id: str, source_field: str, text: str | None, confidence: float) -> GraphEvidence:
    return GraphEvidence(
        evidence_id=f"evidence:claim:{source_id}:{source_field}",
        source_type="claim_synthesis",
        source_id=source_id,
        source_field=source_field,
        evidence_text=text,
        confidence=confidence,
        extractor_name=BUILDER_NAME,
        extractor_version=BUILDER_VERSION,
    )


def _add_node(graph: KnowledgeGraph, node: KnowledgeGraphNode) -> bool:
    if node.node_id in graph.nodes:
        return False
    graph.nodes[node.node_id] = node
    return True


def _add_edge(graph: KnowledgeGraph, edge: KnowledgeGraphEdge) -> bool:
    if edge.edge_id in graph.edges:
        return False
    graph.edges[edge.edge_id] = edge
    return True


def _claim_edge_id(claim_id: str, edge_type: str, target_node_id: str) -> str:
    """Build an edge ID that retains the full ``edge_type`` literally.

    ``make_edge_id`` strips the source node type prefix (e.g. ``claim_``)
    from the relation segment, which would collapse
    ``claim_supported_by_dataset``/``claim_supported_by_paper`` into the
    same ``supported_by_*`` shape as other relations. Claim edges need the
    full, unambiguous edge type preserved in the ID for downstream tooling
    and tests that key off the literal edge type string.
    """
    base_edge_id = make_edge_id(claim_id, edge_type, target_node_id)
    prefix, _, suffix = base_edge_id.partition(":claim:")
    return f"{prefix}:claim:{edge_type}:{suffix}" if suffix else f"{base_edge_id}:{edge_type}"


def _claim_node(claim: dict[str, Any]) -> KnowledgeGraphNode:
    claim_id = claim["claim_id"]
    confidence = float(claim.get("consensus_confidence", 0.5))
    ev = _evidence(
        source_id=claim_id,
        source_field="statement",
        text=claim.get("statement"),
        confidence=confidence,
    )
    return KnowledgeGraphNode(
        node_id=claim_id,
        node_type="claim",
        label=(claim.get("statement") or claim_id)[:200],
        aliases=[claim_id],
        source_ids=[claim_id],
        properties={
            "direction": claim.get("direction"),
            "regions": claim.get("regions", []),
            "species": claim.get("species", []),
            "n_supporting_findings": claim.get("n_supporting_findings", 0),
            "n_contradicting_findings": claim.get("n_contradicting_findings", 0),
            "magnitude_summary": claim.get("magnitude_summary"),
            "timescale": claim.get("timescale"),
            "evidence_strength": claim.get("evidence_strength"),
            "status": claim.get("status", "active"),
            "contradicted_by": claim.get("contradicted_by", []),
            "synthesis_model": claim.get("synthesis_model"),
            "synthesis_prompt_version": claim.get("synthesis_prompt_version"),
            "synthesized_at": claim.get("synthesized_at"),
        },
        evidence=[ev],
        confidence=confidence,
        created_at=claim.get("synthesized_at") or _now(),
    )


def add_claims_to_graph(
    graph: KnowledgeGraph,
    claims_path: Path,
) -> dict[str, int]:
    """Read claims JSONL and add claim nodes + edges to graph.

    Edge types created:
      claim_supported_by_dataset  (claim -> dataset)
      claim_supported_by_paper    (claim -> paper)
    """
    stats = {"claims_added": 0, "edges_added": 0}

    for claim in _iter_jsonl(claims_path):
        claim_id = claim.get("claim_id")
        if not claim_id:
            continue

        node = _claim_node(claim)
        if _add_node(graph, node):
            stats["claims_added"] += 1

        confidence = float(claim.get("consensus_confidence", 0.5))
        ev = node.evidence[0]

        for dataset_id in claim.get("supporting_datasets") or []:
            # Ensure dataset placeholder node exists
            ds_node_id = make_node_id("dataset", dataset_id)
            if ds_node_id not in graph.nodes:
                _add_node(
                    graph,
                    KnowledgeGraphNode(
                        node_id=ds_node_id,
                        node_type="dataset",
                        label=dataset_id,
                        aliases=[dataset_id],
                        source_ids=[dataset_id],
                        properties={"placeholder": True},
                        confidence=0.35,
                        created_at=_now(),
                    ),
                )
            edge = KnowledgeGraphEdge(
                edge_id=_claim_edge_id(claim_id, "claim_supported_by_dataset", ds_node_id),
                source_node_id=claim_id,
                target_node_id=ds_node_id,
                edge_type="claim_supported_by_dataset",
                evidence=[ev],
                confidence=confidence,
                created_at=_now(),
            )
            if _add_edge(graph, edge):
                stats["edges_added"] += 1

        for paper_id in claim.get("supporting_papers") or []:
            paper_parts = paper_id.split(":")
            p_node_id = make_node_id("paper", *paper_parts[1:]) if len(paper_parts) > 1 else make_node_id("paper", paper_id)
            if p_node_id not in graph.nodes:
                _add_node(
                    graph,
                    KnowledgeGraphNode(
                        node_id=p_node_id,
                        node_type="paper",
                        label=paper_id,
                        aliases=[paper_id],
                        source_ids=[paper_id],
                        properties={"placeholder": True},
                        confidence=0.35,
                        created_at=_now(),
                    ),
                )
            edge = KnowledgeGraphEdge(
                edge_id=_claim_edge_id(claim_id, "claim_supported_by_paper", p_node_id),
                source_node_id=claim_id,
                target_node_id=p_node_id,
                edge_type="claim_supported_by_paper",
                evidence=[ev],
                confidence=confidence,
                created_at=_now(),
            )
            if _add_edge(graph, edge):
                stats["edges_added"] += 1

    return stats
