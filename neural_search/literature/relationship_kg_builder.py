"""Add cross-finding and region co-occurrence relationship edges to the knowledge graph.

Follows the exact pattern of neural_search.literature.claim_kg_builder. Reads the
JSONL artifacts produced by neural_search.literature.relationship_builder
(finding_edges.jsonl, region_cooccurrence.jsonl) and materializes them as typed
KnowledgeGraphEdge objects, so cross-paper support/contradiction signals are
traversable from the graph instead of living only in a sidecar file.

Node IDs are constructed identically to neural_search.literature.kg_builder
(``make_node_id("finding", finding_id)`` / ``make_node_id("brain_region",
normalize_node_type(region))``) so these edges attach to the same finding and
brain_region nodes that add_findings_to_graph() creates, regardless of build
order.
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
    normalize_node_type,
)

logger = logging.getLogger(__name__)

BUILDER_NAME = "neural_search.literature.relationship_kg_builder"
BUILDER_VERSION = "v0.1.0"

# relationship_builder.FindingEdge.edge_type -> graph edge_type
_FINDING_EDGE_TYPE_MAP = {
    "supports": "finding_supports_finding",
    "contradicts": "finding_contradicts_finding",
}


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


def _finding_placeholder_node(finding_id: str, paper_id: str) -> KnowledgeGraphNode:
    """Build a finding node matching kg_builder._finding_node's ID scheme.

    Used as a placeholder when relationship edges are ingested before (or
    without) the richer finding nodes from add_findings_to_graph().
    """
    return KnowledgeGraphNode(
        node_id=make_node_id("finding", finding_id),
        node_type="finding",
        label=finding_id,
        aliases=[finding_id],
        source_ids=[finding_id, paper_id] if paper_id else [finding_id],
        properties={"placeholder": True, "paper_id": paper_id},
        confidence=0.35,
        created_at=_now(),
    )


def _region_node(region: str) -> KnowledgeGraphNode:
    """Build a brain_region node matching kg_builder._concept_node's ID scheme."""
    return KnowledgeGraphNode(
        node_id=make_node_id("brain_region", normalize_node_type(region)),
        node_type="brain_region",
        label=region,
        aliases=[region],
        source_ids=[region],
        properties={"placeholder": True},
        confidence=0.5,
        created_at=_now(),
    )


def add_finding_relationships_to_graph(
    graph: KnowledgeGraph,
    finding_edges_path: Path,
) -> dict[str, int]:
    """Read finding_edges.jsonl (supports/contradicts) and add edges to the graph.

    Edge types created:
      finding_supports_finding      (undirected)
      finding_contradicts_finding   (undirected)
    """
    stats = {"finding_nodes_added": 0, "edges_added": 0, "edges_skipped": 0}

    for record in _iter_jsonl(finding_edges_path):
        mapped_type = _FINDING_EDGE_TYPE_MAP.get(record.get("edge_type"))
        finding_id_a = record.get("finding_id_a")
        finding_id_b = record.get("finding_id_b")
        if mapped_type is None or not finding_id_a or not finding_id_b:
            stats["edges_skipped"] += 1
            continue

        node_a = _finding_placeholder_node(finding_id_a, record.get("paper_id_a", ""))
        node_b = _finding_placeholder_node(finding_id_b, record.get("paper_id_b", ""))
        if _add_node(graph, node_a):
            stats["finding_nodes_added"] += 1
        if _add_node(graph, node_b):
            stats["finding_nodes_added"] += 1

        confidence = float(record.get("confidence", 0.5))
        shared_regions = record.get("shared_regions") or []
        evidence = GraphEvidence(
            evidence_id=f"evidence:relationship:{finding_id_a}:{finding_id_b}:{mapped_type}",
            source_type="finding_relationship",
            source_id=f"{finding_id_a}|{finding_id_b}",
            source_field="shared_regions",
            evidence_text=", ".join(shared_regions) or None,
            confidence=confidence,
            extractor_name=BUILDER_NAME,
            extractor_version=BUILDER_VERSION,
        )

        edge = KnowledgeGraphEdge(
            edge_id=make_edge_id(node_a.node_id, mapped_type, node_b.node_id),
            source_node_id=node_a.node_id,
            target_node_id=node_b.node_id,
            edge_type=mapped_type,
            directed=False,
            confidence=confidence,
            evidence=[evidence],
            properties={
                "shared_regions": shared_regions,
                "shared_tasks": record.get("shared_tasks") or [],
                "direction_a": record.get("direction_a"),
                "direction_b": record.get("direction_b"),
            },
            created_at=_now(),
        )
        if _add_edge(graph, edge):
            stats["edges_added"] += 1
        else:
            stats["edges_skipped"] += 1

    return stats


def add_region_cooccurrence_to_graph(
    graph: KnowledgeGraph,
    region_cooccurrence_path: Path,
) -> dict[str, int]:
    """Read region_cooccurrence.jsonl and add region_co_occurs_with_region edges."""
    stats = {"region_nodes_added": 0, "edges_added": 0, "edges_skipped": 0}

    for record in _iter_jsonl(region_cooccurrence_path):
        region_a = record.get("region_a")
        region_b = record.get("region_b")
        if record.get("edge_type") != "region_co_occurs_with" or not region_a or not region_b:
            stats["edges_skipped"] += 1
            continue

        node_a = _region_node(region_a)
        node_b = _region_node(region_b)
        if _add_node(graph, node_a):
            stats["region_nodes_added"] += 1
        if _add_node(graph, node_b):
            stats["region_nodes_added"] += 1

        confidence = float(record.get("confidence", 0.5))
        finding_ids = record.get("finding_ids") or []
        evidence = GraphEvidence(
            evidence_id=f"evidence:region_cooccurrence:{region_a}:{region_b}",
            source_type="region_cooccurrence",
            source_id=f"{region_a}|{region_b}",
            source_field="finding_ids",
            evidence_text=", ".join(finding_ids)[:500] or None,
            confidence=confidence,
            extractor_name=BUILDER_NAME,
            extractor_version=BUILDER_VERSION,
        )

        edge = KnowledgeGraphEdge(
            edge_id=make_edge_id(node_a.node_id, "region_co_occurs_with_region", node_b.node_id),
            source_node_id=node_a.node_id,
            target_node_id=node_b.node_id,
            edge_type="region_co_occurs_with_region",
            directed=False,
            confidence=confidence,
            evidence=[evidence],
            properties={"n_findings": record.get("n_findings", 0)},
            created_at=_now(),
        )
        if _add_edge(graph, edge):
            stats["edges_added"] += 1
        else:
            stats["edges_skipped"] += 1

    return stats
