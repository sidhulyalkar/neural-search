"""High-level field-state memory graph store.

Wraps KnowledgeGraph with upsert, query, export/import, and validation APIs
suited for the field-state memory graph sprint.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_search.graph.schema import (
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_node_id,
)

STORE_VERSION = "v0.9.0"

# Default artifact paths (relative to repo root)
FIELD_STATE_DIR = Path("artifacts/field_state")
NODES_PATH = FIELD_STATE_DIR / "memory_graph_nodes.jsonl"
EDGES_PATH = FIELD_STATE_DIR / "memory_graph_edges.jsonl"
MANIFEST_PATH = FIELD_STATE_DIR / "memory_graph_manifest.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class FieldStateGraphStore:
    """Mutable in-memory field-state graph with JSONL persistence.

    Provides upsert, query, export/import, and invariant validation over a
    KnowledgeGraph.  All mutations are applied in-memory; call ``save()`` to
    flush to disk.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, KnowledgeGraphNode] = {}
        self._edges: dict[str, KnowledgeGraphEdge] = {}
        self._metadata: dict[str, Any] = {
            "store_version": STORE_VERSION,
            "created_at": _utc_now(),
        }

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_node(self, node: KnowledgeGraphNode) -> KnowledgeGraphNode:
        """Insert or update a node; returns the stored node."""
        existing = self._nodes.get(node.node_id)
        if existing is not None:
            updated = existing.model_copy(
                update={
                    "label": node.label,
                    "aliases": list({*existing.aliases, *node.aliases}),
                    "source_ids": list({*existing.source_ids, *node.source_ids}),
                    "properties": {**existing.properties, **node.properties},
                    "evidence": existing.evidence + node.evidence,
                    "confidence": max(existing.confidence, node.confidence),
                    "updated_at": _utc_now(),
                }
            )
            self._nodes[node.node_id] = updated
            return updated
        self._nodes[node.node_id] = node
        return node

    def upsert_edge(self, edge: KnowledgeGraphEdge) -> KnowledgeGraphEdge:
        """Insert or update an edge; returns the stored edge."""
        existing = self._edges.get(edge.edge_id)
        if existing is not None:
            updated = existing.model_copy(
                update={
                    "evidence": existing.evidence + edge.evidence,
                    "confidence": max(existing.confidence, edge.confidence),
                    "properties": {**existing.properties, **edge.properties},
                    "updated_at": _utc_now(),
                }
            )
            self._edges[edge.edge_id] = updated
            return updated
        self._edges[edge.edge_id] = edge
        return edge

    # ------------------------------------------------------------------
    # Get / lookup
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> KnowledgeGraphNode | None:
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> KnowledgeGraphEdge | None:
        return self._edges.get(edge_id)

    def get_neighbors(
        self,
        node_id: str,
        *,
        edge_types: list[str] | None = None,
        direction: str = "both",
    ) -> list[tuple[KnowledgeGraphEdge, KnowledgeGraphNode]]:
        """Return (edge, neighbor_node) pairs for a node.

        Parameters
        ----------
        direction: "out" for outgoing edges, "in" for incoming, "both" for all.
        """
        results: list[tuple[KnowledgeGraphEdge, KnowledgeGraphNode]] = []
        for edge in self._edges.values():
            if edge_types and edge.edge_type not in edge_types:
                continue
            if direction in ("out", "both") and edge.source_node_id == node_id:
                neighbor = self._nodes.get(edge.target_node_id)
                if neighbor:
                    results.append((edge, neighbor))
            if direction in ("in", "both") and edge.target_node_id == node_id:
                neighbor = self._nodes.get(edge.source_node_id)
                if neighbor:
                    results.append((edge, neighbor))
        return results

    # ------------------------------------------------------------------
    # Typed queries
    # ------------------------------------------------------------------

    def query_by_type(self, node_type: str) -> list[KnowledgeGraphNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def query_datasets(self) -> list[KnowledgeGraphNode]:
        return self.query_by_type("dataset")

    def query_by_dataset_id(self, dataset_id: str) -> KnowledgeGraphNode | None:
        node_id = make_node_id("dataset", *dataset_id.split(":")[1:]) if dataset_id.startswith("dataset:") else None
        if node_id:
            node = self.get_node(node_id)
            if node:
                return node
        # Fallback: search properties
        for node in self._nodes.values():
            if node.node_type == "dataset" and node.properties.get("dataset_id") == dataset_id:
                return node
        return None

    def query_by_source_archive(self, source: str) -> list[KnowledgeGraphNode]:
        """Return all dataset nodes belonging to a source archive."""
        archive_node_id = make_node_id("source_archive", source)
        results = []
        for _edge, node in self.get_neighbors(archive_node_id, edge_types=["dataset_from_source"], direction="in"):
            if node.node_type == "dataset":
                results.append(node)
        return results

    def query_by_modality(self, modality_label: str) -> list[KnowledgeGraphNode]:
        modality_id = make_node_id("modality", modality_label.lower().replace(" ", "_"))
        return [
            node
            for edge, node in self.get_neighbors(modality_id, direction="in")
            if edge.edge_type in ("dataset_has_modality",) and node.node_type == "dataset"
        ]

    def query_by_species(self, species_label: str) -> list[KnowledgeGraphNode]:
        species_id = make_node_id("species", species_label.lower().replace(" ", "_"))
        return [
            node
            for edge, node in self.get_neighbors(species_id, direction="in")
            if edge.edge_type == "dataset_has_species" and node.node_type == "dataset"
        ]

    def query_by_region(self, region_label: str) -> list[KnowledgeGraphNode]:
        region_id = make_node_id("brain_region", region_label.lower().replace(" ", "_"))
        return [
            node
            for edge, node in self.get_neighbors(region_id, direction="in")
            if edge.edge_type == "dataset_records_region" and node.node_type == "dataset"
        ]

    def query_by_task(self, task_label: str) -> list[KnowledgeGraphNode]:
        task_id = make_node_id("task", task_label.lower().replace(" ", "_"))
        return [
            node
            for edge, node in self.get_neighbors(task_id, direction="in")
            if edge.edge_type == "dataset_has_task" and node.node_type == "dataset"
        ]

    def query_datasets_supporting_affordance(self, affordance_id: str) -> list[KnowledgeGraphNode]:
        affordance_node_id = make_node_id("analysis_affordance", affordance_id.lower().replace(" ", "_"))
        return [
            node
            for edge, node in self.get_neighbors(affordance_node_id, direction="in")
            if edge.edge_type == "dataset_supports_analysis" and node.node_type == "dataset"
        ]

    def query_datasets_missing_evidence(self) -> list[KnowledgeGraphNode]:
        """Return dataset nodes that have a dataset_lacks_required_evidence edge."""
        dataset_ids_with_gap: set[str] = set()
        for edge in self._edges.values():
            if edge.edge_type == "dataset_lacks_required_evidence":
                dataset_ids_with_gap.add(edge.source_node_id)
        return [self._nodes[nid] for nid in dataset_ids_with_gap if nid in self._nodes]

    def query_contraindicated_datasets(self, query_node_id: str) -> list[KnowledgeGraphNode]:
        return [
            node
            for edge, node in self.get_neighbors(query_node_id, edge_types=["dataset_contraindicated_for"], direction="in")
            if node.node_type == "dataset"
        ]

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def to_knowledge_graph(self) -> KnowledgeGraph:
        return KnowledgeGraph(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            metadata=self._metadata,
        )

    def export_jsonl(self, nodes_path: Path, edges_path: Path) -> None:
        """Write nodes and edges to separate JSONL files."""
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        with nodes_path.open("w", encoding="utf-8") as fh:
            for node in self._nodes.values():
                fh.write(node.model_dump_json() + "\n")
        edges_path.parent.mkdir(parents=True, exist_ok=True)
        with edges_path.open("w", encoding="utf-8") as fh:
            for edge in self._edges.values():
                fh.write(edge.model_dump_json() + "\n")

    @classmethod
    def from_jsonl(cls, nodes_path: Path, edges_path: Path) -> FieldStateGraphStore:
        """Load a store from separate nodes/edges JSONL files."""
        store = cls()
        if nodes_path.exists():
            for line in nodes_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    node = KnowledgeGraphNode.model_validate_json(line)
                    store._nodes[node.node_id] = node
        if edges_path.exists():
            for line in edges_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    edge = KnowledgeGraphEdge.model_validate_json(line)
                    store._edges[edge.edge_id] = edge
        return store

    def write_manifest(self, path: Path, build_id: str = "") -> dict[str, Any]:
        """Write a manifest JSON describing the current graph state."""
        node_counts = {}
        for node in self._nodes.values():
            node_counts[node.node_type] = node_counts.get(node.node_type, 0) + 1
        edge_counts = {}
        for edge in self._edges.values():
            edge_counts[edge.edge_type] = edge_counts.get(edge.edge_type, 0) + 1
        manifest: dict[str, Any] = {
            "store_version": STORE_VERSION,
            "build_id": build_id or _utc_now(),
            "created_at": _utc_now(),
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_counts_by_type": node_counts,
            "edge_counts_by_type": edge_counts,
            "metadata": self._metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_invariants(self) -> list[str]:
        """Return a list of validation errors (empty means graph is valid)."""
        errors: list[str] = []

        # Edge referential integrity
        for edge in self._edges.values():
            if edge.source_node_id not in self._nodes:
                errors.append(f"edge {edge.edge_id}: source {edge.source_node_id!r} not in nodes")
            if edge.target_node_id not in self._nodes:
                errors.append(f"edge {edge.edge_id}: target {edge.target_node_id!r} not in nodes")

        # Every dataset must have at least one source_archive or source in properties
        for node in self._nodes.values():
            if node.node_type != "dataset":
                continue
            has_source = any(
                e.edge_type == "dataset_from_source" and e.source_node_id == node.node_id
                for e in self._edges.values()
            )
            if not has_source and not node.properties.get("source"):
                errors.append(f"dataset node {node.node_id!r} has no source_archive link and no source property")

        return errors

    # ------------------------------------------------------------------
    # Convenience stats
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def __repr__(self) -> str:
        return f"FieldStateGraphStore(nodes={self.node_count}, edges={self.edge_count})"
