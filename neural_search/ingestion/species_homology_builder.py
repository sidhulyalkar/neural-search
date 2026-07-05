"""Build the species homology KG layer from data/ontology/species_homology.yaml."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from neural_search.graph.schema import GraphEdge, GraphNode, KnowledgeGraph

log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "species" / "species_homology.yaml"


def _load_homology() -> dict[str, Any]:
    with open(DATA_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _region_node_id(region_id: str, species: str) -> str:
    return f"ontology_region:{species}:{region_id}"


def build_species_nodes(data: dict[str, Any]) -> list[GraphNode]:
    """Create species nodes from the registry."""
    nodes: list[GraphNode] = []
    for sp in data.get("species", []):
        nodes.append(
            GraphNode(
                node_id=f"species:{sp['id']}",
                node_type="species",
                label=sp["label"],
                properties={
                    "atlas": sp.get("atlas", ""),
                    "common_model": sp.get("common_model", False),
                },
            )
        )
    return nodes


def build_homology_edges(data: dict[str, Any]) -> list[GraphEdge]:
    """Build region_has_homolog_in edges between all member pairs in each group."""
    edges: list[GraphEdge] = []

    for group in data.get("homologs", []):
        group_id = group["group_id"]
        confidence = group.get("confidence", "medium")
        basis = group.get("basis", [])
        divergence = group.get("divergence", "")
        members = group.get("members", [])

        # Create pairwise homolog edges between all species in the group
        for i, src in enumerate(members):
            for tgt in members[i + 1 :]:
                src_region = src["region_id"]
                src_species = src["species"]
                tgt_region = tgt["region_id"]
                tgt_species = tgt["species"]

                src_node_id = _region_node_id(src_region, src_species)
                tgt_node_id = _region_node_id(tgt_region, tgt_species)

                edge_id = (
                    f"edge:homolog:{group_id}:{src_species}:{src_region}"
                    f":{tgt_species}:{tgt_region}"
                )
                edges.append(
                    GraphEdge(
                        edge_id=edge_id,
                        source_node_id=src_node_id,
                        target_node_id=tgt_node_id,
                        edge_type="region_has_homolog_in",
                        directed=False,
                        confidence=_confidence_score(confidence),
                        properties={
                            "group_id": group_id,
                            "confidence": confidence,
                            "basis": basis,
                            "divergence": divergence,
                            "src_notes": src.get("notes", ""),
                            "tgt_notes": tgt.get("notes", ""),
                        },
                    )
                )

    return edges


def _confidence_score(label: str) -> float:
    return {"high": 0.9, "medium": 0.6, "low": 0.3}.get(label, 0.5)


def build_homology_kg() -> KnowledgeGraph:
    data = _load_homology()
    nodes = build_species_nodes(data)
    edges = build_homology_edges(data)
    log.info("Species homology KG: %d nodes, %d edges", len(nodes), len(edges))
    return KnowledgeGraph(nodes=nodes, edges=edges)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kg = build_homology_kg()
    print(f"Species homology KG: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
    high_conf = [
        e for e in kg.edges
        if e.properties.get("confidence") == "high"
    ]
    print(f"  High-confidence homologs: {len(high_conf)}")
    medium_conf = [
        e for e in kg.edges
        if e.properties.get("confidence") == "medium"
    ]
    print(f"  Medium-confidence homologs: {len(medium_conf)}")
