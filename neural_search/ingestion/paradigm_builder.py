"""Build the cross-species paradigm KG layer from data/paradigms/paradigm_registry.yaml.

Creates:
  - paradigm nodes
  - paradigm_validated_cross_species edges (paradigm → species_pair)
  - paradigm_engages_circuit edges
  - paradigm_targets_topic edges
  - paper_uses_paradigm edges (where key_papers are linked)
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

from neural_search.graph.schema import GraphNode, GraphEdge, KnowledgeGraph

log = logging.getLogger(__name__)

DATA_PATH = (
    Path(__file__).parent.parent.parent / "data" / "paradigms" / "paradigm_registry.yaml"
)

VALIDITY_SCORES = {
    "validated": 0.95,
    "adapted": 0.70,
    "analogous": 0.45,
}


def _load_paradigms() -> dict[str, Any]:
    with open(DATA_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _paradigm_node_id(paradigm_id: str) -> str:
    return f"paradigm:{paradigm_id}"


def build_paradigm_nodes(data: dict[str, Any]) -> list[GraphNode]:
    nodes: list[GraphNode] = []
    for p in data.get("paradigms", []):
        species_list = [impl["species"] for impl in p.get("species_implementations", [])]
        nodes.append(
            GraphNode(
                node_id=_paradigm_node_id(p["id"]),
                node_type="paradigm",
                label=p.get("label", p["id"]),
                properties={
                    "aliases": p.get("aliases", []),
                    "cognitive_construct": p.get("cognitive_construct", ""),
                    "description": p.get("description", ""),
                    "species_available": species_list,
                    "circuits_engaged": p.get("circuits_engaged", []),
                    "topics": p.get("topics", []),
                    "key_neural_signal": p.get("key_neural_signal", ""),
                    "key_finding": p.get("key_finding", ""),
                    "formula": p.get("formula", ""),
                },
            )
        )
    return nodes


def build_paradigm_edges(data: dict[str, Any]) -> list[GraphEdge]:
    edges: list[GraphEdge] = []

    for p in data.get("paradigms", []):
        p_id = p["id"]
        p_node_id = _paradigm_node_id(p_id)
        implementations = p.get("species_implementations", [])

        # paradigm → circuit edges
        for circuit_id in p.get("circuits_engaged", []):
            edges.append(
                GraphEdge(
                    edge_id=f"edge:paradigm:{p_id}:circuit:{circuit_id}",
                    source_node_id=p_node_id,
                    target_node_id=f"circuit:{circuit_id}",
                    edge_type="paradigm_engages_circuit",
                )
            )

        # paradigm → topic edges
        for topic_id in p.get("topics", []):
            edges.append(
                GraphEdge(
                    edge_id=f"edge:paradigm:{p_id}:topic:{topic_id}",
                    source_node_id=p_node_id,
                    target_node_id=f"topic:{topic_id}",
                    edge_type="paradigm_targets_topic",
                )
            )

        # cross-species validation edges: pairwise between all implementations
        for impl_a, impl_b in combinations(implementations, 2):
            species_a = impl_a["species"]
            species_b = impl_b["species"]

            # Use the minimum validity of the two implementations
            val_a = VALIDITY_SCORES.get(impl_a.get("cross_species_validity", "analogous"), 0.45)
            val_b = VALIDITY_SCORES.get(impl_b.get("cross_species_validity", "analogous"), 0.45)
            confidence = min(val_a, val_b)

            edge_id = f"edge:paradigm:{p_id}:cross_species:{species_a}:{species_b}"
            edges.append(
                GraphEdge(
                    edge_id=edge_id,
                    source_node_id=f"species:{species_a}",
                    target_node_id=f"species:{species_b}",
                    edge_type="paradigm_validated_cross_species",
                    directed=False,
                    confidence=confidence,
                    properties={
                        "paradigm_id": p_id,
                        "paradigm_label": p.get("label", p_id),
                        "species_a": species_a,
                        "species_b": species_b,
                        "modality_a": impl_a.get("response_modality", ""),
                        "modality_b": impl_b.get("response_modality", ""),
                        "neural_measure_a": impl_a.get("neural_measure", ""),
                        "neural_measure_b": impl_b.get("neural_measure", ""),
                        "platform_a": impl_a.get("platform", ""),
                        "platform_b": impl_b.get("platform", ""),
                        "validated_in": impl_a.get("validated_in", ""),
                    },
                )
            )

            # Also link paradigm → species nodes
            for species in [species_a, species_b]:
                edges.append(
                    GraphEdge(
                        edge_id=f"edge:paradigm:{p_id}:species:{species}",
                        source_node_id=p_node_id,
                        target_node_id=f"species:{species}",
                        edge_type="same_task_cross_species",
                        properties={"paradigm_id": p_id},
                    )
                )

    return edges


def build_paradigm_kg() -> KnowledgeGraph:
    data = _load_paradigms()
    nodes = build_paradigm_nodes(data)
    edges = build_paradigm_edges(data)
    log.info("Paradigm KG: %d nodes, %d edges", len(nodes), len(edges))
    return KnowledgeGraph(nodes=nodes, edges=edges)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kg = build_paradigm_kg()
    print(f"Paradigm KG: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
    cross_species = [e for e in kg.edges if e.edge_type == "paradigm_validated_cross_species"]
    print(f"  Cross-species validated pairs: {len(cross_species)}")
    for e in cross_species[:5]:
        print(f"  {e.properties.get('paradigm_label')}: "
              f"{e.properties.get('species_a')} ↔ {e.properties.get('species_b')} "
              f"(conf={e.confidence:.2f})")
