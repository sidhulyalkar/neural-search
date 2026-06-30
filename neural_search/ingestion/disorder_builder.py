"""Build the Disorder-Circuit KG layer.

Creates nodes and edges mapping neuropsychiatric/neurological disorders to:
  - Disrupted circuits (disorder_disrupts_circuit)
  - Oscillation biomarkers (disorder_has_biomarker)
  - Animal model paradigms (disorder_modeled_by_paradigm)
  - Research topics (concept_related_to_topic)

Data source: data/disorders/disorder_registry.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from neural_search.graph.schema import GraphNode, GraphEdge, KnowledgeGraph

log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "disorders" / "disorder_registry.yaml"


def _load_disorders() -> list[dict[str, Any]]:
    if not DATA_PATH.exists():
        log.warning("Disorder registry not found at %s", DATA_PATH)
        return []
    with open(DATA_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("disorders", [])


def build_disorder_nodes(disorders: list[dict[str, Any]]) -> list[GraphNode]:
    nodes: list[GraphNode] = []
    for d in disorders:
        did = d["id"]
        node_id = f"disorder:{did}"
        nodes.append(
            GraphNode(
                node_id=node_id,
                node_type="disorder",
                label=d["label"],
                properties={
                    "icd11": d.get("icd11", ""),
                    "dsm5": d.get("dsm5", ""),
                    "disorder_type": d.get("type", ""),
                    "diagnostic_methods": d.get("diagnostic_methods", []),
                    "source": "disorder_registry_curated",
                },
            )
        )
    return nodes


def build_disorder_edges(disorders: list[dict[str, Any]]) -> list[GraphEdge]:
    edges: list[GraphEdge] = []

    for d in disorders:
        did = d["id"]
        disorder_id = f"disorder:{did}"

        # Disrupted circuits
        for circuit in d.get("disrupted_circuits", []):
            circuit_node_id = f"ontology_region:{circuit}"
            edges.append(
                GraphEdge(
                    edge_id=f"edge:disorder:{did}:disrupts_circuit:{circuit}",
                    source_node_id=disorder_id,
                    target_node_id=circuit_node_id,
                    edge_type="disorder_disrupts_circuit",
                    confidence=0.85,
                    properties={
                        "source": "disorder_registry_curated",
                        "circuit": circuit,
                    },
                )
            )

        # Oscillation biomarkers
        for bm in d.get("oscillation_biomarkers", []):
            band = bm.get("band", "unknown")
            direction = bm.get("direction", "")
            region = bm.get("region", "")
            note = bm.get("note", "")
            oscillation_id = f"oscillation:{band}"
            edge_id = f"edge:disorder:{did}:has_biomarker:{band}_{direction}"
            edges.append(
                GraphEdge(
                    edge_id=edge_id,
                    source_node_id=disorder_id,
                    target_node_id=oscillation_id,
                    edge_type="disorder_has_biomarker",
                    confidence=0.80,
                    properties={
                        "band": band,
                        "direction": direction,
                        "region": region,
                        "note": note,
                        "source": "disorder_registry_curated",
                    },
                )
            )

        # Species models → paradigm/species nodes
        for model in d.get("species_models", []):
            species = model.get("species", "")
            model_name = model.get("model", "")
            face_validity = model.get("face_validity", "")
            if species and model_name:
                safe_model = model_name.lower().replace(" ", "_").replace("-", "_")
                paradigm_id = f"paradigm:{safe_model}"
                edges.append(
                    GraphEdge(
                        edge_id=f"edge:disorder:{did}:modeled_by:{species}_{safe_model}",
                        source_node_id=disorder_id,
                        target_node_id=paradigm_id,
                        edge_type="disorder_modeled_by_paradigm",
                        confidence=0.75,
                        properties={
                            "species": species,
                            "model_name": model_name,
                            "face_validity": face_validity,
                            "source": "disorder_registry_curated",
                        },
                    )
                )

        # Research topics
        for topic in d.get("topics", []):
            topic_id = f"topic:{topic}"
            edges.append(
                GraphEdge(
                    edge_id=f"edge:disorder:{did}:related_to_topic:{topic}",
                    source_node_id=disorder_id,
                    target_node_id=topic_id,
                    edge_type="concept_related_to_topic",
                    confidence=0.80,
                    properties={"source": "disorder_registry_curated"},
                )
            )

    return edges


def build_disorder_kg() -> KnowledgeGraph:
    disorders = _load_disorders()
    nodes = build_disorder_nodes(disorders)
    edges = build_disorder_edges(disorders)
    log.info(
        "Disorder KG: %d disorder nodes, %d edges (%d circuit, %d biomarker, %d paradigm, %d topic)",
        len(nodes), len(edges),
        sum(1 for e in edges if e.edge_type == "disorder_disrupts_circuit"),
        sum(1 for e in edges if e.edge_type == "disorder_has_biomarker"),
        sum(1 for e in edges if e.edge_type == "disorder_modeled_by_paradigm"),
        sum(1 for e in edges if e.edge_type == "concept_related_to_topic"),
    )
    return KnowledgeGraph(nodes=nodes, edges=edges)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kg = build_disorder_kg()
    print(f"Disorder KG: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
    for node in list(kg.nodes.values())[:5]:
        print(f"  {node.label}")
    circuit_edges = [e for e in kg.edges.values() if e.edge_type == "disorder_disrupts_circuit"]
    print(f"\nSample circuit disruption edges:")
    for e in circuit_edges[:8]:
        disorder = e.source_node_id.split(":")[-1]
        circuit = e.target_node_id.split(":")[-1]
        print(f"  {disorder} -> disrupts -> {circuit}")
