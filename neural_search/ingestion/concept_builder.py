"""Build the Neuroscience Concept Authority KG layer.

Creates nodes and edges for theoretical frameworks, computational models,
and neuroscience principles — the "theory layer" missing from empirical data sources.

Inspired by Scholarpedia's curated concept coverage; all metadata is original curation.

Data source: data/concepts/concept_seed.yaml

Edge types created:
  - concept_narrower_than    (concept → broader concept)
  - concept_motivates_method (concept → method it motivates)
  - concept_related_to_topic (concept → topic)
  - concept_related_to_concept
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from neural_search.graph.schema import GraphEdge, GraphNode, KnowledgeGraph

log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "concepts" / "concept_seed.yaml"


def _load_concepts() -> list[dict[str, Any]]:
    if not DATA_PATH.exists():
        log.warning("Concept seed not found at %s", DATA_PATH)
        return []
    with open(DATA_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("concepts", [])


def build_concept_nodes(concepts: list[dict[str, Any]]) -> list[GraphNode]:
    nodes: list[GraphNode] = []
    for c in concepts:
        cid = c["id"]
        node_id = f"concept:{cid}"
        aliases = c.get("aliases", [])
        nodes.append(
            GraphNode(
                node_id=node_id,
                node_type="concept",
                label=c["label"],
                properties={
                    "definition": c.get("definition", ""),
                    "formula": c.get("formula", ""),
                    "concept_type": c.get("concept_type", "theory_or_principle"),
                    "aliases": aliases,
                    "testable_predictions": c.get("testable_predictions", []),
                    "dataset_affordances": c.get("dataset_affordances", []),
                    "scholarpedia_url": c.get("scholarpedia_url", ""),
                    "source": "concept_seed_curated",
                },
            )
        )
    return nodes


def build_concept_edges(concepts: list[dict[str, Any]]) -> list[GraphEdge]:
    edges: list[GraphEdge] = []

    for c in concepts:
        cid = c["id"]
        node_id = f"concept:{cid}"

        # Hierarchy: narrower_than (this concept → broader concept)
        broader = c.get("broader_concept")
        if broader:
            broader_id = f"concept:{broader}"
            edges.append(
                GraphEdge(
                    edge_id=f"edge:concept:{cid}:narrower_than:{broader}",
                    source_node_id=node_id,
                    target_node_id=broader_id,
                    edge_type="concept_narrower_than",
                    confidence=0.95,
                    properties={"source": "concept_seed"},
                )
            )
            # Reverse: broader_concept → this (concept_broader_than)
            edges.append(
                GraphEdge(
                    edge_id=f"edge:concept:{broader}:broader_than:{cid}",
                    source_node_id=broader_id,
                    target_node_id=node_id,
                    edge_type="concept_broader_than",
                    confidence=0.95,
                    properties={"source": "concept_seed"},
                )
            )

        # Narrower concepts this concept subsumes
        for narrower in c.get("narrower_concepts", []):
            narrower_id = f"concept:{narrower}"
            edges.append(
                GraphEdge(
                    edge_id=f"edge:concept:{narrower}:narrower_than:{cid}",
                    source_node_id=narrower_id,
                    target_node_id=node_id,
                    edge_type="concept_narrower_than",
                    confidence=0.90,
                    properties={"source": "concept_seed"},
                )
            )

        # concept → methods it motivates
        for method in c.get("related_methods", []):
            method_id = f"method:{method}"
            edges.append(
                GraphEdge(
                    edge_id=f"edge:concept:{cid}:motivates_method:{method}",
                    source_node_id=node_id,
                    target_node_id=method_id,
                    edge_type="concept_motivates_method",
                    confidence=0.80,
                    properties={"source": "concept_seed"},
                )
            )

        # concept → topics
        for topic in c.get("topics", []):
            topic_id = f"topic:{topic}"
            edges.append(
                GraphEdge(
                    edge_id=f"edge:concept:{cid}:related_to_topic:{topic}",
                    source_node_id=node_id,
                    target_node_id=topic_id,
                    edge_type="concept_related_to_topic",
                    confidence=0.80,
                    properties={"source": "concept_seed"},
                )
            )

        # concept → regions it's related to
        for region in c.get("related_regions", []):
            region_id = f"ontology_region:{region}"
            edges.append(
                GraphEdge(
                    edge_id=f"edge:concept:{cid}:related_to_region:{region}",
                    source_node_id=node_id,
                    target_node_id=region_id,
                    edge_type="concept_related_to_concept",
                    confidence=0.75,
                    properties={"source": "concept_seed", "relation": "region"},
                )
            )

    return edges


def build_concept_kg() -> KnowledgeGraph:
    concepts = _load_concepts()
    nodes = build_concept_nodes(concepts)
    edges = build_concept_edges(concepts)
    log.info(
        "Concept KG: %d concept nodes, %d edges from %d concepts",
        len(nodes), len(edges), len(concepts),
    )
    return KnowledgeGraph(nodes=nodes, edges=edges)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kg = build_concept_kg()
    print(f"Concept KG: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
    for node in list(kg.nodes.values())[:5]:
        print(f"  [{node.node_type}] {node.label} — {node.properties.get('concept_type','')}")
