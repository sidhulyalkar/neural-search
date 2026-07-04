"""Build the structural connectivity KG layer from HCP priors.

Reads data/hcp/structural_connectivity_priors.yaml (curated priors) and
creates region_structurally_connected edges with FA weights.

When actual HCP data is downloaded (see download_instructions in the YAML),
replace the priors YAML with real connectivity matrices and re-run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from neural_search.graph.schema import GraphNode, GraphEdge, KnowledgeGraph

log = logging.getLogger(__name__)

DATA_PATH = (
    Path(__file__).parent.parent.parent / "data" / "hcp" / "structural_connectivity_priors.yaml"
)

DENSITY_SCORES = {
    "very_high": 0.95,
    "high": 0.80,
    "medium": 0.60,
    "low": 0.40,
}


def _load_priors() -> dict[str, Any]:
    with open(DATA_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _region_node_id(region_id: str) -> str:
    return f"ontology_region:{region_id}"


def _tract_node_id(pathway: str) -> str:
    return f"white_matter_tract:{pathway}"


def build_tract_nodes(data: dict[str, Any]) -> list[GraphNode]:
    """Create white_matter_tract nodes for each unique pathway."""
    nodes: list[GraphNode] = []
    seen: set[str] = set()

    for conn in data.get("structural_connections", []):
        pathway = conn.get("pathway", "")
        if not pathway or pathway in seen:
            continue
        seen.add(pathway)
        nodes.append(
            GraphNode(
                node_id=_tract_node_id(pathway),
                node_type="white_matter_tract",
                label=pathway.replace("_", " ").title(),
                properties={
                    "pathway_id": pathway,
                    "source": "HCP_1200_curated_priors",
                },
            )
        )
    return nodes


def build_connectivity_edges(data: dict[str, Any]) -> list[GraphEdge]:
    edges: list[GraphEdge] = []

    for conn in data.get("structural_connections", []):
        src_id = conn["source"]
        tgt_id = conn["target"]
        pathway = conn.get("pathway", "unknown")
        fa = conn.get("fa_estimate", 0.5)
        density_label = conn.get("streamline_density", "medium")
        density_score = DENSITY_SCORES.get(density_label, 0.6)
        # Use geometric mean of FA and density as overall confidence
        confidence = (fa * density_score) ** 0.5

        src_node_id = _region_node_id(src_id)
        tgt_node_id = _region_node_id(tgt_id)

        # bidirectional structural connection
        edge_id = f"edge:struct:{src_id}:connected:{tgt_id}:{pathway}"
        edges.append(
            GraphEdge(
                edge_id=edge_id,
                source_node_id=src_node_id,
                target_node_id=tgt_node_id,
                edge_type="region_structurally_connected",
                directed=False,
                confidence=min(confidence, 1.0),
                properties={
                    "pathway": pathway,
                    "fa_estimate": fa,
                    "streamline_density": density_label,
                    "functional_relevance": conn.get("functional_relevance", ""),
                    "topics": conn.get("topics", []),
                    "circuits": conn.get("circuits", []),
                    "human_specific": conn.get("human_specific", False),
                    "human_expanded": conn.get("human_expanded", False),
                    "source_dataset": "HCP_1200_subject_release",
                    "notes": conn.get("notes", ""),
                },
            )
        )

        # pathway node linking (tract → region edges)
        if pathway != "unknown":
            tract_node_id = _tract_node_id(pathway)
            edges.append(
                GraphEdge(
                    edge_id=f"edge:tract:{pathway}:connects:{src_id}",
                    source_node_id=tract_node_id,
                    target_node_id=src_node_id,
                    edge_type="region_structurally_adjacent_to",
                    directed=False,
                )
            )
            edges.append(
                GraphEdge(
                    edge_id=f"edge:tract:{pathway}:connects:{tgt_id}",
                    source_node_id=tract_node_id,
                    target_node_id=tgt_node_id,
                    edge_type="region_structurally_adjacent_to",
                    directed=False,
                )
            )

        # circuit annotation edges — link both endpoint regions to the circuit
        # (a GraphEdge's source/target must be node ids; the previous version
        # pointed source_node_id at `edge_id`, an edge id, which can never
        # resolve to a node and always dangled).
        for circuit_id in conn.get("circuits", []):
            circuit_node_id = f"circuit:{circuit_id}"
            for region_id, region_node_id in ((src_id, src_node_id), (tgt_id, tgt_node_id)):
                edges.append(
                    GraphEdge(
                        edge_id=f"edge:struct:{region_id}:circuit:{circuit_id}:{pathway}",
                        source_node_id=region_node_id,
                        target_node_id=circuit_node_id,
                        edge_type="circuit_studied_by_method",
                        properties={"relation": "structural_substrate", "pathway": pathway},
                    )
                )

    return edges


def build_hcp_kg() -> KnowledgeGraph:
    data = _load_priors()
    nodes = build_tract_nodes(data)
    edges = build_connectivity_edges(data)
    log.info("HCP connectivity KG: %d tract nodes, %d edges", len(nodes), len(edges))
    return KnowledgeGraph(nodes=nodes, edges=edges)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kg = build_hcp_kg()
    struct_edges = [e for e in kg.edges if e.edge_type == "region_structurally_connected"]
    print(f"HCP connectivity KG: {len(struct_edges)} structural connections")
    for e in struct_edges[:5]:
        print(
            f"  {e.source_node_id.split(':')[-1]} ↔ {e.target_node_id.split(':')[-1]}"
            f" (FA={e.properties.get('fa_estimate', '?')}, "
            f"conf={e.confidence:.2f})"
        )
