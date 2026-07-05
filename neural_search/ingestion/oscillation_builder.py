"""Build the oscillation signature KG layer from data/oscillations/oscillation_signatures.yaml.

Creates:
  - oscillation nodes (region x frequency_band combos)
  - region_generates_oscillation edges
  - oscillation_measured_by_method edges
  - oscillation_indexes_circuit edges
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from neural_search.graph.schema import GraphEdge, GraphNode, KnowledgeGraph

log = logging.getLogger(__name__)

DATA_PATH = (
    Path(__file__).parent.parent.parent / "data" / "oscillations" / "oscillation_signatures.yaml"
)

# Map our frequency band IDs to method IDs from methods_taxonomy.yaml
BAND_TO_METHOD: dict[str, list[str]] = {
    "theta": ["fft", "multitaper", "wavelet_transform", "hilbert_transform", "plv"],
    "alpha": ["fft", "multitaper"],
    "beta": ["fft", "multitaper", "fooof_specparam", "burst_analysis", "plv"],
    "beta_low": ["fft", "multitaper", "fooof_specparam"],
    "beta_high": ["fft", "multitaper", "burst_analysis"],
    "low_gamma": ["fft", "multitaper"],
    "high_gamma": ["fft", "multitaper", "hilbert_transform"],
    "gamma": ["fft", "multitaper", "pac", "hilbert_transform"],
    "ripple": ["wavelet_transform", "hilbert_transform", "spike_lfp_coupling"],
    "delta": ["fft", "multitaper"],
    "ultra_fast": ["fft", "multitaper"],
}


def _load_signatures() -> dict[str, Any]:
    with open(DATA_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _osc_node_id(region_id: str, band: str) -> str:
    return f"oscillation:{region_id}:{band}"


def _region_node_id(region_id: str) -> str:
    return f"ontology_region:{region_id}"


def build_oscillation_nodes(data: dict[str, Any]) -> list[GraphNode]:
    nodes: list[GraphNode] = []
    seen: set[str] = set()

    for sig in data.get("oscillation_signatures", []):
        region_id = sig["region_id"]
        band = sig["frequency_band"]
        node_id = _osc_node_id(region_id, band)

        if node_id in seen:
            continue
        seen.add(node_id)

        # Find frequency range from band definitions
        band_ranges: dict[str, list[float]] = {}
        for b in data.get("frequency_bands", []):
            if "range_hz" in b:
                band_ranges[b["id"]] = b["range_hz"]

        freq_range = band_ranges.get(band, [])

        nodes.append(
            GraphNode(
                node_id=node_id,
                node_type="oscillation",
                label=f"{band.replace('_', ' ').title()} — {region_id.replace('_', ' ').title()}",
                properties={
                    "region_id": region_id,
                    "frequency_band": band,
                    "frequency_range_hz": freq_range,
                    "functional_role": sig.get("functional_role", ""),
                    "condition": sig.get("condition", ""),
                    "species": sig.get("species", []),
                    "generator": sig.get("generator", ""),
                    "key_papers": sig.get("key_papers", []),
                    "topics": sig.get("topics", []),
                    "pac_coupling": sig.get("pac_coupling", ""),
                    "coherent_with": sig.get("coherent_with", []),
                    "clinical_relevance": sig.get("clinical_relevance", ""),
                    "translational_significance": sig.get("translational_significance", ""),
                },
            )
        )

    return nodes


def build_oscillation_edges(data: dict[str, Any]) -> list[GraphEdge]:
    edges: list[GraphEdge] = []

    for sig in data.get("oscillation_signatures", []):
        region_id = sig["region_id"]
        band = sig["frequency_band"]
        osc_node_id = _osc_node_id(region_id, band)
        region_node_id = _region_node_id(region_id)

        # region → oscillation
        edges.append(
            GraphEdge(
                edge_id=f"edge:osc:{region_id}:generates:{band}",
                source_node_id=region_node_id,
                target_node_id=osc_node_id,
                edge_type="region_generates_oscillation",
                properties={
                    "species": sig.get("species", []),
                    "condition": sig.get("condition", ""),
                    "functional_role": sig.get("functional_role", ""),
                },
            )
        )

        # oscillation → topics (via method_used_for_topic proxy)
        for topic_id in sig.get("topics", []):
            edges.append(
                GraphEdge(
                    edge_id=f"edge:osc:{region_id}:{band}:topic:{topic_id}",
                    source_node_id=osc_node_id,
                    target_node_id=f"topic:{topic_id}",
                    edge_type="finding_advances_topic",
                    properties={"relation": "oscillation_relevant_to_topic"},
                )
            )

        # oscillation → measurement methods
        for method_id in BAND_TO_METHOD.get(band, []):
            edges.append(
                GraphEdge(
                    edge_id=f"edge:osc:{region_id}:{band}:measured_by:{method_id}",
                    source_node_id=osc_node_id,
                    target_node_id=f"method:{method_id}",
                    edge_type="oscillation_measured_by_method",
                )
            )

        # PAC coupling edges
        pac = sig.get("pac_coupling", "")
        if pac:
            edges.append(
                GraphEdge(
                    edge_id=f"edge:osc:{region_id}:{band}:pac",
                    source_node_id=osc_node_id,
                    target_node_id="method:pac",
                    edge_type="oscillation_measured_by_method",
                    properties={"relation": "pac_coupling", "description": pac},
                )
            )

        # coherence edges: oscillation coherent_with region
        for coherent_region in sig.get("coherent_with", []):
            edges.append(
                GraphEdge(
                    edge_id=f"edge:osc:{region_id}:{band}:coherent:{coherent_region}",
                    source_node_id=osc_node_id,
                    target_node_id=_region_node_id(coherent_region),
                    edge_type="region_co_occurs_with_region",
                    properties={"relation": "oscillatory_coherence", "frequency_band": band},
                )
            )

    return edges


def build_oscillation_kg() -> KnowledgeGraph:
    data = _load_signatures()
    nodes = build_oscillation_nodes(data)
    edges = build_oscillation_edges(data)
    log.info("Oscillation KG: %d nodes, %d edges", len(nodes), len(edges))
    return KnowledgeGraph(nodes=nodes, edges=edges)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kg = build_oscillation_kg()
    print(f"Oscillation KG: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
    for n in kg.nodes[:5]:
        print(f"  {n.label}")
