"""Populate `method_supports_analysis` edges from the methodology registry.

Bridges named techniques in `data/methods/methods_taxonomy.yaml` (built into
graph nodes by `neural_search.ingestion.methods_builder.build_methods_kg`) to
the `analysis_affordance` nodes that
`neural_search.graph.builder.build_taxonomy_requirement_subgraph` already
creates and populates with `analysis_requires_*` edges in the real corpus
graph.

Node ID conventions are intentionally matched to what each upstream builder
already emits rather than the canonical `make_node_id` helper, because the
edges in this module must not dangle:

- Technique node ids come from `methods_builder._method_node_id`, which is a
  hand-rolled `f"method:{method_id}"` (NOT `make_node_id("method", ...)`).
- Analysis-affordance node ids come from `graph.builder._taxonomy_analysis_node`,
  which does use `make_node_id("analysis_affordance", analysis_id)`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from neural_search.graph.schema import (
    GraphEdge,
    GraphEvidence,
    KnowledgeGraph,
    make_node_id,
)
from neural_search.kg.schemas.method_registry import MethodRegistry

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "methods"
REGISTRY_PATH = DATA_DIR / "method_registry.yaml"

BUILDER_NAME = "method_registry_builder"
BUILDER_VERSION = "v1.0.0"


def _method_node_id(method_id: str) -> str:
    """Match neural_search.ingestion.methods_builder._method_node_id exactly."""

    return f"method:{method_id}"


def _analysis_affordance_node_id(analysis_family: str) -> str:
    """Match neural_search.graph.builder._taxonomy_analysis_node exactly."""

    return make_node_id("analysis_affordance", analysis_family)


def load_method_registry(path: Path | str = REGISTRY_PATH) -> MethodRegistry:
    with open(path, encoding="utf-8") as fh:
        payload: dict[str, Any] = yaml.safe_load(fh) or {}
    return MethodRegistry.model_validate(payload)


def build_method_registry_edges(registry: MethodRegistry) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for link in registry.links:
        target_id = _analysis_affordance_node_id(link.analysis_family)
        for method_id in link.taxonomy_method_ids:
            source_id = _method_node_id(method_id)
            evidence = GraphEvidence(
                evidence_id=f"evidence:method_registry:{method_id}:{link.analysis_family}",
                source_type="method_registry",
                source_id=link.analysis_family,
                evidence_text=link.rationale.strip(),
                confidence=link.confidence,
                extractor_name=BUILDER_NAME,
                extractor_version=BUILDER_VERSION,
            )
            edges.append(
                GraphEdge(
                    edge_id=f"edge:method:{method_id}:supports_analysis:{link.analysis_family}",
                    source_node_id=source_id,
                    target_node_id=target_id,
                    edge_type="method_supports_analysis",
                    confidence=link.confidence,
                    evidence=[evidence],
                    properties={
                        "rationale": link.rationale.strip(),
                        "cross_ref_affordance_id": link.cross_ref_affordance_id,
                        "cross_ref_ontology_affordance_id": link.cross_ref_ontology_affordance_id,
                        "requires_human_review": link.requires_human_review,
                    },
                )
            )
    return edges


def build_method_registry_subgraph() -> KnowledgeGraph:
    """Build the `method_supports_analysis` edge layer.

    Emits edges only — no nodes. The source `method:*` nodes are created by
    `methods_builder.build_methods_kg()` and the target `analysis_affordance`
    nodes by `graph.builder.build_taxonomy_requirement_subgraph()`; both are
    merged into the same real-corpus graph by `build_graph_from_records`, so
    the edges resolve once merged even though this layer creates no nodes of
    its own.
    """

    registry = load_method_registry()
    edges = build_method_registry_edges(registry)
    log.info(
        "Method registry KG: %d edges from %d analysis_family links",
        len(edges),
        len(registry.links),
    )
    return KnowledgeGraph(nodes=[], edges=edges)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kg = build_method_registry_subgraph()
    print(f"Method registry KG built: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
    for edge in list(kg.edges.values())[:5]:
        print(f"  - {edge.source_node_id} --[{edge.edge_type}]--> {edge.target_node_id}")
