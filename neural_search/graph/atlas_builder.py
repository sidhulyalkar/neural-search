"""Build KG nodes and edges from Allen Brain Atlas structure data.

Converts flattened AllenStructure records into KnowledgeGraphNode and
KnowledgeGraphEdge objects that can be merged into the main neural-search
knowledge graph.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neural_search.graph.schema import (
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
)
from neural_search.ingestion.allen_structures import AllenStructure

logger = logging.getLogger(__name__)

ATLAS_LABEL: dict[int, str] = {
    1: "allen_ccf_mouse",
    10: "allen_human",
}

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts" / "atlas"
ATLAS_GRAPH_PATH = ARTIFACTS_DIR / "atlas_graph.json"

# Max siblings to wire as co-occurrence edges per region (avoids explosion)
_MAX_SIBLINGS = 5


def _node_id(structure: AllenStructure) -> str:
    return make_node_id("brain_region", "allen", str(structure.allen_id))


def build_atlas_nodes(structures: list[AllenStructure]) -> list[KnowledgeGraphNode]:
    """Create brain_region KG nodes enriched with Allen Atlas metadata.

    Each node gets a stable_id like ``node:brain_region:allen:385`` and
    carries Allen-specific properties in its ``properties`` dict.
    """
    nodes: list[KnowledgeGraphNode] = []
    for s in structures:
        atlas_label = ATLAS_LABEL.get(s.atlas_id, f"atlas_{s.atlas_id}")
        node = KnowledgeGraphNode(
            node_id=_node_id(s),
            node_type="brain_region",
            label=s.name or s.acronym,
            aliases=[s.acronym] if s.acronym and s.acronym != s.name else [],
            source_ids=[f"{atlas_label}:{s.allen_id}"],
            properties={
                "allen_id": s.allen_id,
                "acronym": s.acronym,
                "color_hex": f"#{s.color_hex}" if s.color_hex else "",
                "atlas": atlas_label,
                "st_level": s.st_level,
                "graph_order": s.graph_order,
            },
            confidence=1.0,
        )
        nodes.append(node)
    return nodes


def build_atlas_hierarchy_edges(
    structures: list[AllenStructure],
) -> list[KnowledgeGraphEdge]:
    """Create region_is_child_of_region edges for each parent-child pair."""
    id_map = {s.allen_id: s for s in structures}
    edges: list[KnowledgeGraphEdge] = []

    for s in structures:
        if s.parent_id is None or s.parent_id not in id_map:
            continue
        source_id = _node_id(s)
        target_id = _node_id(id_map[s.parent_id])
        edge_id = make_edge_id(source_id, "region_is_child_of_region", target_id)
        edges.append(
            KnowledgeGraphEdge(
                edge_id=edge_id,
                source_node_id=source_id,
                target_node_id=target_id,
                edge_type="region_is_child_of_region",
                directed=True,
                confidence=1.0,
            )
        )
    return edges


def build_atlas_co_region_edges(
    structures: list[AllenStructure],
) -> list[KnowledgeGraphEdge]:
    """Create region_structurally_adjacent_to edges between same-level siblings.

    Siblings share the same parent and the same st_level. Capped at
    _MAX_SIBLINGS neighbours per region (by graph_order proximity) to
    prevent combinatorial explosion.
    """
    # Group children by parent_id + st_level
    from collections import defaultdict

    groups: dict[tuple[int | None, int], list[AllenStructure]] = defaultdict(list)
    for s in structures:
        groups[(s.parent_id, s.st_level)].append(s)

    edges: list[KnowledgeGraphEdge] = []
    seen_pairs: set[frozenset[int]] = set()

    for siblings in groups.values():
        if len(siblings) < 2:
            continue
        sorted_siblings = sorted(siblings, key=lambda x: x.graph_order)
        for idx, s in enumerate(sorted_siblings):
            neighbours = (
                sorted_siblings[max(0, idx - _MAX_SIBLINGS) : idx]
                + sorted_siblings[idx + 1 : idx + 1 + _MAX_SIBLINGS]
            )
            for neighbour in neighbours:
                pair = frozenset({s.allen_id, neighbour.allen_id})
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                src_id = _node_id(s)
                tgt_id = _node_id(neighbour)
                edge_id = make_edge_id(src_id, "region_structurally_adjacent_to", tgt_id)
                edges.append(
                    KnowledgeGraphEdge(
                        edge_id=edge_id,
                        source_node_id=src_id,
                        target_node_id=tgt_id,
                        edge_type="region_structurally_adjacent_to",
                        directed=False,
                        confidence=0.9,
                    )
                )
    return edges


def map_ontology_to_allen(
    ontology_regions: list[dict[str, Any]],
    allen_structures: list[AllenStructure],
) -> dict[str, int]:
    """Map ontology region IDs to Allen structure IDs.

    Uses the ``atlas_refs.allen_ccf_mouse`` field from the YAML, matching
    the stored string value (e.g. '385') to the integer allen_id.

    Returns a dict of ``ontology_id → allen_id``.
    """
    allen_by_id: dict[int, AllenStructure] = {s.allen_id: s for s in allen_structures}

    mapping: dict[str, int] = {}
    for region in ontology_regions:
        region_id = region.get("id", "")
        atlas_refs = region.get("atlas_refs") or {}
        ccf_str = atlas_refs.get("allen_ccf_mouse")
        if not ccf_str:
            continue
        try:
            allen_id = int(ccf_str)
        except (TypeError, ValueError):
            logger.warning("Bad allen_ccf_mouse value for region %s: %r", region_id, ccf_str)
            continue
        if allen_id in allen_by_id:
            mapping[region_id] = allen_id
        else:
            logger.debug("Allen ID %d (region %s) not found in loaded structures", allen_id, region_id)
    return mapping


def save_atlas_graph(
    nodes: list[KnowledgeGraphNode],
    edges: list[KnowledgeGraphEdge],
    output_path: Path,
) -> None:
    """Persist nodes and edges as a lightweight JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": [n.model_dump(mode="json") for n in nodes],
        "edges": [e.model_dump(mode="json") for e in edges],
        "meta": {
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Saved atlas graph: %d nodes, %d edges → %s",
        len(nodes),
        len(edges),
        output_path,
    )


if __name__ == "__main__":
    import yaml  # type: ignore[import]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).parent.parent.parent
    mouse_path = repo_root / "artifacts" / "atlas" / "allen_ccf_mouse_structures.json"
    human_path = repo_root / "artifacts" / "atlas" / "allen_human_structures.json"
    ontology_path = repo_root / "data" / "ontology" / "brain_regions.yaml"

    from neural_search.ingestion.allen_structures import load_structures

    mouse = load_structures(mouse_path)
    human = load_structures(human_path)
    all_structures = mouse + human
    logger.info("Loaded %d mouse + %d human structures", len(mouse), len(human))

    nodes = build_atlas_nodes(all_structures)
    hier_edges = build_atlas_hierarchy_edges(all_structures)
    co_edges = build_atlas_co_region_edges(mouse)  # siblings only for mouse CCF
    logger.info("Built %d nodes, %d hierarchy edges, %d co-region edges",
                len(nodes), len(hier_edges), len(co_edges))

    raw = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    ontology_regions = raw.get("brain_regions", [])
    mapping = map_ontology_to_allen(ontology_regions, mouse)
    logger.info("Mapped %d ontology regions to Allen CCF IDs", len(mapping))

    # Build ontology → atlas edges
    from neural_search.graph.schema import (
        KnowledgeGraphEdge,
        make_edge_id,
        make_node_id,
    )
    ontology_edges: list[KnowledgeGraphEdge] = []
    allen_ids = {s.allen_id for s in all_structures}
    for onto_id, allen_id in mapping.items():
        if allen_id not in allen_ids:
            continue
        src = make_node_id("brain_region", "ontology", onto_id)
        tgt = make_node_id("brain_region", "allen", str(allen_id))
        ontology_edges.append(KnowledgeGraphEdge(
            edge_id=make_edge_id(src, "ontology_region_maps_to_atlas", tgt),
            source_node_id=src,
            target_node_id=tgt,
            edge_type="ontology_region_maps_to_atlas",
            directed=True,
            confidence=1.0,
        ))

    all_edges = hier_edges + co_edges + ontology_edges
    save_atlas_graph(nodes, all_edges, ATLAS_GRAPH_PATH)
    print(f"Atlas graph saved: {len(nodes)} nodes, {len(all_edges)} edges -> {ATLAS_GRAPH_PATH}")
