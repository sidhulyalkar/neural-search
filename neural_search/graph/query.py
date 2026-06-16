"""Lightweight traversal and ranking helpers for knowledge graphs."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from neural_search.graph.builder import dataset_node_id, paper_node_id
from neural_search.graph.schema import (
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_node_id,
    normalize_edge_type,
    normalize_node_type,
)

Direction = Literal["in", "out", "both"]

DEFAULT_WEIGHTS = {
    "shared_task": 3.0,
    "shared_analysis_affordance": 3.0,
    "shared_behavioral_event": 2.5,
    "shared_brain_region": 2.0,
    "shared_modality": 1.5,
    "shared_species": 1.0,
    "shared_data_standard": 0.5,
    "shared_paper": 4.0,
}

DATASET_CONCEPT_EDGES = {
    "required_tasks": ("dataset_has_task", "task"),
    "required_modalities": ("dataset_has_modality", "modality"),
    "required_brain_regions": ("dataset_records_region", "brain_region"),
    "required_species": ("dataset_has_species", "species"),
    "required_behavioral_events": ("dataset_has_behavioral_event", "behavioral_event"),
    "required_analysis_affordances": ("dataset_supports_analysis", "analysis_affordance"),
    "required_data_standards": ("dataset_uses_standard", "data_standard"),
}

RELATED_DATASET_EDGE_WEIGHTS = {
    "dataset_has_task": "shared_task",
    "dataset_supports_analysis": "shared_analysis_affordance",
    "dataset_has_behavioral_event": "shared_behavioral_event",
    "dataset_records_region": "shared_brain_region",
    "dataset_has_modality": "shared_modality",
    "dataset_has_species": "shared_species",
    "dataset_uses_standard": "shared_data_standard",
}

PAPER_DATASET_EDGES = {"paper_uses_dataset", "paper_mentions_dataset"}


@dataclass(frozen=True)
class RelatedItem:
    """Ranked related graph item."""

    node_id: str
    score: float
    reasons: list[str]


def _edge_types(edge_types: Iterable[str] | None) -> set[str] | None:
    if edge_types is None:
        return None
    return {normalize_edge_type(edge_type) for edge_type in edge_types}


def _matches_edge(edge: KnowledgeGraphEdge, edge_types: set[str] | None) -> bool:
    return edge_types is None or edge.edge_type in edge_types


def _concept_node_id(node_type: str, value: str) -> str:
    if value.startswith("node:"):
        return value
    return make_node_id(node_type, normalize_node_type(value))


def _dataset_id(value: str) -> str:
    if value.startswith("node:dataset:"):
        return value
    return dataset_node_id(value)


def _paper_id(value: str) -> str:
    if value.startswith("node:paper:"):
        return value
    return paper_node_id(value)


def _out_edges(
    graph: KnowledgeGraph,
    node_id: str,
    edge_types: Iterable[str] | None = None,
) -> list[KnowledgeGraphEdge]:
    wanted = _edge_types(edge_types)
    return [
        edge
        for edge in graph.edges.values()
        if edge.source_node_id == node_id and _matches_edge(edge, wanted)
    ]


def _in_edges(
    graph: KnowledgeGraph,
    node_id: str,
    edge_types: Iterable[str] | None = None,
) -> list[KnowledgeGraphEdge]:
    wanted = _edge_types(edge_types)
    return [
        edge
        for edge in graph.edges.values()
        if edge.target_node_id == node_id and _matches_edge(edge, wanted)
    ]


def get_node(graph: KnowledgeGraph, node_id: str) -> KnowledgeGraphNode | None:
    """Return a node by ID, or None when absent."""

    return graph.nodes.get(node_id)


def get_neighbors(
    graph: KnowledgeGraph,
    node_id: str,
    edge_types: Iterable[str] | None = None,
    direction: Direction = "both",
) -> list[KnowledgeGraphNode]:
    """Return neighboring nodes for the requested edge types and direction."""

    edges: list[KnowledgeGraphEdge] = []
    if direction in {"out", "both"}:
        edges.extend(_out_edges(graph, node_id, edge_types))
    if direction in {"in", "both"}:
        edges.extend(_in_edges(graph, node_id, edge_types))

    neighbors: list[KnowledgeGraphNode] = []
    seen: set[str] = set()
    for edge in edges:
        neighbor_id = (
            edge.target_node_id if edge.source_node_id == node_id else edge.source_node_id
        )
        if neighbor_id in seen or neighbor_id not in graph.nodes:
            continue
        seen.add(neighbor_id)
        neighbors.append(graph.nodes[neighbor_id])
    return neighbors


def get_edges_between(
    graph: KnowledgeGraph,
    source_id: str,
    target_id: str,
) -> list[KnowledgeGraphEdge]:
    """Return directed edges from source to target."""

    return [
        edge
        for edge in graph.edges.values()
        if edge.source_node_id == source_id and edge.target_node_id == target_id
    ]


def find_nodes_by_type(graph: KnowledgeGraph, node_type: str) -> list[KnowledgeGraphNode]:
    """Find all nodes of a normalized node type."""

    normalized = normalize_node_type(node_type)
    return [node for node in graph.nodes.values() if node.node_type == normalized]


def find_nodes_by_label(
    graph: KnowledgeGraph,
    label_or_alias: str,
) -> list[KnowledgeGraphNode]:
    """Find nodes by exact normalized label or alias."""

    normalized = normalize_node_type(label_or_alias)
    matches: list[KnowledgeGraphNode] = []
    for node in graph.nodes.values():
        values = [node.label, *node.aliases]
        if any(normalize_node_type(value) == normalized for value in values if value):
            matches.append(node)
    return matches


def find_datasets_for_task(graph: KnowledgeGraph, task_id: str) -> list[KnowledgeGraphNode]:
    """Find datasets with a task concept edge."""

    target_id = _concept_node_id("task", task_id)
    return [
        graph.nodes[edge.source_node_id]
        for edge in _in_edges(graph, target_id, ["dataset_has_task"])
        if edge.source_node_id in graph.nodes
    ]


def find_datasets_for_analysis(
    graph: KnowledgeGraph,
    affordance_id: str,
) -> list[KnowledgeGraphNode]:
    """Find datasets that support an analysis affordance."""

    target_id = _concept_node_id("analysis_affordance", affordance_id)
    return [
        graph.nodes[edge.source_node_id]
        for edge in _in_edges(graph, target_id, ["dataset_supports_analysis"])
        if edge.source_node_id in graph.nodes
    ]


def find_papers_for_dataset(
    graph: KnowledgeGraph,
    dataset_id: str,
) -> list[KnowledgeGraphNode]:
    """Find papers linked to a dataset."""

    target_id = _dataset_id(dataset_id)
    return [
        graph.nodes[edge.source_node_id]
        for edge in _in_edges(graph, target_id, PAPER_DATASET_EDGES)
        if edge.source_node_id in graph.nodes
    ]


def find_datasets_for_paper(
    graph: KnowledgeGraph,
    paper_id: str,
) -> list[KnowledgeGraphNode]:
    """Find datasets linked from a paper."""

    source_id = _paper_id(paper_id)
    return [
        graph.nodes[edge.target_node_id]
        for edge in _out_edges(graph, source_id, PAPER_DATASET_EDGES)
        if edge.target_node_id in graph.nodes
    ]


def _dataset_concepts(
    graph: KnowledgeGraph,
    dataset_id: str,
    edge_type: str,
) -> set[str]:
    return {
        edge.target_node_id
        for edge in _out_edges(graph, dataset_id, [edge_type])
        if edge.target_node_id in graph.nodes
    }


def _paper_links_for_dataset(graph: KnowledgeGraph, dataset_id: str) -> set[str]:
    return {
        edge.source_node_id
        for edge in _in_edges(graph, dataset_id, PAPER_DATASET_EDGES)
        if edge.source_node_id in graph.nodes
    }


def find_datasets_with_constraints(
    graph: KnowledgeGraph,
    *,
    required_tasks: Sequence[str] = (),
    required_modalities: Sequence[str] = (),
    required_brain_regions: Sequence[str] = (),
    required_species: Sequence[str] = (),
    required_behavioral_events: Sequence[str] = (),
    required_analysis_affordances: Sequence[str] = (),
    required_data_standards: Sequence[str] = (),
    excluded_tasks: Sequence[str] = (),
    excluded_modalities: Sequence[str] = (),
    excluded_brain_regions: Sequence[str] = (),
    excluded_species: Sequence[str] = (),
    excluded_behavioral_events: Sequence[str] = (),
) -> list[KnowledgeGraphNode]:
    """Find datasets satisfying required and excluded concept constraints."""

    required: dict[str, Sequence[str]] = {
        "required_tasks": required_tasks,
        "required_modalities": required_modalities,
        "required_brain_regions": required_brain_regions,
        "required_species": required_species,
        "required_behavioral_events": required_behavioral_events,
        "required_analysis_affordances": required_analysis_affordances,
        "required_data_standards": required_data_standards,
    }
    excluded: dict[str, tuple[Sequence[str], str, str]] = {
        "excluded_tasks": (excluded_tasks, "dataset_has_task", "task"),
        "excluded_modalities": (excluded_modalities, "dataset_has_modality", "modality"),
        "excluded_brain_regions": (
            excluded_brain_regions,
            "dataset_records_region",
            "brain_region",
        ),
        "excluded_species": (excluded_species, "dataset_has_species", "species"),
        "excluded_behavioral_events": (
            excluded_behavioral_events,
            "dataset_has_behavioral_event",
            "behavioral_event",
        ),
    }

    results: list[KnowledgeGraphNode] = []
    for dataset in find_nodes_by_type(graph, "dataset"):
        ok = True
        for key, values in required.items():
            if not values:
                continue
            edge_type, node_type = DATASET_CONCEPT_EDGES[key]
            required_ids = {_concept_node_id(node_type, value) for value in values}
            actual_ids = _dataset_concepts(graph, dataset.node_id, edge_type)
            if not required_ids <= actual_ids:
                ok = False
                break
        if not ok:
            continue
        for values, edge_type, node_type in excluded.values():
            if not values:
                continue
            excluded_ids = {_concept_node_id(node_type, value) for value in values}
            actual_ids = _dataset_concepts(graph, dataset.node_id, edge_type)
            if excluded_ids & actual_ids:
                ok = False
                break
        if ok:
            results.append(dataset)
    return results


def _adjacent_edges(graph: KnowledgeGraph, node_id: str) -> list[KnowledgeGraphEdge]:
    return [
        edge
        for edge in graph.edges.values()
        if edge.source_node_id == node_id or edge.target_node_id == node_id
    ]


def find_paths(
    graph: KnowledgeGraph,
    source_id: str,
    target_id: str,
    max_depth: int = 3,
) -> list[list[str]]:
    """Find simple node-ID paths between two nodes up to max_depth edges."""

    if source_id not in graph.nodes or target_id not in graph.nodes or max_depth < 1:
        return []
    paths: list[list[str]] = []
    queue: deque[list[str]] = deque([[source_id]])
    while queue:
        path = queue.popleft()
        current = path[-1]
        if len(path) - 1 >= max_depth:
            continue
        for edge in _adjacent_edges(graph, current):
            neighbor = (
                edge.target_node_id if edge.source_node_id == current else edge.source_node_id
            )
            if neighbor in path:
                continue
            next_path = [*path, neighbor]
            if neighbor == target_id:
                paths.append(next_path)
            else:
                queue.append(next_path)
    return paths


def explain_connection(
    graph: KnowledgeGraph,
    source_id: str,
    target_id: str,
    max_depth: int = 3,
) -> dict[str, Any]:
    """Explain how two graph nodes connect using shortest available paths."""

    paths = find_paths(graph, source_id, target_id, max_depth=max_depth)
    best_path = min(paths, key=len) if paths else []
    steps: list[dict[str, Any]] = []
    for left, right in zip(best_path, best_path[1:], strict=False):
        edges = [
            edge
            for edge in graph.edges.values()
            if {edge.source_node_id, edge.target_node_id} == {left, right}
        ]
        if not edges:
            continue
        edge = edges[0]
        steps.append(
            {
                "source": edge.source_node_id,
                "target": edge.target_node_id,
                "edge_type": edge.edge_type,
                "confidence": edge.confidence,
                "evidence": [item.evidence_text for item in edge.evidence if item.evidence_text],
            }
        )
    return {
        "connected": bool(best_path),
        "source_id": source_id,
        "target_id": target_id,
        "path": best_path,
        "steps": steps,
    }


def rank_related_datasets(
    graph: KnowledgeGraph,
    dataset_id: str,
    weights: Mapping[str, float] | None = None,
) -> list[RelatedItem]:
    """Rank datasets related by shared concepts and linked papers."""

    source_id = _dataset_id(dataset_id)
    if source_id not in graph.nodes:
        return []
    score_weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    source_concepts = {
        edge_type: _dataset_concepts(graph, source_id, edge_type)
        for edge_type in RELATED_DATASET_EDGE_WEIGHTS
    }
    source_papers = _paper_links_for_dataset(graph, source_id)
    related: list[RelatedItem] = []

    for dataset in find_nodes_by_type(graph, "dataset"):
        if dataset.node_id == source_id:
            continue
        score = 0.0
        reasons: list[str] = []
        for edge_type, weight_key in RELATED_DATASET_EDGE_WEIGHTS.items():
            shared = source_concepts[edge_type] & _dataset_concepts(
                graph,
                dataset.node_id,
                edge_type,
            )
            if shared:
                weight = score_weights[weight_key]
                score += weight * len(shared)
                reasons.append(f"{weight_key}:{len(shared)}")
        shared_papers = source_papers & _paper_links_for_dataset(graph, dataset.node_id)
        if shared_papers:
            score += score_weights["shared_paper"] * len(shared_papers)
            reasons.append(f"shared_paper:{len(shared_papers)}")
        if score > 0:
            related.append(RelatedItem(dataset.node_id, score, reasons))

    return sorted(related, key=lambda item: (-item.score, item.node_id))


def rank_related_papers(
    graph: KnowledgeGraph,
    paper_id: str,
    weights: Mapping[str, float] | None = None,
) -> list[RelatedItem]:
    """Rank papers related by shared datasets and scientific concepts."""

    source_id = _paper_id(paper_id)
    if source_id not in graph.nodes:
        return []
    score_weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    source_datasets = {
        edge.target_node_id for edge in _out_edges(graph, source_id, PAPER_DATASET_EDGES)
    }
    source_targets = {
        edge.edge_type: {
            item.target_node_id for item in _out_edges(graph, source_id, [edge.edge_type])
        }
        for edge in graph.edges.values()
        if edge.source_node_id == source_id
    }
    related: list[RelatedItem] = []
    for paper in find_nodes_by_type(graph, "paper"):
        if paper.node_id == source_id:
            continue
        score = 0.0
        reasons: list[str] = []
        shared_datasets = source_datasets & {
            edge.target_node_id for edge in _out_edges(graph, paper.node_id, PAPER_DATASET_EDGES)
        }
        if shared_datasets:
            score += score_weights["shared_paper"] * len(shared_datasets)
            reasons.append(f"shared_dataset:{len(shared_datasets)}")
        for edge_type, source_nodes in source_targets.items():
            shared = source_nodes & {
                edge.target_node_id for edge in _out_edges(graph, paper.node_id, [edge_type])
            }
            if shared:
                score += 1.0 * len(shared)
                reasons.append(f"shared_{edge_type}:{len(shared)}")
        if score > 0:
            related.append(RelatedItem(paper.node_id, score, reasons))
    return sorted(related, key=lambda item: (-item.score, item.node_id))


_CROSS_DATASET_EDGE_TYPES: frozenset[str] = frozenset({
    "same_region_same_task",
    "same_region_cross_modality",
    "same_task_cross_species",
})

_RELATION_LABEL: dict[str, str] = {
    "same_region_same_task": "same region + task",
    "same_region_cross_modality": "same region, different modality",
    "same_task_cross_species": "same task, different species",
}


def find_similar_datasets(
    graph: KnowledgeGraph,
    dataset_id: str,
    *,
    limit: int = 6,
    edge_types: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Return datasets similar to dataset_id via cross-dataset graph edges.

    Args:
        graph: Loaded KnowledgeGraph.
        dataset_id: The source dataset ID (e.g. 'allen:ecephys_715093703').
        limit: Max results to return.
        edge_types: Which edge types to traverse (default: all cross-dataset types).

    Returns:
        List of dicts with keys: dataset_id, title, relation, relation_label, weight.
        Sorted descending by weight.
    """
    allowed = edge_types if edge_types is not None else _CROSS_DATASET_EDGE_TYPES
    # Construct node_id directly: graph stores datasets as "node:dataset:{source}:{id}"
    # _dataset_id() normalizes colons to underscores, which breaks archive:id format.
    node_id = f"node:dataset:{dataset_id}"

    similar: list[dict[str, Any]] = []
    for edge in _adjacent_edges(graph, node_id):
        if edge.edge_type not in allowed:
            continue
        other_node_id = (
            edge.target_node_id if edge.source_node_id == node_id else edge.source_node_id
        )
        other_node = get_node(graph, other_node_id)
        if other_node is None:
            continue
        # Extract archive dataset_id from node_id: "node:dataset:{source}:{id}"
        other_dataset_id = other_node_id.removeprefix("node:dataset:")
        similar.append({
            "dataset_id": other_dataset_id,
            "title": other_node.label,
            "relation": edge.edge_type,
            "relation_label": _RELATION_LABEL.get(edge.edge_type, edge.edge_type),
            "weight": edge.confidence,
        })

    # Deduplicate by dataset_id, keeping highest-weight entry
    by_id: dict[str, dict[str, Any]] = {}
    for entry in similar:
        did = entry["dataset_id"]
        if did not in by_id or entry["weight"] > by_id[did]["weight"]:
            by_id[did] = entry

    return sorted(by_id.values(), key=lambda x: -x["weight"])[:limit]
