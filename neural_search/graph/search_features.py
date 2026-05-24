"""Optional graph-derived features for retrieval results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neural_search.graph.builder import dataset_node_id
from neural_search.graph.query import (
    find_papers_for_dataset,
    get_neighbors,
)
from neural_search.graph.schema import KnowledgeGraph, read_graph_json

DEFAULT_GRAPH_SEARCH_WEIGHTS = {
    "linked_paper": 0.04,
    "analysis_affordance": 0.03,
    "task_match": 0.03,
    "modality_match": 0.02,
    "brain_region_match": 0.02,
    "degree": 0.01,
}


def load_graph_if_exists(path: str | Path | None) -> KnowledgeGraph | None:
    """Load a graph JSON file if present, otherwise return None."""

    if not path:
        return None
    graph_path = Path(path)
    if not graph_path.exists():
        return None
    return read_graph_json(graph_path)


def _result_node_id(result_id: str) -> str:
    if result_id.startswith("node:dataset:"):
        return result_id
    return dataset_node_id(result_id)


def _resolve_dataset_node_id(graph: KnowledgeGraph, result_id: str) -> str:
    candidate = _result_node_id(result_id)
    if candidate in graph.nodes:
        return candidate
    normalized_result = str(result_id)
    for node in graph.nodes.values():
        if node.node_type != "dataset":
            continue
        aliases = {
            node.node_id,
            *node.aliases,
            *node.source_ids,
            str(node.properties.get("source_id", "")),
        }
        if normalized_result in aliases:
            return node.node_id
    return candidate


def _neighbor_labels(
    graph: KnowledgeGraph,
    node_id: str,
    edge_type: str,
) -> list[str]:
    return [
        node.label
        for node in get_neighbors(graph, node_id, [edge_type], direction="out")
    ]


def compute_graph_features_for_result(
    graph: KnowledgeGraph | None,
    result_id: str,
    query_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return graph features for a result without requiring graph availability."""

    if graph is None:
        return {
            "graph_available": False,
            "graph_degree": 0,
            "linked_papers": [],
            "analysis_affordances": [],
            "tasks": [],
            "modalities": [],
            "brain_regions": [],
            "matched_query_context": {},
        }

    node_id = _resolve_dataset_node_id(graph, result_id)
    if node_id not in graph.nodes:
        return {
            "graph_available": True,
            "graph_degree": 0,
            "linked_papers": [],
            "analysis_affordances": [],
            "tasks": [],
            "modalities": [],
            "brain_regions": [],
            "matched_query_context": {},
        }

    graph_degree = sum(
        1
        for edge in graph.edges.values()
        if edge.source_node_id == node_id or edge.target_node_id == node_id
    )
    linked_papers = [paper.label for paper in find_papers_for_dataset(graph, node_id)]
    analysis_affordances = _neighbor_labels(graph, node_id, "dataset_supports_analysis")
    tasks = _neighbor_labels(graph, node_id, "dataset_has_task")
    modalities = _neighbor_labels(graph, node_id, "dataset_has_modality")
    brain_regions = _neighbor_labels(graph, node_id, "dataset_records_region")
    context = query_context or {}
    matched_query_context = {
        "tasks": sorted(set(context.get("tasks", [])) & set(tasks)),
        "modalities": sorted(set(context.get("modalities", [])) & set(modalities)),
        "brain_regions": sorted(
            set(context.get("brain_regions", [])) & set(brain_regions)
        ),
        "analysis": sorted(
            set(context.get("analysis", [])) & set(analysis_affordances)
        ),
    }

    return {
        "graph_available": True,
        "graph_degree": graph_degree,
        "linked_papers": linked_papers,
        "analysis_affordances": analysis_affordances,
        "tasks": tasks,
        "modalities": modalities,
        "brain_regions": brain_regions,
        "matched_query_context": matched_query_context,
    }


def graph_context_score(
    graph: KnowledgeGraph | None,
    result_id: str,
    query_context: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """Return a small optional graph score; absent graphs contribute zero."""

    if graph is None:
        return 0.0
    score_weights = {**DEFAULT_GRAPH_SEARCH_WEIGHTS, **(weights or {})}
    features = compute_graph_features_for_result(graph, result_id, query_context)
    if not features["graph_available"]:
        return 0.0

    matched = features["matched_query_context"]
    score = 0.0
    score += min(len(features["linked_papers"]), 3) * score_weights["linked_paper"]
    score += min(len(features["analysis_affordances"]), 3) * score_weights["analysis_affordance"]
    score += len(matched["tasks"]) * score_weights["task_match"]
    score += len(matched["modalities"]) * score_weights["modality_match"]
    score += len(matched["brain_regions"]) * score_weights["brain_region_match"]
    score += min(int(features["graph_degree"]), 10) * score_weights["degree"]
    return round(min(score, 0.25), 4)
