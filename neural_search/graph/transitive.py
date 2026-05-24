"""Transitive concept expansion using graph paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neural_search.graph.query import (
    find_nodes_by_label,
    get_neighbors,
)
from neural_search.graph.schema import KnowledgeGraph, normalize_node_type


@dataclass
class TransitiveMatch:
    """A concept matched via graph traversal."""

    concept_id: str
    concept_label: str
    concept_type: str
    path_length: int
    path_explanation: list[str]
    confidence_decay: float  # Confidence decreases with path length


# Confidence decay per hop
CONFIDENCE_DECAY_RATES = {
    1: 0.80,  # 1-hop: 80% confidence
    2: 0.60,  # 2-hop: 60% confidence
    3: 0.40,  # 3-hop: 40% confidence (rarely used)
}


def find_transitive_concepts(
    graph: KnowledgeGraph,
    seed_concept: str,
    target_type: str,
    max_hops: int = 2,
) -> list[TransitiveMatch]:
    """Find related concepts via graph traversal.

    Args:
        graph: The knowledge graph
        seed_concept: Starting concept label/ID
        target_type: Node type to find (e.g., "task", "modality")
        max_hops: Maximum path length (1-3)

    Returns:
        List of TransitiveMatch sorted by confidence
    """
    matches: list[TransitiveMatch] = []
    seed_nodes = find_nodes_by_label(graph, seed_concept)

    if not seed_nodes:
        return matches

    # Track visited to avoid duplicates
    visited: set[str] = {node.node_id for node in seed_nodes}
    normalized_target = normalize_node_type(target_type)

    # BFS to find related concepts
    current_level = [(node, [node.label]) for node in seed_nodes]

    for hop in range(1, min(max_hops + 1, 4)):
        next_level: list[tuple[Any, list[str]]] = []
        decay = CONFIDENCE_DECAY_RATES.get(hop, 0.30)

        for current_node, path in current_level:
            neighbors = get_neighbors(graph, current_node.node_id, direction="both")

            for neighbor in neighbors:
                if neighbor.node_id in visited:
                    continue
                visited.add(neighbor.node_id)

                new_path = path + [neighbor.label]

                # Check if this is a target type
                if neighbor.node_type == normalized_target:
                    matches.append(
                        TransitiveMatch(
                            concept_id=neighbor.node_id,
                            concept_label=neighbor.label,
                            concept_type=neighbor.node_type,
                            path_length=hop,
                            path_explanation=new_path,
                            confidence_decay=decay,
                        )
                    )
                else:
                    # Continue traversal for next hop
                    next_level.append((neighbor, new_path))

        current_level = next_level

    # Sort by confidence (higher is better) then path length (shorter is better)
    return sorted(matches, key=lambda m: (-m.confidence_decay, m.path_length))


def find_related_tasks(
    graph: KnowledgeGraph,
    task_id: str,
    max_hops: int = 2,
) -> list[TransitiveMatch]:
    """Find tasks related to the given task via shared concepts."""
    return find_transitive_concepts(graph, task_id, "task", max_hops)


def find_related_affordances(
    graph: KnowledgeGraph,
    affordance_id: str,
    max_hops: int = 1,
) -> list[TransitiveMatch]:
    """Find analysis affordances related to the given affordance."""
    return find_transitive_concepts(graph, affordance_id, "analysis_affordance", max_hops)


def expand_query_with_graph(
    graph: KnowledgeGraph | None,
    parsed_query: dict[str, Any],
    max_hops: int = 2,
) -> dict[str, Any]:
    """Expand query terms using graph relationships.

    Args:
        graph: Optional knowledge graph
        parsed_query: Parsed query with tasks, affordances, etc.
        max_hops: Maximum traversal depth

    Returns:
        Expanded parsed_query with transitive_matches
    """
    if graph is None:
        return parsed_query

    expanded = dict(parsed_query)
    transitive_matches: list[dict[str, Any]] = []

    # Expand task concepts
    for task_id in parsed_query.get("tasks", []):
        for match in find_related_tasks(graph, task_id, max_hops):
            transitive_matches.append(
                {
                    "source": task_id,
                    "target": match.concept_label,
                    "target_id": match.concept_id,
                    "type": "task",
                    "confidence": match.confidence_decay,
                    "path": match.path_explanation,
                    "hops": match.path_length,
                }
            )

    # Expand analysis affordances (1 hop only for precision)
    for affordance_id in parsed_query.get("affordances", []):
        for match in find_related_affordances(graph, affordance_id, max_hops=1):
            transitive_matches.append(
                {
                    "source": affordance_id,
                    "target": match.concept_label,
                    "target_id": match.concept_id,
                    "type": "analysis_affordance",
                    "confidence": match.confidence_decay,
                    "path": match.path_explanation,
                    "hops": match.path_length,
                }
            )

    # Find behavioral events related to tasks
    for task_id in parsed_query.get("tasks", []):
        task_nodes = find_nodes_by_label(graph, task_id)
        for task_node in task_nodes:
            event_neighbors = get_neighbors(
                graph,
                task_node.node_id,
                edge_types=["task_has_behavioral_event"],
                direction="out",
            )
            for event in event_neighbors:
                transitive_matches.append(
                    {
                        "source": task_id,
                        "target": event.label,
                        "target_id": event.node_id,
                        "type": "behavioral_event",
                        "confidence": 0.85,
                        "path": [task_id, event.label],
                        "hops": 1,
                    }
                )

    # Find required modalities for affordances
    for affordance_id in parsed_query.get("affordances", []):
        aff_nodes = find_nodes_by_label(graph, affordance_id)
        for aff_node in aff_nodes:
            modality_neighbors = get_neighbors(
                graph,
                aff_node.node_id,
                edge_types=["analysis_requires_modality"],
                direction="out",
            )
            for modality in modality_neighbors:
                transitive_matches.append(
                    {
                        "source": affordance_id,
                        "target": modality.label,
                        "target_id": modality.node_id,
                        "type": "required_modality",
                        "confidence": 0.90,
                        "path": [affordance_id, modality.label],
                        "hops": 1,
                    }
                )

    expanded["transitive_matches"] = transitive_matches
    expanded["graph_expanded"] = bool(transitive_matches)

    return expanded


def get_transitive_boost(
    transitive_matches: list[dict[str, Any]],
    dataset_labels: set[str],
) -> float:
    """Calculate a boost score based on transitive matches.

    Args:
        transitive_matches: List of transitive match dicts
        dataset_labels: Set of normalized labels from the dataset

    Returns:
        Boost score (0.0 to 0.15)
    """
    if not transitive_matches:
        return 0.0

    boost = 0.0
    matched_count = 0

    for match in transitive_matches:
        target_normalized = match["target"].lower().replace("_", " ")
        if target_normalized in dataset_labels or match["target_id"] in dataset_labels:
            boost += match["confidence"] * 0.05  # 5% per match
            matched_count += 1
            if matched_count >= 3:
                break  # Cap at 3 matches

    return min(boost, 0.15)  # Cap total boost at 15%
