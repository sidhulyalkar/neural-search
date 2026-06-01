"""Optional graph-derived features for retrieval results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neural_search.graph.builder import dataset_node_id
from neural_search.graph.query import (
    find_papers_for_dataset,
    get_neighbors,
)
from neural_search.graph.schema import (
    KnowledgeGraph,
    KnowledgeGraphEdge,
    read_graph_json,
)

DEFAULT_GRAPH_SEARCH_WEIGHTS = {
    "linked_paper": 0.04,
    "analysis_affordance": 0.03,
    "requirement_match": 0.01,
    "analysis_requirement_coverage": 0.015,
    "task_match": 0.03,
    "modality_match": 0.02,
    "species_match": 0.02,
    "taxon_match": 0.01,
    "brain_region_match": 0.02,
    "degree": 0.01,
}

REQUIREMENT_GROUPS = {
    "modality": "modality",
    "behavioral_event": "behavioral_event",
    "data_standard": "data_standard",
    "required_signal": "required_signal",
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


def _get_edges_for_node(
    graph: KnowledgeGraph,
    node_id: str,
    edge_type: str | None = None,
    direction: str = "out",
) -> list[KnowledgeGraphEdge]:
    """Return edges connected to a node, optionally filtered by type."""
    edges = []
    for edge in graph.edges.values():
        if direction in ("out", "both") and edge.source_node_id == node_id:
            if edge_type is None or edge.edge_type == edge_type:
                edges.append(edge)
        elif direction in ("in", "both") and edge.target_node_id == node_id:
            if edge_type is None or edge.edge_type == edge_type:
                edges.append(edge)
    # Sort by confidence descending
    return sorted(edges, key=lambda e: e.confidence, reverse=True)


def _empty_features(*, graph_available: bool) -> dict[str, Any]:
    return {
        "graph_available": graph_available,
        "graph_degree": 0,
        "linked_papers": [],
        "analysis_affordances": [],
        "tasks": [],
        "modalities": [],
        "species": [],
        "species_context": {
            "taxon_groups": [],
            "model_roles": [],
            "animal_types": [],
        },
        "brain_regions": [],
        "matched_query_context": {},
        "requirement_matches": {
            "modality": [],
            "behavioral_event": [],
            "data_standard": [],
            "required_signal": [],
        },
    }


def _norm(value: str) -> str:
    return " ".join(str(value).casefold().replace("_", " ").replace("-", " ").split())


def _node_terms(graph: KnowledgeGraph, node_id: str) -> set[str]:
    node = graph.nodes.get(node_id)
    if node is None:
        return set()
    terms = {node.node_id, node.label, *node.aliases, *node.source_ids}
    for value in node.properties.values():
        if isinstance(value, str):
            terms.add(value)
        elif isinstance(value, list):
            terms.update(str(item) for item in value)
    return {_norm(term) for term in terms if str(term).strip()}


def _dataset_concept_terms(
    graph: KnowledgeGraph,
    dataset_node_id: str,
) -> dict[str, set[str]]:
    edge_to_group = {
        "dataset_has_modality": "modality",
        "dataset_has_behavioral_event": "behavioral_event",
        "dataset_uses_standard": "data_standard",
    }
    terms: dict[str, set[str]] = {group: set() for group in REQUIREMENT_GROUPS}
    for edge in _get_edges_for_node(graph, dataset_node_id, direction="out"):
        group = edge_to_group.get(edge.edge_type)
        if group:
            terms[group].update(_node_terms(graph, edge.target_node_id))
        if edge.edge_type == "dataset_supports_analysis":
            terms["required_signal"].update(_node_terms(graph, edge.target_node_id))
    terms["required_signal"].update(_node_terms(graph, dataset_node_id))
    return terms


def _requirement_group(edge: KnowledgeGraphEdge, target_type: str) -> str | None:
    if edge.edge_type == "analysis_requires_modality":
        return "modality"
    if edge.edge_type == "analysis_requires_behavioral_event":
        return "behavioral_event"
    if target_type == "data_standard":
        return "data_standard"
    if target_type == "required_signal":
        return "required_signal"
    return REQUIREMENT_GROUPS.get(target_type)


def _requirement_matches(
    graph: KnowledgeGraph,
    dataset_node_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """Return matched analysis requirements for a dataset."""

    grouped: dict[str, list[dict[str, Any]]] = {
        "modality": [],
        "behavioral_event": [],
        "data_standard": [],
        "required_signal": [],
    }
    dataset_terms = _dataset_concept_terms(graph, dataset_node_id)
    seen: set[tuple[str, str, str]] = set()
    for analysis_edge in _get_edges_for_node(
        graph,
        dataset_node_id,
        "dataset_supports_analysis",
        direction="out",
    ):
        analysis_node = graph.nodes.get(analysis_edge.target_node_id)
        if analysis_node is None:
            continue
        for requirement_edge in _get_edges_for_node(
            graph,
            analysis_node.node_id,
            direction="out",
        ):
            if not requirement_edge.edge_type.startswith("analysis_requires_"):
                continue
            target = graph.nodes.get(requirement_edge.target_node_id)
            if target is None:
                continue
            group = _requirement_group(requirement_edge, target.node_type)
            if group is None:
                continue
            target_terms = _node_terms(graph, target.node_id)
            if not (target_terms & dataset_terms.get(group, set())):
                continue
            key = (analysis_node.node_id, target.node_id, requirement_edge.edge_type)
            if key in seen:
                continue
            seen.add(key)
            grouped[group].append(
                {
                    "analysis": analysis_node.label,
                    "requirement": target.label,
                    "edge_type": requirement_edge.edge_type,
                    "confidence": round(
                        min(analysis_edge.confidence, requirement_edge.confidence),
                        3,
                    ),
                    "data_form": requirement_edge.properties.get("data_form"),
                    "requirement_type": requirement_edge.properties.get("requirement_type"),
                }
            )
    for values in grouped.values():
        values.sort(
            key=lambda item: (
                str(item["analysis"]),
                str(item["requirement"]),
                str(item["edge_type"]),
            )
        )
    return grouped


def _species_context(
    graph: KnowledgeGraph,
    dataset_node_id: str,
) -> dict[str, list[str]]:
    """Return broader organism context linked from dataset species nodes."""

    species_nodes = get_neighbors(
        graph,
        dataset_node_id,
        ["dataset_has_species"],
        direction="out",
    )
    context = {
        "taxon_groups": [],
        "model_roles": [],
        "animal_types": [],
    }
    for species_node in species_nodes:
        context["taxon_groups"].extend(
            _neighbor_labels(graph, species_node.node_id, "species_in_taxon_group")
        )
        context["model_roles"].extend(
            _neighbor_labels(graph, species_node.node_id, "species_has_model_role")
        )
        context["animal_types"].extend(
            _neighbor_labels(graph, species_node.node_id, "species_has_animal_type")
        )
    return {
        key: sorted(dict.fromkeys(values))
        for key, values in context.items()
    }


def compute_graph_features_for_result(
    graph: KnowledgeGraph | None,
    result_id: str,
    query_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return graph features for a result without requiring graph availability."""

    if graph is None:
        return _empty_features(graph_available=False)

    node_id = _resolve_dataset_node_id(graph, result_id)
    if node_id not in graph.nodes:
        return _empty_features(graph_available=True)

    graph_degree = sum(
        1
        for edge in graph.edges.values()
        if edge.source_node_id == node_id or edge.target_node_id == node_id
    )
    linked_papers = [paper.label for paper in find_papers_for_dataset(graph, node_id)]
    analysis_affordances = _neighbor_labels(graph, node_id, "dataset_supports_analysis")
    tasks = _neighbor_labels(graph, node_id, "dataset_has_task")
    modalities = _neighbor_labels(graph, node_id, "dataset_has_modality")
    species = _neighbor_labels(graph, node_id, "dataset_has_species")
    species_context = _species_context(graph, node_id)
    brain_regions = _neighbor_labels(graph, node_id, "dataset_records_region")
    context = query_context or {}
    query_species = {_norm(value) for value in context.get("species", [])}
    species_terms = {_norm(value) for value in species}
    broader_species_terms = {
        _norm(value)
        for value in (
            *species_context["taxon_groups"],
            *species_context["animal_types"],
            *species_context["model_roles"],
        )
    }
    matched_query_context = {
        "tasks": sorted(set(context.get("tasks", [])) & set(tasks)),
        "modalities": sorted(set(context.get("modalities", [])) & set(modalities)),
        "species": sorted(query_species & species_terms),
        "taxon_groups": sorted(query_species & broader_species_terms),
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
        "species": species,
        "species_context": species_context,
        "brain_regions": brain_regions,
        "matched_query_context": matched_query_context,
        "requirement_matches": _requirement_matches(graph, node_id),
    }


def graph_context_score(
    graph: KnowledgeGraph | None,
    result_id: str,
    query_context: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    use_edge_confidence: bool = False,
) -> float:
    """Return a small optional graph score; absent graphs contribute zero.

    Args:
        graph: Knowledge graph instance or None
        result_id: Dataset result identifier
        query_context: Query context with matched tasks/modalities/regions
        weights: Custom scoring weights (merged with defaults)
        use_edge_confidence: Weight scores by edge confidence values

    Returns:
        Graph context score between 0.0 and 0.25
    """
    if graph is None:
        return 0.0
    score_weights = {**DEFAULT_GRAPH_SEARCH_WEIGHTS, **(weights or {})}
    features = compute_graph_features_for_result(graph, result_id, query_context)
    if not features["graph_available"]:
        return 0.0

    matched = features["matched_query_context"]
    requirement_matches = features.get("requirement_matches", {})
    score = 0.0
    node_id = _resolve_dataset_node_id(graph, result_id)
    requirement_count = sum(len(values) for values in requirement_matches.values())
    covered_requirement_groups = len(
        [values for values in requirement_matches.values() if values]
    )

    if use_edge_confidence:
        # Weight linked paper edges by confidence
        paper_edges = _get_edges_for_node(graph, node_id, "paper_uses_dataset", direction="in")
        paper_edges.extend(_get_edges_for_node(graph, node_id, "paper_mentions_dataset", direction="in"))
        for edge in paper_edges[:3]:  # Cap at 3 highest-confidence papers
            score += edge.confidence * score_weights["linked_paper"]

        # Weight analysis affordance edges by confidence
        affordance_edges = _get_edges_for_node(graph, node_id, "dataset_supports_analysis", direction="out")
        for edge in affordance_edges[:3]:
            score += edge.confidence * score_weights["analysis_affordance"]

        # Weight task edges by confidence for matched tasks
        matched_tasks = matched.get("tasks", [])
        if matched_tasks:
            task_edges = _get_edges_for_node(graph, node_id, "dataset_has_task", direction="out")
            for edge in task_edges:
                target_node = graph.nodes.get(edge.target_node_id)
                if target_node and target_node.label in matched_tasks:
                    score += edge.confidence * score_weights["task_match"]

        # Weight modality edges by confidence for matched modalities
        matched_modalities = matched.get("modalities", [])
        if matched_modalities:
            modality_edges = _get_edges_for_node(graph, node_id, "dataset_has_modality", direction="out")
            for edge in modality_edges:
                target_node = graph.nodes.get(edge.target_node_id)
                if target_node and target_node.label in matched_modalities:
                    score += edge.confidence * score_weights["modality_match"]

        # Weight region edges by confidence for matched regions
        matched_regions = matched.get("brain_regions", [])
        if matched_regions:
            region_edges = _get_edges_for_node(graph, node_id, "dataset_records_region", direction="out")
            for edge in region_edges:
                target_node = graph.nodes.get(edge.target_node_id)
                if target_node and target_node.label in matched_regions:
                    score += edge.confidence * score_weights["brain_region_match"]

        # Degree bonus unchanged (no edge-level confidence applies)
        score += min(int(features["graph_degree"]), 10) * score_weights["degree"]
        score += len(matched.get("species", [])) * score_weights.get("species_match", 0.0)
        score += len(matched.get("taxon_groups", [])) * score_weights.get("taxon_match", 0.0)
        score += (
            min(requirement_count, 5)
            * score_weights.get("requirement_match", 0.0)
        )
        score += (
            min(covered_requirement_groups, 4)
            * score_weights.get("analysis_requirement_coverage", 0.0)
        )
    else:
        # Original counting-based scoring
        score += min(len(features["linked_papers"]), 3) * score_weights["linked_paper"]
        score += min(len(features["analysis_affordances"]), 3) * score_weights["analysis_affordance"]
        score += min(requirement_count, 5) * score_weights.get("requirement_match", 0.0)
        score += (
            min(covered_requirement_groups, 4)
            * score_weights.get("analysis_requirement_coverage", 0.0)
        )
        score += len(matched["tasks"]) * score_weights["task_match"]
        score += len(matched["modalities"]) * score_weights["modality_match"]
        score += len(matched.get("species", [])) * score_weights.get("species_match", 0.0)
        score += len(matched.get("taxon_groups", [])) * score_weights.get("taxon_match", 0.0)
        score += len(matched["brain_regions"]) * score_weights["brain_region_match"]
        score += min(int(features["graph_degree"]), 10) * score_weights["degree"]

    return round(min(score, 0.25), 4)
