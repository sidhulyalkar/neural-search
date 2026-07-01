"""Optional graph-derived features for retrieval results."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
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
    "relationship_edge": 0.012,
    "reanalysis_edge": 0.018,
    "requirement_match": 0.01,
    "analysis_requirement_coverage": 0.015,
    "task_match": 0.03,
    "modality_match": 0.02,
    "species_match": 0.02,
    "taxon_match": 0.01,
    "brain_region_match": 0.02,
    "degree": 0.01,
}

RELATIONSHIP_EDGE_TYPES = {
    "dataset_similar_to_dataset",
    "same_region_cross_modality",
    "same_task_cross_species",
    "same_region_same_task",
    "dataset_reanalysis_bridge_dataset",
}

REANALYSIS_EDGE_TYPES = {
    "dataset_old_dataset_new_method_candidate",
    "dataset_reinterpretation_candidate",
    "dataset_reprocessing_candidate",
}

_LOCAL_EDGE_INDEX_CACHE: dict[
    int,
    tuple[int, dict[str, list[KnowledgeGraphEdge]], dict[str, list[KnowledgeGraphEdge]]],
] = {}

# Reverse alias index: graph_id → {alias/source_id/node_id → canonical node_id}
# Built once per graph load, makes _resolve_dataset_node_id O(1) instead of O(N)
_ALIAS_INDEX_CACHE: dict[int, dict[str, str]] = {}

REQUIREMENT_GROUPS = {
    "modality": "modality",
    "recording_scale": "recording_scale",
    "behavioral_event": "behavioral_event",
    "data_standard": "data_standard",
    "required_signal": "required_signal",
}


@lru_cache(maxsize=8)
def load_graph_if_exists(path: str | None) -> KnowledgeGraph | None:
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


def _build_alias_index(graph: KnowledgeGraph) -> dict[str, str]:
    """Build {alias → canonical_node_id} for all dataset nodes. Called once per graph."""
    index: dict[str, str] = {}
    for node in graph.nodes.values():
        if node.node_type != "dataset":
            continue
        nid = node.node_id
        index[nid] = nid
        for alias in node.aliases:
            index[alias] = nid
        for sid in node.source_ids:
            index[sid] = nid
        raw_sid = str(node.properties.get("source_id", ""))
        if raw_sid:
            index[raw_sid] = nid
    return index


def _get_alias_index(graph: KnowledgeGraph) -> dict[str, str]:
    gid = id(graph)
    if gid not in _ALIAS_INDEX_CACHE:
        _ALIAS_INDEX_CACHE[gid] = _build_alias_index(graph)
    return _ALIAS_INDEX_CACHE[gid]


def _resolve_dataset_node_id(graph: KnowledgeGraph, result_id: str) -> str:
    candidate = _result_node_id(result_id)
    if candidate in graph.nodes:
        return candidate
    alias_index = _get_alias_index(graph)
    return alias_index.get(result_id, candidate)


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
    out_index, in_index = _local_edge_indexes(graph)
    if direction in ("out", "both"):
        for edge in out_index.get(node_id, []):
            if edge_type is None or edge.edge_type == edge_type:
                edges.append(edge)
    if direction in ("in", "both"):
        for edge in in_index.get(node_id, []):
            if edge_type is None or edge.edge_type == edge_type:
                edges.append(edge)
    # Sort by confidence descending
    return sorted(edges, key=lambda e: e.confidence, reverse=True)


def _local_edge_indexes(
    graph: KnowledgeGraph,
) -> tuple[dict[str, list[KnowledgeGraphEdge]], dict[str, list[KnowledgeGraphEdge]]]:
    cache_key = id(graph)
    edge_count = len(graph.edges)
    cached = _LOCAL_EDGE_INDEX_CACHE.get(cache_key)
    if cached and cached[0] == edge_count:
        return cached[1], cached[2]
    out_index: dict[str, list[KnowledgeGraphEdge]] = {}
    in_index: dict[str, list[KnowledgeGraphEdge]] = {}
    for edge in graph.edges.values():
        out_index.setdefault(edge.source_node_id, []).append(edge)
        in_index.setdefault(edge.target_node_id, []).append(edge)
    _LOCAL_EDGE_INDEX_CACHE[cache_key] = (edge_count, out_index, in_index)
    return out_index, in_index


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
        "relationship_edges": [],
        "reanalysis_edges": [],
        "recording_scales": [],
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
        "dataset_has_recording_scale": "recording_scale",
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


def _relationship_summaries(
    graph: KnowledgeGraph,
    dataset_node_id: str,
    edge_types: set[str],
) -> list[dict[str, Any]]:
    """Return compact summaries for cross-dataset and reanalysis edges."""

    summaries: list[dict[str, Any]] = []
    for edge in _get_edges_for_node(graph, dataset_node_id, direction="both"):
        if edge.edge_type not in edge_types:
            continue
        other_id = (
            edge.target_node_id
            if edge.source_node_id == dataset_node_id
            else edge.source_node_id
        )
        other = graph.nodes.get(other_id)
        summaries.append(
            {
                "edge_type": edge.edge_type,
                "dataset": other.label if other else other_id,
                "dataset_id": other_id,
                "confidence": round(edge.confidence, 3),
                "relationship_type": edge.properties.get("relationship_type"),
                "method": edge.properties.get("method"),
                "explanation": edge.properties.get("explanation")
                or edge.properties.get("context"),
            }
        )
    summaries.sort(
        key=lambda item: (
            -float(item["confidence"]),
            str(item["edge_type"]),
            str(item["dataset_id"]),
        )
    )
    return summaries


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

    out_index, in_index = _local_edge_indexes(graph)
    graph_degree = len(out_index.get(node_id, [])) + len(in_index.get(node_id, []))
    linked_papers = [paper.label for paper in find_papers_for_dataset(graph, node_id)]
    analysis_affordances = _neighbor_labels(graph, node_id, "dataset_supports_analysis")
    tasks = _neighbor_labels(graph, node_id, "dataset_has_task")
    modalities = _neighbor_labels(graph, node_id, "dataset_has_modality")
    recording_scales = _neighbor_labels(graph, node_id, "dataset_has_recording_scale")
    species = _neighbor_labels(graph, node_id, "dataset_has_species")
    species_context = _species_context(graph, node_id)
    brain_regions = _neighbor_labels(graph, node_id, "dataset_records_region")
    relationship_edges = _relationship_summaries(graph, node_id, RELATIONSHIP_EDGE_TYPES)
    reanalysis_edges = _relationship_summaries(graph, node_id, REANALYSIS_EDGE_TYPES)
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
        "recording_scales": sorted(
            set(context.get("recording_scales", [])) & set(recording_scales)
        ),
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
        "recording_scales": recording_scales,
        "species": species,
        "species_context": species_context,
        "brain_regions": brain_regions,
        "relationship_edges": relationship_edges,
        "reanalysis_edges": reanalysis_edges,
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
    relationship_count = len(features.get("relationship_edges", []))
    reanalysis_count = len(features.get("reanalysis_edges", []))

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
        score += min(relationship_count, 5) * score_weights.get("relationship_edge", 0.0)
        score += min(reanalysis_count, 5) * score_weights.get("reanalysis_edge", 0.0)
    else:
        # Original counting-based scoring
        score += min(len(features["linked_papers"]), 3) * score_weights["linked_paper"]
        score += min(len(features["analysis_affordances"]), 3) * score_weights["analysis_affordance"]
        score += min(requirement_count, 5) * score_weights.get("requirement_match", 0.0)
        score += (
            min(covered_requirement_groups, 4)
            * score_weights.get("analysis_requirement_coverage", 0.0)
        )
        score += len(matched.get("tasks", [])) * score_weights["task_match"]
        score += len(matched.get("modalities", [])) * score_weights["modality_match"]
        score += len(matched.get("species", [])) * score_weights.get("species_match", 0.0)
        score += len(matched.get("taxon_groups", [])) * score_weights.get("taxon_match", 0.0)
        score += len(matched.get("brain_regions", [])) * score_weights["brain_region_match"]
        score += min(relationship_count, 5) * score_weights.get("relationship_edge", 0.0)
        score += min(reanalysis_count, 5) * score_weights.get("reanalysis_edge", 0.0)
        score += min(int(features["graph_degree"]), 10) * score_weights["degree"]

    return round(min(score, 0.25), 4)


# ---------------------------------------------------------------------------
# KG layer indexes — built lazily from artifact JSONL files
# ---------------------------------------------------------------------------

_COMPOSED_KG_PATH = Path("artifacts/kg/composed_kg.jsonl")
_NER_KG_PATH = Path("artifacts/ner/ner_kg.jsonl")
_CITATION_EDGES_PATH = Path("artifacts/citations/citation_edges.jsonl")

_CITATION_INDEGREE_CAP = 50


@lru_cache(maxsize=1)
def _load_neurosynth_index() -> dict[str, list[float]]:
    """Build region→[confidence, ...] index from topic_activates_region edges."""
    region_weights: dict[str, list[float]] = defaultdict(list)
    if not _COMPOSED_KG_PATH.exists():
        return {}
    with _COMPOSED_KG_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record_type") != "edge":
                continue
            edge = rec.get("edge", {})
            if edge.get("edge_type") != "topic_activates_region":
                continue
            target = edge.get("target_node_id", "")
            region_key = target.split(":", 1)[-1]
            region_weights[region_key].append(float(edge.get("confidence", 0.0)))
    return dict(region_weights)


@lru_cache(maxsize=1)
def _load_ner_index() -> dict[str, dict[str, list[str]]]:
    """Build paper_id→{regions, disorders} index from NER KG edges."""
    paper_entities: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"regions": [], "disorders": []}
    )
    if not _NER_KG_PATH.exists():
        return {}
    with _NER_KG_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record_type") != "edge":
                continue
            edge = rec.get("edge", {})
            edge_type = edge.get("edge_type", "")
            src = edge.get("source_node_id", "")
            tgt = edge.get("target_node_id", "")
            region_key = tgt.split(":", 1)[-1]
            if edge_type == "paper_mentions_region":
                paper_entities[src]["regions"].append(region_key)
            elif edge_type == "paper_involves_disorder":
                paper_entities[src]["disorders"].append(region_key)
    return dict(paper_entities)


@lru_cache(maxsize=1)
def _load_citation_indegree() -> dict[str, int]:
    """Build paper_id→in-degree index from citation edges JSONL."""
    indegree: dict[str, int] = defaultdict(int)
    if not _CITATION_EDGES_PATH.exists():
        return {}
    with _CITATION_EDGES_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cited = rec.get("cited_paper_id")
            if cited:
                indegree[cited] += 1
    return dict(indegree)


def _norm_region(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------

def neurosynth_region_score(
    graph: KnowledgeGraph | None,
    parsed_query: dict[str, Any],
    result_id: str,
) -> float:
    """Return [0,1] boost when queried brain regions activate strongly in NeuroSynth."""
    queried_regions = parsed_query.get("brain_regions", [])
    if not queried_regions:
        return 0.0
    try:
        index = _load_neurosynth_index()
    except Exception:
        return 0.0
    if not index:
        return 0.0
    weights: list[float] = []
    for region in queried_regions:
        key = _norm_region(str(region))
        if key in index:
            weights.extend(index[key])
    if not weights:
        return 0.0
    return round(min(sum(weights) / len(weights), 1.0), 4)


def ner_entity_coverage_score(
    graph: KnowledgeGraph | None,
    parsed_query: dict[str, Any],
    result_id: str,
) -> float:
    """Return fraction of queried regions/disorders covered by NER paper mentions."""
    queried_regions = [_norm_region(r) for r in parsed_query.get("brain_regions", [])]
    queried_entities = set(queried_regions)
    if not queried_entities:
        return 0.0

    paper_node_ids: list[str] = []
    if graph is not None:
        try:
            node_id = _resolve_dataset_node_id(graph, result_id)
            paper_node_ids = [p.node_id for p in find_papers_for_dataset(graph, node_id)]
        except Exception:
            pass

    if not paper_node_ids:
        return 0.0

    try:
        ner_index = _load_ner_index()
    except Exception:
        return 0.0
    if not ner_index:
        return 0.0

    covered: set[str] = set()
    for paper_node_id in paper_node_ids:
        ner_key = paper_node_id.removeprefix("node:")
        entities = ner_index.get(ner_key, {})
        covered.update(_norm_region(r) for r in entities.get("regions", []))
        covered.update(_norm_region(d) for d in entities.get("disorders", []))

    matched = queried_entities & covered
    return round(len(matched) / len(queried_entities), 4)


def citation_authority_score(
    graph: KnowledgeGraph | None,
    result_id: str,
    paper_node_ids: list[str],
) -> float:
    """Return [0,1] score based on max in-degree of linked papers in citation graph."""
    if not paper_node_ids:
        return 0.0
    try:
        indegree = _load_citation_indegree()
    except Exception:
        return 0.0
    if not indegree:
        return 0.0

    scores: list[float] = []
    for paper_node_id in paper_node_ids:
        openalex_id = paper_node_id.rsplit(":", 1)[-1]
        cited_count = indegree.get(openalex_id, 0)
        scores.append(min(cited_count, _CITATION_INDEGREE_CAP) / _CITATION_INDEGREE_CAP)

    return round(max(scores), 4)


# ---------------------------------------------------------------------------
# Allen CCF hierarchy helpers
# ---------------------------------------------------------------------------

_ALLEN_CCF_CACHE: tuple[dict[str, set[str]], dict[str, set[str]]] | None = None

_ALLEN_CCF_DEFAULT_PATH = Path(__file__).parents[2] / "artifacts" / "atlas" / "allen_ccf_mouse_structures.json"


def _norm_ccf(value: str) -> str:
    """Normalize a region name for CCF lookup: lowercase, spaces/hyphens → underscores."""
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def load_allen_ccf_hierarchy(
    path: str | Path | None = None,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Load the Allen CCF mouse brain atlas and build ancestor/descendant maps.

    Returns a pair ``(ancestors_map, descendants_map)`` where each key is a
    normalized structure name or acronym and each value is the set of normalized
    names of all ancestors (resp. descendants) of that structure.

    Results are cached in ``_ALLEN_CCF_CACHE`` so the file is only parsed once
    per process.
    """
    global _ALLEN_CCF_CACHE
    if _ALLEN_CCF_CACHE is not None:
        return _ALLEN_CCF_CACHE

    atlas_path = Path(path or _ALLEN_CCF_DEFAULT_PATH)
    if not atlas_path.exists():
        _ALLEN_CCF_CACHE = ({}, {})
        return _ALLEN_CCF_CACHE

    try:
        with atlas_path.open(encoding="utf-8") as fh:
            structures: list[dict] = json.loads(fh.read())
    except Exception:
        _ALLEN_CCF_CACHE = ({}, {})
        return _ALLEN_CCF_CACHE

    # Build allen_id → structure and allen_id → parent_id lookup tables.
    by_id: dict[int, dict] = {s["allen_id"]: s for s in structures}
    parent_of: dict[int, int | None] = {s["allen_id"]: s.get("parent_id") for s in structures}

    # Build all ancestors for each allen_id (walk up the tree).
    def _ancestors(aid: int) -> set[int]:
        result: set[int] = set()
        current = parent_of.get(aid)
        while current is not None:
            result.add(current)
            current = parent_of.get(current)
        return result

    # Build all descendants for each allen_id (walk down via children_ids).
    def _descendants(aid: int) -> set[int]:
        result: set[int] = set()
        stack = list(by_id[aid].get("children_ids") or [])
        while stack:
            child_id = stack.pop()
            if child_id in result:
                continue
            result.add(child_id)
            if child_id in by_id:
                stack.extend(by_id[child_id].get("children_ids") or [])
        return result

    # Helper: collect the normalized name keys for a structure.
    def _keys(s: dict) -> list[str]:
        keys = []
        if s.get("name"):
            keys.append(_norm_ccf(s["name"]))
        if s.get("acronym") and s["acronym"] != s.get("name"):
            keys.append(_norm_ccf(s["acronym"]))
        return keys

    # Build name→set-of-ancestor-names and name→set-of-descendant-names.
    ancestors_map: dict[str, set[str]] = {}
    descendants_map: dict[str, set[str]] = {}

    for s in structures:
        aid = s["allen_id"]
        anc_ids = _ancestors(aid)
        desc_ids = _descendants(aid)

        anc_names: set[str] = set()
        for anc_id in anc_ids:
            if anc_id in by_id:
                anc_names.update(_keys(by_id[anc_id]))

        desc_names: set[str] = set()
        for desc_id in desc_ids:
            if desc_id in by_id:
                desc_names.update(_keys(by_id[desc_id]))

        for key in _keys(s):
            ancestors_map[key] = anc_names
            descendants_map[key] = desc_names

    _ALLEN_CCF_CACHE = (ancestors_map, descendants_map)
    return _ALLEN_CCF_CACHE


def expand_region_query(
    regions: list[str],
    include_descendants: bool = True,
) -> set[str]:
    """Return *regions* expanded to include all their descendants via Allen CCF.

    If the atlas is unavailable the original labels are returned unchanged.
    """
    _, descendants_map = load_allen_ccf_hierarchy()
    expanded: set[str] = set()
    for r in regions:
        key = _norm_ccf(r)
        expanded.add(key)
        if include_descendants and key in descendants_map:
            expanded.update(descendants_map[key])
    return expanded


_CCF_EXPAND_CACHE: dict[str, set[str]] = {}


def _ccf_expand_key(key: str, ancestors_map: dict[str, set[str]]) -> set[str]:
    """Return all CCF keys semantically matching ``key`` (memoized per normalized key).

    Rules (short CCF keys <4 chars excluded from substring/prefix checks):
    1. Exact match
    2. key is a non-trivial substring of ccf_key (≥5 chars shared)
    3. Longest common prefix ≥5 chars (hippocampus↔hippocampal)
    4. First underscore-token match if token ≥4 chars
    """
    if key in _CCF_EXPAND_CACHE:
        return _CCF_EXPAND_CACHE[key]
    matches: set[str] = set()
    if not key or len(key) < 2:
        _CCF_EXPAND_CACHE[key] = matches
        return matches
    key_tokens = key.split("_")
    for ccf_key in ancestors_map:
        if ccf_key == key:
            matches.add(ccf_key)
            continue
        if len(ccf_key) < 4:
            continue
        # Substring: key embedded in ccf_key
        if len(key) >= 4 and key in ccf_key:
            matches.add(ccf_key)
            continue
        # Common leading prefix ≥5 chars
        common = 0
        for a, b in zip(key, ccf_key):
            if a == b:
                common += 1
            else:
                break
        if common >= 5:
            matches.add(ccf_key)
            continue
        # First token match (≥4 chars) — "prefrontal" matches "prefrontal_area_*"
        ccf_first = ccf_key.split("_")[0]
        if len(ccf_first) >= 4 and ccf_first == key_tokens[0]:
            matches.add(ccf_key)
    _CCF_EXPAND_CACHE[key] = matches
    return matches


def region_hierarchy_score(
    dataset_regions: list[str],
    query_regions: list[str],
) -> float:
    """Return [0, 1] score reflecting Allen CCF hierarchy overlap.

    Scoring per (dataset_region, query_region) pair:
    - Direct match: 1.0
    - Query term (or a CCF synonym) is an ancestor of dataset region: 0.7
    - Dataset region is an ancestor of query term (or synonym): 0.5

    Substring expansion handles cases where query uses "hippocampus" but
    Allen CCF has "hippocampal_region" / "hippocampal_formation".
    Returns the maximum score across all pairs, capped at 1.0.
    """
    if not dataset_regions or not query_regions:
        return 0.0

    ancestors_map, _ = load_allen_ccf_hierarchy()
    if not ancestors_map:
        return 0.0

    ds_keys = [_norm_ccf(r) for r in dataset_regions]
    q_keys = [_norm_ccf(r) for r in query_regions]

    best = 0.0
    for ds in ds_keys:
        for q in q_keys:
            # Level 0: exact string match — always score 1.0
            if ds == q:
                best = 1.0
                break

            # Level 1+: CCF hierarchy traversal
            ds_expanded = _ccf_expand_key(ds, ancestors_map)
            q_expanded = _ccf_expand_key(q, ancestors_map)

            if ds_expanded or q_expanded:
                # Direct CCF synonym overlap
                if ds_expanded & q_expanded:
                    best = max(best, 1.0)
                    break

                # Compute combined ancestor set for dataset
                anc_of_ds: set[str] = set()
                for dk in ds_expanded:
                    anc_of_ds.update(ancestors_map.get(dk, set()))

                # Any query synonym is ancestor of dataset region → query is broader
                if q_expanded & anc_of_ds:
                    best = max(best, 0.7)
                    continue

                # Dataset region is ancestor of any query synonym → dataset is broader
                for qk in q_expanded:
                    anc_of_q = ancestors_map.get(qk, set())
                    if ds_expanded & anc_of_q:
                        best = max(best, 0.5)
                        break

            # Level 2: prefix fallback — same first token is weak positive signal
            elif len(ds) >= 4 and ds.split("_")[0] == q.split("_")[0]:
                best = max(best, 0.4)

        if best >= 1.0:
            break

    return round(min(best, 1.0), 4)


KG_LAYER_WEIGHTS: dict[str, float] = {
    "neurosynth_region": 0.025,
    "ner_coverage": 0.020,
    "citation_authority": 0.015,
    "kg_concept": 0.08,
    "region_hierarchy": 0.03,
}

_KG_CONCEPT_INDEX: dict[str, list[str]] | None = None


def load_kg_concept_index(path: str | Path | None = None) -> dict[str, list[str]]:
    """Load (and cache) the dataset→concept index built by corpus_kg_linker."""
    global _KG_CONCEPT_INDEX
    if _KG_CONCEPT_INDEX is not None:
        return _KG_CONCEPT_INDEX
    index_path = Path(path or "data/kg/dataset_concept_index.jsonl")
    if not index_path.exists():
        _KG_CONCEPT_INDEX = {}
        return _KG_CONCEPT_INDEX
    try:
        from neural_search.ingestion.corpus_kg_linker import load_dataset_concept_index
        _KG_CONCEPT_INDEX = load_dataset_concept_index(index_path)
    except Exception:
        _KG_CONCEPT_INDEX = {}
    return _KG_CONCEPT_INDEX


def concept_overlap_score(
    dataset_id: str,
    parsed_query: dict[str, Any],
    concept_index: dict[str, list[str]] | None = None,
) -> float:
    """Return [0, 0.3] score based on concept overlap between query and dataset.

    Uses Scholarpedia-expanded concepts from ``parsed_query["concepts"]`` and
    checks against the dataset's mapped KG concept nodes.
    """
    if concept_index is None:
        concept_index = load_kg_concept_index()
    if not concept_index:
        return 0.0

    query_concepts: list[str] = list(parsed_query.get("concepts") or [])
    if not query_concepts:
        return 0.0

    dataset_concepts = concept_index.get(dataset_id, [])
    if not dataset_concepts:
        bare = dataset_id.removeprefix("dataset:")
        dataset_concepts = concept_index.get(bare, [])
    if not dataset_concepts:
        prefixed = dataset_id if dataset_id.startswith("dataset:") else f"dataset:{dataset_id}"
        dataset_concepts = concept_index.get(prefixed, [])
    if not dataset_concepts:
        return 0.0

    def _slug_norm(s: str) -> str:
        return s.casefold().replace("concept:", "").replace("-", "_").replace(" ", "_")

    ds_slugs = {_slug_norm(c) for c in dataset_concepts}
    q_slugs = [_slug_norm(c) for c in query_concepts]
    matched = sum(1 for q in q_slugs if q in ds_slugs)
    raw = matched / max(len(q_slugs), 1)
    return round(min(raw * 0.3, 0.3), 4)


def compute_kg_layer_scores(
    graph: KnowledgeGraph | None,
    result_id: str,
    parsed_query: dict[str, Any],
    concept_index: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    """Compute NeuroSynth, NER, citation, concept-overlap, and region-hierarchy scores."""
    ns_score = neurosynth_region_score(graph, parsed_query, result_id)
    ner_score = ner_entity_coverage_score(graph, parsed_query, result_id)

    paper_node_ids: list[str] = []
    dataset_regions: list[str] = []
    if graph is not None:
        try:
            node_id = _resolve_dataset_node_id(graph, result_id)
            paper_node_ids = [p.node_id for p in find_papers_for_dataset(graph, node_id)]
            dataset_regions = _neighbor_labels(graph, node_id, "dataset_records_region")
        except Exception:
            pass
    cit_score = citation_authority_score(graph, result_id, paper_node_ids)
    kg_concept = concept_overlap_score(result_id, parsed_query, concept_index)

    query_regions = list(parsed_query.get("brain_regions") or parsed_query.get("regions") or [])
    rh_score = region_hierarchy_score(dataset_regions, query_regions) if query_regions else 0.0

    weighted_total = (
        KG_LAYER_WEIGHTS["neurosynth_region"] * ns_score
        + KG_LAYER_WEIGHTS["ner_coverage"] * ner_score
        + KG_LAYER_WEIGHTS["citation_authority"] * cit_score
        + KG_LAYER_WEIGHTS["kg_concept"] * kg_concept
        + KG_LAYER_WEIGHTS["region_hierarchy"] * rh_score
    )
    return {
        "neurosynth_region_score": round(ns_score, 4),
        "ner_entity_coverage_score": round(ner_score, 4),
        "citation_authority_score": round(cit_score, 4),
        "kg_concept_score": round(kg_concept, 4),
        "region_hierarchy_score": round(rh_score, 4),
        "kg_layer_weighted_total": round(weighted_total, 4),
    }
