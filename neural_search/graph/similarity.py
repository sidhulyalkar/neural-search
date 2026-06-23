"""Dataset similarity computation using shared graph concepts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from neural_search.graph.query import get_neighbors
from neural_search.graph.schema import (
    KnowledgeGraph,
    KnowledgeGraphEdge,
    make_edge_id,
)

# dataset_similar_to_dataset is the one latent (derived-not-observed) edge
# type the graph currently produces. Lifecycle metadata keeps it from being
# mistaken for an observed/asserted relationship: review_status tracks human
# curation, calibration_bin buckets confidence for calibration studies, and
# refresh_due flags when the underlying concept overlap should be
# recomputed (e.g. after new dataset metadata lands).
SIMILARITY_DERIVATION_METHOD = "shared_concept_jaccard"
SIMILARITY_REFRESH_INTERVAL_DAYS = 90


def _calibration_bin(score: float) -> str:
    """Bucket a similarity score for calibration tracking, relative to the
    default min_similarity=0.4 acceptance threshold."""
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _refresh_due(*, from_time: datetime | None = None) -> str:
    base = from_time or datetime.now(UTC)
    return (base + timedelta(days=SIMILARITY_REFRESH_INTERVAL_DAYS)).isoformat()


@dataclass
class SimilarityWeights:
    """Weights for different concept types in similarity computation."""

    task: float = 0.30
    modality: float = 0.20
    recording_scale: float = 0.12
    brain_region: float = 0.18
    behavioral_event: float = 0.15
    analysis_affordance: float = 0.10


DEFAULT_SIMILARITY_WEIGHTS = SimilarityWeights()


@dataclass
class DatasetSimilarity:
    """Similarity score between two datasets with breakdown."""

    source_id: str
    target_id: str
    similarity_score: float
    shared_tasks: list[str]
    shared_modalities: list[str]
    shared_recording_scales: list[str]
    shared_regions: list[str]
    shared_events: list[str]
    shared_affordances: list[str]
    explanation: str


def _get_dataset_concepts(
    graph: KnowledgeGraph,
    dataset_node_id: str,
) -> dict[str, set[str]]:
    """Extract all connected concepts for a dataset node."""
    concepts: dict[str, set[str]] = {
        "task": set(),
        "modality": set(),
        "recording_scale": set(),
        "brain_region": set(),
        "behavioral_event": set(),
        "analysis_affordance": set(),
    }

    # Map edge types to concept categories
    edge_type_mapping = {
        "dataset_has_task": "task",
        "dataset_has_modality": "modality",
        "dataset_has_recording_scale": "recording_scale",
        "dataset_records_region": "brain_region",
        "dataset_has_behavioral_event": "behavioral_event",
        "dataset_supports_analysis": "analysis_affordance",
    }

    for edge_type, category in edge_type_mapping.items():
        neighbors = get_neighbors(
            graph,
            dataset_node_id,
            edge_types=[edge_type],
            direction="out",
        )
        concepts[category].update(n.label for n in neighbors)

    return concepts


def compute_dataset_similarity(
    graph: KnowledgeGraph,
    source_node_id: str,
    target_node_id: str,
    weights: SimilarityWeights | None = None,
) -> DatasetSimilarity:
    """Compute similarity between two datasets based on shared concepts.

    Args:
        graph: Knowledge graph containing datasets
        source_node_id: Source dataset node ID
        target_node_id: Target dataset node ID
        weights: Optional custom weights for concept types

    Returns:
        DatasetSimilarity with score and breakdown
    """
    weights = weights or DEFAULT_SIMILARITY_WEIGHTS

    source_concepts = _get_dataset_concepts(graph, source_node_id)
    target_concepts = _get_dataset_concepts(graph, target_node_id)

    # Compute shared concepts
    shared = {
        category: source_concepts[category] & target_concepts[category]
        for category in source_concepts
    }

    # Compute weighted Jaccard-like similarity
    total_score = 0.0
    weight_sum = 0.0

    weight_map = {
        "task": weights.task,
        "modality": weights.modality,
        "recording_scale": weights.recording_scale,
        "brain_region": weights.brain_region,
        "behavioral_event": weights.behavioral_event,
        "analysis_affordance": weights.analysis_affordance,
    }

    for category, weight in weight_map.items():
        source_set = source_concepts[category]
        target_set = target_concepts[category]
        union = source_set | target_set

        if union:
            jaccard = len(shared[category]) / len(union)
            total_score += jaccard * weight
            weight_sum += weight

    # Normalize
    similarity = total_score / weight_sum if weight_sum > 0 else 0.0

    # Generate explanation
    explanation_parts = []
    if shared["task"]:
        explanation_parts.append(f"tasks: {', '.join(sorted(shared['task']))}")
    if shared["modality"]:
        explanation_parts.append(f"modalities: {', '.join(sorted(shared['modality']))}")
    if shared["recording_scale"]:
        explanation_parts.append(
            f"recording scales: {', '.join(sorted(shared['recording_scale']))}"
        )
    if shared["brain_region"]:
        explanation_parts.append(f"regions: {', '.join(sorted(shared['brain_region']))}")

    explanation = f"Shared {'; '.join(explanation_parts)}" if explanation_parts else "No shared concepts"

    return DatasetSimilarity(
        source_id=source_node_id,
        target_id=target_node_id,
        similarity_score=round(similarity, 4),
        shared_tasks=sorted(shared["task"]),
        shared_modalities=sorted(shared["modality"]),
        shared_recording_scales=sorted(shared["recording_scale"]),
        shared_regions=sorted(shared["brain_region"]),
        shared_events=sorted(shared["behavioral_event"]),
        shared_affordances=sorted(shared["analysis_affordance"]),
        explanation=explanation,
    )


def find_similar_datasets(
    graph: KnowledgeGraph,
    dataset_node_id: str,
    min_similarity: float = 0.3,
    top_k: int = 10,
    weights: SimilarityWeights | None = None,
) -> list[DatasetSimilarity]:
    """Find datasets similar to a given dataset.

    Args:
        graph: Knowledge graph
        dataset_node_id: Source dataset node ID
        min_similarity: Minimum similarity threshold
        top_k: Maximum number of results
        weights: Optional custom weights

    Returns:
        List of DatasetSimilarity sorted by score descending
    """
    similarities = []

    for node in graph.nodes.values():
        if node.node_type != "dataset":
            continue
        if node.node_id == dataset_node_id:
            continue

        sim = compute_dataset_similarity(
            graph, dataset_node_id, node.node_id, weights
        )

        if sim.similarity_score >= min_similarity:
            similarities.append(sim)

    # Sort by similarity descending
    similarities.sort(key=lambda s: s.similarity_score, reverse=True)

    return similarities[:top_k]


def build_similarity_edges(
    graph: KnowledgeGraph,
    min_similarity: float = 0.4,
    min_shared_concepts: int = 2,
    weights: SimilarityWeights | None = None,
) -> list[KnowledgeGraphEdge]:
    """Build similarity edges between all dataset pairs.

    Args:
        graph: Knowledge graph
        min_similarity: Minimum similarity score to create edge
        min_shared_concepts: Minimum shared concepts required
        weights: Optional custom weights

    Returns:
        List of new similarity edges
    """
    edges = []
    dataset_nodes = [
        node for node in graph.nodes.values()
        if node.node_type == "dataset"
    ]

    # Compare all pairs
    for i, source in enumerate(dataset_nodes):
        for target in dataset_nodes[i + 1:]:
            sim = compute_dataset_similarity(
                graph, source.node_id, target.node_id, weights
            )

            # Check thresholds
            total_shared = (
                len(sim.shared_tasks)
                + len(sim.shared_modalities)
                + len(sim.shared_recording_scales)
                + len(sim.shared_regions)
                + len(sim.shared_events)
                + len(sim.shared_affordances)
            )

            if sim.similarity_score >= min_similarity and total_shared >= min_shared_concepts:
                edge = KnowledgeGraphEdge(
                    edge_id=make_edge_id(
                        source.node_id,
                        "dataset_similar_to_dataset",
                        target.node_id,
                    ),
                    source_node_id=source.node_id,
                    target_node_id=target.node_id,
                    edge_type="dataset_similar_to_dataset",
                    directed=False,
                    confidence=sim.similarity_score,
                    properties={
                        "shared_tasks": sim.shared_tasks,
                        "shared_modalities": sim.shared_modalities,
                        "shared_regions": sim.shared_regions,
                        "explanation": sim.explanation,
                        "derivation_method": SIMILARITY_DERIVATION_METHOD,
                        "review_status": "unreviewed",
                        "calibration_bin": _calibration_bin(sim.similarity_score),
                        "refresh_due": _refresh_due(),
                    },
                )
                edges.append(edge)

    return edges


def add_similarity_edges_to_graph(
    graph: KnowledgeGraph,
    min_similarity: float = 0.4,
    min_shared_concepts: int = 2,
) -> int:
    """Add similarity edges to an existing graph in place.

    Args:
        graph: Knowledge graph to modify
        min_similarity: Minimum similarity threshold
        min_shared_concepts: Minimum shared concepts

    Returns:
        Number of edges added
    """
    new_edges = build_similarity_edges(
        graph,
        min_similarity=min_similarity,
        min_shared_concepts=min_shared_concepts,
    )

    for edge in new_edges:
        if edge.edge_id not in graph.edges:
            graph.edges[edge.edge_id] = edge

    return len(new_edges)
