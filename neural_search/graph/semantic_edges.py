"""Semantic similarity edges using learned embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neural_search.graph.schema import (
    KnowledgeGraph,
    KnowledgeGraphEdge,
    make_edge_id,
    make_node_id,
)

if TYPE_CHECKING:
    from neural_search.embeddings.concept_embeddings import (
        ConceptEmbeddingIndex,
    )
    from neural_search.embeddings.semantic_fingerprint import (
        SemanticDatasetFingerprint,
        SemanticSimilarity,
    )


@dataclass
class SemanticEdgeConfig:
    """Configuration for semantic edge creation."""

    # Dataset similarity thresholds
    min_combined_similarity: float = 0.5
    min_task_similarity: float = 0.6
    min_modality_similarity: float = 0.6
    min_behavior_similarity: float = 0.5

    # Concept similarity thresholds
    min_concept_similarity: float = 0.65

    # Maximum edges per dataset
    max_similar_datasets_per_node: int = 5

    # Edge type prefix for semantic edges
    edge_type_prefix: str = "semantic"


DEFAULT_SEMANTIC_CONFIG = SemanticEdgeConfig()


@dataclass
class SemanticEdgeResult:
    """Result of adding semantic edges to graph."""

    dataset_similarity_edges_added: int
    concept_similarity_edges_added: int
    total_edges_added: int
    edge_ids: list[str]


def _compute_fingerprint_similarity(
    source: SemanticDatasetFingerprint,
    target: SemanticDatasetFingerprint,
) -> SemanticSimilarity:
    """Compute semantic similarity between two fingerprints."""
    from neural_search.embeddings.semantic_fingerprint import (
        compute_semantic_similarity,
    )

    return compute_semantic_similarity(source, target)


def build_semantic_dataset_edges(
    fingerprints: list[SemanticDatasetFingerprint],
    config: SemanticEdgeConfig | None = None,
) -> list[KnowledgeGraphEdge]:
    """Build semantic similarity edges between datasets from fingerprints.

    Args:
        fingerprints: List of semantic dataset fingerprints
        config: Optional edge configuration

    Returns:
        List of new semantic similarity edges
    """
    config = config or DEFAULT_SEMANTIC_CONFIG
    edges: list[KnowledgeGraphEdge] = []

    # Index fingerprints by dataset ID
    fp_by_id = {fp.dataset_id: fp for fp in fingerprints}

    # Compare all pairs
    for i, source_fp in enumerate(fingerprints):
        similar_targets: list[tuple[str, SemanticSimilarity]] = []

        for target_fp in fingerprints[i + 1:]:
            sim = _compute_fingerprint_similarity(source_fp, target_fp)

            # Check if meets thresholds
            if sim.combined_similarity >= config.min_combined_similarity:
                similar_targets.append((target_fp.dataset_id, sim))

        # Sort by similarity and limit
        similar_targets.sort(key=lambda x: x[1].combined_similarity, reverse=True)
        similar_targets = similar_targets[:config.max_similar_datasets_per_node]

        # Create edges for top similar datasets
        for target_id, sim in similar_targets:
            source_node_id = make_node_id("dataset", source_fp.dataset_id)
            target_node_id = make_node_id("dataset", target_id)

            # Determine similarity type based on strongest dimension
            sim_type = "general"
            if sim.task_similarity >= config.min_task_similarity:
                sim_type = "task"
            elif sim.modality_similarity >= config.min_modality_similarity:
                sim_type = "modality"
            elif sim.behavior_similarity >= config.min_behavior_similarity:
                sim_type = "behavior"

            edge = KnowledgeGraphEdge(
                edge_id=make_edge_id(
                    source_node_id,
                    f"{config.edge_type_prefix}_similar_{sim_type}",
                    target_node_id,
                ),
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                edge_type=f"{config.edge_type_prefix}_similar_{sim_type}",
                directed=False,
                confidence=round(sim.combined_similarity, 3),
                properties={
                    "combined_similarity": round(sim.combined_similarity, 3),
                    "text_similarity": round(sim.text_similarity, 3),
                    "task_similarity": round(sim.task_similarity, 3),
                    "modality_similarity": round(sim.modality_similarity, 3),
                    "behavior_similarity": round(sim.behavior_similarity, 3),
                    "analysis_similarity": round(sim.analysis_similarity, 3),
                    "region_similarity": round(sim.region_similarity, 3),
                    "design_similarity": round(sim.design_similarity, 3),
                    "similarity_type": sim_type,
                    "source_design": fp_by_id[source_fp.dataset_id].design_type,
                    "target_design": fp_by_id[target_id].design_type,
                },
            )
            edges.append(edge)

    return edges


def build_concept_similarity_edges(
    concept_index: ConceptEmbeddingIndex,
    config: SemanticEdgeConfig | None = None,
) -> list[KnowledgeGraphEdge]:
    """Build similarity edges between concepts from embedding index.

    Args:
        concept_index: Concept embedding index
        config: Optional edge configuration

    Returns:
        List of concept similarity edges
    """
    config = config or DEFAULT_SEMANTIC_CONFIG
    edges: list[KnowledgeGraphEdge] = []

    # Get all concepts from index
    for concept_type in concept_index.concept_types:
        concepts = concept_index.by_type.get(concept_type, [])

        for source in concepts:
            # Find similar concepts of same type
            similar = concept_index.find_similar(
                source.concept_id,
                concept_type=concept_type,
                k=5,
                min_similarity=config.min_concept_similarity,
            )

            for target_sim in similar:
                # Create concept similarity edge
                source_node_id = make_node_id(concept_type, source.label)
                target_node_id = make_node_id(concept_type, target_sim.label)

                edge = KnowledgeGraphEdge(
                    edge_id=make_edge_id(
                        source_node_id,
                        f"{config.edge_type_prefix}_concept_similar",
                        target_node_id,
                    ),
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    edge_type=f"{config.edge_type_prefix}_concept_similar",
                    directed=False,
                    confidence=round(target_sim.similarity, 3),
                    properties={
                        "similarity": round(target_sim.similarity, 3),
                        "concept_type": concept_type,
                        "source_id": source.concept_id,
                        "target_id": target_sim.concept_id,
                    },
                )
                edges.append(edge)

    return edges


def add_semantic_edges_to_graph(
    graph: KnowledgeGraph,
    fingerprints: list[SemanticDatasetFingerprint] | None = None,
    concept_index: ConceptEmbeddingIndex | None = None,
    config: SemanticEdgeConfig | None = None,
) -> SemanticEdgeResult:
    """Add semantic similarity edges to an existing graph.

    Args:
        graph: Knowledge graph to modify
        fingerprints: Optional dataset fingerprints for dataset similarity
        concept_index: Optional concept index for concept similarity
        config: Optional edge configuration

    Returns:
        SemanticEdgeResult with counts and edge IDs
    """
    config = config or DEFAULT_SEMANTIC_CONFIG
    edge_ids: list[str] = []
    dataset_edges = 0
    concept_edges = 0

    # Add dataset similarity edges
    if fingerprints:
        new_dataset_edges = build_semantic_dataset_edges(fingerprints, config)

        for edge in new_dataset_edges:
            # Only add if both nodes exist in graph
            if (
                edge.source_node_id in graph.nodes
                and edge.target_node_id in graph.nodes
                and edge.edge_id not in graph.edges
            ):
                graph.edges[edge.edge_id] = edge
                edge_ids.append(edge.edge_id)
                dataset_edges += 1

    # Add concept similarity edges
    if concept_index:
        new_concept_edges = build_concept_similarity_edges(concept_index, config)

        for edge in new_concept_edges:
            # Only add if both nodes exist in graph
            if (
                edge.source_node_id in graph.nodes
                and edge.target_node_id in graph.nodes
                and edge.edge_id not in graph.edges
            ):
                graph.edges[edge.edge_id] = edge
                edge_ids.append(edge.edge_id)
                concept_edges += 1

    return SemanticEdgeResult(
        dataset_similarity_edges_added=dataset_edges,
        concept_similarity_edges_added=concept_edges,
        total_edges_added=len(edge_ids),
        edge_ids=edge_ids,
    )


def load_and_add_semantic_edges(
    graph: KnowledgeGraph,
    fingerprints_path: str | Path | None = None,
    concept_embeddings_path: str | Path | None = None,
    config: SemanticEdgeConfig | None = None,
) -> SemanticEdgeResult:
    """Load embeddings and add semantic edges to graph.

    Args:
        graph: Knowledge graph to modify
        fingerprints_path: Path to semantic fingerprints JSONL
        concept_embeddings_path: Path to concept embeddings JSONL
        config: Optional edge configuration

    Returns:
        SemanticEdgeResult
    """
    fingerprints = None
    concept_index = None

    if fingerprints_path:
        from neural_search.embeddings.semantic_fingerprint import (
            read_semantic_fingerprints,
        )

        fp_path = Path(fingerprints_path)
        if fp_path.exists():
            fingerprints = read_semantic_fingerprints(fp_path)

    if concept_embeddings_path:
        from neural_search.embeddings.concept_embeddings import load_concept_index

        ce_path = Path(concept_embeddings_path)
        if ce_path.exists():
            concept_index = load_concept_index(ce_path)

    return add_semantic_edges_to_graph(
        graph,
        fingerprints=fingerprints,
        concept_index=concept_index,
        config=config,
    )


def get_semantic_neighbors(
    graph: KnowledgeGraph,
    node_id: str,
    edge_type_prefix: str = "semantic",
    min_similarity: float = 0.5,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Get semantically similar neighbors for a node.

    Args:
        graph: Knowledge graph
        node_id: Source node ID
        edge_type_prefix: Prefix for semantic edge types
        min_similarity: Minimum similarity threshold

    Returns:
        List of (neighbor_node_id, similarity, properties) tuples
    """
    results: list[tuple[str, float, dict[str, Any]]] = []

    for edge in graph.edges.values():
        if not edge.edge_type.startswith(edge_type_prefix):
            continue

        neighbor_id = None
        if edge.source_node_id == node_id:
            neighbor_id = edge.target_node_id
        elif edge.target_node_id == node_id and not edge.directed:
            neighbor_id = edge.source_node_id

        if neighbor_id and edge.confidence >= min_similarity:
            results.append((neighbor_id, edge.confidence, edge.properties))

    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results
