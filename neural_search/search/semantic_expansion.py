"""Semantic query expansion using concept embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_search.embeddings.concept_embeddings import (
        ConceptEmbeddingIndex,
        ConceptSimilarity,
    )


@dataclass
class SemanticExpansion:
    """Result of semantic query expansion."""

    original_tasks: list[str]
    original_modalities: list[str]
    original_behaviors: list[str]
    original_affordances: list[str]

    expanded_tasks: list[tuple[str, float]] = field(default_factory=list)
    expanded_modalities: list[tuple[str, float]] = field(default_factory=list)
    expanded_behaviors: list[tuple[str, float]] = field(default_factory=list)
    expanded_affordances: list[tuple[str, float]] = field(default_factory=list)

    @property
    def total_expansions(self) -> int:
        """Total number of expanded concepts."""
        return (
            len(self.expanded_tasks)
            + len(self.expanded_modalities)
            + len(self.expanded_behaviors)
            + len(self.expanded_affordances)
        )

    def all_task_ids(self) -> list[str]:
        """Get all task IDs including expansions."""
        return self.original_tasks + [t[0] for t in self.expanded_tasks]

    def all_modality_ids(self) -> list[str]:
        """Get all modality IDs including expansions."""
        return self.original_modalities + [m[0] for m in self.expanded_modalities]

    def all_behavior_ids(self) -> list[str]:
        """Get all behavior IDs including expansions."""
        return self.original_behaviors + [b[0] for b in self.expanded_behaviors]

    def all_affordance_ids(self) -> list[str]:
        """Get all affordance IDs including expansions."""
        return self.original_affordances + [a[0] for a in self.expanded_affordances]


def expand_query_with_concepts(
    parsed_query: dict[str, Any],
    concept_index: "ConceptEmbeddingIndex",
    min_similarity: float = 0.65,
    max_expansions_per_concept: int = 3,
) -> SemanticExpansion:
    """Expand query concepts with semantically related concepts.

    Args:
        parsed_query: Parsed query with extracted concepts
        concept_index: Concept embedding index
        min_similarity: Minimum similarity for expansion
        max_expansions_per_concept: Maximum expansions per source concept

    Returns:
        SemanticExpansion with original and expanded concepts
    """
    original_tasks = parsed_query.get("tasks", [])
    original_modalities = parsed_query.get("modalities", [])
    original_behaviors = parsed_query.get("behaviors", [])
    original_affordances = parsed_query.get("affordances", [])

    expanded_tasks = []
    expanded_modalities = []
    expanded_behaviors = []
    expanded_affordances = []

    # Expand tasks
    for task in original_tasks:
        similar = _find_related_concepts(
            task, "task", concept_index, min_similarity, max_expansions_per_concept
        )
        for concept_id, similarity in similar:
            if concept_id not in original_tasks and concept_id not in [t[0] for t in expanded_tasks]:
                expanded_tasks.append((concept_id, similarity))

    # Expand modalities
    for modality in original_modalities:
        similar = _find_related_concepts(
            modality, "modality", concept_index, min_similarity, max_expansions_per_concept
        )
        for concept_id, similarity in similar:
            if concept_id not in original_modalities and concept_id not in [m[0] for m in expanded_modalities]:
                expanded_modalities.append((concept_id, similarity))

    # Expand behaviors
    for behavior in original_behaviors:
        similar = _find_related_concepts(
            behavior, "behavior", concept_index, min_similarity, max_expansions_per_concept
        )
        for concept_id, similarity in similar:
            if concept_id not in original_behaviors and concept_id not in [b[0] for b in expanded_behaviors]:
                expanded_behaviors.append((concept_id, similarity))

    # Expand affordances
    for affordance in original_affordances:
        similar = _find_related_concepts(
            affordance, "analysis", concept_index, min_similarity, max_expansions_per_concept
        )
        for concept_id, similarity in similar:
            if concept_id not in original_affordances and concept_id not in [a[0] for a in expanded_affordances]:
                expanded_affordances.append((concept_id, similarity))

    return SemanticExpansion(
        original_tasks=original_tasks,
        original_modalities=original_modalities,
        original_behaviors=original_behaviors,
        original_affordances=original_affordances,
        expanded_tasks=expanded_tasks,
        expanded_modalities=expanded_modalities,
        expanded_behaviors=expanded_behaviors,
        expanded_affordances=expanded_affordances,
    )


def _find_related_concepts(
    concept_label: str,
    concept_type: str,
    index: "ConceptEmbeddingIndex",
    min_similarity: float,
    max_results: int,
) -> list[tuple[str, float]]:
    """Find concepts related to a given concept.

    Args:
        concept_label: Concept label or ID to find related concepts for
        concept_type: Type of concept (task, modality, behavior, analysis)
        index: Concept embedding index
        min_similarity: Minimum similarity threshold
        max_results: Maximum results to return

    Returns:
        List of (concept_id, similarity) tuples
    """
    # Try to find concept by ID first
    concept_id = f"{concept_type}:{concept_label}"
    source = index.get(concept_id)

    # Try by label if not found by ID
    if source is None:
        source = index.get_by_label(concept_label)

    if source is None:
        return []

    # Find similar concepts
    similar = index.find_similar(
        source.concept_id,
        concept_type=concept_type,
        k=max_results,
        min_similarity=min_similarity,
    )

    return [(s.concept_id.split(":")[-1] if ":" in s.concept_id else s.concept_id, s.similarity) for s in similar]


def merge_expansion_into_query(
    parsed_query: dict[str, Any],
    expansion: SemanticExpansion,
    include_expansion_metadata: bool = True,
) -> dict[str, Any]:
    """Merge semantic expansion into parsed query.

    Args:
        parsed_query: Original parsed query
        expansion: Semantic expansion result
        include_expansion_metadata: Whether to include expansion metadata

    Returns:
        Updated parsed query with expanded concepts
    """
    result = dict(parsed_query)

    # Add expanded concepts with lower weight than originals
    result["expanded_tasks"] = [t[0] for t in expansion.expanded_tasks]
    result["expanded_modalities"] = [m[0] for m in expansion.expanded_modalities]
    result["expanded_behaviors"] = [b[0] for b in expansion.expanded_behaviors]
    result["expanded_affordances"] = [a[0] for a in expansion.expanded_affordances]

    # Optionally include metadata
    if include_expansion_metadata:
        result["semantic_expansion"] = {
            "tasks": [
                {"id": t[0], "similarity": round(t[1], 3)} for t in expansion.expanded_tasks
            ],
            "modalities": [
                {"id": m[0], "similarity": round(m[1], 3)} for m in expansion.expanded_modalities
            ],
            "behaviors": [
                {"id": b[0], "similarity": round(b[1], 3)} for b in expansion.expanded_behaviors
            ],
            "affordances": [
                {"id": a[0], "similarity": round(a[1], 3)} for a in expansion.expanded_affordances
            ],
            "total_expansions": expansion.total_expansions,
        }

    return result


def enrich_query_with_semantic_context(
    query: str,
    parsed_query: dict[str, Any],
    concept_index: "ConceptEmbeddingIndex | None" = None,
    min_similarity: float = 0.65,
    max_expansions: int = 3,
) -> tuple[dict[str, Any], list[str]]:
    """Enrich query with semantic concept expansion.

    Args:
        query: Original query string
        parsed_query: Parsed query with extracted concepts
        concept_index: Optional concept embedding index
        min_similarity: Minimum similarity for expansion
        max_expansions: Maximum expansions per concept

    Returns:
        Tuple of (enriched_parsed_query, context_tokens)
    """
    context_tokens = []

    if concept_index is None:
        return parsed_query, context_tokens

    # Perform semantic expansion
    expansion = expand_query_with_concepts(
        parsed_query,
        concept_index,
        min_similarity=min_similarity,
        max_expansions_per_concept=max_expansions,
    )

    # Merge into query
    enriched = merge_expansion_into_query(parsed_query, expansion)

    # Build context tokens from expansions
    for task, sim in expansion.expanded_tasks[:2]:
        context_tokens.append(f"related_task:{task}({sim:.2f})")

    for mod, sim in expansion.expanded_modalities[:2]:
        context_tokens.append(f"related_modality:{mod}({sim:.2f})")

    return enriched, context_tokens


def compute_expansion_boost(
    dataset_labels: dict[str, list[str]],
    expansion: SemanticExpansion,
    boost_factor: float = 0.5,
) -> float:
    """Compute score boost from semantic expansion matches.

    Args:
        dataset_labels: Dict mapping label types to dataset labels
        expansion: Semantic expansion result
        boost_factor: Maximum boost factor (0-1)

    Returns:
        Score boost from expansion matches (0 to boost_factor)
    """
    matches = 0
    total_expansions = expansion.total_expansions

    if total_expansions == 0:
        return 0.0

    dataset_tasks = set(t.lower() for t in dataset_labels.get("tasks", []))
    dataset_modalities = set(m.lower() for m in dataset_labels.get("modalities", []))
    dataset_behaviors = set(b.lower() for b in dataset_labels.get("behaviors", []))
    dataset_affordances = set(a.lower() for a in dataset_labels.get("affordances", []))

    # Check expansion matches
    for task, _ in expansion.expanded_tasks:
        if task.lower() in dataset_tasks:
            matches += 1

    for mod, _ in expansion.expanded_modalities:
        if mod.lower() in dataset_modalities:
            matches += 1

    for behavior, _ in expansion.expanded_behaviors:
        if behavior.lower() in dataset_behaviors:
            matches += 1

    for affordance, _ in expansion.expanded_affordances:
        if affordance.lower() in dataset_affordances:
            matches += 1

    # Compute proportional boost
    match_ratio = matches / total_expansions if total_expansions > 0 else 0.0
    return boost_factor * match_ratio
