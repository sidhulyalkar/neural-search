"""Semantic similarity functions for concept matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neural_search.embeddings.concept_embeddings import (
    ConceptEmbeddingIndex,
    ConceptSimilarity,
)
from neural_search.embeddings.index import cosine_similarity


@dataclass
class SemanticMatch:
    """A semantically matched concept."""

    query_term: str
    matched_concept_id: str
    matched_label: str
    concept_type: str
    similarity: float
    is_exact: bool = False


@dataclass
class QuerySemanticExpansion:
    """Semantic expansion of a query."""

    original_query: str
    expanded_tasks: list[SemanticMatch] = field(default_factory=list)
    expanded_modalities: list[SemanticMatch] = field(default_factory=list)
    expanded_behaviors: list[SemanticMatch] = field(default_factory=list)
    expanded_analyses: list[SemanticMatch] = field(default_factory=list)
    expanded_regions: list[SemanticMatch] = field(default_factory=list)

    def all_expansions(self) -> list[SemanticMatch]:
        """Get all expansions as a flat list."""
        return (
            self.expanded_tasks
            + self.expanded_modalities
            + self.expanded_behaviors
            + self.expanded_analyses
            + self.expanded_regions
        )

    def expansion_count(self) -> int:
        """Count total expansions."""
        return len(self.all_expansions())


def concept_similarity(
    concept_a: str,
    concept_b: str,
    index: ConceptEmbeddingIndex,
) -> float:
    """Compute semantic similarity between two concepts.

    Args:
        concept_a: First concept ID or label
        concept_b: Second concept ID or label
        index: Concept embedding index

    Returns:
        Cosine similarity (0-1)
    """
    # Try to find by ID first, then by label
    emb_a = index.get(concept_a) or index.get_by_label(concept_a)
    emb_b = index.get(concept_b) or index.get_by_label(concept_b)

    if emb_a is None or emb_b is None:
        return 0.0

    return cosine_similarity(emb_a.embedding, emb_b.embedding)


def find_semantically_similar(
    concept_id: str,
    index: ConceptEmbeddingIndex,
    concept_type: str | None = None,
    min_similarity: float = 0.6,
    max_results: int = 10,
) -> list[ConceptSimilarity]:
    """Find concepts semantically similar to the given one.

    Args:
        concept_id: Source concept ID or label
        index: Concept embedding index
        concept_type: Optional type filter
        min_similarity: Minimum similarity threshold
        max_results: Maximum results to return

    Returns:
        List of similar concepts
    """
    # Try to find by ID first, then by label
    source = index.get(concept_id) or index.get_by_label(concept_id)
    if source is None:
        return []

    return index.find_similar(
        source.concept_id,
        concept_type=concept_type,
        k=max_results,
        min_similarity=min_similarity,
    )


def expand_query_semantically(
    parsed_query: dict[str, Any],
    index: ConceptEmbeddingIndex,
    min_similarity: float = 0.65,
    max_expansions_per_type: int = 3,
) -> QuerySemanticExpansion:
    """Expand query with semantically similar concepts.

    Args:
        parsed_query: Parsed query with tasks, modalities, etc.
        index: Concept embedding index
        min_similarity: Minimum similarity for expansion
        max_expansions_per_type: Max expansions per concept type

    Returns:
        QuerySemanticExpansion with all expanded concepts
    """
    expansion = QuerySemanticExpansion(
        original_query=parsed_query.get("query", "")
    )

    # Expand tasks
    for task_id in parsed_query.get("tasks", []):
        # First add exact match
        exact = index.get(f"task:{task_id}") or index.get_by_label(task_id)
        if exact is not None:
            expansion.expanded_tasks.append(
                SemanticMatch(
                    query_term=task_id,
                    matched_concept_id=exact.concept_id,
                    matched_label=exact.label,
                    concept_type="task",
                    similarity=1.0,
                    is_exact=True,
                )
            )

            # Find similar tasks
            similar = index.find_similar(
                exact.concept_id,
                concept_type="task",
                k=max_expansions_per_type,
                min_similarity=min_similarity,
            )
            for sim in similar:
                expansion.expanded_tasks.append(
                    SemanticMatch(
                        query_term=task_id,
                        matched_concept_id=sim.concept_id,
                        matched_label=sim.label,
                        concept_type="task",
                        similarity=sim.similarity,
                        is_exact=False,
                    )
                )

    # Expand modalities
    for mod_id in parsed_query.get("modalities", []):
        exact = index.get(f"modality:{mod_id}") or index.get_by_label(mod_id)
        if exact is not None:
            expansion.expanded_modalities.append(
                SemanticMatch(
                    query_term=mod_id,
                    matched_concept_id=exact.concept_id,
                    matched_label=exact.label,
                    concept_type="modality",
                    similarity=1.0,
                    is_exact=True,
                )
            )

            similar = index.find_similar(
                exact.concept_id,
                concept_type="modality",
                k=max_expansions_per_type,
                min_similarity=min_similarity,
            )
            for sim in similar:
                expansion.expanded_modalities.append(
                    SemanticMatch(
                        query_term=mod_id,
                        matched_concept_id=sim.concept_id,
                        matched_label=sim.label,
                        concept_type="modality",
                        similarity=sim.similarity,
                        is_exact=False,
                    )
                )

    # Expand behaviors
    for behavior_id in parsed_query.get("behaviors", []):
        exact = index.get(f"behavior:{behavior_id}") or index.get_by_label(behavior_id)
        if exact is not None:
            expansion.expanded_behaviors.append(
                SemanticMatch(
                    query_term=behavior_id,
                    matched_concept_id=exact.concept_id,
                    matched_label=exact.label,
                    concept_type="behavior",
                    similarity=1.0,
                    is_exact=True,
                )
            )

            similar = index.find_similar(
                exact.concept_id,
                concept_type="behavior",
                k=max_expansions_per_type,
                min_similarity=min_similarity,
            )
            for sim in similar:
                expansion.expanded_behaviors.append(
                    SemanticMatch(
                        query_term=behavior_id,
                        matched_concept_id=sim.concept_id,
                        matched_label=sim.label,
                        concept_type="behavior",
                        similarity=sim.similarity,
                        is_exact=False,
                    )
                )

    # Expand affordances/analyses
    for affordance_id in parsed_query.get("affordances", []):
        exact = index.get(f"analysis:{affordance_id}") or index.get_by_label(affordance_id)
        if exact is not None:
            expansion.expanded_analyses.append(
                SemanticMatch(
                    query_term=affordance_id,
                    matched_concept_id=exact.concept_id,
                    matched_label=exact.label,
                    concept_type="analysis",
                    similarity=1.0,
                    is_exact=True,
                )
            )

            similar = index.find_similar(
                exact.concept_id,
                concept_type="analysis",
                k=max_expansions_per_type,
                min_similarity=min_similarity,
            )
            for sim in similar:
                expansion.expanded_analyses.append(
                    SemanticMatch(
                        query_term=affordance_id,
                        matched_concept_id=sim.concept_id,
                        matched_label=sim.label,
                        concept_type="analysis",
                        similarity=sim.similarity,
                        is_exact=False,
                    )
                )

    # Expand brain regions
    for region_id in parsed_query.get("brain_regions", []):
        exact = index.get(f"region:{region_id}") or index.get_by_label(region_id)
        if exact is not None:
            expansion.expanded_regions.append(
                SemanticMatch(
                    query_term=region_id,
                    matched_concept_id=exact.concept_id,
                    matched_label=exact.label,
                    concept_type="region",
                    similarity=1.0,
                    is_exact=True,
                )
            )

            similar = index.find_similar(
                exact.concept_id,
                concept_type="region",
                k=max_expansions_per_type,
                min_similarity=min_similarity,
            )
            for sim in similar:
                expansion.expanded_regions.append(
                    SemanticMatch(
                        query_term=region_id,
                        matched_concept_id=sim.concept_id,
                        matched_label=sim.label,
                        concept_type="region",
                        similarity=sim.similarity,
                        is_exact=False,
                    )
                )

    return expansion


def compute_query_concept_similarity(
    query_embedding: np.ndarray,
    index: ConceptEmbeddingIndex,
    concept_type: str,
    top_k: int = 5,
    min_similarity: float = 0.5,
) -> list[ConceptSimilarity]:
    """Find concepts similar to a query embedding.

    Args:
        query_embedding: Query embedding vector
        index: Concept embedding index
        concept_type: Type of concepts to search
        top_k: Number of results
        min_similarity: Minimum similarity threshold

    Returns:
        List of similar concepts
    """
    return index.find_similar_to_vector(
        query_vec=query_embedding,
        concept_type=concept_type,
        k=top_k,
        min_similarity=min_similarity,
    )


def merge_query_with_expansion(
    parsed_query: dict[str, Any],
    expansion: QuerySemanticExpansion,
) -> dict[str, Any]:
    """Merge original query with semantic expansion.

    Args:
        parsed_query: Original parsed query
        expansion: Semantic expansion

    Returns:
        Merged query with expansion info
    """
    merged = dict(parsed_query)

    # Add expanded concepts
    merged["semantic_expansion"] = {
        "tasks": [
            {
                "query_term": m.query_term,
                "matched_id": m.matched_concept_id,
                "matched_label": m.matched_label,
                "similarity": round(m.similarity, 3),
                "is_exact": m.is_exact,
            }
            for m in expansion.expanded_tasks
        ],
        "modalities": [
            {
                "query_term": m.query_term,
                "matched_id": m.matched_concept_id,
                "matched_label": m.matched_label,
                "similarity": round(m.similarity, 3),
                "is_exact": m.is_exact,
            }
            for m in expansion.expanded_modalities
        ],
        "behaviors": [
            {
                "query_term": m.query_term,
                "matched_id": m.matched_concept_id,
                "matched_label": m.matched_label,
                "similarity": round(m.similarity, 3),
                "is_exact": m.is_exact,
            }
            for m in expansion.expanded_behaviors
        ],
        "analyses": [
            {
                "query_term": m.query_term,
                "matched_id": m.matched_concept_id,
                "matched_label": m.matched_label,
                "similarity": round(m.similarity, 3),
                "is_exact": m.is_exact,
            }
            for m in expansion.expanded_analyses
        ],
        "regions": [
            {
                "query_term": m.query_term,
                "matched_id": m.matched_concept_id,
                "matched_label": m.matched_label,
                "similarity": round(m.similarity, 3),
                "is_exact": m.is_exact,
            }
            for m in expansion.expanded_regions
        ],
        "total_expansions": expansion.expansion_count(),
    }

    # Add expanded IDs for scoring
    merged["expanded_task_ids"] = [
        m.matched_concept_id for m in expansion.expanded_tasks if not m.is_exact
    ]
    merged["expanded_modality_ids"] = [
        m.matched_concept_id for m in expansion.expanded_modalities if not m.is_exact
    ]
    merged["expanded_behavior_ids"] = [
        m.matched_concept_id for m in expansion.expanded_behaviors if not m.is_exact
    ]
    merged["expanded_analysis_ids"] = [
        m.matched_concept_id for m in expansion.expanded_analyses if not m.is_exact
    ]
    merged["expanded_region_ids"] = [
        m.matched_concept_id for m in expansion.expanded_regions if not m.is_exact
    ]

    return merged
