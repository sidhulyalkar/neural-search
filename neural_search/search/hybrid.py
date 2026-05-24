"""Hybrid retrieval combining sparse (ontology) and dense (embedding) signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FusionMethod(Enum):
    """Methods for fusing sparse and dense retrieval scores."""

    WEIGHTED_SUM = "weighted_sum"
    RECIPROCAL_RANK = "reciprocal_rank"
    MAX_SCORE = "max_score"


@dataclass
class HybridRetrievalConfig:
    """Configuration for hybrid sparse-dense retrieval."""

    sparse_weight: float = 0.6  # Weight for ontology/graph/metadata scores
    dense_weight: float = 0.4   # Weight for embedding-based scores
    fusion_method: FusionMethod = FusionMethod.WEIGHTED_SUM
    normalize_scores: bool = True
    min_dense_contribution: float = 0.0  # Minimum dense score to include

    def __post_init__(self):
        total = self.sparse_weight + self.dense_weight
        if abs(total - 1.0) > 0.001:
            # Normalize weights
            self.sparse_weight /= total
            self.dense_weight /= total


@dataclass
class HybridScore:
    """Combined score with breakdown by retrieval type."""

    final_score: float
    sparse_score: float
    dense_score: float
    fusion_method: str
    sparse_components: dict[str, float] = field(default_factory=dict)
    dense_components: dict[str, float] = field(default_factory=dict)


def compute_sparse_score(score_breakdown: dict[str, float]) -> float:
    """Extract sparse (non-embedding) score components.

    Sparse signals include:
    - ontology_score
    - behavior score
    - modality score
    - graph_score
    - metadata score
    - provenance_score
    """
    sparse_keys = {
        "ontology_score",
        "ontology",
        "behavior",
        "modality",
        "graph_score",
        "metadata",
        "provenance_score",
        "paper_confidence",
        "affordance",
        "affordance_score",
        "usability_score",
        "readiness",
    }

    sparse_total = 0.0
    for key in sparse_keys:
        if key in score_breakdown:
            sparse_total += float(score_breakdown[key])

    return min(sparse_total, 1.0)


def compute_dense_score(score_breakdown: dict[str, float]) -> float:
    """Extract dense (embedding-based) score components.

    Dense signals include:
    - semantic_score (embedding similarity)
    - field_semantic_score
    - embedding_semantic
    """
    dense_keys = {
        "semantic_score",
        "semantic",
        "field_semantic_score",
        "embedding_semantic",
    }

    dense_total = 0.0
    for key in dense_keys:
        if key in score_breakdown:
            dense_total += float(score_breakdown[key])

    return min(dense_total, 1.0)


def hybrid_fusion(
    sparse_score: float,
    dense_score: float,
    config: HybridRetrievalConfig,
) -> float:
    """Fuse sparse and dense scores using configured method.

    Args:
        sparse_score: Score from ontology/graph/metadata
        dense_score: Score from embeddings
        config: Hybrid retrieval configuration

    Returns:
        Combined hybrid score
    """
    if config.fusion_method == FusionMethod.WEIGHTED_SUM:
        return (
            config.sparse_weight * sparse_score
            + config.dense_weight * dense_score
        )

    elif config.fusion_method == FusionMethod.MAX_SCORE:
        return max(sparse_score, dense_score)

    elif config.fusion_method == FusionMethod.RECIPROCAL_RANK:
        # RRF-style fusion (assumes scores represent ranks)
        # Higher scores = better, so invert to get pseudo-ranks
        k = 60  # Standard RRF constant
        sparse_rank = 1.0 / (sparse_score + 0.01) if sparse_score > 0 else 1000
        dense_rank = 1.0 / (dense_score + 0.01) if dense_score > 0 else 1000
        return 1.0 / (k + sparse_rank) + 1.0 / (k + dense_rank)

    return sparse_score  # Fallback


def compute_hybrid_score(
    score_breakdown: dict[str, float],
    config: HybridRetrievalConfig | None = None,
) -> HybridScore:
    """Compute hybrid score from existing score breakdown.

    Args:
        score_breakdown: Dict of individual score components
        config: Optional hybrid config (uses defaults if None)

    Returns:
        HybridScore with breakdown
    """
    config = config or HybridRetrievalConfig()

    sparse_score = compute_sparse_score(score_breakdown)
    dense_score = compute_dense_score(score_breakdown)

    # Apply minimum dense contribution threshold
    if dense_score < config.min_dense_contribution:
        dense_score = 0.0

    final_score = hybrid_fusion(sparse_score, dense_score, config)

    return HybridScore(
        final_score=round(final_score, 4),
        sparse_score=round(sparse_score, 4),
        dense_score=round(dense_score, 4),
        fusion_method=config.fusion_method.value,
        sparse_components={
            k: v for k, v in score_breakdown.items()
            if k in {"ontology_score", "behavior", "modality", "graph_score", "metadata"}
        },
        dense_components={
            k: v for k, v in score_breakdown.items()
            if k in {"semantic_score", "field_semantic_score", "embedding_semantic"}
        },
    )


def rerank_with_hybrid_scores(
    results: list[Any],
    config: HybridRetrievalConfig | None = None,
) -> list[Any]:
    """Rerank search results using hybrid scoring.

    Args:
        results: List of SearchResult objects with score_breakdown
        config: Hybrid retrieval configuration

    Returns:
        Results sorted by hybrid score (descending)
    """
    config = config or HybridRetrievalConfig()

    scored_results = []
    for result in results:
        breakdown = getattr(result, "score_breakdown", {})
        hybrid = compute_hybrid_score(breakdown, config)

        # Update result with hybrid score
        breakdown["hybrid_score"] = hybrid.final_score
        breakdown["sparse_score"] = hybrid.sparse_score
        breakdown["dense_score"] = hybrid.dense_score

        scored_results.append((hybrid.final_score, result))

    # Sort by hybrid score descending
    scored_results.sort(key=lambda x: x[0], reverse=True)

    return [r for _, r in scored_results]


# Intent-aware weight profiles for hybrid retrieval
INTENT_HYBRID_PROFILES: dict[str, HybridRetrievalConfig] = {
    "dataset_lookup": HybridRetrievalConfig(
        sparse_weight=0.8,  # Metadata matching is key
        dense_weight=0.2,
    ),
    "task_search": HybridRetrievalConfig(
        sparse_weight=0.55,  # Balance ontology and semantics
        dense_weight=0.45,
    ),
    "analysis_search": HybridRetrievalConfig(
        sparse_weight=0.5,
        dense_weight=0.5,  # Semantic similarity helps here
    ),
    "paper_link": HybridRetrievalConfig(
        sparse_weight=0.7,  # Graph links are important
        dense_weight=0.3,
    ),
    "graph_reasoning": HybridRetrievalConfig(
        sparse_weight=0.75,  # Graph features dominate
        dense_weight=0.25,
    ),
}


def get_hybrid_config_for_intent(intent: str) -> HybridRetrievalConfig:
    """Get hybrid config optimized for a query intent.

    Args:
        intent: Query intent type (from intent classification)

    Returns:
        HybridRetrievalConfig tuned for the intent
    """
    return INTENT_HYBRID_PROFILES.get(
        intent,
        HybridRetrievalConfig(),  # Default balanced config
    )
