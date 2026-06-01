"""Reciprocal Rank Fusion (RRF) utility for combining multiple ranked lists.

This module implements RRF and related fusion methods for combining candidate
lists from different retrieval signals (sparse, semantic, ontology, graph).

Mathematical foundation:
    RRF(d) = sum_{r in rankers} 1 / (k + rank_r(d))

    Where:
    - k is a constant (typically 60) that mitigates the impact of high rankings
    - rank_r(d) is the rank of document d in ranker r's list (1-indexed)

RRF Properties:
    1. Rank-based: Robust to score scale differences between rankers
    2. Diminishing returns: High ranks contribute more than low ranks
    3. Consensus rewarding: Documents ranked highly by multiple sources score higher

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
individual Rank Learning Methods" (SIGIR 2009)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class FusionMethod(StrEnum):
    """Available fusion methods."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED_SUM = "weighted_sum"  # Simple weighted combination
    BORDA = "borda"  # Borda count
    COMBMNZ = "combmnz"  # CombMNZ (multiply by number of lists containing doc)


@dataclass
class RankedItem:
    """An item in a ranked list with optional metadata."""

    id: str
    score: float
    rank: int
    source: str  # Which ranker produced this ranking
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedCandidate:
    """A candidate after fusion with full provenance."""

    id: str
    fused_score: float
    source_ranks: dict[str, int]  # source -> rank in that source
    source_scores: dict[str, float]  # source -> original score
    source_contributions: dict[str, float]  # source -> contribution to fused score
    num_sources: int  # Number of sources that included this candidate
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Complete result of a fusion operation."""

    candidates: list[FusedCandidate]
    method: FusionMethod
    parameters: dict[str, Any]
    source_weights: dict[str, float]
    num_sources: int


class RankedListProtocol(Protocol):
    """Protocol for objects that can be treated as ranked lists."""

    @property
    def id(self) -> str:
        ...

    @property
    def score(self) -> float:
        ...


def _extract_ranked_items(
    ranked_list: Sequence[Any],
    source_name: str,
    id_field: str = "id",
    score_field: str = "score",
) -> list[RankedItem]:
    """Extract RankedItem objects from various input formats."""
    items: list[RankedItem] = []

    for rank, item in enumerate(ranked_list, start=1):
        if isinstance(item, RankedItem):
            items.append(RankedItem(
                id=item.id,
                score=item.score,
                rank=rank,
                source=source_name,
                metadata=item.metadata,
            ))
        elif isinstance(item, Mapping):
            item_id = str(item.get(id_field, item.get("dataset_id", "")))
            score = float(item.get(score_field, 0.0))
            items.append(RankedItem(
                id=item_id,
                score=score,
                rank=rank,
                source=source_name,
                metadata={k: v for k, v in item.items() if k not in (id_field, score_field)},
            ))
        elif hasattr(item, id_field) and hasattr(item, score_field):
            items.append(RankedItem(
                id=str(getattr(item, id_field)),
                score=float(getattr(item, score_field)),
                rank=rank,
                source=source_name,
                metadata={},
            ))
        else:
            # Assume it's just an ID
            items.append(RankedItem(
                id=str(item),
                score=1.0 / rank,  # Use reciprocal rank as default score
                rank=rank,
                source=source_name,
                metadata={},
            ))

    return items


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[Any]],
    k: int = 60,
    source_weights: Mapping[str, float] | None = None,
    id_field: str = "id",
    score_field: str = "score",
    top_k: int | None = None,
) -> FusionResult:
    """Perform Reciprocal Rank Fusion on multiple ranked lists.

    Args:
        ranked_lists: Mapping of source name to ranked list of items.
        k: RRF constant (default 60). Higher values reduce the impact of
           very high rankings.
        source_weights: Optional weights for each source (default: equal weight).
        id_field: Field name for item ID extraction.
        score_field: Field name for score extraction.
        top_k: Maximum number of candidates to return (None = all).

    Returns:
        FusionResult with fused candidates sorted by score.

    Example:
        >>> sparse_results = [{"id": "d1", "score": 0.9}, {"id": "d2", "score": 0.7}]
        >>> semantic_results = [{"id": "d2", "score": 0.85}, {"id": "d3", "score": 0.6}]
        >>> result = reciprocal_rank_fusion({
        ...     "sparse": sparse_results,
        ...     "semantic": semantic_results,
        ... })
    """
    if not ranked_lists:
        return FusionResult(
            candidates=[],
            method=FusionMethod.RRF,
            parameters={"k": k},
            source_weights={},
            num_sources=0,
        )

    # Normalize source weights
    sources = list(ranked_lists.keys())
    if source_weights is None:
        weights = dict.fromkeys(sources, 1.0)
    else:
        weights = {s: source_weights.get(s, 1.0) for s in sources}

    # Extract ranked items from all sources
    all_items: dict[str, dict[str, RankedItem]] = {}  # id -> {source -> item}

    for source_name, ranked_list in ranked_lists.items():
        items = _extract_ranked_items(ranked_list, source_name, id_field, score_field)
        for item in items:
            if item.id not in all_items:
                all_items[item.id] = {}
            all_items[item.id][source_name] = item

    # Compute RRF scores
    candidates: list[FusedCandidate] = []

    for item_id, source_items in all_items.items():
        fused_score = 0.0
        source_ranks: dict[str, int] = {}
        source_scores: dict[str, float] = {}
        source_contributions: dict[str, float] = {}
        merged_metadata: dict[str, Any] = {}

        for source_name, item in source_items.items():
            contribution = weights[source_name] / (k + item.rank)
            fused_score += contribution
            source_ranks[source_name] = item.rank
            source_scores[source_name] = item.score
            source_contributions[source_name] = contribution
            merged_metadata.update(item.metadata)

        candidates.append(FusedCandidate(
            id=item_id,
            fused_score=fused_score,
            source_ranks=source_ranks,
            source_scores=source_scores,
            source_contributions=source_contributions,
            num_sources=len(source_items),
            metadata=merged_metadata,
        ))

    # Sort by fused score descending, then by ID for stability
    candidates.sort(key=lambda c: (-c.fused_score, c.id))

    if top_k is not None:
        candidates = candidates[:top_k]

    return FusionResult(
        candidates=candidates,
        method=FusionMethod.RRF,
        parameters={"k": k},
        source_weights=weights,
        num_sources=len(sources),
    )


def weighted_sum_fusion(
    ranked_lists: Mapping[str, Sequence[Any]],
    source_weights: Mapping[str, float] | None = None,
    normalize_scores: bool = True,
    id_field: str = "id",
    score_field: str = "score",
    top_k: int | None = None,
) -> FusionResult:
    """Perform weighted sum fusion on multiple ranked lists.

    This method directly combines normalized scores from each source,
    weighted by configurable source weights.

    Args:
        ranked_lists: Mapping of source name to ranked list of items.
        source_weights: Weights for each source (default: equal weight).
        normalize_scores: Whether to min-max normalize scores per source.
        id_field: Field name for item ID extraction.
        score_field: Field name for score extraction.
        top_k: Maximum number of candidates to return.

    Returns:
        FusionResult with fused candidates sorted by score.
    """
    if not ranked_lists:
        return FusionResult(
            candidates=[],
            method=FusionMethod.WEIGHTED_SUM,
            parameters={"normalize_scores": normalize_scores},
            source_weights={},
            num_sources=0,
        )

    sources = list(ranked_lists.keys())
    if source_weights is None:
        weights = dict.fromkeys(sources, 1.0)
    else:
        weights = {s: source_weights.get(s, 1.0) for s in sources}

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {s: w / total_weight for s, w in weights.items()}

    # Extract and optionally normalize scores
    all_items: dict[str, dict[str, RankedItem]] = {}

    for source_name, ranked_list in ranked_lists.items():
        items = _extract_ranked_items(ranked_list, source_name, id_field, score_field)

        if normalize_scores and items:
            scores = [item.score for item in items]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            if score_range > 0:
                for item in items:
                    item.score = (item.score - min_score) / score_range
            else:
                for item in items:
                    item.score = 1.0

        for item in items:
            if item.id not in all_items:
                all_items[item.id] = {}
            all_items[item.id][source_name] = item

    # Compute weighted sum scores
    candidates: list[FusedCandidate] = []

    for item_id, source_items in all_items.items():
        fused_score = 0.0
        source_ranks: dict[str, int] = {}
        source_scores: dict[str, float] = {}
        source_contributions: dict[str, float] = {}
        merged_metadata: dict[str, Any] = {}

        for source_name, item in source_items.items():
            contribution = weights[source_name] * item.score
            fused_score += contribution
            source_ranks[source_name] = item.rank
            source_scores[source_name] = item.score
            source_contributions[source_name] = contribution
            merged_metadata.update(item.metadata)

        candidates.append(FusedCandidate(
            id=item_id,
            fused_score=fused_score,
            source_ranks=source_ranks,
            source_scores=source_scores,
            source_contributions=source_contributions,
            num_sources=len(source_items),
            metadata=merged_metadata,
        ))

    candidates.sort(key=lambda c: (-c.fused_score, c.id))

    if top_k is not None:
        candidates = candidates[:top_k]

    return FusionResult(
        candidates=candidates,
        method=FusionMethod.WEIGHTED_SUM,
        parameters={"normalize_scores": normalize_scores},
        source_weights=weights,
        num_sources=len(sources),
    )


def borda_count_fusion(
    ranked_lists: Mapping[str, Sequence[Any]],
    source_weights: Mapping[str, float] | None = None,
    id_field: str = "id",
    score_field: str = "score",
    top_k: int | None = None,
) -> FusionResult:
    """Perform Borda count fusion on multiple ranked lists.

    Each item receives points based on its rank: (n - rank + 1) where n is
    the list length. Points are summed across lists with optional weighting.

    Args:
        ranked_lists: Mapping of source name to ranked list of items.
        source_weights: Weights for each source.
        id_field: Field name for item ID extraction.
        score_field: Field name for score extraction.
        top_k: Maximum number of candidates to return.

    Returns:
        FusionResult with fused candidates sorted by score.
    """
    if not ranked_lists:
        return FusionResult(
            candidates=[],
            method=FusionMethod.BORDA,
            parameters={},
            source_weights={},
            num_sources=0,
        )

    sources = list(ranked_lists.keys())
    if source_weights is None:
        weights = dict.fromkeys(sources, 1.0)
    else:
        weights = {s: source_weights.get(s, 1.0) for s in sources}

    all_items: dict[str, dict[str, RankedItem]] = {}
    list_lengths: dict[str, int] = {}

    for source_name, ranked_list in ranked_lists.items():
        items = _extract_ranked_items(ranked_list, source_name, id_field, score_field)
        list_lengths[source_name] = len(items)

        for item in items:
            if item.id not in all_items:
                all_items[item.id] = {}
            all_items[item.id][source_name] = item

    # Compute Borda scores
    candidates: list[FusedCandidate] = []

    for item_id, source_items in all_items.items():
        fused_score = 0.0
        source_ranks: dict[str, int] = {}
        source_scores: dict[str, float] = {}
        source_contributions: dict[str, float] = {}
        merged_metadata: dict[str, Any] = {}

        for source_name, item in source_items.items():
            n = list_lengths[source_name]
            borda_points = n - item.rank + 1
            contribution = weights[source_name] * borda_points
            fused_score += contribution
            source_ranks[source_name] = item.rank
            source_scores[source_name] = item.score
            source_contributions[source_name] = contribution
            merged_metadata.update(item.metadata)

        candidates.append(FusedCandidate(
            id=item_id,
            fused_score=fused_score,
            source_ranks=source_ranks,
            source_scores=source_scores,
            source_contributions=source_contributions,
            num_sources=len(source_items),
            metadata=merged_metadata,
        ))

    candidates.sort(key=lambda c: (-c.fused_score, c.id))

    if top_k is not None:
        candidates = candidates[:top_k]

    return FusionResult(
        candidates=candidates,
        method=FusionMethod.BORDA,
        parameters={},
        source_weights=weights,
        num_sources=len(sources),
    )


def combmnz_fusion(
    ranked_lists: Mapping[str, Sequence[Any]],
    source_weights: Mapping[str, float] | None = None,
    normalize_scores: bool = True,
    id_field: str = "id",
    score_field: str = "score",
    top_k: int | None = None,
) -> FusionResult:
    """Perform CombMNZ fusion on multiple ranked lists.

    CombMNZ multiplies the sum of normalized scores by the number of lists
    containing the document. This rewards documents appearing in multiple lists.

    Args:
        ranked_lists: Mapping of source name to ranked list of items.
        source_weights: Weights for each source.
        normalize_scores: Whether to min-max normalize scores per source.
        id_field: Field name for item ID extraction.
        score_field: Field name for score extraction.
        top_k: Maximum number of candidates to return.

    Returns:
        FusionResult with fused candidates sorted by score.
    """
    if not ranked_lists:
        return FusionResult(
            candidates=[],
            method=FusionMethod.COMBMNZ,
            parameters={"normalize_scores": normalize_scores},
            source_weights={},
            num_sources=0,
        )

    sources = list(ranked_lists.keys())
    if source_weights is None:
        weights = dict.fromkeys(sources, 1.0)
    else:
        weights = {s: source_weights.get(s, 1.0) for s in sources}

    all_items: dict[str, dict[str, RankedItem]] = {}

    for source_name, ranked_list in ranked_lists.items():
        items = _extract_ranked_items(ranked_list, source_name, id_field, score_field)

        if normalize_scores and items:
            scores = [item.score for item in items]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            if score_range > 0:
                for item in items:
                    item.score = (item.score - min_score) / score_range
            else:
                for item in items:
                    item.score = 1.0

        for item in items:
            if item.id not in all_items:
                all_items[item.id] = {}
            all_items[item.id][source_name] = item

    # Compute CombMNZ scores
    candidates: list[FusedCandidate] = []

    for item_id, source_items in all_items.items():
        sum_score = 0.0
        source_ranks: dict[str, int] = {}
        source_scores: dict[str, float] = {}
        source_contributions: dict[str, float] = {}
        merged_metadata: dict[str, Any] = {}

        for source_name, item in source_items.items():
            contribution = weights[source_name] * item.score
            sum_score += contribution
            source_ranks[source_name] = item.rank
            source_scores[source_name] = item.score
            source_contributions[source_name] = contribution
            merged_metadata.update(item.metadata)

        # CombMNZ multiplies by number of sources
        num_sources = len(source_items)
        fused_score = sum_score * num_sources

        candidates.append(FusedCandidate(
            id=item_id,
            fused_score=fused_score,
            source_ranks=source_ranks,
            source_scores=source_scores,
            source_contributions=source_contributions,
            num_sources=num_sources,
            metadata=merged_metadata,
        ))

    candidates.sort(key=lambda c: (-c.fused_score, c.id))

    if top_k is not None:
        candidates = candidates[:top_k]

    return FusionResult(
        candidates=candidates,
        method=FusionMethod.COMBMNZ,
        parameters={"normalize_scores": normalize_scores},
        source_weights=weights,
        num_sources=len(sources),
    )


def fuse(
    ranked_lists: Mapping[str, Sequence[Any]],
    method: FusionMethod = FusionMethod.RRF,
    source_weights: Mapping[str, float] | None = None,
    id_field: str = "id",
    score_field: str = "score",
    top_k: int | None = None,
    **kwargs: Any,
) -> FusionResult:
    """Generic fusion function that dispatches to the appropriate method.

    Args:
        ranked_lists: Mapping of source name to ranked list of items.
        method: Fusion method to use.
        source_weights: Weights for each source.
        id_field: Field name for item ID extraction.
        score_field: Field name for score extraction.
        top_k: Maximum number of candidates to return.
        **kwargs: Additional method-specific parameters.

    Returns:
        FusionResult with fused candidates.

    Example:
        >>> result = fuse(
        ...     {"sparse": sparse_results, "semantic": semantic_results},
        ...     method=FusionMethod.RRF,
        ...     source_weights={"sparse": 1.0, "semantic": 0.8},
        ...     top_k=20,
        ... )
    """
    if method == FusionMethod.RRF:
        k = kwargs.get("k", 60)
        return reciprocal_rank_fusion(
            ranked_lists, k=k, source_weights=source_weights,
            id_field=id_field, score_field=score_field, top_k=top_k,
        )
    elif method == FusionMethod.WEIGHTED_SUM:
        normalize = kwargs.get("normalize_scores", True)
        return weighted_sum_fusion(
            ranked_lists, source_weights=source_weights, normalize_scores=normalize,
            id_field=id_field, score_field=score_field, top_k=top_k,
        )
    elif method == FusionMethod.BORDA:
        return borda_count_fusion(
            ranked_lists, source_weights=source_weights,
            id_field=id_field, score_field=score_field, top_k=top_k,
        )
    elif method == FusionMethod.COMBMNZ:
        normalize = kwargs.get("normalize_scores", True)
        return combmnz_fusion(
            ranked_lists, source_weights=source_weights, normalize_scores=normalize,
            id_field=id_field, score_field=score_field, top_k=top_k,
        )
    else:
        raise ValueError(f"Unknown fusion method: {method}")


def explain_fusion(
    result: FusionResult,
    candidate_id: str,
) -> dict[str, Any]:
    """Explain the fusion score for a specific candidate.

    Args:
        result: The fusion result to examine.
        candidate_id: ID of the candidate to explain.

    Returns:
        Dictionary with detailed explanation.
    """
    for candidate in result.candidates:
        if candidate.id == candidate_id:
            return {
                "candidate_id": candidate_id,
                "fused_score": round(candidate.fused_score, 6),
                "method": result.method.value,
                "parameters": result.parameters,
                "num_sources_total": result.num_sources,
                "num_sources_containing": candidate.num_sources,
                "source_weights": result.source_weights,
                "source_ranks": candidate.source_ranks,
                "source_scores": {k: round(v, 6) for k, v in candidate.source_scores.items()},
                "source_contributions": {
                    k: round(v, 6) for k, v in candidate.source_contributions.items()
                },
            }

    return {"error": f"Candidate {candidate_id} not found in fusion result"}
