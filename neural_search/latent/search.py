"""Latent similarity search for neural/behavioral features.

This module provides placeholder implementations for searching datasets
by neural activity patterns or behavioral signatures. Future implementations
will use actual vector similarity search.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neural_search.latent.schema import (
    FeatureType,
    LatentSearchResult,
    SessionFeatures,
)


def search_by_neural_similarity(
    query_features: SessionFeatures,
    index_sessions: list[SessionFeatures],
    feature_types: list[FeatureType] | None = None,
    top_k: int = 10,
) -> list[LatentSearchResult]:
    """Search for sessions with similar neural activity patterns.

    This is a placeholder implementation using simple cosine similarity.
    Future versions will use optimized vector search.

    Args:
        query_features: Features of the query session
        index_sessions: Sessions to search through
        feature_types: Optional filter for specific feature types
        top_k: Number of results to return

    Returns:
        List of search results ranked by similarity
    """
    if feature_types is None:
        # Default to neural feature types
        feature_types = [
            FeatureType.NEURAL_SUMMARY_STATISTICS,
            FeatureType.FIRING_RATE_HISTOGRAM,
            FeatureType.SPIKE_TRAIN_STATISTICS,
            FeatureType.CALCIUM_TRACE_SUMMARY,
            FeatureType.NEURAL_EMBEDDING,
        ]

    results: list[LatentSearchResult] = []

    for session in index_sessions:
        if session.dataset_id == query_features.dataset_id:
            continue  # Skip self-match

        similarity, matched_types, why = _compute_similarity(
            query_features, session, feature_types
        )

        if similarity > 0:
            results.append(
                LatentSearchResult(
                    dataset_id=session.dataset_id,
                    session_id=session.session_id,
                    similarity_score=similarity,
                    matched_feature_types=matched_types,
                    why_similar=why,
                )
            )

    # Sort by similarity descending
    results.sort(key=lambda r: r.similarity_score, reverse=True)
    return results[:top_k]


def search_by_behavior_pattern(
    query_features: SessionFeatures,
    index_sessions: list[SessionFeatures],
    top_k: int = 10,
) -> list[LatentSearchResult]:
    """Search for sessions with similar behavioral patterns.

    Args:
        query_features: Features of the query session
        index_sessions: Sessions to search through
        top_k: Number of results to return

    Returns:
        List of search results ranked by behavioral similarity
    """
    behavior_types = [
        FeatureType.EVENT_HISTOGRAM,
        FeatureType.BEHAVIOR_TRANSITION_SUMMARY,
        FeatureType.TASK_STATE_LABELS,
        FeatureType.BEHAVIOR_TRANSITION_MATRIX,
        FeatureType.TASK_STATE_SEQUENCE,
        FeatureType.TRIAL_OUTCOME_DISTRIBUTION,
        FeatureType.BEHAVIOR_EMBEDDING,
    ]

    return search_by_neural_similarity(
        query_features,
        index_sessions,
        feature_types=behavior_types,
        top_k=top_k,
    )


def search_by_task_performance(
    target_performance: float,
    index_sessions: list[SessionFeatures],
    tolerance: float = 0.1,
    top_k: int = 10,
) -> list[LatentSearchResult]:
    """Search for sessions with similar task performance.

    Args:
        target_performance: Target performance level (0-1)
        index_sessions: Sessions to search through
        tolerance: Acceptable deviation from target
        top_k: Number of results to return

    Returns:
        List of search results matching performance criteria
    """
    results: list[LatentSearchResult] = []

    for session in index_sessions:
        # Find task state sequence feature
        task_feature = None
        for feature in session.features:
            if feature.feature_type == FeatureType.TASK_STATE_SEQUENCE:
                task_feature = feature
                break

        if task_feature is None or len(task_feature.values) < 2:
            continue

        # Second value is performance in our placeholder schema
        performance = task_feature.values[1]
        distance = abs(performance - target_performance)

        if distance <= tolerance:
            similarity = 1.0 - (distance / tolerance)
            results.append(
                LatentSearchResult(
                    dataset_id=session.dataset_id,
                    session_id=session.session_id,
                    similarity_score=similarity,
                    matched_feature_types=[FeatureType.TASK_STATE_SEQUENCE],
                    why_similar=[f"Performance {performance:.1%} matches target {target_performance:.1%}"],
                )
            )

    results.sort(key=lambda r: r.similarity_score, reverse=True)
    return results[:top_k]


def hybrid_search(
    query: str,
    query_features: SessionFeatures | None,
    index_sessions: list[SessionFeatures],
    ontology_results: list[dict[str, Any]],
    neural_weight: float = 0.3,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Combine ontology search with latent similarity search.

    This is a placeholder for hybrid retrieval that combines:
    1. Metadata/ontology-based search (existing system)
    2. Neural/behavioral similarity search (latent features)

    Args:
        query: Text query for ontology search
        query_features: Optional latent features for similarity search
        index_sessions: Sessions with extracted features
        ontology_results: Results from existing ontology search
        neural_weight: Weight for neural similarity (0-1)
        top_k: Number of final results

    Returns:
        Combined search results with both scores
    """
    # Start with ontology results
    combined: dict[str, dict[str, Any]] = {}
    for result in ontology_results:
        dataset_id = result.get("dataset_id", result.get("id"))
        combined[dataset_id] = {
            "dataset_id": dataset_id,
            "ontology_score": result.get("score", 0),
            "neural_score": 0.0,
            "combined_score": 0.0,
            "neural_match_reasons": [],
            **result,
        }

    # Add latent similarity if features provided
    if query_features is not None:
        latent_results = search_by_neural_similarity(
            query_features, index_sessions, top_k=top_k * 2
        )

        for lr in latent_results:
            if lr.dataset_id in combined:
                combined[lr.dataset_id]["neural_score"] = lr.similarity_score
                combined[lr.dataset_id]["neural_match_reasons"] = lr.why_similar
            else:
                combined[lr.dataset_id] = {
                    "dataset_id": lr.dataset_id,
                    "ontology_score": 0.0,
                    "neural_score": lr.similarity_score,
                    "combined_score": 0.0,
                    "neural_match_reasons": lr.why_similar,
                }

    # Compute combined scores
    ontology_weight = 1.0 - neural_weight
    for item in combined.values():
        item["combined_score"] = (
            ontology_weight * item["ontology_score"]
            + neural_weight * item["neural_score"]
        )

    # Sort and return top-k
    results = sorted(
        combined.values(),
        key=lambda x: x["combined_score"],
        reverse=True,
    )
    return results[:top_k]


def _compute_similarity(
    query: SessionFeatures,
    target: SessionFeatures,
    feature_types: list[FeatureType],
) -> tuple[float, list[FeatureType], list[str]]:
    """Compute similarity between two sessions.

    Args:
        query: Query session features
        target: Target session features
        feature_types: Feature types to compare

    Returns:
        Tuple of (similarity_score, matched_types, explanations)
    """
    similarities: list[float] = []
    matched_types: list[FeatureType] = []
    explanations: list[str] = []

    for ft in feature_types:
        q_feature = next((f for f in query.features if f.feature_type == ft), None)
        t_feature = next((f for f in target.features if f.feature_type == ft), None)

        if q_feature is None or t_feature is None:
            continue

        if len(q_feature.values) == 0 or len(t_feature.values) == 0:
            continue

        # Compute cosine similarity
        q_vec = np.array(q_feature.values)
        t_vec = np.array(t_feature.values)

        # Handle dimension mismatch
        min_dim = min(len(q_vec), len(t_vec))
        q_vec = q_vec[:min_dim]
        t_vec = t_vec[:min_dim]

        q_norm = np.linalg.norm(q_vec)
        t_norm = np.linalg.norm(t_vec)

        if q_norm > 0 and t_norm > 0:
            sim = float(np.dot(q_vec, t_vec) / (q_norm * t_norm))
            sim = (sim + 1) / 2  # Normalize to [0, 1]
            similarities.append(sim)
            matched_types.append(ft)
            explanations.append(f"{ft.value}: {sim:.2f} similarity")

    if not similarities:
        return 0.0, [], []

    avg_similarity = sum(similarities) / len(similarities)
    return avg_similarity, matched_types, explanations
