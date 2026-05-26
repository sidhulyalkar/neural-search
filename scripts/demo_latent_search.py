#!/usr/bin/env python3
"""Demonstrate latent neural/behavioral search capabilities.

This script shows how the latent search module can find datasets
with similar neural activity patterns or behavioral signatures.

Usage:
    python scripts/demo_latent_search.py
"""

from __future__ import annotations

from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.latent import (
    SessionFeatures,
    extract_session_features,
)
from neural_search.latent.indexing import build_default_indices
from neural_search.latent.search import (
    hybrid_search,
    search_by_behavior_pattern,
    search_by_neural_similarity,
    search_by_task_performance,
)


def main() -> int:
    """Run latent search demonstration."""
    print("=" * 70)
    print("NEURAL SEARCH: LATENT SIMILARITY DEMONSTRATION")
    print("=" * 70)

    # Load demo datasets
    print("\n[1/4] Loading demo datasets and extracting features...")
    records = build_demo_seed()
    sessions: list[SessionFeatures] = []

    for record in records:
        dataset = record["dataset"]
        features = extract_session_features(dataset)
        sessions.append(features)
        print(f"  - {features.dataset_id}: {len(features.features)} feature types")

    print(f"\n  Total: {len(sessions)} sessions indexed")

    # Build indices
    print("\n[2/4] Building latent feature indices...")
    indices = build_default_indices(sessions)
    for name, index in indices.items():
        print(f"  - {name}: {index.num_sessions} sessions, dim={index.embedding_dim}")

    # Demo: Find similar datasets to a query dataset
    print("\n[3/4] Searching by neural/behavioral similarity...")

    # Use first dataset as query
    query_session = sessions[0]
    print(f"\n  Query dataset: {query_session.dataset_id}")

    # Search by neural similarity
    print("\n  Neural similarity results:")
    neural_results = search_by_neural_similarity(
        query_session, sessions, top_k=5
    )
    for i, result in enumerate(neural_results, 1):
        reasons = ", ".join(result.why_similar[:2]) if result.why_similar else "N/A"
        print(f"    {i}. {result.dataset_id} (score: {result.similarity_score:.3f})")
        print(f"       Why: {reasons}")

    # Search by behavior pattern
    print("\n  Behavioral similarity results:")
    behavior_results = search_by_behavior_pattern(
        query_session, sessions, top_k=5
    )
    for i, result in enumerate(behavior_results, 1):
        reasons = ", ".join(result.why_similar[:2]) if result.why_similar else "N/A"
        print(f"    {i}. {result.dataset_id} (score: {result.similarity_score:.3f})")
        print(f"       Why: {reasons}")

    # Demo: Search by task performance
    print("\n  Task performance similarity (target=0.75):")
    perf_results = search_by_task_performance(
        target_performance=0.75,
        index_sessions=sessions,
        tolerance=0.3,
        top_k=3,
    )
    for i, result in enumerate(perf_results, 1):
        print(f"    {i}. {result.dataset_id} (score: {result.similarity_score:.3f})")

    # Demo: Hybrid search combining ontology and latent
    print("\n[4/4] Hybrid search (ontology + latent)...")

    # Mock ontology results (in real usage, these come from the main search)
    ontology_results = [
        {"dataset_id": sessions[1].dataset_id, "score": 0.8},
        {"dataset_id": sessions[2].dataset_id, "score": 0.7},
        {"dataset_id": sessions[3].dataset_id, "score": 0.6},
    ]

    hybrid_results = hybrid_search(
        query="Find reversal learning datasets",
        query_features=query_session,
        index_sessions=sessions,
        ontology_results=ontology_results,
        neural_weight=0.3,
        top_k=5,
    )

    print("\n  Hybrid search results (70% ontology + 30% neural):")
    for i, result in enumerate(hybrid_results, 1):
        print(
            f"    {i}. {result['dataset_id']} "
            f"(combined: {result['combined_score']:.3f}, "
            f"ontology: {result['ontology_score']:.2f}, "
            f"neural: {result['neural_score']:.2f})"
        )

    print("\n" + "=" * 70)
    print("Latent search demonstration complete!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
