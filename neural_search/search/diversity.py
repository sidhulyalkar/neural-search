"""Source diversity enforcement for search results.

Without diversity enforcement a single large corpus (NeuroVault: 2,000+ fMRI
datasets, NeuroMorpho: 100,000+ morphologies) can dominate all top-K slots
for queries that happen to match its data well. This makes the ranking look
high-quality but is not actually useful — the researcher sees 10 variations
of the same dataset type from one lab.

Algorithm (greedy windowed):
  1. Walk sorted results in score order.
  2. Accept a result if its source has appeared fewer than `max_per_source` times.
  3. Defer results that exceed the cap.
  4. Return accepted results followed by deferred results, truncated to `limit`.

This preserves score ordering within the accepted window and ensures deferred
results are still reachable if the caller requests more than `limit` items.
"""
from __future__ import annotations

from neural_search.schemas import SearchResult

_DEFAULT_MAX_PER_SOURCE = 3
_UNKNOWN_SOURCE = "unknown"


def apply_source_diversity(
    results: list[SearchResult],
    max_per_source: int = _DEFAULT_MAX_PER_SOURCE,
    limit: int | None = None,
) -> list[SearchResult]:
    """Reorder results to enforce per-source diversity within the top-N window.

    Args:
        results: Score-sorted results (highest first).
        max_per_source: Maximum results from any single source in the diverse window.
        limit: If given, truncate final list to this length.

    Returns:
        Reordered results list. Items beyond `max_per_source` for their source
        are moved to the back; the overall list is otherwise score-ordered.
    """
    if max_per_source <= 0:
        return results[:limit] if limit is not None else list(results)

    source_counts: dict[str, int] = {}
    accepted: list[SearchResult] = []
    deferred: list[SearchResult] = []

    for result in results:
        src = result.source or _UNKNOWN_SOURCE
        count = source_counts.get(src, 0)
        if count < max_per_source:
            source_counts[src] = count + 1
            accepted.append(result)
        else:
            deferred.append(result)

    combined = accepted + deferred
    return combined[:limit] if limit is not None else combined


def diversity_stats(results: list[SearchResult]) -> dict[str, int]:
    """Return per-source result counts for diagnostics."""
    counts: dict[str, int] = {}
    for result in results:
        src = result.source or _UNKNOWN_SOURCE
        counts[src] = counts.get(src, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
