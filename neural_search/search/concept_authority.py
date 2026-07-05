"""Scholarpedia concept authority index for query expansion and disambiguation."""

from __future__ import annotations

import re
from functools import lru_cache

from neural_search.ingestion.scholarpedia_builder import SCHOLARPEDIA_CONCEPTS


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


@lru_cache(maxsize=1)
def load_scholarpedia_index() -> dict[str, str]:
    """Return a mapping of {normalized_alias: canonical_concept_slug}.

    Cached for the lifetime of the process.
    """
    index: dict[str, str] = {}
    for slug, entry in SCHOLARPEDIA_CONCEPTS.items():
        index[_normalize(slug.replace("_", " "))] = slug
        index[_normalize(slug)] = slug
        for alias in entry.get("aliases", []):
            normalized = _normalize(alias)
            if normalized:
                index[normalized] = slug
    return index


def get_concept_aliases(concept_slug: str) -> list[str]:
    """Return all aliases for a canonical concept slug, or empty list if not found."""
    entry = SCHOLARPEDIA_CONCEPTS.get(concept_slug)
    if entry is None:
        return []
    return list(entry.get("aliases", []))


def expand_query_with_concepts(query: str) -> list[str]:
    """Match query tokens against the Scholarpedia alias index.

    Returns canonical concept slugs found in the query, in appearance order,
    without duplicates. Longest aliases are matched first to avoid substring
    collisions.
    """
    if not query or not query.strip():
        return []

    index = load_scholarpedia_index()
    normalized_query = _normalize(query)
    matched: list[str] = []
    seen: set[str] = set()

    for alias_norm, slug in sorted(index.items(), key=lambda kv: -len(kv[0])):
        if not alias_norm:
            continue
        pattern = rf"(?<!\w){re.escape(alias_norm)}(?!\w)"
        if re.search(pattern, normalized_query) and slug not in seen:
            matched.append(slug)
            seen.add(slug)

    return matched
