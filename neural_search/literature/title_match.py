"""Title normalization/similarity utilities shared across all
literature-linking sources (OpenAlex, Crossref, Semantic Scholar, PubMed)
and the Bluesky discourse layer's dataset-title matching.

Factored out of neural_search.literature.linking, which re-imports these
names for backward compatibility with existing callers.
"""

from __future__ import annotations

import difflib
import re

TITLE_STOPWORDS = {
    "and",
    "are",
    "based",
    "data",
    "dataset",
    "datasets",
    "during",
    "for",
    "from",
    "human",
    "mouse",
    "neural",
    "neuronal",
    "of",
    "the",
    "using",
    "with",
}


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def title_tokens(title: str | None) -> list[str]:
    normalized = normalize_title(title)
    return [
        token
        for token in normalized.split()
        if len(token) >= 4 and token not in TITLE_STOPWORDS
    ]


def title_similarity(a: str, b: str) -> float:
    """Ratio in [0, 1] via difflib.SequenceMatcher, case-insensitive."""

    return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()
