"""Tokenization helpers for latent feature summaries."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize_labels(values: Iterable[str]) -> list[str]:
    """Normalize labels into stable lowercase tokens."""

    tokens: list[str] = []
    for value in values:
        tokens.extend(TOKEN_RE.findall(str(value).casefold()))
    return tokens


def normalized_histogram(tokens: Iterable[str], vocabulary: list[str]) -> list[float]:
    """Return a normalized count vector over a fixed vocabulary."""

    counts = Counter(tokens)
    total = sum(counts[token] for token in vocabulary)
    if total == 0:
        return [0.0 for _ in vocabulary]
    return [counts[token] / total for token in vocabulary]
