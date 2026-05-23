"""Small vector helpers used by retrieval tests and local search."""

from __future__ import annotations

import math


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity clipped to a 0-1 retrieval signal."""

    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min((dot / (left_norm * right_norm) + 1.0) / 2.0, 1.0))
