"""Intent classification for usefulness-focused retrieval.

Distinct from neural_search.search.intent which handles retrieval-head
weight overrides. This module classifies the *usefulness relationship*
the user seeks (replication, pipeline reuse, method transfer, etc.).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class UsefulnessIntent(Enum):
    STRICT_LOOKUP = "strict_lookup"
    REPLICATION = "replication"
    META_ANALYSIS = "meta_analysis"
    PIPELINE_REUSE = "pipeline_reuse"
    CROSS_DATASET_COMPARISON = "cross_dataset_comparison"
    EXPLORATION = "exploration"
    METHOD_TRANSFER = "method_transfer"


@dataclass
class IntentClassification:
    intent: UsefulnessIntent
    confidence: float
    matched_patterns: list[str] = field(default_factory=list)
    explanation: str = ""


# Each intent maps to a list of (regex_pattern, confidence_boost) tuples.
_PATTERNS: list[tuple[UsefulnessIntent, str, float]] = [
    # REPLICATION
    (UsefulnessIntent.REPLICATION, r"\breplicate\b", 0.90),
    (UsefulnessIntent.REPLICATION, r"\breproduce\b", 0.88),
    (UsefulnessIntent.REPLICATION, r"\bsame experiment as\b", 0.85),
    (UsefulnessIntent.REPLICATION, r"\breplic", 0.82),
    # PIPELINE_REUSE
    (UsefulnessIntent.PIPELINE_REUSE, r"datasets?\s+like\s+\w", 0.90),
    (UsefulnessIntent.PIPELINE_REUSE, r"similar\s+to\s+dandi", 0.88),
    (UsefulnessIntent.PIPELINE_REUSE, r"same\s+pipeline", 0.85),
    (UsefulnessIntent.PIPELINE_REUSE, r"reuse", 0.80),
    # CROSS_DATASET_COMPARISON
    (UsefulnessIntent.CROSS_DATASET_COMPARISON, r"compare\s+\w+\s+and\s+\w+", 0.88),
    (UsefulnessIntent.CROSS_DATASET_COMPARISON, r"cross[\s-]species", 0.85),
    (UsefulnessIntent.CROSS_DATASET_COMPARISON, r"\bcompar", 0.75),
    (UsefulnessIntent.CROSS_DATASET_COMPARISON, r"across\s+(species|datasets?|studies)", 0.80),
    # META_ANALYSIS
    (UsefulnessIntent.META_ANALYSIS, r"meta[\s-]analysis", 0.92),
    (UsefulnessIntent.META_ANALYSIS, r"pool\s+datasets?", 0.88),
    (UsefulnessIntent.META_ANALYSIS, r"systematic\s+review", 0.85),
    (UsefulnessIntent.META_ANALYSIS, r"aggregate\s+(across|multiple)", 0.82),
    # METHOD_TRANSFER
    (UsefulnessIntent.METHOD_TRANSFER, r"model\s+fitting", 0.88),
    (UsefulnessIntent.METHOD_TRANSFER, r"q[\s-]?learning", 0.85),
    (UsefulnessIntent.METHOD_TRANSFER, r"methods?\s+transfer", 0.85),
    (UsefulnessIntent.METHOD_TRANSFER, r"apply\s+(this\s+)?method", 0.82),
    (UsefulnessIntent.METHOD_TRANSFER, r"datasets?\s+for\s+\w+\s+(model|fitting|algorithm)", 0.80),
    # EXPLORATION
    (UsefulnessIntent.EXPLORATION, r"surprising", 0.85),
    (UsefulnessIntent.EXPLORATION, r"find\s+(related|unexpected|novel)", 0.82),
    (UsefulnessIntent.EXPLORATION, r"what\s+(other|else)", 0.78),
    (UsefulnessIntent.EXPLORATION, r"explore\b", 0.78),
    (UsefulnessIntent.EXPLORATION, r"discover", 0.75),
]

_EXPLANATIONS: dict[UsefulnessIntent, str] = {
    UsefulnessIntent.STRICT_LOOKUP: "Query targets specific dataset features; exact match weights prioritized.",
    UsefulnessIntent.REPLICATION: "Query seeks to replicate a prior study; task/species/region alignment emphasized.",
    UsefulnessIntent.META_ANALYSIS: "Query seeks datasets suitable for meta-analysis; provenance and statistical power emphasized.",
    UsefulnessIntent.PIPELINE_REUSE: "Query seeks datasets compatible with an existing analysis pipeline.",
    UsefulnessIntent.CROSS_DATASET_COMPARISON: "Query seeks to compare across datasets; comparability emphasized.",
    UsefulnessIntent.EXPLORATION: "Query seeks to discover unexpected related datasets; graph proximity emphasized.",
    UsefulnessIntent.METHOD_TRANSFER: "Query seeks datasets to apply a specific analysis method to.",
}


def classify_query_intent(
    query: str,
    parsed_constraints: "Any | None" = None,
) -> IntentClassification:
    """Classify the latent usefulness intent of a query using deterministic rules."""
    lower = query.lower()
    hits: list[tuple[UsefulnessIntent, float, str]] = []

    for intent, pattern, confidence in _PATTERNS:
        if re.search(pattern, lower):
            hits.append((intent, confidence, pattern))

    if not hits:
        return IntentClassification(
            intent=UsefulnessIntent.STRICT_LOOKUP,
            confidence=0.55,
            matched_patterns=[],
            explanation=_EXPLANATIONS[UsefulnessIntent.STRICT_LOOKUP],
        )

    # Pick highest-confidence hit; if tie, prefer first defined
    hits.sort(key=lambda x: x[1], reverse=True)
    best_intent, best_conf, _ = hits[0]
    matched = [h[2] for h in hits if h[0] == best_intent]

    return IntentClassification(
        intent=best_intent,
        confidence=best_conf,
        matched_patterns=matched,
        explanation=_EXPLANATIONS[best_intent],
    )
