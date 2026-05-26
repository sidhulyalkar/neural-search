"""Query intent classification for optimized retrieval profiles."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from neural_search.ontology import normalize_text

# Default path to intent profiles config
DEFAULT_INTENT_PROFILES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "config" / "intent_profiles.yaml"
)


class QueryIntent(Enum):
    """Types of search queries with different retrieval profiles."""

    DATASET_LOOKUP = "dataset_lookup"  # "DANDI 000026", "OpenNeuro ds000117"
    TASK_SEARCH = "task_search"  # "reversal learning datasets"
    ANALYSIS_SEARCH = "analysis_search"  # "choice decoding ready"
    PAPER_LINK = "paper_link"  # "data from Steinmetz 2019"
    DESIGN_QUERY = "design_query"  # "design auditory reversal learning"
    GRAPH_REASONING = "graph_reasoning"  # "datasets from labs that published..."
    COMPOUND_CONSTRAINT = "compound"  # Multiple constraints with AND/NOT


@dataclass
class IntentClassification:
    """Result of query intent classification."""

    primary_intent: QueryIntent
    confidence: float
    secondary_intents: list[QueryIntent]
    weight_overrides: dict[str, float]


# Patterns for detecting specific intents
INTENT_PATTERNS: dict[QueryIntent, list[str]] = {
    QueryIntent.DATASET_LOOKUP: [
        r"dandi\s*\d+",
        r"openneuro\s*ds\d+",
        r"dataset\s+(id|#|number)\s*\w+",
        r"^[A-Z_]+\d+$",  # ID-like patterns
    ],
    QueryIntent.PAPER_LINK: [
        r"data\s+from\s+\w+\s+\d{4}",
        r"papers?\s+using",
        r"publications?\s+citing",
        r"dataset\s+from\s+\w+\s+lab",
        r"steinmetz|churchland|allen\s+institute",
        r"\w+\s+et\s+al",
    ],
    QueryIntent.ANALYSIS_SEARCH: [
        r"ready\s+for\s+\w+",
        r"\w+\s+decoding\s+ready",
        r"supporting\s+\w+\s+analysis",
        r"suitable\s+for\s+\w+\s+modeling",
        r"can\s+i\s+do\s+\w+\s+with",
        r"datasets?\s+for\s+\w+\s+analysis",
    ],
    QueryIntent.DESIGN_QUERY: [
        r"design\s+\w+\s+task",
        r"paradigm\s+for",
        r"protocol\s+for",
        r"setup\s+for",
        r"how\s+to\s+run",
        r"experiment\s+design",
    ],
    QueryIntent.GRAPH_REASONING: [
        r"datasets?\s+from\s+labs?\s+that",
        r"share\s+authors?\s+with",
        r"citing\s+same",
        r"similar\s+to",
        r"related\s+to\s+\w+\s+paper",
        r"same\s+\w+\s+as",
        r"labs?\s+that\s+study",
    ],
}

@dataclass
class IntentProfile:
    """Configuration for an intent-specific scoring profile."""

    name: str
    description: str
    weights: dict[str, float]
    scoring_mode: str = "weighted_sum"
    min_confidence: float = 0.70
    result_limit: int | None = None
    strict_exclusion: bool = False
    use_graph_expansion: bool = False
    use_semantic_fingerprint: bool = False
    penalties: dict[str, float] = field(default_factory=dict)


@lru_cache(maxsize=1)
def load_intent_profiles(
    path: str | Path | None = None,
) -> dict[str, IntentProfile]:
    """Load intent profiles from YAML configuration.

    Args:
        path: Path to intent profiles YAML file

    Returns:
        Dict mapping profile name to IntentProfile
    """
    if path is None:
        path = DEFAULT_INTENT_PROFILES_PATH

    path = Path(path)
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    profiles = {}
    for name, config in data.get("profiles", {}).items():
        profiles[name] = IntentProfile(
            name=name,
            description=config.get("description", ""),
            weights=config.get("weights", {}),
            scoring_mode=config.get("scoring_mode", "weighted_sum"),
            min_confidence=config.get("min_confidence", 0.70),
            result_limit=config.get("result_limit"),
            strict_exclusion=config.get("strict_exclusion", False),
            use_graph_expansion=config.get("use_graph_expansion", False),
            use_semantic_fingerprint=config.get("use_semantic_fingerprint", False),
            penalties=config.get("penalties", {}),
        )

    return profiles


def get_intent_profile(intent: QueryIntent) -> IntentProfile | None:
    """Get the profile for a given intent.

    Args:
        intent: QueryIntent enum value

    Returns:
        IntentProfile if found, None otherwise
    """
    profiles = load_intent_profiles()
    intent_name = intent.value  # e.g., "task_search"
    return profiles.get(intent_name)


def get_weights_for_intent(intent: QueryIntent) -> dict[str, float]:
    """Get weight overrides for a given intent.

    Args:
        intent: QueryIntent enum value

    Returns:
        Weight dictionary from profile, or fallback to hardcoded weights
    """
    profile = get_intent_profile(intent)
    if profile:
        return profile.weights

    # Fall back to hardcoded weights
    return INTENT_WEIGHT_PROFILES.get(intent, {})


# Weight profiles per intent type - these override base weights (legacy/fallback)
INTENT_WEIGHT_PROFILES: dict[QueryIntent, dict[str, float]] = {
    QueryIntent.DATASET_LOOKUP: {
        "metadata": 0.25,
        "semantic": 0.20,
        "ontology": 0.15,
        "graph": 0.02,
    },
    QueryIntent.TASK_SEARCH: {
        "ontology": 0.32,
        "behavior": 0.22,
        "affordance": 0.12,
        "graph": 0.04,
    },
    QueryIntent.ANALYSIS_SEARCH: {
        "affordance": 0.28,
        "readiness": 0.18,
        "ontology": 0.18,
        "behavior": 0.12,
    },
    QueryIntent.PAPER_LINK: {
        "graph": 0.12,
        "paper_confidence": 0.18,
        "semantic": 0.15,
        "metadata": 0.12,
    },
    QueryIntent.GRAPH_REASONING: {
        "graph": 0.18,
        "paper_confidence": 0.12,
        "ontology": 0.18,
        "semantic": 0.12,
    },
    QueryIntent.DESIGN_QUERY: {
        "ontology": 0.25,
        "behavior": 0.20,
        "affordance": 0.15,
        "readiness": 0.12,
    },
    QueryIntent.COMPOUND_CONSTRAINT: {
        # Keep base weights but ensure constraint handling
        "ontology": 0.28,
        "behavior": 0.20,
    },
}


def classify_query_intent(
    query: str,
    parsed_query: Mapping[str, Any] | None = None,
) -> IntentClassification:
    """Classify query intent and return weight overrides.

    Args:
        query: The raw query string
        parsed_query: Optional pre-parsed query with extracted intents

    Returns:
        IntentClassification with primary intent, confidence, and weight overrides
    """
    normalized = normalize_text(query)
    matches: list[tuple[QueryIntent, float]] = []

    # Pattern matching for explicit intent signals
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                matches.append((intent, 0.85))
                break
            # Also check original query for case-sensitive patterns
            if re.search(pattern, query, re.IGNORECASE):
                matches.append((intent, 0.80))
                break

    # Heuristic signals from parsed query
    if parsed_query:
        # Check for exclusions indicating compound constraint
        has_exclusions = any(
            [
                parsed_query.get("excluded_modalities"),
                parsed_query.get("excluded_species"),
                parsed_query.get("excluded_tasks"),
                parsed_query.get("excluded_regions"),
            ]
        )
        if has_exclusions:
            matches.append((QueryIntent.COMPOUND_CONSTRAINT, 0.75))

        # Strong affordance signals
        if parsed_query.get("affordances"):
            matches.append((QueryIntent.ANALYSIS_SEARCH, 0.72))

        # Multiple task matches suggest task search
        if len(parsed_query.get("tasks", [])) >= 2:
            matches.append((QueryIntent.TASK_SEARCH, 0.70))

    # Check for explicit analysis keywords
    analysis_keywords = [
        "decoding",
        "classification",
        "modeling",
        "fitting",
        "detection",
        "prediction",
    ]
    if any(kw in normalized for kw in analysis_keywords):
        matches.append((QueryIntent.ANALYSIS_SEARCH, 0.68))

    # Default to task search if no strong signals
    if not matches:
        matches.append((QueryIntent.TASK_SEARCH, 0.55))

    # Sort by confidence and deduplicate
    matches.sort(key=lambda x: x[1], reverse=True)
    seen_intents: set[QueryIntent] = set()
    unique_matches: list[tuple[QueryIntent, float]] = []
    for intent, conf in matches:
        if intent not in seen_intents:
            seen_intents.add(intent)
            unique_matches.append((intent, conf))

    primary = unique_matches[0][0]
    primary_conf = unique_matches[0][1]

    # Get weight overrides from YAML profile or fallback to hardcoded
    weight_overrides = get_weights_for_intent(primary)

    return IntentClassification(
        primary_intent=primary,
        confidence=primary_conf,
        secondary_intents=[m[0] for m in unique_matches[1:3]],
        weight_overrides=weight_overrides,
    )


def blend_weights(
    base_weights: dict[str, float],
    intent_overrides: dict[str, float],
    confidence: float,
    confidence_threshold: float = 0.70,
) -> dict[str, float]:
    """Blend base weights with intent-specific overrides.

    Args:
        base_weights: Default retrieval weights
        intent_overrides: Intent-specific weight adjustments
        confidence: Classification confidence (0-1)
        confidence_threshold: Minimum confidence to apply overrides

    Returns:
        Blended weight dictionary
    """
    if confidence < confidence_threshold or not intent_overrides:
        return base_weights

    # Blend factor based on confidence
    blend = (confidence - confidence_threshold) / (1.0 - confidence_threshold)
    blend = min(blend, 0.5)  # Cap at 50% override influence

    result = dict(base_weights)
    for key, override_value in intent_overrides.items():
        if key in result:
            result[key] = result[key] * (1 - blend) + override_value * blend

    return result
