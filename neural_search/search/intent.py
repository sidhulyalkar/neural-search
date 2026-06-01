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
    """Types of search queries with different retrieval profiles.

    Intent names correspond to profile names in intent_profiles.yaml
    """

    # Core content intents
    DATASET_LOOKUP = "dataset_lookup"      # "DANDI 000026", "OpenNeuro ds000117"
    PAPER_LOOKUP = "paper_lookup"          # "Steinmetz 2019 paper"
    TASK_SEARCH = "task_search"            # "reversal learning datasets"
    MODALITY_SEARCH = "modality_search"    # "neuropixels recordings"
    SPECIES_REGION = "species_region"      # "mouse hippocampus"
    ANALYSIS_SEARCH = "analysis_affordance"  # "choice decoding ready"

    # Linking and similarity intents
    PAPER_TO_DATASET = "paper_to_dataset"  # "datasets from Steinmetz lab"
    SIMILAR_DATASET = "similar_dataset"    # "datasets like 000026"

    # Constraint intents
    HARD_NEGATIVE = "hard_negative"        # "neuropixels NOT calcium"
    MULTI_CONSTRAINT = "multi_constraint"  # Complex multi-faceted queries

    # Exploratory intent
    EXPLORATORY = "exploratory"            # "what datasets study attention?"

    # Legacy aliases (for backwards compatibility)
    PAPER_LINK = "paper_to_dataset"
    DESIGN_QUERY = "task_search"
    GRAPH_REASONING = "paper_to_dataset"
    COMPOUND_CONSTRAINT = "hard_negative"


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
        r"\bds\d{5,}\b",
        r"\b\d{6}\b",  # 6-digit IDs
    ],
    QueryIntent.PAPER_LOOKUP: [
        r"\w+\s+et\s+al\.?\s*\d{4}",
        r"\w+\s+\d{4}\s+paper",
        r"doi:\s*10\.\d+",
        r"pmid:\s*\d+",
        r"find\s+paper",
    ],
    QueryIntent.PAPER_TO_DATASET: [
        r"data\s+from\s+\w+\s+\d{4}",
        r"papers?\s+using",
        r"publications?\s+citing",
        r"dataset\s+from\s+\w+\s+lab",
        r"steinmetz|churchland|allen\s+institute",
        r"datasets?\s+from\s+labs?\s+that",
        r"labs?\s+that\s+study",
    ],
    QueryIntent.ANALYSIS_SEARCH: [
        r"ready\s+for\s+\w+",
        r"\w+\s+decoding\s+ready",
        r"supporting\s+\w+\s+analysis",
        r"suitable\s+for\s+\w+\s+modeling",
        r"can\s+i\s+do\s+\w+\s+with",
        r"datasets?\s+for\s+\w+\s+analysis",
        r"supports?\s+\w+\s+analysis",
    ],
    QueryIntent.MODALITY_SEARCH: [
        r"neuropixels?\s+recordings?",
        r"calcium\s+imaging",
        r"two[\s-]?photon",
        r"fmri\s+data",
        r"\beeg\b\s+data",
        r"\bmeg\b\s+data",
        r"ecog\s+recordings?",
        r"patch[\s-]?clamp",
        r"fiber\s+photometry",
    ],
    QueryIntent.SPECIES_REGION: [
        r"mouse\s+(hippocampus|cortex|striatum|thalamus)",
        r"human\s+(brain|cortex|hippocampus)",
        r"macaque\s+(v1|v4|prefrontal|motor)",
        r"rat\s+(hippocampus|cortex)",
        r"(hippocampus|cortex|striatum)\s+in\s+(mouse|rat|human)",
    ],
    QueryIntent.SIMILAR_DATASET: [
        r"similar\s+to\s+\w+",
        r"like\s+\w+\s*\d+",
        r"datasets?\s+like",
        r"related\s+to\s+\w+\s+dataset",
        r"same\s+\w+\s+as",
    ],
    QueryIntent.HARD_NEGATIVE: [
        r"\bNOT\b",
        r"\bwithout\b",
        r"\bexcluding\b",
        r"\bexcept\b",
        r"\bnot\s+including\b",
        r"\b(?<!do\s)not\s+\w+",  # "not calcium" but not "do not"
    ],
    QueryIntent.EXPLORATORY: [
        r"^what\s+datasets?",
        r"^which\s+datasets?",
        r"^how\s+many\s+datasets?",
        r"^list\s+all",
        r"^show\s+me\s+all",
        r"^find\s+all",
        r"available\s+datasets?",
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
# Note: Profiles from intent_profiles.yaml take precedence over these
INTENT_WEIGHT_PROFILES: dict[QueryIntent, dict[str, float]] = {
    QueryIntent.DATASET_LOOKUP: {
        "metadata": 0.60,
        "semantic": 0.15,
        "ontology": 0.15,
    },
    QueryIntent.PAPER_LOOKUP: {
        "paper_confidence": 0.40,
        "metadata": 0.30,
        "semantic": 0.15,
    },
    QueryIntent.TASK_SEARCH: {
        "ontology": 0.35,
        "behavior": 0.25,
        "semantic": 0.15,
        "affordance": 0.10,
    },
    QueryIntent.MODALITY_SEARCH: {
        "modality": 0.40,
        "ontology": 0.20,
        "semantic": 0.15,
    },
    QueryIntent.SPECIES_REGION: {
        "metadata": 0.35,
        "ontology": 0.25,
        "semantic": 0.15,
    },
    QueryIntent.ANALYSIS_SEARCH: {
        "affordance": 0.40,
        "ontology": 0.20,
        "readiness": 0.15,
        "semantic": 0.15,
    },
    QueryIntent.PAPER_TO_DATASET: {
        "paper_confidence": 0.35,
        "semantic": 0.20,
        "ontology": 0.15,
        "metadata": 0.15,
    },
    QueryIntent.SIMILAR_DATASET: {
        "semantic": 0.40,
        "ontology": 0.25,
        "behavior": 0.15,
    },
    QueryIntent.HARD_NEGATIVE: {
        "ontology": 0.25,
        "modality": 0.25,
        "semantic": 0.15,
        "behavior": 0.15,
    },
    QueryIntent.MULTI_CONSTRAINT: {
        "ontology": 0.25,
        "modality": 0.20,
        "metadata": 0.15,
        "semantic": 0.15,
        "behavior": 0.15,
    },
    QueryIntent.EXPLORATORY: {
        "semantic": 0.30,
        "ontology": 0.25,
        "metadata": 0.20,
        "readiness": 0.15,
    },
}


def classify_query_intent(
    query: str,
    parsed_query: Mapping[str, Any] | None = None,
) -> IntentClassification:
    """Classify query intent and return weight overrides.

    Uses a multi-signal approach:
    1. Pattern matching for explicit intent signals
    2. Keyword detection for analysis/exploratory queries
    3. Parsed query features (exclusions, modalities, etc.)
    4. Fallback to exploratory or task search

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
        # Check for exclusions indicating hard negative
        has_exclusions = any(
            [
                parsed_query.get("excluded_modalities"),
                parsed_query.get("excluded_species"),
                parsed_query.get("excluded_tasks"),
                parsed_query.get("excluded_regions"),
            ]
        )
        if has_exclusions:
            matches.append((QueryIntent.HARD_NEGATIVE, 0.88))

        # Strong affordance signals
        if parsed_query.get("affordances"):
            matches.append((QueryIntent.ANALYSIS_SEARCH, 0.78))

        # Strong modality signals
        modalities = parsed_query.get("modalities", [])
        if modalities and len(modalities) >= 1:
            matches.append((QueryIntent.MODALITY_SEARCH, 0.72))

        # Species + region combination
        species = parsed_query.get("species", [])
        regions = parsed_query.get("brain_regions", [])
        if species and regions:
            matches.append((QueryIntent.SPECIES_REGION, 0.75))

        # Multiple constraints suggest multi-constraint
        constraint_count = sum([
            1 if parsed_query.get("tasks") else 0,
            1 if parsed_query.get("modalities") else 0,
            1 if parsed_query.get("species") else 0,
            1 if parsed_query.get("brain_regions") else 0,
            1 if parsed_query.get("behaviors") else 0,
        ])
        if constraint_count >= 3:
            matches.append((QueryIntent.MULTI_CONSTRAINT, 0.70))

        # Task matches
        if len(parsed_query.get("tasks", [])) >= 1:
            matches.append((QueryIntent.TASK_SEARCH, 0.68))

    # Check for explicit analysis keywords
    analysis_keywords = [
        "decoding",
        "classification",
        "modeling",
        "fitting",
        "detection",
        "prediction",
        "suitable for",
        "ready for",
    ]
    if any(kw in normalized for kw in analysis_keywords):
        matches.append((QueryIntent.ANALYSIS_SEARCH, 0.72))

    # Check for exploratory question patterns
    exploratory_starters = ["what", "which", "how many", "list", "show", "find all"]
    if any(normalized.startswith(s) for s in exploratory_starters):
        matches.append((QueryIntent.EXPLORATORY, 0.65))

    # Check for similarity queries
    if "similar" in normalized or "like" in normalized:
        matches.append((QueryIntent.SIMILAR_DATASET, 0.70))

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
