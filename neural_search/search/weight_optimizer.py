"""Adaptive weight optimization for hybrid search."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from neural_search.search.intent import QueryIntent


class QueryComplexity(Enum):
    """Query complexity levels affect weight distributions."""

    SIMPLE = "simple"  # Single constraint (e.g., "reversal learning")
    MODERATE = "moderate"  # 2-3 constraints
    COMPLEX = "complex"  # 4+ constraints or graph reasoning needed


@dataclass
class WeightProfile:
    """A weight configuration with metadata."""

    name: str
    weights: dict[str, float]
    description: str = ""
    applicable_intents: list[QueryIntent] = field(default_factory=list)
    complexity_bias: QueryComplexity | None = None


# Base weight profiles for different search scenarios
WEIGHT_PROFILES: dict[str, WeightProfile] = {
    "balanced": WeightProfile(
        name="balanced",
        weights={
            "ontology": 0.25,
            "behavior": 0.18,
            "modality": 0.12,
            "affordance": 0.10,
            "metadata": 0.08,
            "semantic": 0.08,
            "field_semantic": 0.06,
            "graph": 0.04,
            "readiness": 0.06,
            "paper_confidence": 0.03,
        },
        description="Balanced weights for general queries",
    ),
    "task_focused": WeightProfile(
        name="task_focused",
        weights={
            "ontology": 0.32,
            "behavior": 0.20,
            "modality": 0.10,
            "affordance": 0.08,
            "metadata": 0.06,
            "semantic": 0.06,
            "field_semantic": 0.05,
            "graph": 0.04,
            "readiness": 0.06,
            "paper_confidence": 0.03,
        },
        description="Emphasize task/behavior matching for behavioral queries",
        applicable_intents=[QueryIntent.TASK_SEARCH],
    ),
    "analysis_focused": WeightProfile(
        name="analysis_focused",
        weights={
            "ontology": 0.18,
            "behavior": 0.12,
            "modality": 0.10,
            "affordance": 0.18,
            "metadata": 0.06,
            "semantic": 0.06,
            "field_semantic": 0.08,
            "graph": 0.06,
            "readiness": 0.12,
            "paper_confidence": 0.04,
        },
        description="Emphasize affordances and readiness for analysis queries",
        applicable_intents=[QueryIntent.ANALYSIS_SEARCH],
    ),
    "graph_reasoning": WeightProfile(
        name="graph_reasoning",
        weights={
            "ontology": 0.20,
            "behavior": 0.12,
            "modality": 0.10,
            "affordance": 0.10,
            "metadata": 0.06,
            "semantic": 0.08,
            "field_semantic": 0.06,
            "graph": 0.14,
            "readiness": 0.06,
            "paper_confidence": 0.08,
        },
        description="Emphasize graph context for relationship queries",
        applicable_intents=[QueryIntent.GRAPH_REASONING],
    ),
    "paper_lookup": WeightProfile(
        name="paper_lookup",
        weights={
            "ontology": 0.15,
            "behavior": 0.10,
            "modality": 0.08,
            "affordance": 0.08,
            "metadata": 0.06,
            "semantic": 0.12,
            "field_semantic": 0.08,
            "graph": 0.10,
            "readiness": 0.06,
            "paper_confidence": 0.17,
        },
        description="Emphasize paper links and semantic matching",
        applicable_intents=[QueryIntent.PAPER_LINK],
    ),
    "dataset_lookup": WeightProfile(
        name="dataset_lookup",
        weights={
            "ontology": 0.12,
            "behavior": 0.08,
            "modality": 0.08,
            "affordance": 0.06,
            "metadata": 0.25,
            "semantic": 0.18,
            "field_semantic": 0.10,
            "graph": 0.04,
            "readiness": 0.05,
            "paper_confidence": 0.04,
        },
        description="Emphasize metadata for direct dataset lookups",
        applicable_intents=[QueryIntent.DATASET_LOOKUP],
    ),
    "semantic_heavy": WeightProfile(
        name="semantic_heavy",
        weights={
            "ontology": 0.15,
            "behavior": 0.10,
            "modality": 0.08,
            "affordance": 0.08,
            "metadata": 0.06,
            "semantic": 0.18,
            "field_semantic": 0.20,
            "graph": 0.06,
            "readiness": 0.05,
            "paper_confidence": 0.04,
        },
        description="Emphasize semantic similarity for fuzzy queries",
        complexity_bias=QueryComplexity.SIMPLE,
    ),
}


@dataclass
class QueryAnalysis:
    """Analysis of query characteristics for weight selection."""

    complexity: QueryComplexity
    primary_intent: QueryIntent
    constraint_count: int
    has_task_constraints: bool
    has_modality_constraints: bool
    has_affordance_constraints: bool
    has_species_constraints: bool
    has_region_constraints: bool
    has_graph_patterns: bool
    suggested_profile: str


def analyze_query_for_weights(
    parsed_query: dict[str, Any],
    intent: QueryIntent | None = None,
) -> QueryAnalysis:
    """Analyze query to determine optimal weight profile.

    Args:
        parsed_query: Parsed query dict from parse_query()
        intent: Optional pre-classified intent

    Returns:
        QueryAnalysis with profile recommendation
    """
    # Count constraints
    tasks = parsed_query.get("tasks", [])
    modalities = parsed_query.get("modalities", [])
    affordances = parsed_query.get("affordances", [])
    species = parsed_query.get("species", [])
    regions = parsed_query.get("brain_regions", [])

    constraint_count = (
        (1 if tasks else 0)
        + (1 if modalities else 0)
        + (1 if affordances else 0)
        + (1 if species else 0)
        + (1 if regions else 0)
    )

    # Determine complexity
    if constraint_count <= 1:
        complexity = QueryComplexity.SIMPLE
    elif constraint_count <= 3:
        complexity = QueryComplexity.MODERATE
    else:
        complexity = QueryComplexity.COMPLEX

    # Check for graph patterns
    query_intent_info = parsed_query.get("query_intent", {})
    has_graph_patterns = query_intent_info.get("primary") == "graph_reasoning"

    # Determine primary intent
    if intent is None:
        intent_str = query_intent_info.get("primary", "task_search")
        try:
            primary_intent = QueryIntent(intent_str)
        except ValueError:
            primary_intent = QueryIntent.TASK_SEARCH
    else:
        primary_intent = intent

    # Select profile
    if primary_intent == QueryIntent.DATASET_LOOKUP:
        suggested_profile = "dataset_lookup"
    elif primary_intent == QueryIntent.PAPER_LINK:
        suggested_profile = "paper_lookup"
    elif primary_intent == QueryIntent.GRAPH_REASONING or has_graph_patterns:
        suggested_profile = "graph_reasoning"
    elif primary_intent == QueryIntent.ANALYSIS_SEARCH or affordances:
        suggested_profile = "analysis_focused"
    elif primary_intent == QueryIntent.TASK_SEARCH and tasks:
        suggested_profile = "task_focused"
    elif complexity == QueryComplexity.SIMPLE:
        suggested_profile = "semantic_heavy"
    else:
        suggested_profile = "balanced"

    return QueryAnalysis(
        complexity=complexity,
        primary_intent=primary_intent,
        constraint_count=constraint_count,
        has_task_constraints=bool(tasks),
        has_modality_constraints=bool(modalities),
        has_affordance_constraints=bool(affordances),
        has_species_constraints=bool(species),
        has_region_constraints=bool(regions),
        has_graph_patterns=has_graph_patterns,
        suggested_profile=suggested_profile,
    )


def get_adaptive_weights(
    parsed_query: dict[str, Any],
    base_weights: dict[str, float] | None = None,
    intent: QueryIntent | None = None,
) -> dict[str, float]:
    """Get optimized weights based on query analysis.

    Args:
        parsed_query: Parsed query dict
        base_weights: Optional base weights to blend with
        intent: Optional pre-classified intent

    Returns:
        Optimized weight dict
    """
    analysis = analyze_query_for_weights(parsed_query, intent)
    profile = WEIGHT_PROFILES.get(analysis.suggested_profile, WEIGHT_PROFILES["balanced"])

    if base_weights is None:
        return dict(profile.weights)

    # Blend base weights with profile weights
    # Use 60% profile, 40% base for adaptive behavior
    blend_factor = 0.6
    result: dict[str, float] = {}

    all_keys = set(base_weights.keys()) | set(profile.weights.keys())
    for key in all_keys:
        base_val = base_weights.get(key, 0.0)
        profile_val = profile.weights.get(key, base_val)
        result[key] = base_val * (1 - blend_factor) + profile_val * blend_factor

    return result


@dataclass
class WeightSensitivity:
    """Sensitivity analysis for a weight parameter."""

    weight_name: str
    current_value: float
    low_impact: float  # Score change with weight -50%
    high_impact: float  # Score change with weight +50%
    sensitivity_score: float  # Absolute average impact


def compute_weight_sensitivity(
    weight_name: str,
    current_weights: dict[str, float],
    score_fn: callable,  # type: ignore
    delta: float = 0.5,
) -> WeightSensitivity:
    """Compute sensitivity of final score to a weight change.

    Args:
        weight_name: Weight to analyze
        current_weights: Current weight dict
        score_fn: Function(weights) -> score to evaluate
        delta: Fractional change to test (0.5 = +/- 50%)

    Returns:
        WeightSensitivity analysis
    """
    current_val = current_weights.get(weight_name, 0.0)
    base_score = score_fn(current_weights)

    # Test low value
    low_weights = dict(current_weights)
    low_weights[weight_name] = current_val * (1 - delta)
    low_score = score_fn(low_weights)
    low_impact = low_score - base_score

    # Test high value
    high_weights = dict(current_weights)
    high_weights[weight_name] = current_val * (1 + delta)
    high_score = score_fn(high_weights)
    high_impact = high_score - base_score

    sensitivity = (abs(low_impact) + abs(high_impact)) / 2

    return WeightSensitivity(
        weight_name=weight_name,
        current_value=current_val,
        low_impact=low_impact,
        high_impact=high_impact,
        sensitivity_score=sensitivity,
    )


def analyze_weight_sensitivity(
    weights: dict[str, float],
    score_fn: callable,  # type: ignore
    delta: float = 0.5,
) -> list[WeightSensitivity]:
    """Analyze sensitivity of all weights.

    Args:
        weights: Weight dict to analyze
        score_fn: Function(weights) -> score to evaluate
        delta: Fractional change to test

    Returns:
        List of WeightSensitivity sorted by impact
    """
    results = []
    for name in weights:
        sensitivity = compute_weight_sensitivity(name, weights, score_fn, delta)
        results.append(sensitivity)

    # Sort by sensitivity (most sensitive first)
    return sorted(results, key=lambda s: s.sensitivity_score, reverse=True)


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize weights to sum to 1.0.

    Args:
        weights: Weight dict

    Returns:
        Normalized weight dict
    """
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def interpolate_weights(
    weights_a: dict[str, float],
    weights_b: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    """Interpolate between two weight configurations.

    Args:
        weights_a: First weight dict
        weights_b: Second weight dict
        alpha: Interpolation factor (0 = a, 1 = b)

    Returns:
        Interpolated weights
    """
    alpha = max(0.0, min(1.0, alpha))
    all_keys = set(weights_a.keys()) | set(weights_b.keys())
    result: dict[str, float] = {}

    for key in all_keys:
        val_a = weights_a.get(key, 0.0)
        val_b = weights_b.get(key, val_a)
        result[key] = val_a * (1 - alpha) + val_b * alpha

    return result


def boost_weights_for_constraints(
    base_weights: dict[str, float],
    parsed_query: dict[str, Any],
    boost_factor: float = 0.3,
) -> dict[str, float]:
    """Boost weights for constraint types present in query.

    Args:
        base_weights: Base weight dict
        parsed_query: Parsed query with constraints
        boost_factor: Fractional boost to apply

    Returns:
        Boosted weight dict
    """
    result = dict(base_weights)

    # Map constraint types to weight names
    constraint_weight_map = {
        "tasks": "ontology",
        "behaviors": "behavior",
        "modalities": "modality",
        "affordances": "affordance",
        "species": "metadata",
        "brain_regions": "metadata",
    }

    boosted_weights: set[str] = set()
    for constraint_key, weight_name in constraint_weight_map.items():
        if parsed_query.get(constraint_key):
            boosted_weights.add(weight_name)

    # Apply boosts
    for weight_name in boosted_weights:
        if weight_name in result:
            result[weight_name] *= 1 + boost_factor

    # Reduce unboosted weights proportionally to maintain sum ~ 1.0
    unboosted_sum = sum(result[k] for k in result if k not in boosted_weights)
    boosted_sum = sum(result[k] for k in boosted_weights)
    target_sum = sum(base_weights.values())

    if unboosted_sum > 0 and boosted_sum < target_sum:
        reduction = (boosted_sum + unboosted_sum - target_sum) / unboosted_sum
        for key in result:
            if key not in boosted_weights:
                result[key] *= max(0, 1 - reduction)

    return result
