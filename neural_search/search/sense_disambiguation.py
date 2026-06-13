"""Query sense disambiguation for overloaded scientific terms.

This module addresses a critical problem in neuroscience dataset search:
many terms have multiple scientific meanings that require different datasets.

Example: "delay" can mean:
- delay discounting (intertemporal choice, reward valuation)
- motor delay (delayed reach, instructed delay period)
- signal delay (propagation delay, conduction latency)
- working memory delay (delay period maintenance)

A lexical search for "delay" would return all of these mixed together.
This module identifies the intended sense and penalizes mismatches.

Usage:
    from neural_search.search.sense_disambiguation import disambiguate_query

    result = disambiguate_query("delay discounting datasets with choices")
    print(result.primary_sense)  # "delay_discounting"
    print(result.negative_senses)  # ["motor_delay", "signal_delay", ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# =============================================================================
# Sense Definitions
# =============================================================================

class SenseCategory(StrEnum):
    """Categories of scientific sense disambiguation."""

    DELAY = "delay"
    REWARD = "reward"
    CHOICE = "choice"
    MEMORY = "memory"
    SPIKE = "spike"
    SIGNAL = "signal"
    MODEL = "model"


@dataclass
class SenseDefinition:
    """Definition of a specific sense for an overloaded term."""

    sense_id: str
    category: str
    label: str
    description: str

    # Terms that indicate this sense
    positive_terms: list[str] = field(default_factory=list)

    # Context words that must appear for this sense
    required_context_any: list[str] = field(default_factory=list)

    # Terms that contradict this sense
    negative_terms: list[str] = field(default_factory=list)

    # Other senses that are mutually exclusive
    exclusive_senses: list[str] = field(default_factory=list)

    # Associated affordances
    associated_affordances: list[str] = field(default_factory=list)

    # Associated tasks
    associated_tasks: list[str] = field(default_factory=list)


@dataclass
class DisambiguationResult:
    """Result of query sense disambiguation."""

    query: str
    detected_category: str | None
    primary_sense: str | None
    primary_confidence: float
    secondary_senses: list[str]
    negative_senses: list[str]
    matched_positive_terms: list[str]
    matched_negative_terms: list[str]
    context_matches: list[str]
    disambiguation_notes: str


# =============================================================================
# Built-in Sense Definitions
# =============================================================================

DELAY_SENSES = {
    "delay_discounting": SenseDefinition(
        sense_id="delay_discounting",
        category="delay",
        label="Delay discounting / Intertemporal choice",
        description="Temporal discounting of reward value, choosing between immediate and delayed rewards",
        positive_terms=[
            "delay discounting",
            "temporal discounting",
            "intertemporal choice",
            "delayed reward",
            "immediate reward",
            "impulsive choice",
            "impulsivity",
            "discounting rate",
            "hyperbolic discounting",
            "k parameter",
            "discount factor",
            "sooner smaller",
            "larger later",
            "delay aversion",
            "time preference",
        ],
        required_context_any=[
            "reward",
            "choice",
            "value",
            "magnitude",
            "decision",
            "prefer",
            "option",
            "impuls",
        ],
        negative_terms=[
            "reach",
            "grasp",
            "motor",
            "movement",
            "propagation",
            "conduction",
            "latency",
            "synaptic",
            "working memory",
            "maintenance",
            "retention",
        ],
        exclusive_senses=["motor_delay", "signal_delay", "working_memory_delay"],
        associated_affordances=["delay_discounting_modeling", "q_learning"],
        associated_tasks=["delay_discounting", "intertemporal_choice", "impulsivity"],
    ),
    "motor_delay": SenseDefinition(
        sense_id="motor_delay",
        category="delay",
        label="Motor delay / Instructed delay",
        description="Delay period in motor tasks, delayed reach, movement preparation",
        positive_terms=[
            "delayed reach",
            "instructed delay",
            "motor planning",
            "movement preparation",
            "delay period activity",
            "preparatory activity",
            "reach to grasp",
            "delayed saccade",
            "go cue",
            "hold period",
        ],
        required_context_any=[
            "reach",
            "grasp",
            "motor",
            "movement",
            "saccade",
            "arm",
            "hand",
            "target",
        ],
        negative_terms=[
            "discount",
            "value",
            "impulsive",
            "magnitude",
            "propagation",
            "memory",
        ],
        exclusive_senses=["delay_discounting", "signal_delay"],
        associated_affordances=["motor_decoding", "choice_decoding"],
        associated_tasks=["delayed_reach", "center_out", "instructed_delay"],
    ),
    "signal_delay": SenseDefinition(
        sense_id="signal_delay",
        category="delay",
        label="Signal propagation delay",
        description="Neural signal propagation time, conduction delay, latency",
        positive_terms=[
            "propagation delay",
            "conduction delay",
            "signal latency",
            "transmission delay",
            "synaptic delay",
            "axonal delay",
            "response latency",
            "onset latency",
        ],
        required_context_any=[
            "propagat",
            "conduct",
            "latenc",
            "transmis",
            "synap",
            "axon",
        ],
        negative_terms=[
            "discount",
            "choice",
            "reward",
            "motor",
            "reach",
            "memory",
        ],
        exclusive_senses=["delay_discounting", "motor_delay"],
        associated_affordances=["functional_connectivity", "cross_area_interaction"],
        associated_tasks=[],
    ),
    "working_memory_delay": SenseDefinition(
        sense_id="working_memory_delay",
        category="delay",
        label="Working memory delay period",
        description="Delay period in working memory tasks, maintenance, retention",
        positive_terms=[
            "delay period",
            "working memory",
            "maintenance",
            "retention interval",
            "memory delay",
            "delay match",
            "delayed match to sample",
            "dmts",
            "dms task",
        ],
        required_context_any=[
            "memory",
            "maintain",
            "retain",
            "sample",
            "match",
            "remember",
        ],
        negative_terms=[
            "discount",
            "reward",
            "motor",
            "propagation",
        ],
        exclusive_senses=["delay_discounting", "signal_delay"],
        associated_affordances=["trial_aligned_neural_analysis"],
        associated_tasks=["working_memory", "delayed_match_to_sample"],
    ),
}

REWARD_SENSES = {
    "reward_value": SenseDefinition(
        sense_id="reward_value",
        category="reward",
        label="Reward value / Decision value",
        description="Subjective value of rewards for decision making",
        positive_terms=[
            "reward value",
            "subjective value",
            "decision value",
            "expected value",
            "reward prediction",
            "value signal",
            "value representation",
        ],
        required_context_any=["value", "decision", "choice", "expect"],
        negative_terms=["reward delivery", "juice", "water", "pellet"],
        exclusive_senses=["reward_delivery"],
        associated_affordances=["q_learning", "delay_discounting_modeling"],
        associated_tasks=["value_based_decision"],
    ),
    "reward_delivery": SenseDefinition(
        sense_id="reward_delivery",
        category="reward",
        label="Reward delivery / Consummatory",
        description="Physical reward delivery, consumption, reinforcement",
        positive_terms=[
            "reward delivery",
            "juice reward",
            "water reward",
            "food reward",
            "pellet",
            "sucrose",
            "reward consumption",
        ],
        required_context_any=["deliver", "juice", "water", "food", "lick", "consum"],
        negative_terms=["value", "subjective", "expected"],
        exclusive_senses=["reward_value"],
        associated_affordances=["q_learning"],
        associated_tasks=["reinforcement_learning"],
    ),
}

MEMORY_SENSES = {
    "working_memory": SenseDefinition(
        sense_id="working_memory",
        category="memory",
        label="Working memory",
        description="Short-term maintenance of information for task performance",
        positive_terms=[
            "working memory",
            "short term memory",
            "maintenance",
            "delay period",
            "memory load",
            "memory capacity",
            "n-back",
            "delayed match",
        ],
        required_context_any=["maintain", "load", "capacity", "delay", "short"],
        negative_terms=["episodic", "hippocampus", "long term", "consolidation"],
        exclusive_senses=["episodic_memory", "spatial_memory"],
        associated_affordances=["trial_aligned_neural_analysis"],
        associated_tasks=["working_memory", "n_back"],
    ),
    "episodic_memory": SenseDefinition(
        sense_id="episodic_memory",
        category="memory",
        label="Episodic memory",
        description="Memory for specific events, hippocampal memory",
        positive_terms=[
            "episodic memory",
            "hippocampal memory",
            "memory encoding",
            "memory retrieval",
            "recall",
            "recognition memory",
            "autobiographical",
        ],
        required_context_any=["hippocamp", "encod", "retriev", "recall", "episod"],
        negative_terms=["working", "maintenance", "delay period", "short term"],
        exclusive_senses=["working_memory"],
        associated_affordances=["trial_aligned_neural_analysis"],
        associated_tasks=["episodic_memory", "recognition"],
    ),
    "spatial_memory": SenseDefinition(
        sense_id="spatial_memory",
        category="memory",
        label="Spatial memory / Navigation",
        description="Memory for spatial locations, place cells, navigation",
        positive_terms=[
            "spatial memory",
            "place cell",
            "grid cell",
            "navigation",
            "spatial map",
            "cognitive map",
            "head direction",
            "path integration",
        ],
        required_context_any=["place", "grid", "navigat", "spatial", "locat"],
        negative_terms=["working", "episodic", "verbal"],
        exclusive_senses=["working_memory"],
        associated_affordances=["pose_neural_correlation"],
        associated_tasks=["spatial_navigation", "foraging"],
    ),
}

# Combine all senses
ALL_SENSES: dict[str, SenseDefinition] = {
    **DELAY_SENSES,
    **REWARD_SENSES,
    **MEMORY_SENSES,
}


# =============================================================================
# Disambiguation Logic
# =============================================================================

def _normalize_query(query: str) -> str:
    """Normalize query for matching."""
    return query.lower().strip()


def _count_term_matches(text: str, terms: list[str]) -> tuple[int, list[str]]:
    """Count how many terms appear in text and return matched terms."""
    text_lower = text.lower()
    matched = []
    for term in terms:
        if term.lower() in text_lower:
            matched.append(term)
    return len(matched), matched


def _check_context(text: str, context_terms: list[str]) -> tuple[bool, list[str]]:
    """Check if any context terms appear in text."""
    text_lower = text.lower()
    matched = []
    for term in context_terms:
        # Allow partial matches for context (e.g., "impuls" matches "impulsive")
        if term.lower() in text_lower:
            matched.append(term)
    return len(matched) > 0, matched


def score_sense(query: str, sense: SenseDefinition) -> tuple[float, dict[str, Any]]:
    """Score how well a query matches a sense definition.

    Returns:
        Tuple of (score, details dict with matched terms)
    """
    query_lower = _normalize_query(query)

    # Count positive term matches
    pos_count, pos_matched = _count_term_matches(query_lower, sense.positive_terms)

    # Check context requirements
    context_ok, context_matched = _check_context(query_lower, sense.required_context_any)

    # Count negative term matches (penalties)
    neg_count, neg_matched = _count_term_matches(query_lower, sense.negative_terms)

    # Scoring:
    # - Each positive match: +0.3
    # - Context match bonus: +0.2
    # - Each negative match: -0.4
    score = (pos_count * 0.3) + (0.2 if context_ok else 0.0) - (neg_count * 0.4)

    # Normalize to 0-1 range
    score = max(0.0, min(1.0, score))

    # Boost if we have strong positive evidence
    if pos_count >= 2:
        score = min(1.0, score + 0.2)

    details = {
        "positive_matches": pos_matched,
        "context_matches": context_matched,
        "negative_matches": neg_matched,
        "context_satisfied": context_ok,
    }

    return score, details


def disambiguate_query(query: str) -> DisambiguationResult:
    """Disambiguate a query to identify scientific sense.

    This is the main entry point for sense disambiguation. Given a query,
    it identifies:
    - The most likely intended sense
    - Confidence in that sense
    - Alternative senses that might apply
    - Negative senses that should be penalized in retrieval

    Args:
        query: The search query to disambiguate

    Returns:
        DisambiguationResult with primary sense, alternatives, and negatives
    """
    # Score all senses
    sense_scores: list[tuple[str, float, dict[str, Any]]] = []

    for sense_id, sense in ALL_SENSES.items():
        score, details = score_sense(query, sense)
        if score > 0:
            sense_scores.append((sense_id, score, details))

    # Sort by score descending
    sense_scores.sort(key=lambda x: x[1], reverse=True)

    # Determine results
    if not sense_scores:
        return DisambiguationResult(
            query=query,
            detected_category=None,
            primary_sense=None,
            primary_confidence=0.0,
            secondary_senses=[],
            negative_senses=[],
            matched_positive_terms=[],
            matched_negative_terms=[],
            context_matches=[],
            disambiguation_notes="No specific sense detected",
        )

    primary_sense_id, primary_score, primary_details = sense_scores[0]
    primary_sense = ALL_SENSES[primary_sense_id]

    # Get secondary senses (score > 0.1 but not primary)
    secondary = [s[0] for s in sense_scores[1:] if s[1] > 0.1]

    # Get negative senses (exclusive to primary)
    negative_senses = primary_sense.exclusive_senses.copy()

    # Also add senses that had negative matches
    for sense_id, _score, details in sense_scores:
        if details.get("negative_matches") and sense_id != primary_sense_id:
            if sense_id not in negative_senses:
                negative_senses.append(sense_id)

    return DisambiguationResult(
        query=query,
        detected_category=primary_sense.category,
        primary_sense=primary_sense_id,
        primary_confidence=primary_score,
        secondary_senses=secondary,
        negative_senses=negative_senses,
        matched_positive_terms=primary_details.get("positive_matches", []),
        matched_negative_terms=primary_details.get("negative_matches", []),
        context_matches=primary_details.get("context_matches", []),
        disambiguation_notes=f"Detected {primary_sense.label} with confidence {primary_score:.2f}",
    )


def get_sense_penalties(
    result: DisambiguationResult,
) -> dict[str, float]:
    """Get penalty scores for senses that should be downweighted.

    Returns a dict mapping sense_id to penalty (0.0 = no penalty, 1.0 = full penalty)
    """
    penalties = {}

    for sense_id in result.negative_senses:
        # Full penalty for explicitly negative senses
        penalties[sense_id] = 0.8

    return penalties


def get_associated_affordances(result: DisambiguationResult) -> list[str]:
    """Get affordances associated with the detected sense."""
    if not result.primary_sense:
        return []

    sense = ALL_SENSES.get(result.primary_sense)
    if not sense:
        return []

    return sense.associated_affordances


def get_associated_tasks(result: DisambiguationResult) -> list[str]:
    """Get tasks associated with the detected sense."""
    if not result.primary_sense:
        return []

    sense = ALL_SENSES.get(result.primary_sense)
    if not sense:
        return []

    return sense.associated_tasks


# =============================================================================
# YAML Configuration Loading
# =============================================================================

@lru_cache(maxsize=1)
def load_sense_config(config_path: str | Path | None = None) -> dict[str, SenseDefinition]:
    """Load sense definitions from YAML config.

    If no path provided, returns built-in senses.
    """
    if config_path is None:
        return ALL_SENSES

    path = Path(config_path)
    if not path.exists():
        return ALL_SENSES

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse YAML into SenseDefinition objects
    senses = {}
    for category, category_senses in config.items():
        if not isinstance(category_senses, dict):
            continue
        for sense_id, sense_data in category_senses.items():
            if not isinstance(sense_data, dict):
                continue
            senses[sense_id] = SenseDefinition(
                sense_id=sense_id,
                category=category,
                label=sense_data.get("label", sense_id),
                description=sense_data.get("description", ""),
                positive_terms=sense_data.get("positive_terms", []),
                required_context_any=sense_data.get("required_context_any", []),
                negative_terms=sense_data.get("negative_terms", []),
                exclusive_senses=sense_data.get("exclusive_senses", []),
                associated_affordances=sense_data.get("associated_affordances", []),
                associated_tasks=sense_data.get("associated_tasks", []),
            )

    return senses


def list_senses() -> list[str]:
    """List all registered sense IDs."""
    return list(ALL_SENSES.keys())


def get_sense(sense_id: str) -> SenseDefinition | None:
    """Get a sense definition by ID."""
    return ALL_SENSES.get(sense_id)


def get_senses_by_category(category: str) -> list[SenseDefinition]:
    """Get all senses in a category."""
    return [s for s in ALL_SENSES.values() if s.category == category]
