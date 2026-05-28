"""Query Understanding and Retrieval Planning.

This module transforms natural language queries into structured retrieval plans.
The planner:

1. Classifies query intent (find datasets, link papers, propose experiments, etc.)
2. Extracts constraints (modality, species, task, region, analysis)
3. Determines which retrieval stages to use and how to weight them
4. Decides if graph expansion would help
5. Outputs a structured QueryPlan that the retrieval pipeline executes

Supported query intents:
- find_datasets: Search for datasets matching criteria
- find_papers: Search for papers matching criteria
- link_papers_to_datasets: Find dataset-paper connections
- find_analysis_datasets: Find datasets suitable for a specific analysis
- find_related_paradigms: Find related experimental paradigms
- find_methods_for_dataset: Find applicable methods
- find_similar_experiments: Find experiments like a reference
- find_multimodal_datasets: Find datasets with multiple modalities
- find_latent_modeling_data: Find data for latent-state modeling
- find_evidence_for_claim: Find supporting evidence
- propose_experiment: Suggest future experiment directions
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from neural_search.ontology import (
    expand_query_terms,
    match_affordances,
    match_behavior_labels,
    match_brain_regions,
    match_modalities,
    match_tasks,
    normalize_text,
)


class QueryIntent(StrEnum):
    """Primary intent categories for search queries."""

    FIND_DATASETS = "find_datasets"
    FIND_PAPERS = "find_papers"
    LINK_PAPERS_TO_DATASETS = "link_papers_to_datasets"
    FIND_ANALYSIS_DATASETS = "find_analysis_datasets"
    FIND_RELATED_PARADIGMS = "find_related_paradigms"
    FIND_METHODS_FOR_DATASET = "find_methods_for_dataset"
    FIND_SIMILAR_EXPERIMENTS = "find_similar_experiments"
    FIND_MULTIMODAL_DATASETS = "find_multimodal_datasets"
    FIND_LATENT_MODELING_DATA = "find_latent_modeling_data"
    FIND_EVIDENCE_FOR_CLAIM = "find_evidence_for_claim"
    PROPOSE_EXPERIMENT = "propose_experiment"
    EXPLORATORY = "exploratory"


class ConstraintType(StrEnum):
    """Types of query constraints."""

    SPECIES = "species"
    MODALITY = "modality"
    TASK = "task"
    BRAIN_REGION = "brain_region"
    BEHAVIOR = "behavior"
    ANALYSIS = "analysis"
    DATA_STANDARD = "data_standard"
    FILE_FORMAT = "file_format"
    SOURCE = "source"
    YEAR = "year"
    AUTHOR = "author"


class QueryConstraint(BaseModel):
    """A single constraint extracted from the query."""

    constraint_type: ConstraintType
    value: str
    required: bool = True        # Must match (vs. preferred)
    negated: bool = False        # NOT this value
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)
    source: str = "query"        # Where this constraint came from


class RetrievalStage(StrEnum):
    """Stages in the retrieval pipeline."""

    LEXICAL = "lexical"
    METADATA_FILTER = "metadata_filter"
    ONTOLOGY_MATCH = "ontology_match"
    EMBEDDING_SEARCH = "embedding_search"
    GRAPH_EXPANSION = "graph_expansion"
    PAPER_LINKING = "paper_linking"
    AFFORDANCE_MATCH = "affordance_match"
    RERANK = "rerank"
    EXPLAIN = "explain"


class StageConfig(BaseModel):
    """Configuration for a retrieval stage."""

    stage: RetrievalStage
    enabled: bool = True
    weight: float = Field(ge=0.0, le=1.0, default=0.1)
    top_k: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class QueryPlan(BaseModel):
    """A structured retrieval plan output by the query planner.

    This is the primary output of query understanding. It tells the
    retrieval pipeline exactly what to do and how.
    """

    # Original query
    query_text: str
    normalized_query: str

    # Intent classification
    primary_intent: QueryIntent
    intent_confidence: float = Field(ge=0.0, le=1.0)
    secondary_intents: list[QueryIntent] = Field(default_factory=list)

    # Target object type
    target_record_type: Literal["dataset", "paper", "any"] = "dataset"

    # Extracted constraints
    constraints: list[QueryConstraint] = Field(default_factory=list)
    required_modalities: list[str] = Field(default_factory=list)
    required_species: list[str] = Field(default_factory=list)
    required_tasks: list[str] = Field(default_factory=list)
    required_regions: list[str] = Field(default_factory=list)
    required_analyses: list[str] = Field(default_factory=list)

    # Exclusions
    excluded_modalities: list[str] = Field(default_factory=list)
    excluded_species: list[str] = Field(default_factory=list)
    excluded_tasks: list[str] = Field(default_factory=list)
    excluded_sources: list[str] = Field(default_factory=list)

    # Retrieval configuration
    stages: list[StageConfig] = Field(default_factory=list)
    use_graph_expansion: bool = False
    graph_expansion_hops: int = 2
    use_paper_linking: bool = False

    # Scoring weights (overrides defaults based on intent)
    weight_overrides: dict[str, float] = Field(default_factory=dict)

    # Query understanding metadata
    detected_keywords: list[str] = Field(default_factory=list)
    ontology_matches: dict[str, list[str]] = Field(default_factory=dict)
    expanded_terms: list[str] = Field(default_factory=list)

    # Quality signals
    is_precise_query: bool = True       # vs. exploratory
    needs_explanation: bool = True
    uncertainty_high: bool = False

    # Planning metadata
    planner_version: str = "v0.5.0"
    planning_notes: list[str] = Field(default_factory=list)

    def get_stage_weight(self, stage: RetrievalStage) -> float:
        """Get the weight for a specific stage."""
        for s in self.stages:
            if s.stage == stage:
                return s.weight
        return 0.0

    def is_stage_enabled(self, stage: RetrievalStage) -> bool:
        """Check if a stage is enabled."""
        for s in self.stages:
            if s.stage == stage:
                return s.enabled
        return False


# Intent detection patterns
INTENT_PATTERNS: dict[QueryIntent, list[str]] = {
    QueryIntent.FIND_DATASETS: [
        r"find\s+datasets?",
        r"search\s+for\s+datasets?",
        r"datasets?\s+(?:with|for|containing)",
        r"looking\s+for\s+(?:a\s+)?dataset",
        r"neural\s+data\s+(?:with|for)",
    ],
    QueryIntent.FIND_PAPERS: [
        r"find\s+papers?",
        r"search\s+for\s+papers?",
        r"papers?\s+(?:about|on|studying)",
        r"publications?\s+(?:about|on)",
    ],
    QueryIntent.LINK_PAPERS_TO_DATASETS: [
        r"link\s+papers?\s+to\s+datasets?",
        r"papers?\s+(?:that\s+)?use[sd]?\s+(?:this\s+)?dataset",
        r"datasets?\s+used\s+in\s+papers?",
        r"publications?\s+for\s+(?:this\s+)?dataset",
    ],
    QueryIntent.FIND_ANALYSIS_DATASETS: [
        r"datasets?\s+(?:for|suitable\s+for)\s+(?:a\s+)?(?:an\s+)?analysis",
        r"data\s+(?:for|suitable\s+for)\s+(?:doing\s+)?(?:a\s+)?(?:an\s+)?",
        r"suitable\s+for\s+(?:spike\s+sorting|decoding|classification|modeling)",
        r"(?:can\s+)?(?:do|run|perform)\s+(?:a\s+)?(?:an\s+)?(?:\w+\s+)?analysis\s+on",
    ],
    QueryIntent.FIND_LATENT_MODELING_DATA: [
        r"latent[\s-]?state",
        r"latent[\s-]?variable",
        r"hidden[\s-]?state",
        r"dimensionality\s+reduction",
        r"neural\s+(?:population|ensemble)\s+(?:dynamics|activity)",
        r"trial[\s-]?(?:by[\s-]?trial|averaged?)\s+(?:analysis|decoding)",
    ],
    QueryIntent.FIND_MULTIMODAL_DATASETS: [
        r"multimodal",
        r"multi[\s-]?modal",
        r"multiple\s+modalities",
        r"(?:neural\s+)?(?:and|with)\s+behavior",
        r"combined\s+(?:ephys|electrophysiology|imaging)\s+(?:and|with)",
    ],
    QueryIntent.FIND_SIMILAR_EXPERIMENTS: [
        r"similar\s+(?:to|experiments?|datasets?)",
        r"like\s+(?:this|that)\s+(?:experiment|dataset)",
        r"experiments?\s+similar",
    ],
    QueryIntent.FIND_EVIDENCE_FOR_CLAIM: [
        r"evidence\s+(?:for|that)",
        r"support(?:s|ing)?\s+(?:the\s+)?(?:claim|hypothesis)",
        r"datasets?\s+(?:that\s+)?show",
    ],
    QueryIntent.PROPOSE_EXPERIMENT: [
        r"propose\s+(?:a\s+)?(?:an\s+)?experiment",
        r"what\s+(?:experiment|analysis)\s+(?:could|should|can)",
        r"future\s+(?:experiment|research|direction)",
        r"next\s+steps?\s+(?:for|with)",
    ],
}

# Analysis keywords that indicate specific intents
ANALYSIS_KEYWORDS: dict[str, QueryIntent] = {
    "spike sorting": QueryIntent.FIND_ANALYSIS_DATASETS,
    "decoding": QueryIntent.FIND_ANALYSIS_DATASETS,
    "classification": QueryIntent.FIND_ANALYSIS_DATASETS,
    "latent state": QueryIntent.FIND_LATENT_MODELING_DATA,
    "latent variable": QueryIntent.FIND_LATENT_MODELING_DATA,
    "dimensionality reduction": QueryIntent.FIND_LATENT_MODELING_DATA,
    "neural dynamics": QueryIntent.FIND_LATENT_MODELING_DATA,
    "trial aligned": QueryIntent.FIND_LATENT_MODELING_DATA,
    "pose estimation": QueryIntent.FIND_ANALYSIS_DATASETS,
    "behavior segmentation": QueryIntent.FIND_ANALYSIS_DATASETS,
}

# Default stage weights by intent
INTENT_STAGE_WEIGHTS: dict[QueryIntent, dict[str, float]] = {
    QueryIntent.FIND_DATASETS: {
        "ontology": 0.25,
        "behavior": 0.18,
        "modality": 0.15,
        "affordance": 0.12,
        "metadata": 0.10,
        "semantic": 0.08,
        "graph": 0.06,
        "readiness": 0.06,
    },
    QueryIntent.FIND_ANALYSIS_DATASETS: {
        "affordance": 0.28,
        "ontology": 0.20,
        "behavior": 0.15,
        "modality": 0.12,
        "readiness": 0.10,
        "metadata": 0.08,
        "graph": 0.04,
        "semantic": 0.03,
    },
    QueryIntent.FIND_LATENT_MODELING_DATA: {
        "affordance": 0.30,
        "behavior": 0.20,
        "ontology": 0.15,
        "modality": 0.12,
        "readiness": 0.10,
        "metadata": 0.08,
        "graph": 0.05,
    },
    QueryIntent.LINK_PAPERS_TO_DATASETS: {
        "graph": 0.30,
        "paper_confidence": 0.25,
        "ontology": 0.15,
        "semantic": 0.12,
        "modality": 0.08,
        "metadata": 0.05,
        "behavior": 0.05,
    },
    QueryIntent.FIND_PAPERS: {
        "semantic": 0.25,
        "ontology": 0.20,
        "graph": 0.15,
        "metadata": 0.15,
        "modality": 0.10,
        "behavior": 0.10,
        "affordance": 0.05,
    },
}


def _classify_intent(query: str) -> tuple[QueryIntent, float, list[QueryIntent]]:
    """Classify the primary intent of a query."""
    normalized = normalize_text(query)
    scores: dict[QueryIntent, float] = {}

    # Check regex patterns
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                scores[intent] = scores.get(intent, 0) + 0.3

    # Check analysis keywords
    for keyword, intent in ANALYSIS_KEYWORDS.items():
        if normalize_text(keyword) in normalized:
            scores[intent] = scores.get(intent, 0) + 0.4

    # Default to find_datasets if no strong signal
    if not scores:
        return QueryIntent.FIND_DATASETS, 0.5, []

    # Sort by score
    sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_intents[0][0]
    confidence = min(sorted_intents[0][1], 1.0)

    # Secondary intents
    secondary = [
        intent for intent, score in sorted_intents[1:3]
        if score >= 0.2
    ]

    return primary, confidence, secondary


def _extract_constraints(
    query: str,
    ontology_matches: dict[str, list[Any]],
) -> list[QueryConstraint]:
    """Extract constraints from query text and ontology matches."""
    constraints: list[QueryConstraint] = []
    normalized = normalize_text(query)

    # Extract from ontology matches
    for match in ontology_matches.get("tasks", []):
        constraints.append(QueryConstraint(
            constraint_type=ConstraintType.TASK,
            value=match.id if hasattr(match, "id") else str(match),
            confidence=match.confidence if hasattr(match, "confidence") else 0.9,
        ))

    for match in ontology_matches.get("modalities", []):
        constraints.append(QueryConstraint(
            constraint_type=ConstraintType.MODALITY,
            value=match.id if hasattr(match, "id") else str(match),
            confidence=match.confidence if hasattr(match, "confidence") else 0.9,
        ))

    for match in ontology_matches.get("regions", []):
        constraints.append(QueryConstraint(
            constraint_type=ConstraintType.BRAIN_REGION,
            value=match.id if hasattr(match, "id") else str(match),
            confidence=match.confidence if hasattr(match, "confidence") else 0.9,
        ))

    for match in ontology_matches.get("behaviors", []):
        constraints.append(QueryConstraint(
            constraint_type=ConstraintType.BEHAVIOR,
            value=match.id if hasattr(match, "id") else str(match),
            confidence=match.confidence if hasattr(match, "confidence") else 0.9,
        ))

    for match in ontology_matches.get("affordances", []):
        constraints.append(QueryConstraint(
            constraint_type=ConstraintType.ANALYSIS,
            value=match.id if hasattr(match, "id") else str(match),
            confidence=match.confidence if hasattr(match, "confidence") else 0.9,
        ))

    # Detect negations
    negation_patterns = [
        r"not\s+(\w+)",
        r"without\s+(\w+)",
        r"excluding\s+(\w+)",
        r"no\s+(\w+)\s+(?:data|recordings?)",
    ]

    for pattern in negation_patterns:
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            term = match.group(1).strip()
            # Add negated constraint
            constraints.append(QueryConstraint(
                constraint_type=ConstraintType.MODALITY,  # Default to modality
                value=term,
                negated=True,
                confidence=0.85,
            ))

    return constraints


def _determine_stages(
    intent: QueryIntent,
    constraints: list[QueryConstraint],
) -> list[StageConfig]:
    """Determine which retrieval stages to use based on intent."""
    stages: list[StageConfig] = []

    # Always do ontology matching
    stages.append(StageConfig(
        stage=RetrievalStage.ONTOLOGY_MATCH,
        enabled=True,
        weight=INTENT_STAGE_WEIGHTS.get(intent, {}).get("ontology", 0.20),
    ))

    # Always do lexical search
    stages.append(StageConfig(
        stage=RetrievalStage.LEXICAL,
        enabled=True,
        weight=0.08,
    ))

    # Add embedding search
    stages.append(StageConfig(
        stage=RetrievalStage.EMBEDDING_SEARCH,
        enabled=True,
        weight=INTENT_STAGE_WEIGHTS.get(intent, {}).get("semantic", 0.10),
    ))

    # Add metadata filter if constraints present
    has_hard_constraints = any(c.required and not c.negated for c in constraints)
    stages.append(StageConfig(
        stage=RetrievalStage.METADATA_FILTER,
        enabled=has_hard_constraints,
        weight=0.15,
    ))

    # Add affordance matching for analysis-focused intents
    analysis_intents = {
        QueryIntent.FIND_ANALYSIS_DATASETS,
        QueryIntent.FIND_LATENT_MODELING_DATA,
    }
    stages.append(StageConfig(
        stage=RetrievalStage.AFFORDANCE_MATCH,
        enabled=intent in analysis_intents,
        weight=INTENT_STAGE_WEIGHTS.get(intent, {}).get("affordance", 0.15),
    ))

    # Add graph expansion for linking intents
    linking_intents = {
        QueryIntent.LINK_PAPERS_TO_DATASETS,
        QueryIntent.FIND_SIMILAR_EXPERIMENTS,
        QueryIntent.FIND_RELATED_PARADIGMS,
    }
    stages.append(StageConfig(
        stage=RetrievalStage.GRAPH_EXPANSION,
        enabled=intent in linking_intents,
        weight=INTENT_STAGE_WEIGHTS.get(intent, {}).get("graph", 0.10),
    ))

    # Add paper linking stage
    stages.append(StageConfig(
        stage=RetrievalStage.PAPER_LINKING,
        enabled=intent == QueryIntent.LINK_PAPERS_TO_DATASETS,
        weight=0.20,
    ))

    # Always do reranking and explanation
    stages.append(StageConfig(stage=RetrievalStage.RERANK, enabled=True, weight=0.0))
    stages.append(StageConfig(stage=RetrievalStage.EXPLAIN, enabled=True, weight=0.0))

    return stages


def parse_and_plan_query(
    query: str,
    retrieval_config: dict[str, Any] | None = None,
) -> QueryPlan:
    """Parse a natural language query and produce a structured retrieval plan.

    This is the main entry point for query understanding. It:
    1. Normalizes the query text
    2. Classifies query intent
    3. Extracts ontology matches (tasks, modalities, regions, behaviors)
    4. Identifies constraints (required and excluded)
    5. Determines retrieval stages and weights
    6. Returns a complete QueryPlan

    Args:
        query: Natural language search query
        retrieval_config: Optional retrieval configuration overrides

    Returns:
        QueryPlan with all information needed for retrieval
    """
    normalized = normalize_text(query)

    # Classify intent
    primary_intent, intent_confidence, secondary_intents = _classify_intent(query)

    # Match ontology terms
    task_matches = match_tasks(query)
    behavior_matches = match_behavior_labels(query)
    modality_matches = match_modalities(query)
    region_matches = match_brain_regions(query)
    affordance_matches = match_affordances(query)

    ontology_matches = {
        "tasks": task_matches,
        "behaviors": behavior_matches,
        "modalities": modality_matches,
        "regions": region_matches,
        "affordances": affordance_matches,
    }

    # Expand terms
    expanded = expand_query_terms(query)
    expanded_terms = expanded.get("terms", [])

    # Extract constraints
    constraints = _extract_constraints(query, ontology_matches)

    # Organize constraints by type
    required_modalities = [
        c.value for c in constraints
        if c.constraint_type == ConstraintType.MODALITY and not c.negated
    ]
    required_species = [
        c.value for c in constraints
        if c.constraint_type == ConstraintType.SPECIES and not c.negated
    ]
    required_tasks = [
        c.value for c in constraints
        if c.constraint_type == ConstraintType.TASK and not c.negated
    ]
    required_regions = [
        c.value for c in constraints
        if c.constraint_type == ConstraintType.BRAIN_REGION and not c.negated
    ]
    required_analyses = [
        c.value for c in constraints
        if c.constraint_type == ConstraintType.ANALYSIS and not c.negated
    ]

    excluded_modalities = [
        c.value for c in constraints
        if c.constraint_type == ConstraintType.MODALITY and c.negated
    ]
    excluded_species = [
        c.value for c in constraints
        if c.constraint_type == ConstraintType.SPECIES and c.negated
    ]
    excluded_tasks = [
        c.value for c in constraints
        if c.constraint_type == ConstraintType.TASK and c.negated
    ]

    # Determine retrieval stages
    stages = _determine_stages(primary_intent, constraints)

    # Determine if graph expansion is useful
    use_graph_expansion = primary_intent in {
        QueryIntent.LINK_PAPERS_TO_DATASETS,
        QueryIntent.FIND_SIMILAR_EXPERIMENTS,
        QueryIntent.FIND_RELATED_PARADIGMS,
    }

    use_paper_linking = primary_intent == QueryIntent.LINK_PAPERS_TO_DATASETS

    # Get weight overrides for this intent
    weight_overrides = INTENT_STAGE_WEIGHTS.get(primary_intent, {})

    # Determine query characteristics
    is_precise = len(constraints) >= 2 or intent_confidence >= 0.7
    needs_explanation = True  # Always explain for now
    uncertainty_high = intent_confidence < 0.6

    # Planning notes
    planning_notes = []
    if uncertainty_high:
        planning_notes.append("Low intent confidence; results may need refinement")
    if not constraints:
        planning_notes.append("No specific constraints detected; broad search")
    if use_graph_expansion:
        planning_notes.append("Graph expansion enabled for relationship discovery")

    return QueryPlan(
        query_text=query,
        normalized_query=normalized,
        primary_intent=primary_intent,
        intent_confidence=intent_confidence,
        secondary_intents=secondary_intents,
        target_record_type="dataset",  # Default to datasets
        constraints=constraints,
        required_modalities=required_modalities,
        required_species=required_species,
        required_tasks=required_tasks,
        required_regions=required_regions,
        required_analyses=required_analyses,
        excluded_modalities=excluded_modalities,
        excluded_species=excluded_species,
        excluded_tasks=excluded_tasks,
        excluded_sources=[],
        stages=stages,
        use_graph_expansion=use_graph_expansion,
        graph_expansion_hops=2,
        use_paper_linking=use_paper_linking,
        weight_overrides=weight_overrides,
        detected_keywords=expanded_terms[:20],
        ontology_matches={
            "tasks": [m.id for m in task_matches],
            "behaviors": [m.id for m in behavior_matches],
            "modalities": [m.id for m in modality_matches],
            "regions": [m.id for m in region_matches],
            "affordances": [m.id for m in affordance_matches],
        },
        expanded_terms=expanded_terms,
        is_precise_query=is_precise,
        needs_explanation=needs_explanation,
        uncertainty_high=uncertainty_high,
        planning_notes=planning_notes,
    )
