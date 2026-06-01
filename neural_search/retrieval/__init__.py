"""Retrieval module for query processing and search.

This module provides:
- ConstraintParser: Boolean constraint parsing for complex queries
- Negative filtering: Post-retrieval filtering for exclusion terms
- Implicit constraints: Domain knowledge-based query expansion

Usage:
    from neural_search.retrieval import (
        ConstraintParser,
        parse_query,
        apply_negative_filter,
    )

    # Parse a query with Boolean operators
    result = parse_query("mouse decision-making NOT fMRI")

    # Filter results
    filtered = apply_negative_filter(results, result.negative_terms)
"""

from neural_search.retrieval.constraint_parser import (
    ConstraintParser,
    ConstraintType,
    OperatorType,
    ParsedConstraint,
    ParsedQuery,
    ParsedTerm,
    apply_negative_filter,
    parse_query,
)
from neural_search.retrieval.query_intent import (
    UsefulnessIntent,
    IntentClassification,
    classify_query_intent,
)
from neural_search.retrieval.usefulness_scorer import (
    DatasetContext,
    UsefulnessScore,
    INTENT_WEIGHT_PROFILES,
    score_usefulness,
)
from neural_search.retrieval.graph_usefulness import (
    affordance_overlap,
    pipeline_overlap,
    complementarity_score,
    normalized_metapath_score,
    graph_usefulness_features,
)

__all__ = [
    "ConstraintParser",
    "ConstraintType",
    "OperatorType",
    "ParsedConstraint",
    "ParsedQuery",
    "ParsedTerm",
    "apply_negative_filter",
    "parse_query",
    "UsefulnessIntent",
    "IntentClassification",
    "classify_query_intent",
    "DatasetContext",
    "UsefulnessScore",
    "INTENT_WEIGHT_PROFILES",
    "score_usefulness",
    "affordance_overlap",
    "pipeline_overlap",
    "complementarity_score",
    "normalized_metapath_score",
    "graph_usefulness_features",
]
