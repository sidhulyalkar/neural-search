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

__all__ = [
    "ConstraintParser",
    "ConstraintType",
    "OperatorType",
    "ParsedConstraint",
    "ParsedQuery",
    "ParsedTerm",
    "apply_negative_filter",
    "parse_query",
]
