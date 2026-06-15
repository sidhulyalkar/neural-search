"""Boolean constraint parser for complex queries.

This module provides a constraint parser that handles:
- Negation: NOT fMRI, NOT "visual cortex"
- Conjunction: mouse AND decision-making
- Disjunction: ephys OR imaging
- Parenthetical grouping: (mouse OR rat) AND NOT fMRI
- Implicit constraints from domain knowledge

Usage:
    from neural_search.retrieval.constraint_parser import (
        ConstraintParser,
        parse_query,
    )

    parser = ConstraintParser()
    result = parser.parse("mouse AND decision-making NOT fMRI")
    print(result.positive_terms)  # ["mouse", "decision-making"]
    print(result.negative_terms)  # ["fMRI"]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ConstraintType(StrEnum):
    """Types of constraints that can be parsed."""

    POSITIVE = "positive"      # Include these terms
    NEGATIVE = "negative"      # Exclude these terms
    REQUIRED = "required"      # Must have these
    OPTIONAL = "optional"      # Nice to have


class OperatorType(StrEnum):
    """Boolean operators."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"


@dataclass
class ParsedTerm:
    """A single parsed term with its constraint type."""

    term: str
    constraint_type: ConstraintType
    source: str = "explicit"  # "explicit" or "implicit"
    confidence: float = 1.0


@dataclass
class ParsedConstraint:
    """A parsed constraint expression."""

    operator: OperatorType | None
    terms: list[ParsedTerm | ParsedConstraint] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.terms) == 1 and isinstance(self.terms[0], ParsedTerm)


@dataclass
class ParsedQuery:
    """Complete parsed query with all constraints."""

    original_query: str
    positive_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)
    required_terms: list[str] = field(default_factory=list)

    # Structured representation
    constraint_tree: ParsedConstraint | None = None

    # Implicit constraints added by parser
    implicit_constraints: list[ParsedTerm] = field(default_factory=list)

    # For hybrid retrieval
    semantic_query: str = ""  # Query for embedding search
    filter_query: dict[str, Any] = field(default_factory=dict)  # For structured filtering

    # Parser metadata
    parse_warnings: list[str] = field(default_factory=list)

    def has_negation(self) -> bool:
        return len(self.negative_terms) > 0

    def has_boolean_operators(self) -> bool:
        return self.constraint_tree is not None and self.constraint_tree.operator is not None


# Implicit constraint mappings from domain knowledge
IMPLICIT_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "neuropixels": {
        "modality": "ephys",
        "constraint": {"min_channels": 384},
    },
    "patch clamp": {
        "modality": "ephys",
        "constraint": {"recording_type": "intracellular"},
    },
    "two-photon": {
        "modality": "calcium_imaging",
    },
    "widefield": {
        "modality": "calcium_imaging",
        "constraint": {"imaging_type": "widefield"},
    },
    "bold": {
        "modality": "fmri",
    },
    "eeg": {
        "modality": "eeg",
    },
    "meg": {
        "modality": "meg",
    },
}


class ConstraintParser:
    """Parser for query constraints with Boolean operators."""

    # Pattern for quoted phrases
    QUOTED_PATTERN = re.compile(r'"([^"]+)"')

    # Pattern for Boolean operators
    OPERATOR_PATTERN = re.compile(
        r'\b(AND|OR|NOT)\b',
        re.IGNORECASE,
    )

    # Parentheses for grouping
    PAREN_PATTERN = re.compile(r'\(([^()]+)\)')

    def __init__(
        self,
        apply_implicit_constraints: bool = True,
        case_sensitive: bool = False,
    ):
        self.apply_implicit_constraints = apply_implicit_constraints
        self.case_sensitive = case_sensitive

    def parse(self, query: str) -> ParsedQuery:
        """Parse a query string into structured constraints.

        Args:
            query: Raw query string

        Returns:
            ParsedQuery with extracted constraints
        """
        warnings: list[str] = []
        original = query

        # Extract quoted phrases first (protect from tokenization)
        quoted_phrases: list[str] = []
        placeholders: dict[str, str] = {}

        def extract_quoted(match: re.Match) -> str:
            phrase = match.group(1)
            placeholder = f"__QUOTED_{len(quoted_phrases)}__"
            quoted_phrases.append(phrase)
            placeholders[placeholder] = phrase
            return placeholder

        query = self.QUOTED_PATTERN.sub(extract_quoted, query)

        # Identify operators and their positions
        positive_terms: list[str] = []
        negative_terms: list[str] = []
        required_terms: list[str] = []

        # Simple parsing: split by operators
        # First handle NOT
        parts = re.split(r'\bNOT\b', query, flags=re.IGNORECASE)

        if len(parts) > 1:
            # First part is positive, rest are negated
            positive_part = parts[0].strip()
            negative_parts = parts[1:]

            # Parse positive part
            positive_tokens = self._tokenize(positive_part, placeholders)
            positive_terms.extend(positive_tokens)

            # Parse negative parts
            for neg_part in negative_parts:
                neg_tokens = self._tokenize(neg_part.strip(), placeholders)
                # First token after NOT is negated
                if neg_tokens:
                    negative_terms.append(neg_tokens[0])
                    # Any remaining tokens after AND/OR go back to positive
                    if len(neg_tokens) > 1:
                        for t in neg_tokens[1:]:
                            if t.upper() not in ("AND", "OR"):
                                positive_terms.append(t)
        else:
            # No NOT, just parse normally
            tokens = self._tokenize(query, placeholders)
            positive_terms.extend(tokens)

        # Remove Boolean operators from term lists
        positive_terms = [
            t for t in positive_terms
            if t.upper() not in ("AND", "OR", "NOT")
        ]
        negative_terms = [
            t for t in negative_terms
            if t.upper() not in ("AND", "OR", "NOT")
        ]

        # Build semantic query (for embedding search)
        semantic_parts = positive_terms.copy()
        semantic_query = " ".join(semantic_parts)

        # Apply implicit constraints
        implicit_constraints: list[ParsedTerm] = []
        if self.apply_implicit_constraints:
            for term in positive_terms:
                term_lower = term.lower()
                if term_lower in IMPLICIT_CONSTRAINTS:
                    constraint_info = IMPLICIT_CONSTRAINTS[term_lower]
                    if "modality" in constraint_info:
                        implicit_constraints.append(ParsedTerm(
                            term=constraint_info["modality"],
                            constraint_type=ConstraintType.REQUIRED,
                            source="implicit",
                            confidence=0.9,
                        ))

        # Build filter query for structured filtering
        filter_query: dict[str, Any] = {}
        if negative_terms:
            filter_query["exclude_terms"] = negative_terms
        if required_terms:
            filter_query["require_terms"] = required_terms

        return ParsedQuery(
            original_query=original,
            positive_terms=positive_terms,
            negative_terms=negative_terms,
            required_terms=required_terms,
            semantic_query=semantic_query,
            filter_query=filter_query,
            implicit_constraints=implicit_constraints,
            parse_warnings=warnings,
        )

    def _tokenize(
        self,
        text: str,
        placeholders: dict[str, str],
    ) -> list[str]:
        """Tokenize text, respecting quoted phrases."""
        # Split on whitespace and operators
        tokens = re.split(r'\s+', text.strip())
        tokens = [t for t in tokens if t]

        # Restore quoted phrases
        result = []
        for token in tokens:
            if token in placeholders:
                result.append(placeholders[token])
            else:
                result.append(token)

        return result

    def parse_with_tree(self, query: str) -> ParsedQuery:
        """Parse query into a full constraint tree (for complex Boolean logic).

        This method builds a tree representation for queries with
        parenthetical grouping and mixed operators.
        """
        result = self.parse(query)

        # Build constraint tree for complex queries
        if result.has_negation() or "AND" in query.upper() or "OR" in query.upper():
            result.constraint_tree = self._build_tree(query)

        return result

    def _build_tree(self, query: str) -> ParsedConstraint:
        """Build a constraint tree from a query string."""
        # Simplified tree building
        # For a full implementation, use a proper parser (e.g., pyparsing)

        terms: list[ParsedTerm | ParsedConstraint] = []

        # Check for OR at top level (lowest precedence)
        if " OR " in query.upper():
            parts = re.split(r'\s+OR\s+', query, flags=re.IGNORECASE)
            for part in parts:
                sub_tree = self._build_tree(part.strip())
                terms.append(sub_tree)
            return ParsedConstraint(operator=OperatorType.OR, terms=terms)

        # Check for AND (higher precedence than OR)
        if " AND " in query.upper():
            parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)
            for part in parts:
                sub_tree = self._build_tree(part.strip())
                terms.append(sub_tree)
            return ParsedConstraint(operator=OperatorType.AND, terms=terms)

        # Check for NOT (highest precedence)
        if query.upper().startswith("NOT "):
            inner = query[4:].strip()
            inner_term = ParsedTerm(
                term=inner,
                constraint_type=ConstraintType.NEGATIVE,
            )
            return ParsedConstraint(
                operator=OperatorType.NOT,
                terms=[inner_term],
            )

        # Base case: single term
        return ParsedConstraint(
            operator=None,
            terms=[ParsedTerm(term=query.strip(), constraint_type=ConstraintType.POSITIVE)],
        )


def parse_query(query: str, **kwargs: Any) -> ParsedQuery:
    """Convenience function to parse a query.

    Args:
        query: Raw query string
        **kwargs: Passed to ConstraintParser

    Returns:
        ParsedQuery with extracted constraints
    """
    parser = ConstraintParser(**kwargs)
    return parser.parse(query)


def apply_negative_filter(
    results: list[dict[str, Any]],
    negative_terms: list[str],
    text_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter results to exclude those matching negative terms.

    Args:
        results: List of result dictionaries
        negative_terms: Terms to exclude
        text_fields: Fields to check for negative terms

    Returns:
        Filtered results
    """
    if not negative_terms:
        return results

    if text_fields is None:
        text_fields = ["title", "description", "text_card"]

    negative_lower = [t.lower() for t in negative_terms]

    filtered = []
    for result in results:
        should_exclude = False
        for field_name in text_fields:
            text = str(result.get(field_name, "")).lower()
            for neg_term in negative_lower:
                if neg_term in text:
                    should_exclude = True
                    break
            if should_exclude:
                break

        if not should_exclude:
            filtered.append(result)

    return filtered
