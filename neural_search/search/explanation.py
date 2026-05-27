"""Rich explanation generation for search results.

This module provides improved explanation quality by:
1. Grouping related matches into coherent sections
2. Adding query context (what was asked vs what was found)
3. Generating human-readable summary explanations
4. Supporting multiple explanation styles (brief, detailed, technical)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MatchGroup:
    """A group of related matches for explanation."""

    category: str  # "task", "modality", "species", etc.
    query_terms: list[str]  # What was requested
    matched_terms: list[str]  # What was found
    partial_matches: list[str] = field(default_factory=list)  # Related but not exact
    match_quality: str = "full"  # "full", "partial", "inferred", "none"


@dataclass
class ExplanationContext:
    """Context for generating explanations."""

    query_text: str
    dataset_title: str
    dataset_id: str
    match_groups: list[MatchGroup]
    score: float
    score_breakdown: dict[str, float]
    warnings: list[str]
    missing_metadata: list[str]
    graph_context: dict[str, Any] | None = None
    linked_papers: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExplanationResult:
    """Result of explanation generation."""

    brief: str  # One-sentence summary
    detailed: str  # Multi-line explanation
    technical: str  # Score breakdown explanation
    match_summary: dict[str, Any]  # Structured match data
    quality_grade: str  # "excellent", "good", "moderate", "weak"


def _pluralize(word: str, count: int) -> str:
    """Simple pluralization."""
    if count == 1:
        return word
    return word + "s"


def _format_term_list(terms: Sequence[str], max_items: int = 3) -> str:
    """Format a list of terms for display."""
    if not terms:
        return "none"
    if len(terms) <= max_items:
        if len(terms) == 1:
            return terms[0]
        return ", ".join(terms[:-1]) + f" and {terms[-1]}"
    return ", ".join(terms[:max_items]) + f" (+{len(terms) - max_items} more)"


def _calculate_quality_grade(
    context: ExplanationContext,
) -> str:
    """Calculate overall match quality grade."""
    score = context.score
    num_matches = sum(len(g.matched_terms) for g in context.match_groups)
    has_warnings = bool(context.warnings)
    missing_count = len(context.missing_metadata)

    if score >= 70 and num_matches >= 3 and not has_warnings:
        return "excellent"
    if score >= 50 and num_matches >= 2:
        return "good"
    if score >= 30 or num_matches >= 1:
        return "moderate"
    return "weak"


def _generate_brief_explanation(context: ExplanationContext) -> str:
    """Generate a one-sentence explanation."""
    matched_categories = [
        g.category for g in context.match_groups if g.matched_terms
    ]

    if not matched_categories:
        return f"Dataset '{context.dataset_title}' has limited overlap with query."

    all_matched = []
    for group in context.match_groups:
        all_matched.extend(group.matched_terms)

    if len(matched_categories) == 1:
        cat = matched_categories[0]
        terms = _format_term_list(
            [g.matched_terms for g in context.match_groups if g.category == cat][0],
            max_items=2,
        )
        return f"Matches {cat}: {terms}."

    # Multiple categories
    cat_summary = ", ".join(matched_categories[:-1]) + f" and {matched_categories[-1]}"
    return f"Matches query on {cat_summary}."


def _generate_detailed_explanation(context: ExplanationContext) -> str:
    """Generate a multi-line detailed explanation."""
    lines = []

    # Opening line with overall context
    quality = _calculate_quality_grade(context)
    quality_desc = {
        "excellent": "strongly matches",
        "good": "matches well with",
        "moderate": "partially matches",
        "weak": "has limited overlap with",
    }
    lines.append(
        f"This dataset {quality_desc[quality]} your query for '{context.query_text}'."
    )
    lines.append("")

    # Group matches by category
    for group in context.match_groups:
        if not group.matched_terms and not group.partial_matches:
            continue

        category_title = group.category.replace("_", " ").title()

        if group.matched_terms:
            match_type = "Exact match" if len(group.matched_terms) == 1 else "Matches"
            terms = _format_term_list(group.matched_terms, max_items=3)
            lines.append(f"• {category_title}: {match_type} - {terms}")

        if group.partial_matches:
            partial = _format_term_list(group.partial_matches, max_items=2)
            lines.append(f"  Related: {partial}")

    # Graph context if available
    if context.graph_context:
        paper_count = len(context.linked_papers)
        if paper_count > 0:
            lines.append("")
            lines.append(
                f"• Linked to {paper_count} "
                f"{_pluralize('publication', paper_count)} "
                "for provenance tracking."
            )

    # Warnings
    if context.warnings:
        lines.append("")
        for warning in context.warnings[:2]:
            lines.append(f"⚠ {warning}")

    return "\n".join(lines)


def _generate_technical_explanation(context: ExplanationContext) -> str:
    """Generate a technical explanation with score breakdown."""
    lines = []

    lines.append(f"Dataset: {context.dataset_id}")
    lines.append(f"Overall Score: {context.score:.1f}")
    lines.append("")
    lines.append("Score Components:")

    for component, value in sorted(
        context.score_breakdown.items(),
        key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0,
    ):
        if isinstance(value, (int, float)) and value > 0:
            lines.append(f"  {component}: {value:.3f}")

    lines.append("")
    lines.append("Match Details:")

    for group in context.match_groups:
        if group.query_terms:
            matched_count = len(group.matched_terms)
            total_count = len(group.query_terms)
            lines.append(
                f"  {group.category}: {matched_count}/{total_count} "
                f"({group.match_quality})"
            )
            if group.matched_terms:
                lines.append(f"    Matched: {', '.join(group.matched_terms)}")
            if group.partial_matches:
                lines.append(f"    Partial: {', '.join(group.partial_matches)}")

    if context.missing_metadata:
        lines.append("")
        lines.append(f"Missing Metadata: {', '.join(context.missing_metadata[:5])}")

    return "\n".join(lines)


def generate_explanation(
    context: ExplanationContext,
) -> ExplanationResult:
    """Generate comprehensive explanation from context.

    Args:
        context: ExplanationContext with all match information

    Returns:
        ExplanationResult with multiple explanation formats
    """
    quality_grade = _calculate_quality_grade(context)

    # Build match summary
    match_summary = {
        "quality_grade": quality_grade,
        "total_matches": sum(len(g.matched_terms) for g in context.match_groups),
        "categories_matched": [
            g.category for g in context.match_groups if g.matched_terms
        ],
        "score": context.score,
        "has_warnings": bool(context.warnings),
        "missing_metadata_count": len(context.missing_metadata),
    }

    return ExplanationResult(
        brief=_generate_brief_explanation(context),
        detailed=_generate_detailed_explanation(context),
        technical=_generate_technical_explanation(context),
        match_summary=match_summary,
        quality_grade=quality_grade,
    )


def build_explanation_context(
    query: str,
    dataset: Mapping[str, Any],
    parsed_query: Mapping[str, Any],
    result: Any,  # SearchResult
) -> ExplanationContext:
    """Build explanation context from search result.

    Args:
        query: Original query text
        dataset: Dataset record
        parsed_query: Parsed query with extracted terms
        result: SearchResult object

    Returns:
        ExplanationContext ready for explanation generation
    """
    # Extract match groups from matched_terms and parsed_query
    match_groups = []

    # Tasks
    query_tasks = parsed_query.get("tasks", [])
    matched_tasks = [
        t for t in result.matched_terms
        if any(t.lower() in qt.lower() or qt.lower() in t.lower() for qt in query_tasks)
    ] if query_tasks else []
    if query_tasks or matched_tasks:
        match_groups.append(MatchGroup(
            category="task",
            query_terms=list(query_tasks),
            matched_terms=matched_tasks,
            match_quality="full" if matched_tasks else "none",
        ))

    # Modalities
    query_modalities = parsed_query.get("modalities", [])
    matched_modalities = [
        m for m in result.matched_terms
        if any(
            m.lower().replace("_", " ") in qm.lower()
            or qm.lower() in m.lower()
            for qm in query_modalities
        )
    ] if query_modalities else []
    if query_modalities or matched_modalities:
        match_groups.append(MatchGroup(
            category="modality",
            query_terms=list(query_modalities),
            matched_terms=matched_modalities,
            match_quality="full" if matched_modalities else "none",
        ))

    # Species
    query_species = parsed_query.get("species", [])
    matched_species = [
        s for s in result.matched_terms
        if any(
            s.lower() in qs.lower() or qs.lower() in s.lower()
            for qs in query_species
        )
    ] if query_species else []
    if query_species or matched_species:
        match_groups.append(MatchGroup(
            category="species",
            query_terms=list(query_species),
            matched_terms=matched_species,
            match_quality="full" if matched_species else "none",
        ))

    # Brain regions
    query_regions = parsed_query.get("brain_regions", [])
    matched_regions = [
        r for r in result.matched_terms
        if any(
            r.lower() in qr.lower() or qr.lower() in r.lower()
            for qr in query_regions
        )
    ] if query_regions else []
    if query_regions or matched_regions:
        match_groups.append(MatchGroup(
            category="brain_region",
            query_terms=list(query_regions),
            matched_terms=matched_regions,
            match_quality="full" if matched_regions else "none",
        ))

    # Behaviors
    query_behaviors = parsed_query.get("behaviors", [])
    matched_behaviors = [
        b for b in result.matched_terms
        if any(
            b.lower() in qb.lower() or qb.lower() in b.lower()
            for qb in query_behaviors
        )
    ] if query_behaviors else []
    if query_behaviors or matched_behaviors:
        match_groups.append(MatchGroup(
            category="behavior",
            query_terms=list(query_behaviors),
            matched_terms=matched_behaviors,
            match_quality="full" if matched_behaviors else "none",
        ))

    # Affordances
    query_affordances = parsed_query.get("affordances", [])
    matched_affordances = [
        a for a in result.matched_terms
        if any(
            a.lower() in qa.lower() or qa.lower() in a.lower()
            for qa in query_affordances
        )
    ] if query_affordances else []
    if query_affordances or matched_affordances:
        match_groups.append(MatchGroup(
            category="affordance",
            query_terms=list(query_affordances),
            matched_terms=matched_affordances,
            match_quality="full" if matched_affordances else "none",
        ))

    dataset_title = dataset.get("title", str(result.dataset_id))

    return ExplanationContext(
        query_text=query,
        dataset_title=dataset_title,
        dataset_id=str(result.dataset_id),
        match_groups=match_groups,
        score=result.score,
        score_breakdown=dict(result.score_breakdown),
        warnings=list(result.warnings),
        missing_metadata=list(result.missing_metadata),
        graph_context=result.graph_context,
        linked_papers=list(result.linked_papers),
    )


def generate_result_explanation(
    query: str,
    dataset: Mapping[str, Any],
    parsed_query: Mapping[str, Any],
    result: Any,
    style: str = "detailed",
) -> str:
    """Generate explanation for a search result.

    Args:
        query: Original query text
        dataset: Dataset record
        parsed_query: Parsed query
        result: SearchResult
        style: "brief", "detailed", or "technical"

    Returns:
        Explanation string in the requested style
    """
    context = build_explanation_context(query, dataset, parsed_query, result)
    explanation = generate_explanation(context)

    if style == "brief":
        return explanation.brief
    if style == "technical":
        return explanation.technical
    return explanation.detailed
