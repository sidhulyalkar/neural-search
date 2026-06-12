"""Concept-grounded result explanations for dataset retrieval.

Generates structured, serializable explanations showing which concepts matched,
what evidence supports them, what is missing, and how the final score was derived.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx  # type: ignore[import-untyped]

from neural_search.field_state.concept_memory.graph_builder import (
    build_concept_graph,
    read_concept_artifacts,
)
from neural_search.field_state.concept_memory.reranker import (
    _build_dataset_adjacency,
    rerank_datasets,
)
from neural_search.field_state.concept_memory.schema import (
    ConceptExplanation,
    ConceptNode,
    ConceptRerankedResult,
    EvidenceLink,
)

_EXPECTED_CONCEPT_TYPES = frozenset({
    "task",
    "modality",
    "species",
    "brain_region",
    "method",
    "analysis_affordance",
})


def _find_dataset_concept(
    dataset_id: str,
    concepts: list[ConceptNode],
) -> ConceptNode | None:
    """Find a dataset ConceptNode by original dataset_id or concept_id."""
    for c in concepts:
        if c.concept_type != "dataset":
            continue
        if c.concept_id == dataset_id:
            return c
        if dataset_id in c.source_ids:
            return c
        # Also check if the canonical_name matches (for robustness)
    return None


def _identify_missing_evidence(
    matched_types: set[str],
) -> list[str]:
    """Identify expected concept types that have no matched evidence."""
    missing = []
    for ctype in sorted(_EXPECTED_CONCEPT_TYPES):
        if ctype not in matched_types:
            missing.append(f"No '{ctype}' concept match found for this dataset.")
    return missing


def _identify_hard_negative_conflicts(
    dataset_concept_id: str,
    adjacency: dict[str, list[tuple[str, EvidenceLink]]],
    graph: nx.DiGraph,
) -> list[str]:
    """Describe any failure-mode concept links that act as hard-negative conflicts."""
    conflicts = []
    connections = adjacency.get(dataset_concept_id, [])
    for connected_id, link in connections:
        if connected_id in graph:
            ctype = graph.nodes[connected_id].get("concept_type", "")
            if ctype == "failure_mode":
                name = graph.nodes[connected_id].get("canonical_name", connected_id)
                conflicts.append(
                    f"Failure mode conflict: '{name}' "
                    f"(relation: {link.relation_type}, confidence: {link.confidence:.2f})"
                )
    return conflicts


def _render_explanation_markdown(explanation: ConceptExplanation) -> str:
    """Render a ConceptExplanation to human-readable Markdown."""
    lines: list[str] = [
        "## Concept Memory Explanation",
        "",
        f"**Dataset:** {explanation.dataset_title}  ",
        f"**Dataset ID:** `{explanation.dataset_id}`  ",
        f"**Query:** {explanation.query}",
        "",
    ]

    if explanation.score_breakdown is not None:
        sb = explanation.score_breakdown
        lines += [
            "### Score Breakdown",
            "",
            "| Component | Value |",
            "|-----------|-------|",
            f"| Base score (lexical) | {sb.base_score:.4f} |",
            f"| Concept boost | {sb.concept_boost:.4f} |",
            f"| Evidence boost | {sb.evidence_boost:.4f} |",
            f"| Hard-negative penalty | -{sb.hard_negative_penalty:.4f} |",
            f"| **Final score** | **{sb.final_score:.4f}** |",
            f"| Matched concepts | {sb.matched_concept_count} |",
            "",
        ]

    if explanation.matched_concepts:
        lines += ["### Matched Concepts", ""]
        for mc in explanation.matched_concepts:
            lines.append(
                f"- **[{mc.concept_type}]** {mc.canonical_name}"
                f" (match score: {mc.match_score:.3f})"
            )
            if mc.evidence_texts:
                for et in mc.evidence_texts[:2]:
                    lines.append(f"  - *Evidence:* {et}")
            if mc.relation_types:
                lines.append(f"  - *Relations:* {', '.join(mc.relation_types)}")
        lines.append("")
    else:
        lines += ["### Matched Concepts", "", "_None — base lexical score only._", ""]

    if explanation.missing_evidence:
        lines += ["### Missing Evidence", ""]
        for msg in explanation.missing_evidence:
            lines.append(f"- {msg}")
        lines.append("")

    if explanation.hard_negative_conflicts:
        lines += ["### Hard-Negative Conflicts", ""]
        for msg in explanation.hard_negative_conflicts:
            lines.append(f"- {msg}")
        lines.append("")
    else:
        lines += ["### Hard-Negative Conflicts", "", "_None detected._", ""]

    return "\n".join(lines)


def explain_result(
    query: str,
    dataset_id: str,
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
    graph: nx.DiGraph | None = None,
    reranked_result: ConceptRerankedResult | None = None,
) -> ConceptExplanation:
    """Generate a structured explanation for a specific dataset against a query.

    Args:
        query: Natural language search query.
        dataset_id: The dataset to explain (original source ID or concept_id).
        concepts: All concept nodes from the concept memory.
        evidence_links: All evidence links from the concept memory.
        graph: Pre-built concept graph (built from concepts/evidence_links if None).
        reranked_result: Pre-computed reranking result (avoids recomputing scores).

    Returns:
        ConceptExplanation with matched concepts, score breakdown, and missing evidence.
    """
    if graph is None:
        graph = build_concept_graph(concepts, evidence_links)

    dataset_concept = _find_dataset_concept(dataset_id, concepts)
    if dataset_concept is None:
        return ConceptExplanation(
            dataset_id=dataset_id,
            dataset_title=dataset_id,
            query=query,
            explanation_markdown=(
                f"## Concept Memory Explanation\n\n"
                f"**Dataset ID:** `{dataset_id}`\n\n"
                f"_Dataset not found in concept memory. "
                f"Run `concept-build` to index the corpus._"
            ),
        )

    # If a precomputed result is provided, reuse its scores and matched concepts
    if reranked_result is not None and reranked_result.dataset_id == dataset_id:
        matched_concepts = reranked_result.matched_concepts
        score_breakdown = reranked_result.score_breakdown()
    else:
        # Compute scores on demand
        results = rerank_datasets(
            query=query,
            concepts=concepts,
            evidence_links=evidence_links,
            graph=graph,
            limit=10000,  # no cutoff
        )
        result = next(
            (r for r in results if r.dataset_id == dataset_id or r.dataset_title == dataset_concept.canonical_name),
            None,
        )
        if result is not None:
            matched_concepts = result.matched_concepts
            score_breakdown = result.score_breakdown()
        else:
            matched_concepts = []
            score_breakdown = None

    matched_types = {m.concept_type for m in matched_concepts}
    adjacency = _build_dataset_adjacency(concepts, evidence_links, graph)
    hard_negative_conflicts = _identify_hard_negative_conflicts(
        dataset_concept.concept_id, adjacency, graph
    )
    missing_evidence = _identify_missing_evidence(matched_types)

    explanation = ConceptExplanation(
        dataset_id=dataset_id,
        dataset_title=dataset_concept.canonical_name,
        query=query,
        matched_concepts=matched_concepts,
        score_breakdown=score_breakdown,
        missing_evidence=missing_evidence,
        hard_negative_conflicts=hard_negative_conflicts,
    )
    explanation = explanation.model_copy(
        update={"explanation_markdown": _render_explanation_markdown(explanation)}
    )
    return explanation


def explain_from_artifacts(
    query: str,
    dataset_id: str,
    root: Path | None = None,
) -> ConceptExplanation:
    """Load artifacts from disk and generate an explanation."""
    concepts, evidence_links = read_concept_artifacts(root)
    graph = build_concept_graph(concepts, evidence_links)
    return explain_result(
        query=query,
        dataset_id=dataset_id,
        concepts=concepts,
        evidence_links=evidence_links,
        graph=graph,
    )
