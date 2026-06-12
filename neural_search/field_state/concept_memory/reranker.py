"""Concept Memory reranking layer for dataset retrieval.

Reranks dataset candidates using concept graph proximity and evidence strength.
All scoring is transparent and decomposable into named components.

Disabled by default; enable explicitly via CLI --concept-boost flag or config.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import networkx as nx  # type: ignore[import-untyped]

from neural_search.field_state.concept_memory.graph_builder import (
    build_concept_graph,
    read_concept_artifacts,
)
from neural_search.field_state.concept_memory.retrieval import (
    _lexical_score,
    search_concepts,
)
from neural_search.field_state.concept_memory.schema import (
    ConceptNode,
    ConceptRerankedResult,
    EvidenceLink,
    MatchedConceptInfo,
)

# ---------------------------------------------------------------------------
# Type weights for concept boost
# ---------------------------------------------------------------------------

CONCEPT_TYPE_BOOST_WEIGHTS: dict[str, float] = {
    "task": 0.40,
    "modality": 0.30,
    "species": 0.15,
    "brain_region": 0.15,
    "method": 0.10,
    "analysis_affordance": 0.10,
    "experimental_protocol": 0.08,
    "neuroscience_concept": 0.08,
    "metric": 0.05,
    "model": 0.05,
    "tool": 0.05,
    "failure_mode": 0.00,  # never boost for failure modes
    "claim": 0.05,
    "benchmark_gap": 0.03,
    "opportunity": 0.03,
    "open_problem": 0.03,
    "paper": 0.02,
    "dataset": 0.00,  # dataset-to-dataset links are not concept boosts
}

_EVIDENCE_STRENGTH_SCORES: dict[str, float] = {
    "strong": 0.05,
    "moderate": 0.03,
    "weak": 0.01,
    "none": 0.0,
}

_DEFAULT_CONCEPT_BOOST_SCALE = 0.30
_DEFAULT_EVIDENCE_BOOST_SCALE = 0.10


# ---------------------------------------------------------------------------
# Adjacency index builder
# ---------------------------------------------------------------------------


def _build_dataset_adjacency(
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
    graph: nx.DiGraph,
) -> dict[str, list[tuple[str, EvidenceLink]]]:
    """Build {dataset_concept_id: [(connected_concept_id, link), ...]}."""
    dataset_ids = {c.concept_id for c in concepts if c.concept_type == "dataset"}
    adjacency: dict[str, list[tuple[str, EvidenceLink]]] = defaultdict(list)

    for link in evidence_links:
        src = link.source_concept_id
        tgt = link.target_concept_id
        if src in dataset_ids and tgt is not None:
            adjacency[src].append((tgt, link))
        elif tgt is not None and tgt in dataset_ids:
            adjacency[tgt].append((src, link))

    return dict(adjacency)


# ---------------------------------------------------------------------------
# Per-dataset scoring helpers
# ---------------------------------------------------------------------------


def _score_concept_boost(
    dataset_concept_id: str,
    adjacency: dict[str, list[tuple[str, EvidenceLink]]],
    concept_match_scores: dict[str, float],
    graph: nx.DiGraph,
    boost_scale: float,
) -> tuple[float, list[MatchedConceptInfo]]:
    """Compute concept boost and matched concept list for one dataset."""
    connections = adjacency.get(dataset_concept_id, [])
    matched: list[MatchedConceptInfo] = []
    raw_boost = 0.0

    seen_concept_ids: set[str] = set()
    for connected_id, link in connections:
        if connected_id not in concept_match_scores:
            continue
        if connected_id in seen_concept_ids:
            continue
        seen_concept_ids.add(connected_id)

        ctype = ""
        if connected_id in graph:
            ctype = graph.nodes[connected_id].get("concept_type", "")
        type_weight = CONCEPT_TYPE_BOOST_WEIGHTS.get(ctype, 0.05)

        match_score = concept_match_scores[connected_id]
        contribution = match_score * type_weight * link.confidence
        raw_boost += contribution

        matched.append(
            MatchedConceptInfo(
                concept_id=connected_id,
                canonical_name=graph.nodes[connected_id].get("canonical_name", connected_id)
                if connected_id in graph
                else connected_id,
                concept_type=ctype,
                match_score=round(match_score, 6),
                evidence_texts=[link.evidence_text] if link.evidence_text else [],
                relation_types=[link.relation_type],
            )
        )

    # Cap and scale
    capped = min(raw_boost, 1.0) * boost_scale
    return round(capped, 6), matched


def _score_evidence_boost(
    dataset_concept_id: str,
    adjacency: dict[str, list[tuple[str, EvidenceLink]]],
    concept_match_scores: dict[str, float],
    evidence_boost_scale: float,
) -> float:
    """Compute evidence strength bonus for links to query-matched concepts."""
    connections = adjacency.get(dataset_concept_id, [])
    raw = 0.0
    for connected_id, link in connections:
        if connected_id not in concept_match_scores:
            continue
        strength_label = link.metadata.get("evidence_strength", "none")
        score = _EVIDENCE_STRENGTH_SCORES.get(str(strength_label), 0.0)
        if link.review_status != "unreviewed":
            score = max(score, 0.01)
        raw += score
    capped = min(raw, 1.0) * evidence_boost_scale
    return round(capped, 6)


def _score_hard_negative_penalty(
    dataset_concept: ConceptNode,
    adjacency: dict[str, list[tuple[str, EvidenceLink]]],
    graph: nx.DiGraph,
) -> float:
    """Small penalty when a dataset is linked to failure-mode concepts."""
    penalty = 0.0
    connections = adjacency.get(dataset_concept.concept_id, [])
    for connected_id, _link in connections:
        if connected_id in graph:
            ctype = graph.nodes[connected_id].get("concept_type", "")
            if ctype == "failure_mode":
                penalty += 0.05
    return round(min(penalty, 0.10), 6)


# ---------------------------------------------------------------------------
# Public reranking API
# ---------------------------------------------------------------------------


def rerank_datasets(
    query: str,
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
    graph: nx.DiGraph | None = None,
    limit: int = 10,
    concept_boost_scale: float = _DEFAULT_CONCEPT_BOOST_SCALE,
    evidence_boost_scale: float = _DEFAULT_EVIDENCE_BOOST_SCALE,
    enable_concept_boost: bool = True,
    enable_evidence_boost: bool = True,
    enable_hard_negative_penalty: bool = True,
    min_base_score: float = 0.0,
) -> list[ConceptRerankedResult]:
    """Rerank dataset concepts against a query using concept memory signals.

    Algorithm (all terms are named and decomposable):
      base_score         = lexical similarity between query and dataset name/description
      concept_boost      = sum of (query-concept-match × type-weight × link-confidence)
                           for all concepts connected to this dataset, capped and scaled
      evidence_boost     = bonus for reviewed/strong-evidence links to matched concepts
      hard_negative_penalty = penalty for linked failure-mode concepts
      final_score = base_score + concept_boost + evidence_boost - hard_negative_penalty

    Returns up to `limit` results sorted by final_score descending.
    """
    if graph is None:
        graph = build_concept_graph(concepts, evidence_links)

    # Find top query-concept matches (non-dataset types)
    query_concept_results = search_concepts(
        query=query,
        concepts=concepts,
        evidence_links=evidence_links,
        graph=graph,
        limit=100,
        min_score=0.05,
    )
    concept_match_scores = {r.concept_id: r.score for r in query_concept_results}

    # Build adjacency index once for all datasets
    adjacency = _build_dataset_adjacency(concepts, evidence_links, graph)

    results: list[ConceptRerankedResult] = []
    for concept in concepts:
        if concept.concept_type != "dataset":
            continue

        base = round(_lexical_score(query, concept), 6)
        if base < min_base_score and not concept_match_scores:
            continue

        c_boost, matched = _score_concept_boost(
            concept.concept_id,
            adjacency,
            concept_match_scores,
            graph,
            concept_boost_scale if enable_concept_boost else 0.0,
        )
        e_boost = (
            _score_evidence_boost(
                concept.concept_id, adjacency, concept_match_scores, evidence_boost_scale
            )
            if enable_evidence_boost
            else 0.0
        )
        penalty = (
            _score_hard_negative_penalty(concept, adjacency, graph)
            if enable_hard_negative_penalty
            else 0.0
        )

        final = round(max(base + c_boost + e_boost - penalty, 0.0), 6)

        # Build a brief explanation summary
        type_names = {m.concept_type for m in matched}
        if matched:
            names = ", ".join(m.canonical_name for m in matched[:3])
            summary = f"Matched {len(matched)} concept(s): {names}"
            if len(matched) > 3:
                summary += f" (+{len(matched) - 3} more)"
            summary += f" [{', '.join(sorted(type_names))}]"
        else:
            summary = "No concept graph matches; base lexical score only."

        # Original dataset ID from source_ids (first entry) or concept_id
        dataset_id = concept.source_ids[0] if concept.source_ids else concept.concept_id

        results.append(
            ConceptRerankedResult(
                dataset_id=dataset_id,
                dataset_title=concept.canonical_name,
                base_score=base,
                concept_boost=c_boost,
                evidence_boost=e_boost,
                hard_negative_penalty=penalty,
                final_score=final,
                matched_concepts=matched,
                explanation_summary=summary,
            )
        )

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results[:limit]


def rerank_from_artifacts(
    query: str,
    limit: int = 10,
    root: Path | None = None,
    enable_concept_boost: bool = True,
    enable_evidence_boost: bool = True,
    enable_hard_negative_penalty: bool = True,
) -> list[ConceptRerankedResult]:
    """Load artifacts from disk and rerank datasets for the given query."""
    concepts, evidence_links = read_concept_artifacts(root)
    graph = build_concept_graph(concepts, evidence_links)
    return rerank_datasets(
        query=query,
        concepts=concepts,
        evidence_links=evidence_links,
        graph=graph,
        limit=limit,
        enable_concept_boost=enable_concept_boost,
        enable_evidence_boost=enable_evidence_boost,
        enable_hard_negative_penalty=enable_hard_negative_penalty,
    )
