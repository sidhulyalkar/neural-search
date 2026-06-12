"""Retrieval layer for Graph-Indexed Concept Memory.

Supports lexical search with an optional graph connectivity boost.
No LLM calls, no network calls.
"""

from __future__ import annotations

import re
from math import log1p

import networkx as nx  # type: ignore[import-untyped]

from neural_search.field_state.concept_memory.evidence import (
    get_concept_evidence_links,
    get_incoming_evidence_links,
)
from neural_search.field_state.concept_memory.schema import (
    ConceptNode,
    ConceptSearchResult,
    EvidenceLink,
)

_TOKEN_RE = re.compile(r"[^\w]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace/punctuation."""
    return [t for t in _TOKEN_RE.split(text.lower()) if t]


def _lexical_score(query: str, concept: ConceptNode) -> float:
    """Score a concept against a query using simple lexical matching.

    Algorithm:
    1. Normalize query to lowercase tokens (split on whitespace + punctuation).
    2. For each token check presence in:
       - canonical_name  (weight 3.0)
       - any alias       (weight 2.0)
       - description     (weight 1.0)
       - tags            (weight 1.5)
    3. Sum matched token weights, normalize by number of query tokens.
    4. Return float clipped to [0.0, 1.0].
    """
    tokens = _tokenize(query)
    if not tokens:
        return 0.0

    canonical_lower = concept.canonical_name.lower()
    aliases_lower = [a.lower() for a in concept.aliases]
    description_lower = (concept.description or "").lower()
    tags_lower = [t.lower() for t in concept.tags]

    total_weight = 0.0
    for token in tokens:
        matched = 0.0
        if token in canonical_lower:
            matched = max(matched, 3.0)
        for alias in aliases_lower:
            if token in alias:
                matched = max(matched, 2.0)
        if token in description_lower:
            matched = max(matched, 1.0)
        for tag in tags_lower:
            if token in tag:
                matched = max(matched, 1.5)
        total_weight += matched

    # Normalise by max possible weight (3.0 per token) then clip
    max_possible = 3.0 * len(tokens)
    score = total_weight / max_possible
    return min(score, 1.0)


def _matched_terms(query: str, concept: ConceptNode) -> list[str]:
    """Return query tokens that matched indexed text fields."""
    tokens = _tokenize(query)
    canonical_lower = concept.canonical_name.lower()
    aliases_lower = [a.lower() for a in concept.aliases]
    description_lower = (concept.description or "").lower()
    tags_lower = [t.lower() for t in concept.tags]
    matched: list[str] = []
    for token in tokens:
        if (
            token in canonical_lower
            or any(token in alias for alias in aliases_lower)
            or token in description_lower
            or any(token in tag for tag in tags_lower)
        ):
            matched.append(token)
    return list(dict.fromkeys(matched))


def _graph_boost_components(
    concept: ConceptNode,
    evidence_links: list[EvidenceLink],
    graph: nx.DiGraph | None = None,
) -> tuple[float, float]:
    """Compute a small boost based on graph connectivity.

    Boost rules (additive, capped at 0.3, then degree-normalized):
    + 0.05 for each reviewed evidence link (max 0.15)
    + 0.02 for each connected claim (max 0.10)
    + 0.02 for each connected dataset (max 0.10)
    """
    outgoing = get_concept_evidence_links(concept.concept_id, evidence_links)
    incoming = get_incoming_evidence_links(concept.concept_id, evidence_links)
    all_links = outgoing + incoming

    reviewed_count = sum(1 for lnk in all_links if lnk.review_status != "unreviewed")
    reviewed_boost = min(reviewed_count * 0.05, 0.15)

    connected_ids: set[str] = set()
    for lnk in outgoing:
        if lnk.target_concept_id is not None:
            connected_ids.add(lnk.target_concept_id)
    for lnk in incoming:
        connected_ids.add(lnk.source_concept_id)

    claim_count = 0
    dataset_count = 0
    if graph is not None:
        for nid in connected_ids:
            if nid in graph:
                ctype = graph.nodes[nid].get("concept_type", "")
                if ctype == "claim":
                    claim_count += 1
                elif ctype == "dataset":
                    dataset_count += 1

    claim_boost = min(claim_count * 0.02, 0.10)
    dataset_boost = min(dataset_count * 0.02, 0.10)

    raw_boost = min(reviewed_boost + claim_boost + dataset_boost, 0.3)
    degree = len(connected_ids)
    if graph is not None and concept.concept_id in graph:
        degree = max(degree, int(graph.degree(concept.concept_id)))
    normalizer = 1.0 + log1p(max(degree - 1, 0))
    normalized_boost = min(raw_boost / normalizer, 0.3)
    return raw_boost, normalized_boost


def _graph_boost(
    concept: ConceptNode,
    evidence_links: list[EvidenceLink],
    graph: nx.DiGraph | None = None,
) -> float:
    """Return the degree-normalized graph boost."""
    _, normalized = _graph_boost_components(concept, evidence_links, graph)
    return normalized


def _missingness_penalty(concept: ConceptNode) -> tuple[float, list[str]]:
    """Return a bounded penalty and warnings for sparse concept metadata."""
    penalty = 0.0
    warnings: list[str] = []
    if concept.evidence_count == 0:
        penalty += 0.03
        warnings.append("no recorded evidence_count")
    if not concept.description:
        penalty += 0.03
        warnings.append("missing description")
    if not concept.source_ids and not concept.source_artifacts and not concept.source_note_paths:
        penalty += 0.02
        warnings.append("missing source provenance")
    if concept.review_status == "unreviewed":
        penalty += 0.02
        warnings.append("unreviewed concept")
    return min(penalty, 0.10), warnings


def get_top_evidence_texts(
    concept_id_str: str,
    evidence_links: list[EvidenceLink],
    max_items: int = 5,
) -> list[str]:
    """Return up to max_items non-None evidence_text from matching links."""
    texts: list[str] = []
    for lnk in evidence_links:
        if lnk.source_concept_id == concept_id_str and lnk.evidence_text is not None:
            texts.append(lnk.evidence_text)
            if len(texts) >= max_items:
                break
    return texts


def search_concepts(
    query: str,
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
    graph: nx.DiGraph | None = None,
    limit: int = 10,
    min_score: float = 0.01,
    concept_type_filter: str | None = None,
) -> list[ConceptSearchResult]:
    """Search concepts using lexical scoring + graph boost.

    For each concept:
    1. Compute lexical_score
    2. Compute graph_boost
    3. final_score = lexical_score + graph_boost
    4. Skip if final_score < min_score
    5. Build ConceptSearchResult

    Returns up to `limit` results sorted by score descending.
    When concept_type_filter is set, only concepts of that type are scored.
    """
    results: list[ConceptSearchResult] = []

    for concept in concepts:
        if concept_type_filter is not None and concept.concept_type != concept_type_filter:
            continue

        lex = _lexical_score(query, concept)
        if lex <= 0.0:
            # Graph connectivity is a reranking signal, not standalone evidence
            # that a concept answers the user's text query.
            continue
        raw_boost, normalized_boost = _graph_boost_components(concept, evidence_links, graph)
        missingness_penalty, warnings = _missingness_penalty(concept)
        final = max(0.0, lex + normalized_boost - missingness_penalty)

        if final < min_score:
            continue

        top_evidence = get_top_evidence_texts(concept.concept_id, evidence_links)

        outgoing = get_concept_evidence_links(concept.concept_id, evidence_links)
        related_claims: list[str] = []
        related_datasets: list[str] = []
        source_note_paths: list[str] = list(concept.source_note_paths)

        for lnk in outgoing:
            tgt = lnk.target_concept_id
            if tgt is None:
                continue
            if graph is not None and tgt in graph:
                ctype = graph.nodes[tgt].get("concept_type", "")
                if ctype == "claim":
                    related_claims.append(tgt)
                elif ctype == "dataset":
                    related_datasets.append(tgt)
            if lnk.source_note_path and lnk.source_note_path not in source_note_paths:
                source_note_paths.append(lnk.source_note_path)

        results.append(
            ConceptSearchResult(
                concept_id=concept.concept_id,
                canonical_name=concept.canonical_name,
                concept_type=concept.concept_type,
                score=round(final, 6),
                lexical_score=round(lex, 6),
                graph_boost=round(normalized_boost, 6),
                graph_boost_raw=round(raw_boost, 6),
                graph_boost_degree_normalized=round(normalized_boost, 6),
                missingness_penalty=round(missingness_penalty, 6),
                final_score=round(final, 6),
                matched_terms=_matched_terms(query, concept),
                matched_concepts=[concept.concept_id],
                warnings=warnings,
                evidence_count=concept.evidence_count,
                top_evidence=top_evidence,
                related_claims=related_claims,
                related_datasets=related_datasets,
                source_note_paths=source_note_paths,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]
