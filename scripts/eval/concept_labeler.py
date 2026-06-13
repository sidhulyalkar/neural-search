"""Concept-memory weak labeler for silver qrels generation.

Uses Concept Memory outputs (matched concepts, missing evidence, hard-negative
conflicts) to produce a weak relevance vote without circularly treating Concept
Memory as ground truth.

Rules:
- Concept Memory contributes ONE vote among many; it does not dominate.
- Hard-negative conflicts from concept memory override positive concept matches.
- Unreviewed / metadata-derived links produce lower confidence.
- Missing concepts lead to abstain or low confidence, not false negatives.
- Concept coverage gaps are surfaced explicitly.
"""

from __future__ import annotations

from typing import Any

from scripts.eval.silver_qrels_schema import LabelingFunctionVote

# ---------------------------------------------------------------------------
# Types consumed from concept memory outputs
# ---------------------------------------------------------------------------
# The labeler operates on plain dicts so it does not require the full
# neural_search package to be importable in test environments.
#
# Expected structure of a concept_result dict (from ConceptRerankedResult or
# ConceptExplanation serialised to dict):
#   dataset_id: str
#   matched_concepts: list[{concept_id, canonical_name, concept_type, match_score,
#                            evidence_texts, relation_types}]
#   missing_evidence: list[str]       — concept types with no match
#   hard_negative_conflicts: list[str]  — descriptions of HN concept links
#   confidence: float (optional)
#   explanation_summary: str (optional)


_UNREVIEWED_PENALTY = 0.10  # confidence reduction for metadata-only / unreviewed links


def label_from_concept_result(
    query: Any,  # BenchmarkQueryV1 — typed as Any to avoid circular import in tests
    concept_result: dict[str, Any],
) -> LabelingFunctionVote:
    """Produce a weak relevance vote from a concept-memory result dict.

    Parameters
    ----------
    query:
        BenchmarkQueryV1 (or any object with .query_text, .hard_negatives,
        .expected_modalities, etc.).
    concept_result:
        Serialised ConceptRerankedResult / ConceptExplanation dict.

    Returns
    -------
    LabelingFunctionVote with source="concept_memory".
    """
    matched = concept_result.get("matched_concepts") or []
    missing_evidence = concept_result.get("missing_evidence") or []
    hn_conflicts = concept_result.get("hard_negative_conflicts") or []
    explanation = concept_result.get("explanation_summary", "")

    evidence: list[str] = []

    # ------------------------------------------------------------------
    # 1. Hard-negative conflicts override positive concept signals
    # ------------------------------------------------------------------
    if hn_conflicts:
        return LabelingFunctionVote(
            source="concept_memory",
            vote=0,
            confidence=0.80,
            rationale=f"concept hard-negative conflict: {hn_conflicts[:2]}",
            evidence=[f"concept_hn_conflict: {c}" for c in hn_conflicts[:3]],
        )

    # ------------------------------------------------------------------
    # 2. No matches at all → abstain
    # ------------------------------------------------------------------
    if not matched:
        return LabelingFunctionVote(
            source="concept_memory",
            vote=None,
            confidence=0.35,
            rationale="no concept matches; abstaining",
        )

    # ------------------------------------------------------------------
    # 3. Compute match quality and confidence
    # ------------------------------------------------------------------
    match_count = len(matched)
    avg_match_score = sum(
        float(c.get("match_score", 0.5)) for c in matched
    ) / match_count

    # Detect when all evidence comes from corpus metadata (not from papers/notes)
    metadata_only = sum(
        1 for c in matched
        if all(
            any(kw in str(et).lower() for kw in ("modali", "species", "brain_reg", "task"))
            for et in (c.get("evidence_texts") or [""]) or [""]
        )
    )

    confidence_base = min(0.50 + 0.08 * min(match_count, 5) + 0.15 * avg_match_score, 0.85)
    # Penalise for unreviewed / metadata-only links
    if metadata_only >= max(1, match_count // 2):
        confidence_base -= _UNREVIEWED_PENALTY
        evidence.append(f"concept_links_metadata_only: {metadata_only}/{match_count}")

    # Penalise for many missing concept types
    missing_penalty = 0.05 * len(missing_evidence)
    confidence_base = max(0.20, confidence_base - missing_penalty)

    # Build evidence list
    for c in matched[:5]:
        cname = c.get("canonical_name", c.get("concept_id", ""))
        ctype = c.get("concept_type", "")
        score = c.get("match_score", 0.0)
        evidence.append(f"concept_match: {cname} ({ctype}, score={score:.2f})")

    if missing_evidence:
        evidence.append(f"concept_coverage_gaps: {missing_evidence}")

    # ------------------------------------------------------------------
    # 4. Map match quality to vote
    # ------------------------------------------------------------------
    # Avoid vote=3 from concept memory alone — that requires human review.
    if avg_match_score >= 0.80 and match_count >= 3 and not missing_evidence:
        vote = 2
    elif avg_match_score >= 0.50 and match_count >= 1:
        vote = 2 if not missing_evidence else 1
    else:
        vote = 1

    rationale_parts = [
        f"{match_count} concept(s) matched (avg score {avg_match_score:.2f})",
    ]
    if missing_evidence:
        rationale_parts.append(f"missing coverage: {missing_evidence}")
    if explanation:
        rationale_parts.append(explanation[:120])

    return LabelingFunctionVote(
        source="concept_memory",
        vote=vote,
        confidence=max(0.15, min(confidence_base, 0.85)),
        rationale="; ".join(rationale_parts),
        evidence=evidence,
    )
