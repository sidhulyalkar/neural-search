"""Utilities for summarizing evidence from concept nodes and evidence links."""

from __future__ import annotations

from typing import TypedDict

from neural_search.field_state.concept_memory.schema import ConceptNode, EvidenceLink


class EvidenceSummary(TypedDict):
    concept_id: str
    outgoing_links: int
    incoming_links: int
    linked_claims: list[str]
    linked_datasets: list[str]
    linked_papers: list[str]
    linked_opportunities: list[str]
    reviewed_link_count: int
    top_evidence_texts: list[str]


def get_concept_evidence_links(
    concept_id_str: str,
    evidence_links: list[EvidenceLink],
) -> list[EvidenceLink]:
    """Return all evidence links where source_concept_id == concept_id_str."""
    return [lnk for lnk in evidence_links if lnk.source_concept_id == concept_id_str]


def get_incoming_evidence_links(
    concept_id_str: str,
    evidence_links: list[EvidenceLink],
) -> list[EvidenceLink]:
    """Return all evidence links where target_concept_id == concept_id_str."""
    return [lnk for lnk in evidence_links if lnk.target_concept_id == concept_id_str]


def summarize_evidence(
    concept: ConceptNode,
    evidence_links: list[EvidenceLink],
    concepts_by_id: dict[str, ConceptNode],
) -> EvidenceSummary:
    """Return a summary dict of evidence for a concept node.

    Keys:
        concept_id, outgoing_links, incoming_links,
        linked_claims, linked_datasets, linked_papers, linked_opportunities,
        reviewed_link_count, top_evidence_texts
    """
    outgoing = get_concept_evidence_links(concept.concept_id, evidence_links)
    incoming = get_incoming_evidence_links(concept.concept_id, evidence_links)

    linked_claims: list[str] = []
    linked_datasets: list[str] = []
    linked_papers: list[str] = []
    linked_opportunities: list[str] = []

    for lnk in outgoing:
        tgt_id = lnk.target_concept_id
        if tgt_id is None:
            continue
        node = concepts_by_id.get(tgt_id)
        if node is None:
            continue
        if node.concept_type == "claim":
            linked_claims.append(tgt_id)
        elif node.concept_type == "dataset":
            linked_datasets.append(tgt_id)
        elif node.concept_type == "paper":
            linked_papers.append(tgt_id)
        elif node.concept_type == "opportunity":
            linked_opportunities.append(tgt_id)

    all_links = outgoing + incoming
    reviewed_link_count = sum(
        1 for lnk in all_links if lnk.review_status != "unreviewed"
    )

    top_evidence_texts: list[str] = []
    for lnk in all_links:
        if lnk.evidence_text is not None:
            top_evidence_texts.append(lnk.evidence_text)
            if len(top_evidence_texts) >= 5:
                break

    return {
        "concept_id": concept.concept_id,
        "outgoing_links": len(outgoing),
        "incoming_links": len(incoming),
        "linked_claims": linked_claims,
        "linked_datasets": linked_datasets,
        "linked_papers": linked_papers,
        "linked_opportunities": linked_opportunities,
        "reviewed_link_count": reviewed_link_count,
        "top_evidence_texts": top_evidence_texts,
    }


def evidence_strength_from_count(reviewed_count: int, total_count: int) -> str:
    """Conservative evidence strength from reviewed/total link counts.

    - 0 total → "none"
    - total > 0 but reviewed == 0 → "weak"
    - reviewed >= 1 and total >= 2 → "moderate"
    - reviewed >= 3 and total >= 5 → "strong"
    """
    if total_count == 0:
        return "none"
    if reviewed_count == 0:
        return "weak"
    if reviewed_count >= 3 and total_count >= 5:
        return "strong"
    if reviewed_count >= 1 and total_count >= 2:
        return "moderate"
    return "weak"
