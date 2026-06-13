"""Evidence-backed result cards for scientific dataset search.

This module provides rich, auditable result cards that explain:
- Why a dataset matched the query
- What evidence supports the match (claim citations)
- What analysis affordances the dataset supports
- What requirements are met vs missing
- Any hard-negative flags or sense disambiguation issues

The goal is to move from "here are similar datasets" to
"here are reusable datasets with evidence."

Example result card:
    EvidenceResultCard(
        dataset_id="dandi:000026",
        rank=1,
        score=0.92,
        reusability_status="supported",
        matched_constructs=["delay_discounting"],
        matched_affordances=["delay_discounting_modeling"],
        matched_requirements=["choice", "reward_magnitude", "delay_duration"],
        missing_requirements=["reaction_time"],
        hard_negative_flags=[],
        evidence=[
            ClaimSummary(claim_id="claim:...", source="paper_methods", summary="...")
        ],
        explanation="This dataset supports delay discounting model fitting..."
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ReusabilityStatus(StrEnum):
    """Status of a dataset's reusability for a query."""

    SUPPORTED = "supported"        # All requirements met
    PARTIAL = "partial"            # Most requirements met
    UNSUPPORTED = "unsupported"    # Missing critical requirements
    UNKNOWN = "unknown"            # Cannot determine


@dataclass
class ClaimSummary:
    """Summary of a claim supporting a result."""

    claim_id: str
    source_type: str  # paper_methods, file_inspection, archive_metadata, etc.
    predicate: str    # has_task, has_variable, supports_affordance, etc.
    object_label: str  # What was claimed
    summary: str       # Human-readable summary
    confidence: float = 1.0


@dataclass
class AffordanceMatch:
    """Result of affordance validation for a result."""

    affordance_id: str
    affordance_label: str
    status: str  # supported, partial, unsupported
    confidence: float
    matched_requirements: list[str]
    missing_requirements: list[str]
    hard_blockers: list[str]


@dataclass
class SenseMatch:
    """Sense disambiguation result for a result."""

    detected_sense: str | None
    sense_label: str | None
    is_hard_negative: bool = False
    penalty_applied: float = 0.0
    notes: str = ""


class EvidenceResultCard(BaseModel):
    """Evidence-backed result card for a search result.

    This is the rich result format that explains not just what matched,
    but why we believe the dataset is reusable for the target analysis.
    """

    # Identity
    dataset_id: str
    dataset_title: str
    source: str
    source_url: str | None = None

    # Ranking
    rank: int
    score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)

    # Reusability assessment
    reusability_status: ReusabilityStatus = ReusabilityStatus.UNKNOWN

    # Construct matches (what scientific concepts matched)
    matched_constructs: list[str] = Field(default_factory=list)
    partial_constructs: list[str] = Field(default_factory=list)

    # Affordance validation
    matched_affordances: list[str] = Field(default_factory=list)
    affordance_details: list[dict[str, Any]] = Field(default_factory=list)

    # Requirement validation
    matched_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    optional_missing: list[str] = Field(default_factory=list)

    # Hard negatives and sense disambiguation
    hard_negative_flags: list[str] = Field(default_factory=list)
    sense_match: dict[str, Any] | None = None

    # Evidence claims
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_claim_ids: list[str] = Field(default_factory=list)

    # Human-readable explanation
    explanation: str = ""
    brief_explanation: str = ""

    # Quality indicators
    evidence_quality: str = "unknown"  # strong, moderate, weak
    metadata_completeness: float = 0.0

    # Warnings
    warnings: list[str] = Field(default_factory=list)

    # Timestamps
    generated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


def build_evidence_card(
    dataset: dict[str, Any],
    query: str,
    rank: int,
    score: float,
    score_breakdown: dict[str, float] | None = None,
    claims: list[dict[str, Any]] | None = None,
    affordance_results: list[dict[str, Any]] | None = None,
    sense_result: dict[str, Any] | None = None,
    required_constructs: list[str] | None = None,
    must_have: list[str] | None = None,
    should_have: list[str] | None = None,
) -> EvidenceResultCard:
    """Build an evidence-backed result card.

    Args:
        dataset: Dataset metadata dictionary
        query: The search query
        rank: Result rank (1-indexed)
        score: Final relevance score
        score_breakdown: Component scores
        claims: List of claims supporting this result
        affordance_results: List of affordance validation results
        sense_result: Sense disambiguation result
        required_constructs: Constructs the query needs
        must_have: Must-have variables from query
        should_have: Should-have variables from query

    Returns:
        EvidenceResultCard with full evidence trail
    """
    dataset_id = dataset.get("dataset_id", "unknown")
    title = dataset.get("title", "Untitled")
    source = dataset.get("source", "unknown")

    # Process claims
    evidence_list = []
    evidence_claim_ids = []
    if claims:
        for claim in claims:
            evidence_list.append({
                "claim_id": claim.get("claim_id", ""),
                "source_type": claim.get("source_type", "unknown"),
                "predicate": claim.get("predicate", ""),
                "object_label": claim.get("object_label", ""),
                "summary": claim.get("evidence_text", ""),
                "confidence": claim.get("confidence", 0.5),
            })
            if claim.get("claim_id"):
                evidence_claim_ids.append(claim["claim_id"])

    # Process affordance results
    matched_affordances = []
    affordance_details = []
    if affordance_results:
        for result in affordance_results:
            if result.get("supported"):
                matched_affordances.append(result.get("affordance_id", ""))
            affordance_details.append({
                "affordance_id": result.get("affordance_id"),
                "status": result.get("support_level", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "matched": result.get("found_required_features", []),
                "missing": result.get("missing_required_features", []),
            })

    # Check requirements
    matched_requirements = []
    missing_requirements = []
    optional_missing = []

    # Extract dataset variables/features
    dataset_features = set()
    for field_name in ["behavioral_events", "task", "modality", "data_standards"]:
        for item in dataset.get(field_name, []):
            if isinstance(item, str):
                dataset_features.add(item.lower())
            elif hasattr(item, "label"):
                dataset_features.add(item.label.lower())

    # Also check usability flags
    usability = dataset.get("usability", {})
    if isinstance(usability, dict):
        if usability.get("has_trials"):
            dataset_features.add("trial_structure")
            dataset_features.add("trial_id")
        if usability.get("has_behavior"):
            dataset_features.add("behavior")

    if must_have:
        for req in must_have:
            req_lower = req.lower()
            if req_lower in dataset_features or any(req_lower in f for f in dataset_features):
                matched_requirements.append(req)
            else:
                missing_requirements.append(req)

    if should_have:
        for opt in should_have:
            opt_lower = opt.lower()
            if opt_lower not in dataset_features and not any(opt_lower in f for f in dataset_features):
                optional_missing.append(opt)

    # Hard negative detection
    hard_negative_flags = []
    if sense_result:
        for _neg_sense in sense_result.get("negative_senses", []):
            # Check if dataset matches a negative sense
            # This would require checking dataset metadata against sense definitions
            pass  # Placeholder for actual implementation

    # Determine reusability status
    if not missing_requirements and matched_affordances:
        status = ReusabilityStatus.SUPPORTED
    elif len(missing_requirements) <= 1 and (matched_affordances or matched_requirements):
        status = ReusabilityStatus.PARTIAL
    elif hard_negative_flags:
        status = ReusabilityStatus.UNSUPPORTED
    elif missing_requirements:
        status = ReusabilityStatus.UNSUPPORTED
    else:
        status = ReusabilityStatus.UNKNOWN

    # Build explanation
    explanation_parts = []

    if matched_affordances:
        explanation_parts.append(
            f"Supports: {', '.join(matched_affordances)}"
        )

    if matched_requirements:
        explanation_parts.append(
            f"Has: {', '.join(matched_requirements)}"
        )

    if missing_requirements:
        explanation_parts.append(
            f"Missing: {', '.join(missing_requirements)}"
        )

    if evidence_list:
        explanation_parts.append(
            f"Evidence from {len(evidence_list)} source(s)"
        )

    explanation = ". ".join(explanation_parts) if explanation_parts else "Limited evidence available."

    # Brief explanation
    if status == ReusabilityStatus.SUPPORTED:
        brief = "Dataset supports the requested analysis."
    elif status == ReusabilityStatus.PARTIAL:
        brief = f"Dataset partially supports the analysis (missing {len(missing_requirements)} requirement(s))."
    else:
        brief = "Dataset may not fully support the requested analysis."

    # Evidence quality
    if len(evidence_list) >= 3 and all(e.get("confidence", 0) > 0.8 for e in evidence_list):
        evidence_quality = "strong"
    elif evidence_list:
        evidence_quality = "moderate"
    else:
        evidence_quality = "weak"

    # Metadata completeness
    total_fields = len(["title", "description", "species", "modality", "task", "brain_region"])
    filled_fields = sum(1 for f in ["title", "description"] if dataset.get(f))
    filled_fields += sum(1 for f in ["species", "modality", "task", "brain_region"] if dataset.get(f, []))
    metadata_completeness = filled_fields / total_fields if total_fields > 0 else 0.0

    # Warnings
    warnings = []
    if evidence_quality == "weak":
        warnings.append("Limited evidence available for this match")
    if hard_negative_flags:
        warnings.append(f"May match wrong sense of query term: {', '.join(hard_negative_flags)}")
    if metadata_completeness < 0.5:
        warnings.append("Limited metadata available")

    return EvidenceResultCard(
        dataset_id=dataset_id,
        dataset_title=title,
        source=source,
        source_url=dataset.get("source_url"),
        rank=rank,
        score=score,
        score_breakdown=score_breakdown or {},
        reusability_status=status,
        matched_constructs=required_constructs or [],
        matched_affordances=matched_affordances,
        affordance_details=affordance_details,
        matched_requirements=matched_requirements,
        missing_requirements=missing_requirements,
        optional_missing=optional_missing,
        hard_negative_flags=hard_negative_flags,
        sense_match=sense_result,
        evidence=evidence_list,
        evidence_claim_ids=evidence_claim_ids,
        explanation=explanation,
        brief_explanation=brief,
        evidence_quality=evidence_quality,
        metadata_completeness=metadata_completeness,
        warnings=warnings,
    )


def format_evidence_card_text(card: EvidenceResultCard) -> str:
    """Format an evidence card as human-readable text."""
    lines = []

    lines.append(f"[{card.rank}] {card.dataset_title} ({card.dataset_id})")
    lines.append(f"    Score: {card.score:.2f} | Status: {card.reusability_status.value}")

    if card.matched_affordances:
        lines.append(f"    Affordances: {', '.join(card.matched_affordances)}")

    if card.matched_requirements:
        lines.append(f"    ✓ Has: {', '.join(card.matched_requirements)}")

    if card.missing_requirements:
        lines.append(f"    ✗ Missing: {', '.join(card.missing_requirements)}")

    if card.evidence:
        lines.append(f"    Evidence: {len(card.evidence)} claim(s) from {', '.join({e['source_type'] for e in card.evidence})}")

    if card.warnings:
        for warning in card.warnings:
            lines.append(f"    ⚠ {warning}")

    lines.append(f"    {card.explanation}")

    return "\n".join(lines)


def format_evidence_cards_report(cards: list[EvidenceResultCard]) -> str:
    """Format multiple evidence cards as a report."""
    if not cards:
        return "No results found."

    lines = []
    lines.append(f"Found {len(cards)} results:\n")

    for card in cards:
        lines.append(format_evidence_card_text(card))
        lines.append("")  # Blank line between cards

    # Summary statistics
    supported = sum(1 for c in cards if c.reusability_status == ReusabilityStatus.SUPPORTED)
    partial = sum(1 for c in cards if c.reusability_status == ReusabilityStatus.PARTIAL)
    unsupported = sum(1 for c in cards if c.reusability_status == ReusabilityStatus.UNSUPPORTED)

    lines.append("---")
    lines.append(f"Summary: {supported} supported, {partial} partial, {unsupported} unsupported")

    return "\n".join(lines)
