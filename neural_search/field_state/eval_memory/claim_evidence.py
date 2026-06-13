"""Suggest field-claim evidence updates from qrels/eval artifacts."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from neural_search.field_state.eval_memory.qrels_schema import AdjudicatedQrel
from neural_search.field_state.schemas import FieldClaim
from neural_search.field_state.store import (
    ADJUDICATED_QRELS_PATH,
    CLAIM_EVIDENCE_SUGGESTIONS_PATH,
    CLAIM_EVIDENCE_UPDATE_REPORT,
    QRELS_AGREEMENT_PATH,
    read_claims,
    read_jsonl,
    resolve_path,
    write_jsonl,
)


class ClaimEvidenceSuggestion(BaseModel):
    """Suggested update to a field claim."""

    claim_id: str
    current_evidence_level: str
    suggested_evidence_level: str
    reason: str
    supporting_artifacts: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    safe_to_auto_apply: bool = False
    schema_version: str = "0.3"


def _suggest_for_claim(
    claim: FieldClaim,
    adjudicated_count: int,
    root: Path | None,
) -> ClaimEvidenceSuggestion:
    supporting: list[str] = []
    missing: list[str] = []
    if resolve_path(ADJUDICATED_QRELS_PATH, root).exists():
        supporting.append(str(ADJUDICATED_QRELS_PATH))
    else:
        missing.append(str(ADJUDICATED_QRELS_PATH))
    if resolve_path(QRELS_AGREEMENT_PATH, root).exists():
        supporting.append(str(QRELS_AGREEMENT_PATH))
    else:
        missing.append(str(QRELS_AGREEMENT_PATH))

    suggested = str(claim.evidence_level)
    reason = "No qrels-backed evidence update available yet."
    if claim.claim_id == "claim_human_qrels":
        if adjudicated_count >= 20:
            suggested = "supported"
            reason = "Adjudicated qrels exist at a useful threshold for metric credibility."
        elif adjudicated_count > 0:
            suggested = "plausible"
            reason = "Some adjudicated qrels exist, but more labels are needed."
        else:
            reason = "Human qrels claim remains a design requirement until adjudicated qrels exist."
    elif claim.claim_id == "claim_hard_negatives":
        reason = "Needs hard-negative violation metrics linked to adjudicated qrels."
        missing.append("reports/field_state/hard_negative_review.md")
    elif claim.claim_id == "claim_dense_semantic_retrieval":
        reason = "Needs qrels-backed ablation comparing dense retrieval against baselines."
        missing.append("reports/eval/dense_ablation.json")
    elif claim.claim_id == "claim_affordance_extraction":
        reason = "Needs content or human validation of analysis affordance labels."
        missing.append("reports/field_state/affordance_validation.md")
    elif claim.claim_id == "claim_metadata_richness":
        reason = "Needs metadata quality scores correlated with reuse labels."
        missing.append("reports/field_state/metadata_quality_scoring.md")
    return ClaimEvidenceSuggestion(
        claim_id=claim.claim_id,
        current_evidence_level=str(claim.evidence_level),
        suggested_evidence_level=suggested,
        reason=reason,
        supporting_artifacts=sorted(set(supporting)),
        missing_artifacts=sorted(set(missing)),
    )


def build_claim_evidence_suggestions(
    root: Path | None = None,
) -> list[ClaimEvidenceSuggestion]:
    """Build claim evidence suggestions from available eval artifacts."""
    adjudicated = read_jsonl(ADJUDICATED_QRELS_PATH, AdjudicatedQrel, root)
    return [
        _suggest_for_claim(claim, len(adjudicated), root)
        for claim in read_claims(root)
    ]


def render_claim_evidence_suggestions(
    suggestions: list[ClaimEvidenceSuggestion],
) -> str:
    """Render claim evidence suggestions Markdown."""
    lines = ["# Claim Evidence Update Suggestions", ""]
    for suggestion in suggestions:
        lines.extend(
            [
                f"## {suggestion.claim_id}",
                "",
                f"- Current evidence level: `{suggestion.current_evidence_level}`",
                f"- Suggested evidence level: `{suggestion.suggested_evidence_level}`",
                f"- Safe to auto-apply: `{suggestion.safe_to_auto_apply}`",
                f"- Reason: {suggestion.reason}",
                "- Supporting artifacts:",
            ]
        )
        lines.extend(f"- `{item}`" for item in suggestion.supporting_artifacts or ["none"])
        lines.append("- Missing artifacts:")
        lines.extend(f"- `{item}`" for item in suggestion.missing_artifacts or ["none"])
        lines.append("")
    return "\n".join(lines)


def write_claim_evidence_suggestions(root: Path | None = None) -> list[ClaimEvidenceSuggestion]:
    """Write claim evidence suggestions JSONL and Markdown."""
    suggestions = build_claim_evidence_suggestions(root)
    write_jsonl(CLAIM_EVIDENCE_SUGGESTIONS_PATH, suggestions, root)
    report_path = resolve_path(CLAIM_EVIDENCE_UPDATE_REPORT, root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_claim_evidence_suggestions(suggestions), encoding="utf-8")
    return suggestions
