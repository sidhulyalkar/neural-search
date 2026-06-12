"""Multi-judge consensus for neuro_judge.

build_consensus(judgments, ...) → (ConsensusResult | None, ConflictRecord | None)

Rules (from spec):
  - Exact agreement + mean_confidence >= 0.75  → neuro_judge_consensus
  - Label diff == 1 + mean_confidence >= 0.80  → consensus w/ minor_disagreement flag
  - Label diff >= 2                             → conflict
  - HN detection differs                        → conflict
  - Label in {2,3} but required dim missing     → human audit queue (flagged in consensus)
  - High NDCG@10 impact                         → human audit queue

Single-judge inputs produce a consensus result directly (no conflict possible).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from neural_search.eval.neuro_judge.evidence_packet import (
    NEURO_JUDGE_WATERMARK,
    ConflictRecord,
    ConsensusResult,
    NeuroJudgment,
)

# ---------------------------------------------------------------------------
# Routing thresholds (from spec)
# ---------------------------------------------------------------------------

_EXACT_AGREE_CONF_THRESHOLD = 0.75
_MINOR_DIFF_CONF_THRESHOLD = 0.80
_HIGH_NDCG_IMPACT_THRESHOLD = 0.10  # if ndcg_impact > this, flag for human


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mean_confidence(judgments: Sequence[NeuroJudgment]) -> float:
    return statistics.mean(j.confidence for j in judgments)


def _median_label(judgments: Sequence[NeuroJudgment]) -> int:
    labels = sorted(j.label for j in judgments)
    mid = len(labels) // 2
    if len(labels) % 2 == 0:
        return round((labels[mid - 1] + labels[mid]) / 2)
    return labels[mid]


def _merge_lists(judgments: Sequence[NeuroJudgment], field: str) -> list[str]:
    seen: dict[str, None] = {}
    for j in judgments:
        for item in getattr(j, field, []):
            seen[item] = None
    return list(seen)


def _build_consensus_result(
    judgments: Sequence[NeuroJudgment],
    label: int,
    exact_agreement: bool,
    minor_disagreement: bool,
    ndcg_impact: float,
) -> ConsensusResult:
    mean_conf = _mean_confidence(judgments)
    hn_detected = any(j.hard_negative_detected for j in judgments)
    packet_hash = judgments[0].evidence_packet_hash if judgments else ""
    query_id = judgments[0].query_id
    dataset_id = judgments[0].dataset_id

    needs_human_review = (
        ndcg_impact > _HIGH_NDCG_IMPACT_THRESHOLD
        or (
            label >= 2
            and (
                any(j.missing_information for j in judgments)
                or any(j.required_dimensions_missing for j in judgments)
                or any(j.abstain_recommended for j in judgments)
            )
        )
    )

    provenance = "neuro_judge_consensus" if len(judgments) > 1 else (
        "neuro_judge_rag" if judgments[0].label_provenance == "neuro_judge_rag" else "neuro_judge"
    )

    # Annotate if needs human audit
    failure_modes = _merge_lists(judgments, "failure_modes")
    if needs_human_review:
        failure_modes = list(dict.fromkeys(failure_modes + ["needs_human_review"]))
    required_present = _merge_lists(judgments, "required_dimensions_present")
    required_missing = _merge_lists(judgments, "required_dimensions_missing")
    abstain_recommended = any(j.abstain_recommended for j in judgments) or (
        label >= 2 and bool(required_missing)
    )
    abstain_reasons = [
        str(j.abstain_reason)
        for j in judgments
        if j.abstain_reason
    ]

    return ConsensusResult(
        query_id=query_id,
        dataset_id=dataset_id,
        label=label,
        confidence=mean_conf,
        label_provenance=provenance,
        judge_count=len(judgments),
        exact_agreement=exact_agreement,
        minor_disagreement=minor_disagreement,
        hard_negative_detected=hn_detected,
        rationale_short=" | ".join(
            j.rationale_short for j in judgments if j.rationale_short
        )[:512],
        evidence_for=_merge_lists(judgments, "evidence_for"),
        evidence_against=_merge_lists(judgments, "evidence_against"),
        missing_information=_merge_lists(judgments, "missing_information"),
        matched_dimensions=_merge_lists(judgments, "matched_dimensions"),
        failure_modes=failure_modes,
        evidence_completeness=round(
            statistics.mean(j.evidence_completeness for j in judgments), 4
        ),
        required_dimensions_present=required_present,
        required_dimensions_missing=required_missing,
        abstain_recommended=abstain_recommended,
        abstain_reason="; ".join(dict.fromkeys(abstain_reasons))[:256] or None,
        judge_models=[j.judge_model for j in judgments],
        prompt_version=judgments[0].prompt_version if judgments else "v1",
        evidence_packet_hash=packet_hash,
        watermark=NEURO_JUDGE_WATERMARK,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_consensus(
    judgments: list[NeuroJudgment],
    ndcg_impact: float = 0.0,
) -> tuple[ConsensusResult | None, ConflictRecord | None]:
    """Apply consensus rules and return (consensus, conflict).

    Exactly one of the two will be non-None.

    Args:
        judgments: All NeuroJudgment outputs for the same (query_id, dataset_id).
        ndcg_impact: Estimated delta NDCG@10 if this label is changed.
    Returns:
        Tuple of (ConsensusResult | None, ConflictRecord | None).
    """
    if not judgments:
        raise ValueError("judgments must be non-empty")

    # Validate homogeneity
    query_ids = {j.query_id for j in judgments}
    dataset_ids = {j.dataset_id for j in judgments}
    if len(query_ids) > 1 or len(dataset_ids) > 1:
        raise ValueError(
            f"All judgments must share the same (query_id, dataset_id). "
            f"Got query_ids={query_ids}, dataset_ids={dataset_ids}"
        )

    query_id = judgments[0].query_id
    dataset_id = judgments[0].dataset_id
    packet_hash = judgments[0].evidence_packet_hash

    # Single judge → direct consensus
    if len(judgments) == 1:
        j = judgments[0]
        consensus = _build_consensus_result(
            judgments, j.label, exact_agreement=False, minor_disagreement=False, ndcg_impact=ndcg_impact
        )
        return consensus, None

    labels = [j.label for j in judgments]
    label_min, label_max = min(labels), max(labels)
    diff = label_max - label_min
    mean_conf = _mean_confidence(judgments)

    # Check HN detection homogeneity
    hn_detections = {j.hard_negative_detected for j in judgments}
    if len(hn_detections) > 1:
        return None, ConflictRecord(
            query_id=query_id,
            dataset_id=dataset_id,
            judgments=list(judgments),
            conflict_reason="hn_detection_differs",
            ndcg_impact=ndcg_impact,
            evidence_packet_hash=packet_hash,
        )

    # Exact agreement
    if diff == 0 and mean_conf >= _EXACT_AGREE_CONF_THRESHOLD:
        consensus = _build_consensus_result(
            judgments, labels[0], exact_agreement=True, minor_disagreement=False, ndcg_impact=ndcg_impact
        )
        return consensus, None

    # Exact agreement but low confidence → still return consensus, mark minor
    if diff == 0 and mean_conf < _EXACT_AGREE_CONF_THRESHOLD:
        consensus = _build_consensus_result(
            judgments, labels[0], exact_agreement=True, minor_disagreement=True, ndcg_impact=ndcg_impact
        )
        return consensus, None

    # Minor disagreement (diff == 1)
    if diff == 1 and mean_conf >= _MINOR_DIFF_CONF_THRESHOLD:
        consensus = _build_consensus_result(
            judgments,
            _median_label(judgments),
            exact_agreement=False,
            minor_disagreement=True,
            ndcg_impact=ndcg_impact,
        )
        return consensus, None

    # Major disagreement (diff >= 2) or minor diff with low confidence → conflict
    conflict_reason = (
        "label_diff_gte_2" if diff >= 2 else "minor_diff_low_confidence"
    )
    return None, ConflictRecord(
        query_id=query_id,
        dataset_id=dataset_id,
        judgments=list(judgments),
        conflict_reason=conflict_reason,
        ndcg_impact=ndcg_impact,
        evidence_packet_hash=packet_hash,
    )
