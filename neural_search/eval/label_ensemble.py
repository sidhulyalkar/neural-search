"""Confidence-weighted vote aggregation and gold/silver/bronze qrels tiers.

Tier definitions:
  gold   — human-audited label
  silver — ≥3 non-abstaining LFs, avg_confidence ≥ 0.75, variance < 0.5
  bronze — everything else
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from neural_search.eval.evidence import LFVote


@dataclass
class EnsembleResult:
    label: int
    confidence: float
    tier: str                       # "bronze" before human audit
    hard_negative_triggered: bool
    disagreement: float             # variance of active vote labels
    active_vote_count: int
    audit_priority: float
    provenance: list[str]           # lf_names that influenced label


def aggregate_votes(votes: list[LFVote]) -> EnsembleResult:
    """Aggregate LF votes into a single EnsembleResult."""
    hard_neg = [v for v in votes if v.lf_name == "lf_hard_negative" and not v.abstain]
    if hard_neg:
        return EnsembleResult(
            label=0, confidence=0.95, tier="bronze",
            hard_negative_triggered=True, disagreement=0.0,
            active_vote_count=len([v for v in votes if not v.abstain]),
            audit_priority=0.0,
            provenance=["lf_hard_negative"],
        )

    active = [v for v in votes if not v.abstain]
    if not active:
        return EnsembleResult(
            label=1, confidence=0.30, tier="bronze",
            hard_negative_triggered=False, disagreement=0.0,
            active_vote_count=0, audit_priority=0.0, provenance=[],
        )

    total_weight = sum(v.confidence for v in active)
    weighted_label_f = sum(v.label * v.confidence for v in active) / total_weight
    label = round(weighted_label_f)
    avg_conf = total_weight / len(active)

    variance = sum((v.label - weighted_label_f) ** 2 for v in active) / len(active)

    if len(active) >= 3 and avg_conf >= 0.75 and variance < 0.5:
        tier = "silver"
    else:
        tier = "bronze"

    provenance = [v.lf_name for v in active]

    return EnsembleResult(
        label=label,
        confidence=min(avg_conf, 1.0),
        tier=tier,
        hard_negative_triggered=False,
        disagreement=variance,
        active_vote_count=len(active),
        audit_priority=0.0,
        provenance=provenance,
    )


def assign_tier(result: EnsembleResult, human_audited: bool) -> str:
    """Upgrade tier to gold if human-audited; enforce silver requirements."""
    if human_audited:
        return "gold"
    if result.tier == "silver" and result.active_vote_count < 3:
        return "bronze"
    return result.tier


def compute_audit_priority(result: EnsembleResult, min_rank: int) -> float:
    """Higher = more urgent to audit.

    Factors: disagreement, hard-negative flag, proximity to top of rank list.
    """
    rank_boost = 1.0 / (math.log(min_rank + 1) + 1.0)
    hn_factor = 2.0 if result.hard_negative_triggered else 1.0
    return result.disagreement * hn_factor * rank_boost


def make_qrel(
    query_id: str,
    record_id: str,
    result: EnsembleResult,
    tier: str,
) -> dict:
    """Return a JSONL-ready qrel dict."""
    from datetime import datetime, timezone
    return {
        "query_id": query_id,
        "record_id": record_id,
        "label": result.label,
        "confidence": round(result.confidence, 4),
        "source": tier,
        "provenance": result.provenance,
        "hard_negative_triggered": result.hard_negative_triggered,
        "disagreement": round(result.disagreement, 4),
        "created": datetime.now(timezone.utc).isoformat(),
    }


def compute_hard_negative_violations(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[tuple[str, float]]],
    cutoff: int = 10,
) -> dict[str, int]:
    """Count hard-negative (label=0) entries appearing in top-k of run.

    Returns {query_id: violation_count}.
    """
    violations: dict[str, int] = {}
    for qid, ranked in run.items():
        q_qrels = qrels.get(qid, {})
        hard_negatives = {rid for rid, lbl in q_qrels.items() if lbl == 0}
        top_k = [rid for rid, _ in ranked[:cutoff]]
        count = sum(1 for rid in top_k if rid in hard_negatives)
        if count:
            violations[qid] = count
    return violations
