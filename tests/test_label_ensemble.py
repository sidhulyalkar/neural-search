"""Tests for label ensemble and qrels tier assignment."""
from __future__ import annotations

from neural_search.eval.evidence import LFVote
from neural_search.eval.label_ensemble import (
    EnsembleResult,
    aggregate_votes,
    assign_tier,
    compute_audit_priority,
)


def _vote(lf_name: str, label: int, conf: float, abstain: bool = False) -> LFVote:
    return LFVote(lf_name=lf_name, label=label, confidence=conf,
                  rationale="test", abstain=abstain)


class TestAggregateVotes:
    def test_hard_negative_override(self):
        votes = [
            _vote("lf_hard_negative", 0, 0.95),
            _vote("lf_required_modality", 3, 0.90),
        ]
        result = aggregate_votes(votes)
        assert result.label == 0
        assert result.hard_negative_triggered is True

    def test_all_abstain_returns_bronze(self):
        votes = [_vote(f"lf_{i}", 0, 0.0, abstain=True) for i in range(5)]
        result = aggregate_votes(votes)
        assert result.tier == "bronze"
        assert result.label in (0, 1)

    def test_strong_agreement_gives_silver(self):
        votes = [
            _vote("lf_required_modality", 3, 0.90),
            _vote("lf_species_constraint", 3, 0.85),
            _vote("lf_raw_data_available", 3, 0.70),
            _vote("lf_license_reusable", 3, 0.85),
        ]
        result = aggregate_votes(votes)
        assert result.tier == "silver"
        assert result.label == 3

    def test_high_disagreement_gives_bronze(self):
        votes = [
            _vote("lf_a", 0, 0.80),
            _vote("lf_b", 3, 0.80),
            _vote("lf_c", 0, 0.80),
        ]
        result = aggregate_votes(votes)
        assert result.tier == "bronze"

    def test_weighted_average_rounds_correctly(self):
        votes = [
            _vote("lf_a", 2, 0.90),
            _vote("lf_b", 2, 0.90),
            _vote("lf_c", 2, 0.90),
        ]
        result = aggregate_votes(votes)
        assert result.label == 2


class TestAssignTier:
    def test_gold_when_human_audited(self):
        result = EnsembleResult(label=2, confidence=0.9, tier="silver",
                                hard_negative_triggered=False, disagreement=0.1,
                                active_vote_count=3, audit_priority=0.0,
                                provenance=[])
        tier = assign_tier(result, human_audited=True)
        assert tier == "gold"

    def test_silver_requires_3_active_votes(self):
        result = EnsembleResult(label=2, confidence=0.9, tier="silver",
                                hard_negative_triggered=False, disagreement=0.1,
                                active_vote_count=2, audit_priority=0.0,
                                provenance=[])
        tier = assign_tier(result, human_audited=False)
        assert tier == "bronze"


class TestAuditPriority:
    def test_hard_neg_raises_priority(self):
        result_with_hn = EnsembleResult(label=0, confidence=0.5, tier="bronze",
                                        hard_negative_triggered=True, disagreement=1.0,
                                        active_vote_count=2, audit_priority=0.0,
                                        provenance=[])
        result_without_hn = EnsembleResult(label=2, confidence=0.8, tier="silver",
                                           hard_negative_triggered=False, disagreement=0.0,
                                           active_vote_count=4, audit_priority=0.0,
                                           provenance=[])
        p_with = compute_audit_priority(result_with_hn, min_rank=1)
        p_without = compute_audit_priority(result_without_hn, min_rank=100)
        assert p_with > p_without
