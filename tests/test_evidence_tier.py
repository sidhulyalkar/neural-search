"""Tests for the evidence-tier taxonomy schema."""

from __future__ import annotations

import pytest

from neural_search.kg.schemas.evidence_tier import (
    EvidenceTier,
    is_trustworthy,
    tier_rank,
    upgrade_tier,
)


def test_tier_rank_orders_weakest_to_strongest():
    assert tier_rank(EvidenceTier.HEURISTIC_CANDIDATE) < tier_rank(EvidenceTier.EVIDENCE_BACKED_BRIDGE)
    assert tier_rank(EvidenceTier.EVIDENCE_BACKED_BRIDGE) < tier_rank(EvidenceTier.SOURCE_DECLARED)
    assert tier_rank(EvidenceTier.SOURCE_DECLARED) < tier_rank(EvidenceTier.FILE_VALIDATED)
    assert tier_rank(EvidenceTier.FILE_VALIDATED) < tier_rank(EvidenceTier.HUMAN_VALIDATED)
    assert tier_rank(EvidenceTier.HUMAN_VALIDATED) < tier_rank(EvidenceTier.COMPUTED)


def test_only_top_three_tiers_are_trustworthy():
    assert not is_trustworthy(EvidenceTier.HEURISTIC_CANDIDATE)
    assert not is_trustworthy(EvidenceTier.EVIDENCE_BACKED_BRIDGE)
    assert not is_trustworthy(EvidenceTier.SOURCE_DECLARED)
    assert is_trustworthy(EvidenceTier.FILE_VALIDATED)
    assert is_trustworthy(EvidenceTier.HUMAN_VALIDATED)
    assert is_trustworthy(EvidenceTier.COMPUTED)


def test_upgrade_tier_never_downgrades():
    assert upgrade_tier(EvidenceTier.FILE_VALIDATED, EvidenceTier.HEURISTIC_CANDIDATE) == (
        EvidenceTier.FILE_VALIDATED
    )
    assert upgrade_tier(EvidenceTier.HEURISTIC_CANDIDATE, EvidenceTier.FILE_VALIDATED) == (
        EvidenceTier.FILE_VALIDATED
    )


def test_accepts_plain_strings():
    assert tier_rank("computed") == tier_rank(EvidenceTier.COMPUTED)
    assert is_trustworthy("file_validated") is True


def test_rejects_unknown_tier():
    with pytest.raises(ValueError):
        tier_rank("platinum_standard")
