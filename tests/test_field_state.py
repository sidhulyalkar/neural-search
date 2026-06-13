from pathlib import Path

import pytest

from neural_search.field_state.cli import main
from neural_search.field_state.reports import (
    render_benchmark_gaps_report,
    render_top_opportunities_report,
    render_weak_claims_report,
)
from neural_search.field_state.schemas import (
    BenchmarkGap,
    EvidenceLevel,
    FieldClaim,
    FieldOpportunity,
)
from neural_search.field_state.scoring import rank_opportunities, score_opportunity
from neural_search.field_state.seeds import (
    seed_benchmark_gaps,
    seed_claims,
    seed_opportunities,
)
from neural_search.field_state.store import (
    BENCHMARK_GAPS_PATH,
    CLAIMS_PATH,
    LATEST_SNAPSHOT_REPORT,
    OPPORTUNITIES_PATH,
    TOP_OPPORTUNITIES_REPORT,
    WEAK_CLAIMS_REPORT,
    read_jsonl,
    write_jsonl,
)


def test_field_claim_schema_validates_required_fields():
    claim = FieldClaim(
        claim_id="claim_test",
        claim_text="Dense semantic retrieval improves dataset matching.",
        evidence_level=EvidenceLevel.PLAUSIBLE,
        confidence=0.6,
    )

    assert claim.claim_id == "claim_test"
    assert claim.evidence_level == EvidenceLevel.PLAUSIBLE

    with pytest.raises(ValueError, match="must not be empty"):
        FieldClaim(
            claim_id="",
            claim_text="Missing ID should fail.",
            evidence_level=EvidenceLevel.HYPOTHESIS,
            confidence=0.5,
        )

    with pytest.raises(ValueError):
        FieldClaim(
            claim_id="claim_bad_confidence",
            claim_text="Confidence must be bounded.",
            evidence_level=EvidenceLevel.HYPOTHESIS,
            confidence=1.5,
        )


def test_benchmark_gap_schema_validates_required_fields():
    gap = BenchmarkGap(
        gap_id="gap_test",
        title="Human qrels benchmark",
        description="Expert-labeled compatibility qrels.",
        why_it_matters="It makes nDCG/MRR credible.",
        severity=0.9,
    )

    assert gap.gap_id == "gap_test"
    assert gap.severity == 0.9

    with pytest.raises(ValueError, match="must not be empty"):
        BenchmarkGap(
            gap_id="gap_bad",
            title="",
            description="Missing title should fail.",
            why_it_matters="Validation needs readable gaps.",
        )


def test_opportunity_scoring_uses_transparent_formula():
    opportunity = FieldOpportunity(
        opportunity_id="opp_test",
        title="Human qrels benchmark",
        description="Build a compact relevance benchmark.",
        next_step="Label candidate pools.",
        novelty_score=7.0,
        feasibility_score=7.0,
        impact_score=10.0,
        uncertainty_reduction_score=10.0,
        personal_fit_score=9.0,
        risk_score=3.0,
    )

    assert score_opportunity(opportunity) == 7.7
    assert opportunity.total_score == 7.7


def test_rank_opportunities_sorts_descending_by_score():
    lower = FieldOpportunity(
        opportunity_id="opp_lower",
        title="Lower",
        description="Lower score.",
        next_step="Do less urgent work.",
        novelty_score=5,
        feasibility_score=5,
        impact_score=5,
        uncertainty_reduction_score=5,
        personal_fit_score=5,
        risk_score=5,
    )
    higher = FieldOpportunity(
        opportunity_id="opp_higher",
        title="Higher",
        description="Higher score.",
        next_step="Do more urgent work.",
        novelty_score=8,
        feasibility_score=8,
        impact_score=8,
        uncertainty_reduction_score=8,
        personal_fit_score=8,
        risk_score=2,
    )

    assert rank_opportunities([lower, higher])[0].opportunity_id == "opp_higher"


def test_jsonl_read_write_round_trip(tmp_path: Path):
    path = Path("artifacts/field_state/claims.jsonl")
    claims = seed_claims(tmp_path)

    written_path = write_jsonl(path, claims, tmp_path)
    restored = read_jsonl(path, FieldClaim, tmp_path)

    assert written_path.exists()
    assert len(restored) == len(claims)
    assert restored[0].claim_id == claims[0].claim_id


def test_report_rendering_includes_expected_sections():
    claims = seed_claims()
    gaps = seed_benchmark_gaps()
    opportunities = seed_opportunities()

    weak_claims_md = render_weak_claims_report(claims)
    gaps_md = render_benchmark_gaps_report(gaps)
    opportunities_md = render_top_opportunities_report(opportunities)

    assert "# Weak Claims" in weak_claims_md
    assert "Dense semantic retrieval improves dataset matching." in weak_claims_md
    assert "# Benchmark Gaps" in gaps_md
    assert "Human qrels benchmark" in gaps_md
    assert "# Top Opportunities" in opportunities_md
    assert "total_score = 0.20 * novelty_score" in opportunities_md


def test_seeded_field_state_contains_requested_items():
    claims = seed_claims()
    gaps = seed_benchmark_gaps()
    opportunities = seed_opportunities()

    assert len(claims) == 6
    assert len(gaps) == 6
    assert len(opportunities) == 8
    assert claims[0].claim_text == "Dense semantic retrieval improves dataset matching."
    assert any(gap.gap_id == "gap_calibration_ece" for gap in gaps)
    assert opportunities[0].title == (
        "Human qrels benchmark for dataset-method compatibility"
    )


def test_cli_commands_create_artifacts_and_reports(tmp_path: Path):
    assert main(["--root", str(tmp_path), "init"]) == 0
    assert main(["--root", str(tmp_path), "report"]) == 0
    assert main(["--root", str(tmp_path), "opportunities"]) == 0
    assert main(["--root", str(tmp_path), "snapshot"]) == 0

    for path in [
        CLAIMS_PATH,
        BENCHMARK_GAPS_PATH,
        OPPORTUNITIES_PATH,
        WEAK_CLAIMS_REPORT,
        TOP_OPPORTUNITIES_REPORT,
        LATEST_SNAPSHOT_REPORT,
    ]:
        assert (tmp_path / path).exists()
