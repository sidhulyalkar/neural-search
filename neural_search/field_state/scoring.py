"""Transparent heuristic scoring for field-state opportunities."""

from __future__ import annotations

from neural_search.field_state.schemas import FieldOpportunity

OPPORTUNITY_SCORE_WEIGHTS: dict[str, float] = {
    "novelty_score": 0.20,
    "feasibility_score": 0.25,
    "impact_score": 0.20,
    "uncertainty_reduction_score": 0.15,
    "personal_fit_score": 0.15,
    "risk_score": -0.10,
}


def score_opportunity(opportunity: FieldOpportunity) -> float:
    """Return the weighted heuristic score for one opportunity."""
    return round(
        0.20 * opportunity.novelty_score
        + 0.25 * opportunity.feasibility_score
        + 0.20 * opportunity.impact_score
        + 0.15 * opportunity.uncertainty_reduction_score
        + 0.15 * opportunity.personal_fit_score
        - 0.10 * opportunity.risk_score,
        3,
    )


def rank_opportunities(
    opportunities: list[FieldOpportunity],
) -> list[FieldOpportunity]:
    """Sort opportunities by descending total score, then stable ID."""
    return sorted(
        opportunities,
        key=lambda item: (-score_opportunity(item), item.opportunity_id),
    )
