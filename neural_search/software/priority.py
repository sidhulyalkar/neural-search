"""Scientific audit prioritization.

Scores are transparent and deliberately bounded. They rank where to spend review effort;
they are not evidence that a defect exists.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AuditPriority(BaseModel):
    exposure: float = Field(ge=0.0, le=1.0)
    consequence: float = Field(ge=0.0, le=1.0)
    reproducibility: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    verification_feasibility: float = Field(ge=0.0, le=1.0)
    already_addressed_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    mapping_uncertainty_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: list[str] = Field(default_factory=list)

    @property
    def score(self) -> float:
        base = (
            self.exposure
            * self.consequence
            * self.reproducibility
            * self.uncertainty
            * self.verification_feasibility
        ) ** (1 / 5)
        penalty = (1.0 - self.already_addressed_penalty) * (
            1.0 - self.mapping_uncertainty_penalty
        )
        return round(base * penalty, 6)


def score_audit_priority(
    *,
    exposure: float,
    consequence: float,
    reproducibility: float,
    uncertainty: float,
    verification_feasibility: float,
    already_addressed_penalty: float = 0.0,
    mapping_uncertainty_penalty: float = 0.0,
    rationale: list[str] | None = None,
) -> AuditPriority:
    """Create a validated priority record for a candidate code path."""

    return AuditPriority(
        exposure=exposure,
        consequence=consequence,
        reproducibility=reproducibility,
        uncertainty=uncertainty,
        verification_feasibility=verification_feasibility,
        already_addressed_penalty=already_addressed_penalty,
        mapping_uncertainty_penalty=mapping_uncertainty_penalty,
        rationale=rationale or [],
    )
