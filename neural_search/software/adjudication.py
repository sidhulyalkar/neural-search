"""Adversarial adjudication records for software audit hypotheses.

A finding should survive an explicit attempt to disconfirm it before it is promoted to
upstream-ready status. This module stores that review separately from the hypothesis that
motivated the investigation.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class AdjudicationDisposition(StrEnum):
    SURVIVES = "survives"
    REJECTED = "rejected"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    ALREADY_ADDRESSED = "already_addressed"
    INTENTIONAL_BEHAVIOR = "intentional_behavior"
    DUPLICATE = "duplicate"


class AdjudicationCheck(BaseModel):
    name: str
    completed: bool
    evidence_refs: list[str] = Field(default_factory=list)
    notes: str | None = None


class FindingAdjudication(BaseModel):
    adjudication_id: str
    hypothesis_id: str
    independent_reviewer_run_ids: list[str] = Field(default_factory=list)
    checks: list[AdjudicationCheck]
    disposition: AdjudicationDisposition
    rationale: str

    @model_validator(mode="after")
    def require_core_checks(self) -> FindingAdjudication:
        by_name = {check.name: check for check in self.checks}
        required = {
            "documentation",
            "tests",
            "history",
            "existing_issues",
            "existing_pull_requests",
            "release_notes",
            "algorithm_reference",
        }
        missing = sorted(required - set(by_name))
        incomplete = sorted(
            name for name in required if name in by_name and not by_name[name].completed
        )
        if missing:
            raise ValueError(f"adjudication is missing checks: {', '.join(missing)}")
        if incomplete:
            raise ValueError(f"adjudication has incomplete checks: {', '.join(incomplete)}")
        if self.disposition == AdjudicationDisposition.SURVIVES and not self.independent_reviewer_run_ids:
            raise ValueError("surviving AI-assisted hypotheses require an independent reviewer run")
        return self

    @property
    def survived(self) -> bool:
        return self.disposition == AdjudicationDisposition.SURVIVES
