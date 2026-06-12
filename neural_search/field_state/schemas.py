"""Schemas for lightweight field-state artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class EvidenceLevel(StrEnum):
    """Coarse evidence level for a field-level claim."""

    HYPOTHESIS = "hypothesis"
    PLAUSIBLE = "plausible"
    SUPPORTED = "supported"
    VALIDATED = "validated"


class ClaimStatus(StrEnum):
    """Lifecycle state for a field-level claim."""

    ACTIVE = "active"
    NEEDS_VALIDATION = "needs_validation"
    PARTIALLY_TESTED = "partially_tested"
    RETIRED = "retired"


class GapStatus(StrEnum):
    """Lifecycle state for benchmark gaps."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    ADDRESSED = "addressed"


class OpportunityStatus(StrEnum):
    """Lifecycle state for opportunities."""

    CANDIDATE = "candidate"
    NEXT_UP = "next_up"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    DEFERRED = "deferred"


class FieldClaim(BaseModel):
    """A tracked scientific or evaluation claim about Neural Search."""

    claim_id: str = Field(description="Stable claim identifier")
    claim_text: str = Field(description="Human-readable claim")
    evidence_level: EvidenceLevel = Field(description="Current evidence level")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence from 0 to 1")
    related_artifacts: list[str] = Field(default_factory=list)
    missing_tests: list[str] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.NEEDS_VALIDATION
    notes: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    updated_at: str | None = None

    @field_validator("claim_id", "claim_text")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    def to_jsonl(self) -> str:
        """Serialize this claim as one JSONL row."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> FieldClaim:
        """Deserialize a claim from one JSONL row."""
        return cls.model_validate_json(line)


class BenchmarkGap(BaseModel):
    """A missing benchmark, audit, or validation artifact."""

    gap_id: str = Field(description="Stable benchmark gap identifier")
    title: str
    description: str
    why_it_matters: str
    related_claim_ids: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    available_artifacts: list[str] = Field(default_factory=list)
    blocking_questions: list[str] = Field(default_factory=list)
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    status: GapStatus = GapStatus.OPEN
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    updated_at: str | None = None

    @field_validator("gap_id", "title", "description", "why_it_matters")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    def to_jsonl(self) -> str:
        """Serialize this gap as one JSONL row."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> BenchmarkGap:
        """Deserialize a gap from one JSONL row."""
        return cls.model_validate_json(line)


class FieldOpportunity(BaseModel):
    """A ranked next opportunity for improving Neural Search rigor."""

    opportunity_id: str = Field(description="Stable opportunity identifier")
    title: str
    description: str
    linked_claim_ids: list[str] = Field(default_factory=list)
    linked_gap_ids: list[str] = Field(default_factory=list)
    next_step: str
    novelty_score: float = Field(ge=0.0, le=10.0)
    feasibility_score: float = Field(ge=0.0, le=10.0)
    impact_score: float = Field(ge=0.0, le=10.0)
    uncertainty_reduction_score: float = Field(ge=0.0, le=10.0)
    personal_fit_score: float = Field(ge=0.0, le=10.0)
    risk_score: float = Field(ge=0.0, le=10.0)
    total_score: float = Field(
        default=0.0,
        description="Weighted heuristic score computed from score components.",
    )
    status: OpportunityStatus = OpportunityStatus.CANDIDATE
    rationale: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    updated_at: str | None = None

    @field_validator("opportunity_id", "title", "description", "next_step")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @model_validator(mode="after")
    def set_total_score(self) -> Self:
        """Weighted heuristic opportunity score."""
        score = (
            0.20 * self.novelty_score
            + 0.25 * self.feasibility_score
            + 0.20 * self.impact_score
            + 0.15 * self.uncertainty_reduction_score
            + 0.15 * self.personal_fit_score
            - 0.10 * self.risk_score
        )
        self.total_score = round(score, 3)
        return self

    def to_jsonl(self) -> str:
        """Serialize this opportunity as one JSONL row."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> FieldOpportunity:
        """Deserialize an opportunity from one JSONL row."""
        return cls.model_validate_json(line)
