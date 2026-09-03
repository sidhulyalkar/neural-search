"""Typed domain model for scientific software auditing.

The central safety property is that model-generated suspicion, executable verification,
scientific impact, and maintainer adjudication are represented as different states and
records. A language model cannot promote its own hypothesis to verified evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class AuditState(StrEnum):
    DISCOVERED = "discovered"
    TRIAGED = "triaged"
    CODE_REVIEW_HYPOTHESIS = "code_review_hypothesis"
    STATICALLY_SUPPORTED = "statically_supported"
    MINIMAL_REPRODUCER = "minimal_reproducer"
    NUMERICALLY_VERIFIED = "numerically_verified"
    PIPELINE_VERIFIED = "pipeline_verified"
    OPEN_DATA_REANALYZED = "open_data_reanalyzed"
    UPSTREAM_READY = "upstream_ready"
    SUBMITTED = "submitted"
    MAINTAINER_ADJUDICATED = "maintainer_adjudicated"
    FALSE_POSITIVE = "false_positive"
    ALREADY_FIXED = "already_fixed"
    KNOWN_BEHAVIOR = "known_behavior"
    INTENTIONAL_BEHAVIOR = "intentional_behavior"
    NEGLIGIBLE_EFFECT = "negligible_effect"
    UNREPRODUCIBLE = "unreproducible"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    DUPLICATE = "duplicate"
    WITHDRAWN = "withdrawn"


TERMINAL_STATES = {
    AuditState.FALSE_POSITIVE,
    AuditState.ALREADY_FIXED,
    AuditState.KNOWN_BEHAVIOR,
    AuditState.INTENTIONAL_BEHAVIOR,
    AuditState.NEGLIGIBLE_EFFECT,
    AuditState.UNREPRODUCIBLE,
    AuditState.INSUFFICIENT_EVIDENCE,
    AuditState.DUPLICATE,
    AuditState.WITHDRAWN,
    AuditState.MAINTAINER_ADJUDICATED,
}

_PROGRESS = [
    AuditState.DISCOVERED,
    AuditState.TRIAGED,
    AuditState.CODE_REVIEW_HYPOTHESIS,
    AuditState.STATICALLY_SUPPORTED,
    AuditState.MINIMAL_REPRODUCER,
    AuditState.NUMERICALLY_VERIFIED,
    AuditState.PIPELINE_VERIFIED,
    AuditState.OPEN_DATA_REANALYZED,
    AuditState.UPSTREAM_READY,
    AuditState.SUBMITTED,
    AuditState.MAINTAINER_ADJUDICATED,
]


class VerificationLevel(StrEnum):
    STATIC = "static"
    EXECUTABLE = "executable"
    NUMERICAL_ORACLE = "numerical_oracle"
    PIPELINE = "pipeline"
    OPEN_DATA = "open_data"
    INDEPENDENT_REPLICATION = "independent_replication"


class SoftwarePackage(BaseModel):
    package_id: str
    name: str
    repository_url: str
    ecosystem: str | None = None
    scientific_domains: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SoftwareRelease(BaseModel):
    release_id: str
    package_id: str
    version: str
    commit_sha: str | None = None
    released_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SoftwareComponent(BaseModel):
    component_id: str
    package_id: str
    path: str
    symbol: str | None = None
    component_type: str = "source"
    implements_methods: list[str] = Field(default_factory=list)
    scientific_role: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditHypothesis(BaseModel):
    hypothesis_id: str
    component_id: str
    summary: str
    rationale: str
    state: AuditState = AuditState.CODE_REVIEW_HYPOTHESIS
    affected_release_ids: list[str] = Field(default_factory=list)
    source_run_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("state")
    @classmethod
    def hypothesis_state_is_unverified(cls, value: AuditState) -> AuditState:
        if value not in {
            AuditState.DISCOVERED,
            AuditState.TRIAGED,
            AuditState.CODE_REVIEW_HYPOTHESIS,
            AuditState.STATICALLY_SUPPORTED,
        }:
            raise ValueError("AuditHypothesis cannot claim executable or downstream verification")
        return value


class VerificationRun(BaseModel):
    verification_id: str
    hypothesis_id: str
    level: VerificationLevel
    command: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    input_artifact_ids: list[str] = Field(default_factory=list)
    observed: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    passed: bool
    reproducer_path: str | None = None
    output_artifact_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditFinding(BaseModel):
    finding_id: str
    hypothesis_id: str
    component_id: str
    summary: str
    state: AuditState
    verification_ids: list[str] = Field(default_factory=list)
    affected_release_ids: list[str] = Field(default_factory=list)
    exposed_paper_ids: list[str] = Field(default_factory=list)
    candidate_dataset_ids: list[str] = Field(default_factory=list)
    scientific_impact: str | None = None
    created_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def verified_states_need_evidence(self) -> AuditFinding:
        verification_required = {
            AuditState.MINIMAL_REPRODUCER,
            AuditState.NUMERICALLY_VERIFIED,
            AuditState.PIPELINE_VERIFIED,
            AuditState.OPEN_DATA_REANALYZED,
            AuditState.UPSTREAM_READY,
            AuditState.SUBMITTED,
            AuditState.MAINTAINER_ADJUDICATED,
        }
        if self.state in verification_required and not self.verification_ids:
            raise ValueError(f"state {self.state.value} requires at least one verification record")
        return self


class MaintainerDecision(BaseModel):
    decision_id: str
    finding_id: str
    repository_url: str
    channel: str
    disposition: str
    issue_or_pr_url: str | None = None
    rationale: str | None = None
    decided_at: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


def can_transition(current: AuditState, target: AuditState) -> bool:
    """Return whether a state transition is allowed by the audit evidence ladder."""

    if current == target:
        return True
    if current in TERMINAL_STATES:
        return False
    if target in TERMINAL_STATES:
        return True
    return _PROGRESS.index(target) == _PROGRESS.index(current) + 1


def require_transition(current: AuditState, target: AuditState) -> None:
    if not can_transition(current, target):
        raise ValueError(f"invalid audit transition: {current.value} -> {target.value}")
