"""Human-gated contribution packets for upstream scientific software projects."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from neural_search.software.adjudication import FindingAdjudication
from neural_search.software.schema import AuditFinding, AuditState, VerificationRun


class ContributionChannel(StrEnum):
    ISSUE = "issue"
    PULL_REQUEST = "pull_request"
    DISCUSSION = "discussion"
    MAILING_LIST = "mailing_list"
    SUPPORT_FORUM = "support_forum"
    OTHER = "other"


class ContributionPolicy(BaseModel):
    repository_url: str
    preferred_channel: ContributionChannel
    issue_before_pr: bool = False
    behavior_change_requires_discussion: bool = True
    regression_test_required: bool = True
    required_documents: list[str] = Field(
        default_factory=lambda: [
            "README",
            "CONTRIBUTING",
            "CODE_OF_CONDUCT",
            "CHANGELOG_OR_NEWS",
            "ISSUE_TEMPLATE",
            "PULL_REQUEST_TEMPLATE",
        ]
    )
    reviewed_documents: dict[str, str] = Field(default_factory=dict)
    existing_issue_search: str | None = None
    existing_pr_search: str | None = None
    last_checked_revision: str | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def missing_required_documents(self) -> list[str]:
        return [name for name in self.required_documents if name not in self.reviewed_documents]


class ContributionPacket(BaseModel):
    finding: AuditFinding
    verifications: list[VerificationRun]
    adjudication: FindingAdjudication
    policy: ContributionPolicy
    executive_summary: str
    mathematical_or_algorithmic_rationale: str | None = None
    affected_code: list[str] = Field(default_factory=list)
    reproducer_instructions: list[str] = Field(default_factory=list)
    regression_tests: list[str] = Field(default_factory=list)
    patch_refs: list[str] = Field(default_factory=list)
    scientific_impact_caveat: str
    proposed_upstream_message: str
    human_approved: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_readiness(self) -> ContributionPacket:
        if self.finding.state not in {
            AuditState.UPSTREAM_READY,
            AuditState.SUBMITTED,
            AuditState.MAINTAINER_ADJUDICATED,
        }:
            raise ValueError("contribution packet requires an upstream-ready finding")
        if not self.verifications:
            raise ValueError("contribution packet requires executable verification")
        if self.adjudication.hypothesis_id != self.finding.hypothesis_id:
            raise ValueError("adjudication must review the finding's hypothesis")
        if not self.adjudication.survived:
            raise ValueError("contribution packet requires a hypothesis that survived adjudication")
        if self.policy.missing_required_documents:
            missing = ", ".join(self.policy.missing_required_documents)
            raise ValueError(f"contribution policy review is incomplete: {missing}")
        if self.policy.regression_test_required and not self.regression_tests:
            raise ValueError("repository policy requires a regression test")
        return self

    def submission_payload(self) -> dict[str, Any]:
        """Return a public-submission payload only after explicit human approval."""

        if not self.human_approved:
            raise PermissionError("public contribution submission requires human approval")
        return {
            "repository_url": self.policy.repository_url,
            "channel": self.policy.preferred_channel.value,
            "message": self.proposed_upstream_message,
            "patch_refs": self.patch_refs,
            "regression_tests": self.regression_tests,
        }
