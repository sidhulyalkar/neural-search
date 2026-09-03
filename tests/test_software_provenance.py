"""Tests for evidence-gated scientific software provenance."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neural_search.software.contribution import (
    ContributionChannel,
    ContributionPacket,
    ContributionPolicy,
)
from neural_search.software.graph import component_node, package_node, relation
from neural_search.software.priority import score_audit_priority
from neural_search.software.schema import (
    AuditFinding,
    AuditHypothesis,
    AuditState,
    SoftwareComponent,
    SoftwarePackage,
    VerificationLevel,
    VerificationRun,
    can_transition,
)


def test_hypothesis_cannot_claim_numerical_verification() -> None:
    with pytest.raises(ValidationError):
        AuditHypothesis(
            hypothesis_id="h1",
            component_id="c1",
            summary="suspected off-by-one",
            rationale="code reading",
            state=AuditState.NUMERICALLY_VERIFIED,
        )


def test_verified_finding_requires_verification_record_reference() -> None:
    with pytest.raises(ValidationError):
        AuditFinding(
            finding_id="f1",
            hypothesis_id="h1",
            component_id="c1",
            summary="numerical discrepancy",
            state=AuditState.NUMERICALLY_VERIFIED,
        )


def test_state_machine_is_sequential_but_allows_terminal_adjudication() -> None:
    assert can_transition(AuditState.TRIAGED, AuditState.CODE_REVIEW_HYPOTHESIS)
    assert not can_transition(AuditState.TRIAGED, AuditState.NUMERICALLY_VERIFIED)
    assert can_transition(AuditState.TRIAGED, AuditState.FALSE_POSITIVE)
    assert not can_transition(AuditState.FALSE_POSITIVE, AuditState.TRIAGED)


def test_priority_score_penalizes_mapping_uncertainty() -> None:
    baseline = score_audit_priority(
        exposure=0.9,
        consequence=0.8,
        reproducibility=0.9,
        uncertainty=0.7,
        verification_feasibility=0.9,
    )
    uncertain = score_audit_priority(
        exposure=0.9,
        consequence=0.8,
        reproducibility=0.9,
        uncertainty=0.7,
        verification_feasibility=0.9,
        mapping_uncertainty_penalty=0.5,
    )
    assert baseline.score > uncertain.score


def _upstream_ready_finding() -> tuple[AuditFinding, VerificationRun]:
    verification = VerificationRun(
        verification_id="v1",
        hypothesis_id="h1",
        level=VerificationLevel.NUMERICAL_ORACLE,
        command=["python", "reproduce.py"],
        observed={"value": 0.2},
        expected={"value": 0.4},
        passed=True,
        reproducer_path="reproduce.py",
    )
    finding = AuditFinding(
        finding_id="f1",
        hypothesis_id="h1",
        component_id="c1",
        summary="verified numerical discrepancy",
        state=AuditState.UPSTREAM_READY,
        verification_ids=[verification.verification_id],
    )
    return finding, verification


def test_contribution_packet_requires_policy_review_and_human_approval() -> None:
    finding, verification = _upstream_ready_finding()
    reviewed = {
        name: f"https://example.test/{name.lower()}"
        for name in ContributionPolicy(
            repository_url="https://github.com/example/neuro",
            preferred_channel=ContributionChannel.PULL_REQUEST,
        ).required_documents
    }
    policy = ContributionPolicy(
        repository_url="https://github.com/example/neuro",
        preferred_channel=ContributionChannel.PULL_REQUEST,
        reviewed_documents=reviewed,
    )
    packet = ContributionPacket(
        finding=finding,
        verifications=[verification],
        policy=policy,
        executive_summary="A verified discrepancy exists under a narrow condition.",
        affected_code=["module.py:function"],
        reproducer_instructions=["python reproduce.py"],
        regression_tests=["tests/test_regression.py"],
        scientific_impact_caveat="Exposure does not imply that published conclusions are wrong.",
        proposed_upstream_message="This report includes a reproducer and regression test.",
    )

    with pytest.raises(PermissionError):
        packet.submission_payload()

    approved = packet.model_copy(update={"human_approved": True})
    payload = approved.submission_payload()
    assert payload["channel"] == "pull_request"


def test_software_records_project_into_knowledge_graph() -> None:
    package = package_node(
        SoftwarePackage(
            package_id="kilosort",
            name="Kilosort",
            repository_url="https://github.com/MouseLand/Kilosort",
            scientific_domains=["electrophysiology", "spike_sorting"],
        )
    )
    component = component_node(
        SoftwareComponent(
            component_id="kilosort:clustering",
            package_id="kilosort",
            path="kilosort/clustering_qr.py",
            symbol="cluster_qr",
        )
    )
    edge = relation(package, "software_package_has_component", component)

    assert package.node_type == "software_package"
    assert component.node_type == "software_component"
    assert edge.source_node_id == package.node_id
    assert edge.target_node_id == component.node_id
