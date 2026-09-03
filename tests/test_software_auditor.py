"""Tests for model-assisted scientific code hypothesis generation."""

from __future__ import annotations

import json

import httpx

from neural_search.inference import (
    InferenceCapability,
    InferenceRegistry,
    InferenceService,
    ModelProfile,
    ProviderSettings,
)
from neural_search.software.auditor import CodeAuditInput, generate_audit_hypothesis
from neural_search.software.schema import AuditState, SoftwareComponent


def test_code_auditor_creates_only_unverified_hypothesis_with_provenance() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert payload["response_format"]["type"] == "json_schema"
        response = {
            "summary": "Boundary value may enter the wrong bin",
            "rationale": "The comparison excludes equality at the training minimum.",
            "confidence": 0.82,
            "testable_prediction": "An input equal to the minimum maps to a different bin than an epsilon-above input.",
            "affected_conditions": ["value equals lower training bound"],
            "candidate_oracle": "compare against numpy digitize reference",
        }
        return httpx.Response(
            200,
            json={
                "model": "nim-code",
                "choices": [{"message": {"content": json.dumps(response)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8},
            },
        )

    registry = InferenceRegistry(
        providers={
            "nim": ProviderSettings(name="nim", kind="nim", base_url="http://nim.local")
        },
        models={
            "code_reasoning": ModelProfile(
                name="code_reasoning",
                provider="nim",
                model="nim-code",
                capabilities={InferenceCapability.CODE_REASONING},
            )
        },
    )
    service = InferenceService(registry, transports={"nim": httpx.MockTransport(handler)})
    component = SoftwareComponent(
        component_id="suite2p:classifier-bin",
        package_id="suite2p",
        path="suite2p/classification/classifier.py",
        symbol="probability",
    )

    hypothesis = generate_audit_hypothesis(
        service,
        CodeAuditInput(
            component=component,
            source_text="if x > xmin: return 0\nreturn n_bins - 1",
            repository_revision="suite2p@abc123",
        ),
    )

    assert hypothesis.state == AuditState.CODE_REVIEW_HYPOTHESIS
    assert hypothesis.component_id == component.component_id
    assert hypothesis.source_run_ids
    assert hypothesis.metadata["repository_revision"] == "suite2p@abc123"
    assert hypothesis.metadata["confidence"] == 0.82
    service.close()
