"""Tests for provider-neutral NVIDIA NIM integration."""

from __future__ import annotations

import json

import httpx

from neural_search.inference.registry import InferenceRegistry
from neural_search.inference.schemas import (
    InferenceCapability,
    InferenceMessage,
    InferenceRequest,
    ModelProfile,
    ProviderSettings,
)
from neural_search.inference.service import InferenceService


def _transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "nim-test-model"}]})
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == "nim-test-model"
        assert payload["temperature"] == 0.0
        return httpx.Response(
            200,
            json={
                "model": "nim-test-model",
                "choices": [{"message": {"content": "verified response", "tool_calls": []}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            },
        )

    return httpx.MockTransport(handler)


def _registry() -> InferenceRegistry:
    return InferenceRegistry(
        providers={
            "nim": ProviderSettings(
                name="nim",
                kind="nim",
                base_url="http://nim.local",
                api_key="test-secret",
            )
        },
        models={
            "code_reasoning": ModelProfile(
                name="code_reasoning",
                provider="nim",
                model="nim-test-model",
                capabilities={InferenceCapability.CODE_REASONING},
            )
        },
    )


def test_nim_service_routes_by_capability_and_records_manifest() -> None:
    service = InferenceService(_registry(), transports={"nim": _transport()})
    request = InferenceRequest(
        messages=[InferenceMessage(role="user", content="Inspect this numerical routine")],
        capability=InferenceCapability.CODE_REASONING,
        metadata={"input_revision": "repo@abc123", "prompt_template": "audit_numeric_v1"},
    )

    result = service.generate(request)

    assert result.text == "verified response"
    assert result.provider == "nim"
    assert result.manifest.input_revision == "repo@abc123"
    assert result.manifest.prompt_template == "audit_numeric_v1"
    assert len(result.manifest.prompt_hash) == 64
    assert len(result.manifest.request_hash) == 64
    service.close()


def test_nim_health_uses_openai_compatible_models_endpoint() -> None:
    service = InferenceService(_registry(), transports={"nim": _transport()})
    health = service.health()
    assert health["healthy"] is True
    assert health["providers"]["nim"]["models"] == ["nim-test-model"]
    service.close()


def test_capability_mismatch_is_rejected_before_model_call() -> None:
    service = InferenceService(_registry(), transports={"nim": _transport()})
    request = InferenceRequest(
        messages=[InferenceMessage(role="user", content="Extract fields")],
        capability=InferenceCapability.STRUCTURED_EXTRACTION,
        model_profile="code_reasoning",
    )
    try:
        service.generate(request)
    except ValueError as exc:
        assert "does not advertise capability" in str(exc)
    else:
        raise AssertionError("expected capability mismatch to fail")
    finally:
        service.close()
