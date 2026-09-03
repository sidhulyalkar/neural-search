"""OpenAI-compatible inference provider implemented with httpx only."""

from __future__ import annotations

from typing import Any

import httpx

from neural_search.inference.schemas import InferenceRequest, ModelProfile, ProviderSettings


class OpenAICompatibleProvider:
    """Small, dependency-light client for OpenAI-compatible model servers."""

    def __init__(self, settings: ProviderSettings, *, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        headers = {"Content-Type": "application/json", **settings.extra_headers}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self._client = httpx.Client(
            base_url=settings.base_url.rstrip("/"),
            headers=headers,
            timeout=settings.timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any]:
        """Return endpoint health without assuming a vendor-specific health route."""

        try:
            response = self._client.get("/v1/models")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return {"healthy": False, "provider": self.settings.name, "error": str(exc)}
        return {
            "healthy": True,
            "provider": self.settings.name,
            "models": [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)],
        }

    def generate(self, request: InferenceRequest, profile: ModelProfile) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": profile.model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": request.temperature if request.temperature is not None else profile.temperature,
            "max_tokens": request.max_tokens if request.max_tokens is not None else profile.max_tokens,
        }
        if request.tools:
            payload["tools"] = request.tools
        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "neural_search_response",
                    "strict": True,
                    "schema": request.response_schema,
                },
            }

        response = self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("inference provider returned no choices")
        message = choices[0].get("message", {})
        return {
            "text": message.get("content") or "",
            "model": data.get("model", profile.model),
            "usage": data.get("usage", {}),
            "tool_calls": message.get("tool_calls") or [],
            "raw": data,
        }
