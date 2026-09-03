"""Inference service with capability routing and invocation provenance."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from neural_search.inference.providers import NIMProvider, OpenAICompatibleProvider
from neural_search.inference.registry import InferenceRegistry
from neural_search.inference.schemas import (
    EmbeddingRequest,
    EmbeddingResult,
    InferenceCapability,
    InferenceRequest,
    InferenceResult,
    RankedPassage,
    RerankRequest,
    RerankResult,
    RunManifest,
)


class InferenceService:
    """Route provider-neutral requests and attach immutable provenance manifests."""

    def __init__(
        self,
        registry: InferenceRegistry,
        *,
        transports: dict[str, httpx.BaseTransport] | None = None,
    ) -> None:
        self.registry = registry
        self.transports = transports or {}
        self._providers: dict[str, OpenAICompatibleProvider] = {}

    def _provider(self, name: str) -> OpenAICompatibleProvider:
        if name in self._providers:
            return self._providers[name]
        settings = self.registry.get_provider(name)
        transport = self.transports.get(name)
        if settings.kind == "nim":
            provider: OpenAICompatibleProvider = NIMProvider(settings, transport=transport)
        elif settings.kind == "openai_compatible":
            provider = OpenAICompatibleProvider(settings, transport=transport)
        else:
            raise ValueError(f"unsupported provider kind: {settings.kind}")
        self._providers[name] = provider
        return provider

    def generate(self, request: InferenceRequest) -> InferenceResult:
        profile = (
            self.registry.get_model(request.model_profile)
            if request.model_profile
            else self.registry.model_for_capability(request.capability)
        )
        if request.capability not in profile.capabilities:
            raise ValueError(
                f"model profile {profile.name} does not advertise capability "
                f"{request.capability.value}"
            )
        provider = self._provider(profile.provider)
        started_at = datetime.now(UTC)
        response: dict[str, Any] = provider.generate(request, profile)
        completed_at = datetime.now(UTC)
        manifest = RunManifest.build(
            provider=profile.provider,
            model=str(response.get("model") or profile.model),
            model_profile=profile.name,
            capability=request.capability,
            request=request,
            started_at=started_at,
            completed_at=completed_at,
        )
        return InferenceResult(
            text=str(response.get("text", "")),
            model=str(response.get("model") or profile.model),
            provider=profile.provider,
            usage=dict(response.get("usage") or {}),
            tool_calls=list(response.get("tool_calls") or []),
            raw=dict(response.get("raw") or {}),
            manifest=manifest,
        )

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Generate retrieval embeddings with a capability-routed model."""

        profile = (
            self.registry.get_model(request.model_profile)
            if request.model_profile
            else self.registry.model_for_capability(InferenceCapability.EMBEDDING)
        )
        if InferenceCapability.EMBEDDING not in profile.capabilities:
            raise ValueError(f"model profile {profile.name} does not advertise embedding")
        provider = self._provider(profile.provider)
        started_at = datetime.now(UTC)
        response = provider.embed(request, profile)
        completed_at = datetime.now(UTC)
        model = str(response.get("model") or profile.model)
        payload = dict(response.get("payload") or request.model_dump(mode="json", exclude_none=True))
        manifest = RunManifest.from_payload(
            provider=profile.provider,
            model=model,
            model_profile=profile.name,
            capability=InferenceCapability.EMBEDDING,
            payload=payload,
            prompt_material="\n".join(request.inputs),
            metadata=request.metadata,
            started_at=started_at,
            completed_at=completed_at,
        )
        return EmbeddingResult(
            vectors=list(response.get("vectors") or []),
            model=model,
            provider=profile.provider,
            usage=dict(response.get("usage") or {}),
            manifest=manifest,
        )

    def rerank(self, request: RerankRequest) -> RerankResult:
        """Rerank candidate passages with a capability-routed NIM."""

        profile = (
            self.registry.get_model(request.model_profile)
            if request.model_profile
            else self.registry.model_for_capability(InferenceCapability.RERANKING)
        )
        if InferenceCapability.RERANKING not in profile.capabilities:
            raise ValueError(f"model profile {profile.name} does not advertise reranking")
        provider = self._provider(profile.provider)
        started_at = datetime.now(UTC)
        response = provider.rerank(request, profile)
        completed_at = datetime.now(UTC)
        model = str(response.get("model") or profile.model)
        payload = dict(response.get("payload") or request.model_dump(mode="json", exclude_none=True))
        manifest = RunManifest.from_payload(
            provider=profile.provider,
            model=model,
            model_profile=profile.name,
            capability=InferenceCapability.RERANKING,
            payload=payload,
            prompt_material="\n".join([request.query, *request.passages]),
            metadata=request.metadata,
            started_at=started_at,
            completed_at=completed_at,
        )
        rankings = [RankedPassage.model_validate(item) for item in response.get("rankings", [])]
        return RerankResult(
            rankings=rankings,
            model=model,
            provider=profile.provider,
            usage=dict(response.get("usage") or {}),
            manifest=manifest,
        )

    def health(self) -> dict[str, object]:
        results: dict[str, dict[str, Any]] = {}
        for name in self.registry.providers:
            results[name] = self._provider(name).health()
        return {
            "healthy": bool(results) and all(bool(value.get("healthy")) for value in results.values()),
            "providers": results,
        }

    def close(self) -> None:
        for provider in self._providers.values():
            provider.close()
        self._providers.clear()
