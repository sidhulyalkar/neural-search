"""Typed contracts for model routing and inference provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class InferenceCapability(StrEnum):
    """Capabilities used by Neural Search to route work to models."""

    CHAT = "chat"
    CODE_REASONING = "code_reasoning"
    STRUCTURED_EXTRACTION = "structured_extraction"
    MATHEMATICAL_REVIEW = "mathematical_review"
    TOOL_CALLING = "tool_calling"
    EMBEDDING = "embedding"
    RERANKING = "reranking"


class InferenceMessage(BaseModel):
    """Provider-neutral chat message."""

    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported inference role: {value}")
        return normalized


class ProviderSettings(BaseModel):
    """Connection settings for one inference provider."""

    name: str
    kind: str
    base_url: str
    api_key: str | None = None
    timeout_seconds: float = Field(default=120.0, gt=0)
    extra_headers: dict[str, str] = Field(default_factory=dict)


class ModelProfile(BaseModel):
    """A named model plus the capabilities Neural Search may use it for."""

    name: str
    provider: str
    model: str
    capabilities: set[InferenceCapability]
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceRequest(BaseModel):
    """Provider-neutral chat/generation request."""

    messages: list[InferenceMessage]
    capability: InferenceCapability = InferenceCapability.CHAT
    model_profile: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    response_schema: dict[str, Any] | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingRequest(BaseModel):
    """Provider-neutral embedding request for retrieval or indexing."""

    inputs: list[str]
    model_profile: str | None = None
    input_type: str = "passage"
    truncate: str = "END"
    dimensions: int | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("inputs")
    @classmethod
    def nonempty_inputs(cls, value: list[str]) -> list[str]:
        if not value or not all(item.strip() for item in value):
            raise ValueError("embedding inputs must contain non-empty text")
        return value

    @field_validator("input_type")
    @classmethod
    def valid_input_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"query", "passage"}:
            raise ValueError("input_type must be query or passage")
        return normalized


class RerankRequest(BaseModel):
    """Provider-neutral query/passage reranking request."""

    query: str
    passages: list[str]
    model_profile: str | None = None
    truncate: str = "END"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def nonempty_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rerank query cannot be empty")
        return value

    @field_validator("passages")
    @classmethod
    def nonempty_passages(cls, value: list[str]) -> list[str]:
        if not value or not all(item.strip() for item in value):
            raise ValueError("rerank passages must contain non-empty text")
        return value


class RunManifest(BaseModel):
    """Immutable provenance metadata for a single model invocation."""

    run_id: str
    provider: str
    model: str
    model_profile: str
    capability: InferenceCapability
    prompt_hash: str
    request_hash: str
    started_at: str
    completed_at: str
    input_revision: str | None = None
    prompt_template: str | None = None
    parent_run_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        *,
        provider: str,
        model: str,
        model_profile: str,
        capability: InferenceCapability,
        payload: dict[str, Any],
        prompt_material: str,
        metadata: dict[str, Any],
        started_at: datetime,
        completed_at: datetime,
    ) -> RunManifest:
        request_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        request_hash = hashlib.sha256(request_json.encode()).hexdigest()
        prompt_hash = hashlib.sha256(prompt_material.encode()).hexdigest()
        run_id = f"inference:{request_hash[:16]}:{completed_at.timestamp():.0f}"
        return cls(
            run_id=run_id,
            provider=provider,
            model=model,
            model_profile=model_profile,
            capability=capability,
            prompt_hash=prompt_hash,
            request_hash=request_hash,
            started_at=started_at.astimezone(UTC).isoformat(),
            completed_at=completed_at.astimezone(UTC).isoformat(),
            input_revision=metadata.get("input_revision"),
            prompt_template=metadata.get("prompt_template"),
            parent_run_ids=list(metadata.get("parent_run_ids", [])),
            metadata={
                key: value
                for key, value in metadata.items()
                if key not in {"input_revision", "prompt_template", "parent_run_ids"}
            },
        )

    @classmethod
    def build(
        cls,
        *,
        provider: str,
        model: str,
        model_profile: str,
        capability: InferenceCapability,
        request: InferenceRequest,
        started_at: datetime,
        completed_at: datetime,
    ) -> RunManifest:
        payload = request.model_dump(mode="json", exclude_none=True)
        prompt_text = "\n".join(message.content for message in request.messages)
        return cls.from_payload(
            provider=provider,
            model=model,
            model_profile=model_profile,
            capability=capability,
            payload=payload,
            prompt_material=prompt_text,
            metadata=request.metadata,
            started_at=started_at,
            completed_at=completed_at,
        )


class InferenceResult(BaseModel):
    """Normalized generation response plus reproducibility provenance."""

    text: str
    model: str
    provider: str
    usage: dict[str, int | float | str | None] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
    manifest: RunManifest


class EmbeddingResult(BaseModel):
    vectors: list[list[float | int]]
    model: str
    provider: str
    usage: dict[str, int | float | str | None] = Field(default_factory=dict)
    manifest: RunManifest


class RankedPassage(BaseModel):
    index: int = Field(ge=0)
    score: float
    passage: str


class RerankResult(BaseModel):
    rankings: list[RankedPassage]
    model: str
    provider: str
    usage: dict[str, int | float | str | None] = Field(default_factory=dict)
    manifest: RunManifest
