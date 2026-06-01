"""Versioned latent signature schemas for event-aligned neural search."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required_text(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("must not be empty")
    return cleaned


class LatentSourceFile(BaseModel):
    """Source file provenance for a latent signature."""

    path: str
    file_format: str | None = None
    modality: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    asset_id: str | None = None

    @field_validator("path")
    @classmethod
    def require_path(cls, value: str) -> str:
        return _required_text(value)


class LatentQCMetadata(BaseModel):
    """QC metadata for a latent signature extraction."""

    valid: bool = True
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    missing_signals: list[str] = Field(default_factory=list)


class LatentSignatureVector(BaseModel):
    """One fixed-dimensional latent vector block."""

    name: str
    dimensions: int = Field(gt=0)
    values: list[float] = Field(default_factory=list)
    feature_family: str = "event_aligned"
    units: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "feature_family")
    @classmethod
    def require_text(cls, value: str) -> str:
        return _required_text(value)

    @model_validator(mode="after")
    def validate_dimensions(self) -> LatentSignatureVector:
        if self.values and len(self.values) != self.dimensions:
            raise ValueError("values length must match dimensions")
        return self


class EventAlignedLatentSignature(BaseModel):
    """Versioned event-aligned latent signature with provenance and QC."""

    schema_version: str = "v0.9.0"
    dataset_id: str
    session_id: str
    alignment_event: str
    pre_event_ms: float = Field(ge=0.0)
    post_event_ms: float = Field(ge=0.0)
    vectors: list[LatentSignatureVector] = Field(default_factory=list)
    source_files: list[LatentSourceFile] = Field(default_factory=list)
    qc: LatentQCMetadata = Field(default_factory=LatentQCMetadata)
    extractor_name: str
    extractor_version: str
    created_at: str = Field(default_factory=_utc_now)

    @field_validator(
        "schema_version",
        "dataset_id",
        "session_id",
        "alignment_event",
        "extractor_name",
        "extractor_version",
    )
    @classmethod
    def require_text(cls, value: str) -> str:
        return _required_text(value)

    @model_validator(mode="after")
    def validate_signature(self) -> EventAlignedLatentSignature:
        if not self.vectors:
            raise ValueError("at least one latent vector is required")
        if not self.source_files:
            raise ValueError("at least one source file is required")
        if self.pre_event_ms == 0 and self.post_event_ms == 0:
            raise ValueError("event window must have non-zero duration")
        return self

    @property
    def total_dimensions(self) -> int:
        """Return total dimensionality across vector blocks."""

        return sum(vector.dimensions for vector in self.vectors)

    @property
    def has_valid_qc(self) -> bool:
        """Return True when QC is valid and no required signals are missing."""

        return self.qc.valid and not self.qc.missing_signals

    def compact_metadata(self) -> dict[str, Any]:
        """Return agent-friendly metadata without vector values."""

        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "session_id": self.session_id,
            "alignment_event": self.alignment_event,
            "pre_event_ms": self.pre_event_ms,
            "post_event_ms": self.post_event_ms,
            "total_dimensions": self.total_dimensions,
            "vector_names": [vector.name for vector in self.vectors],
            "source_files": [source.path for source in self.source_files],
            "extractor_name": self.extractor_name,
            "extractor_version": self.extractor_version,
            "qc_valid": self.has_valid_qc,
            "warnings": self.qc.warnings,
        }
