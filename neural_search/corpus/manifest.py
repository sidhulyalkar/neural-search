"""Validation helpers for manifest-driven real corpus builds."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

ManifestStatus = Literal["fixture", "ready", "planned", "skipped"]
ManifestRecordType = Literal["dataset", "paper"]


class CorpusManifestEntry(BaseModel):
    """One source record selected for a corpus build."""

    model_config = ConfigDict(extra="forbid")

    source: str
    source_id: str
    record_type: ManifestRecordType
    priority: int = Field(ge=1)
    status: ManifestStatus = "fixture"
    tags: list[str] = Field(default_factory=list)
    fetch: dict[str, Any] = Field(default_factory=dict)
    review_notes: str = ""
    scientific_rationale: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    inspection_paths: list[str] = Field(default_factory=list)

    @field_validator("source", "source_id", "scientific_rationale")
    @classmethod
    def required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        tags: list[str] = []
        for value in values:
            tag = str(value).strip().lower().replace("-", "_")
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
        return tags


class CorpusManifest(BaseModel):
    """Top-level manifest that can drive deterministic fixture ingestion."""

    model_config = ConfigDict(extra="forbid")

    corpus_tag: str = "real_v07"
    description: str = ""
    entries: list[CorpusManifestEntry]

    @field_validator("corpus_tag")
    @classmethod
    def required_tag(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("corpus_tag must not be empty")
        return cleaned

    def source_counts(self) -> dict[str, int]:
        return dict(Counter(entry.source for entry in self.entries))

    def record_type_counts(self) -> dict[str, int]:
        return dict(Counter(entry.record_type for entry in self.entries))


def load_manifest(path: str | Path) -> CorpusManifest:
    """Load and validate a real-corpus manifest from YAML."""

    manifest_path = Path(path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a YAML mapping: {manifest_path}")
    return CorpusManifest.model_validate(payload)
