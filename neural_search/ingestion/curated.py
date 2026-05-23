"""Loader for curated seed sources.

Loads manually curated dataset and paper references from YAML for seeding
the Neural Search index before full live ingestion is reliable.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

CURATED_SOURCES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "seed" / "curated_sources.yaml"
)


class SourceType(str, Enum):
    """Supported source types for curated entries."""

    DANDI = "dandi"
    OPENNEURO = "openneuro"
    OPENALEX = "openalex"
    MANUAL = "manual"


class Priority(str, Enum):
    """Ingestion priority levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CuratedSource(BaseModel):
    """A single curated source entry."""

    source_type: SourceType
    source_id: str
    title: str
    url: str | None = None
    expected_tasks: list[str] = Field(default_factory=list)
    expected_modalities: list[str] = Field(default_factory=list)
    expected_behaviors: list[str] = Field(default_factory=list)
    expected_species: list[str] = Field(default_factory=list)
    notes: str | None = None
    priority: Priority = Priority.MEDIUM

    @field_validator("notes", mode="before")
    @classmethod
    def strip_notes(cls, v: str | None) -> str | None:
        """Strip leading/trailing whitespace from multi-line notes."""
        if v is not None:
            return v.strip()
        return v

    def is_dataset(self) -> bool:
        """Return True if this source represents a dataset (not a paper)."""
        return self.source_type in (SourceType.DANDI, SourceType.OPENNEURO, SourceType.MANUAL)

    def is_paper(self) -> bool:
        """Return True if this source represents a paper."""
        return self.source_type == SourceType.OPENALEX

    def canonical_source(self) -> str:
        """Return the canonical source string for database records."""
        if self.source_type == SourceType.DANDI:
            return "dandi"
        elif self.source_type == SourceType.OPENNEURO:
            return "openneuro"
        elif self.source_type == SourceType.OPENALEX:
            return "openalex"
        else:
            return "manual"


class CuratedSourcesFile(BaseModel):
    """Container for the curated sources YAML file."""

    sources: list[CuratedSource] = Field(default_factory=list)


def load_curated_sources(path: str | Path | None = None) -> list[CuratedSource]:
    """Load curated sources from YAML file.

    Args:
        path: Path to the YAML file. Defaults to data/seed/curated_sources.yaml.

    Returns:
        List of CuratedSource entries.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        ValueError: If the YAML is malformed or doesn't match schema.
    """
    file_path = Path(path) if path else CURATED_SOURCES_PATH

    if not file_path.exists():
        raise FileNotFoundError(f"Curated sources file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict) or "sources" not in raw:
        raise ValueError(f"Invalid curated sources format in {file_path}")

    parsed = CuratedSourcesFile.model_validate(raw)
    return parsed.sources


def load_curated_datasets(path: str | Path | None = None) -> list[CuratedSource]:
    """Load only dataset entries (DANDI, OpenNeuro, manual) from curated sources."""
    return [s for s in load_curated_sources(path) if s.is_dataset()]


def load_curated_papers(path: str | Path | None = None) -> list[CuratedSource]:
    """Load only paper entries (OpenAlex) from curated sources."""
    return [s for s in load_curated_sources(path) if s.is_paper()]


def load_by_priority(
    priority: Priority | str, path: str | Path | None = None
) -> list[CuratedSource]:
    """Load curated sources filtered by priority level.

    Args:
        priority: Priority level to filter by (high, medium, low).
        path: Path to the YAML file.

    Returns:
        List of CuratedSource entries matching the priority.
    """
    if isinstance(priority, str):
        priority = Priority(priority)
    return [s for s in load_curated_sources(path) if s.priority == priority]


def load_high_priority(path: str | Path | None = None) -> list[CuratedSource]:
    """Load only high-priority curated sources."""
    return load_by_priority(Priority.HIGH, path)


def curated_to_dataset_create(source: CuratedSource) -> dict[str, Any]:
    """Convert a CuratedSource to a DatasetCreate-compatible dict.

    This creates a minimal dataset record that can be used to seed the database
    or as a target for live ingestion to enrich.

    Args:
        source: A curated source entry that is_dataset().

    Returns:
        Dict compatible with DatasetCreate schema.
    """
    if not source.is_dataset():
        raise ValueError(f"Cannot convert paper source to dataset: {source.source_id}")

    return {
        "source": source.canonical_source(),
        "source_id": source.source_id,
        "title": source.title,
        "description": source.notes,
        "url": source.url,
        "species": source.expected_species,
        "modalities": source.expected_modalities,
        "tasks": source.expected_tasks,
        "behaviors": source.expected_behaviors,
        "metadata_json": {
            "curated": True,
            "priority": source.priority.value,
        },
    }


def curated_to_paper_stub(source: CuratedSource) -> dict[str, Any]:
    """Convert a CuratedSource to a Paper-compatible dict stub.

    This creates a minimal paper record for linking purposes.

    Args:
        source: A curated source entry that is_paper().

    Returns:
        Dict compatible with Paper model fields.
    """
    if not source.is_paper():
        raise ValueError(f"Cannot convert dataset source to paper: {source.source_id}")

    return {
        "openalex_id": source.source_id,
        "title": source.title,
        "url": source.url,
        "concepts": source.expected_tasks + source.expected_modalities,
        "metadata_json": {
            "curated": True,
            "priority": source.priority.value,
            "expected_species": source.expected_species,
            "expected_behaviors": source.expected_behaviors,
        },
    }


def summarize_curated_sources(path: str | Path | None = None) -> dict[str, Any]:
    """Return a summary of curated sources for diagnostics.

    Returns:
        Dict with counts by source_type, priority, and totals.
    """
    sources = load_curated_sources(path)

    by_type: dict[str, int] = {}
    by_priority: dict[str, int] = {}

    for source in sources:
        by_type[source.source_type.value] = by_type.get(source.source_type.value, 0) + 1
        by_priority[source.priority.value] = by_priority.get(source.priority.value, 0) + 1

    return {
        "total": len(sources),
        "datasets": len([s for s in sources if s.is_dataset()]),
        "papers": len([s for s in sources if s.is_paper()]),
        "by_source_type": by_type,
        "by_priority": by_priority,
    }
