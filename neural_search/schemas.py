"""Pydantic schemas for API, extraction, search, and generation outputs."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class LabelEvidence(BaseModel):
    id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    category: str | None = None


class DatasetCreate(BaseModel):
    source: str
    source_id: str
    title: str
    description: str | None = None
    url: str | None = None
    license: str | None = None
    species: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    brain_regions: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    behaviors: list[str] = Field(default_factory=list)
    data_standards: list[str] = Field(default_factory=list)
    has_behavior: bool = False
    has_trials: bool = False
    has_raw_data: bool = False
    has_processed_data: bool = False
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class DatasetRead(DatasetCreate):
    model_config = ConfigDict(from_attributes=True)

    id: UUID | str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DatasetAssetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID | str
    dataset_id: UUID | str
    path: str
    asset_type: str | None = None
    file_format: str | None = None
    size_bytes: int | None = None
    subject_id: str | None = None
    session_id: str | None = None
    modality: str | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class PaperRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID | str
    openalex_id: str | None = None
    doi: str | None = None
    title: str
    abstract: str | None = None
    publication_year: int | None = None
    authors_json: list[dict[str, Any]] = Field(default_factory=list)
    url: str | None = None
    concepts: list[str] = Field(default_factory=list)
    linked_dataset_ids: list[str] = Field(default_factory=list)
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class OntologyTermRead(BaseModel):
    id: str
    label: str
    category: str | None = None
    definition: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    common_events: list[str] = Field(default_factory=list)
    relevant_modalities: list[str] = Field(default_factory=list)
    relevant_regions: list[str] = Field(default_factory=list)
    suggested_analyses: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    tasks: list[LabelEvidence] = Field(default_factory=list)
    behaviors: list[LabelEvidence] = Field(default_factory=list)
    modalities: list[LabelEvidence] = Field(default_factory=list)
    brain_regions: list[LabelEvidence] = Field(default_factory=list)
    species: list[LabelEvidence] = Field(default_factory=list)
    data_standards: list[LabelEvidence] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class AnalysisReadiness(BaseModel):
    score: int = Field(ge=0, le=100)
    strengths: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class DatasetCardRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID | str | None = None
    dataset_id: UUID | str
    summary: str
    why_relevant: list[str] = Field(default_factory=list)
    scientific_labels: dict[str, Any] = Field(default_factory=dict)
    analysis_readiness: AnalysisReadiness
    missing_fields: list[str] = Field(default_factory=list)
    suggested_analyses: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    card_markdown: str | None = None


class NotebookGenerationResponse(BaseModel):
    dataset_id: UUID | str
    asset_id: UUID | str
    output_path: str
    valid: bool
    warnings: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=10, ge=1, le=100)


class SearchResult(BaseModel):
    dataset_id: UUID | str
    score: float = Field(ge=0.0)
    why_matched: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    dataset_card_preview: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    parsed_query: dict[str, Any] = Field(default_factory=dict)
    results: list[SearchResult] = Field(default_factory=list)

