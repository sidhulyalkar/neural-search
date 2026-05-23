"""Pydantic schemas for API, extraction, search, and generation outputs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

DatasetQAStatus = Literal[
    "unreviewed",
    "auto_generated",
    "needs_review",
    "reviewed",
    "trusted",
    "rejected",
]


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
    qa_status: DatasetQAStatus = "auto_generated"
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
    title: str | None = None
    source: str | None = None
    source_id: str | None = None
    url: str | None = None
    doi: str | None = None
    license: str | None = None
    data_standard: str | None = None
    species: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    brain_regions: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    behaviors: list[str] = Field(default_factory=list)
    assets: list[dict[str, Any]] = Field(default_factory=list)
    related_papers: list[dict[str, Any]] = Field(default_factory=list)
    summary: str
    summary_details: dict[str, Any] = Field(default_factory=dict)
    experimental_structure: dict[str, Any] = Field(default_factory=dict)
    neural_data: dict[str, Any] = Field(default_factory=dict)
    analysis_plan: dict[str, Any] = Field(default_factory=dict)
    linked_literature: dict[str, Any] = Field(default_factory=dict)
    reuse_instructions: dict[str, Any] = Field(default_factory=dict)
    why_relevant: list[str] = Field(default_factory=list)
    scientific_labels: dict[str, Any] = Field(default_factory=dict)
    analysis_readiness: AnalysisReadiness
    readiness: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    missing_metadata: list[str] = Field(default_factory=list)
    suggested_analyses: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    card_markdown: str | None = None
    qa_status: DatasetQAStatus = "auto_generated"
    task_labels_verified: bool = False
    modality_labels_verified: bool = False
    behavior_labels_verified: bool = False
    brain_regions_verified: bool = False
    linked_papers_verified: bool = False
    notebook_tested: bool = False
    reviewer_notes: str = ""
    markdown: str | None = None
    generated_at: datetime | None = None


class NotebookGenerationResponse(BaseModel):
    dataset_id: UUID | str
    asset_id: UUID | str
    output_path: str
    valid: bool
    warnings: list[str] = Field(default_factory=list)


class ExperimentQuery(BaseModel):
    task: list[str] = Field(default_factory=list)
    behavior: list[str] = Field(default_factory=list)
    modality: list[str] = Field(default_factory=list)
    species: list[str] = Field(default_factory=list)
    brain_region: list[str] = Field(default_factory=list)
    data_standard: list[str] = Field(default_factory=list)
    source_archive: list[str] = Field(default_factory=list)
    analysis_goal: list[str] = Field(default_factory=list)
    min_analysis_readiness_score: int | None = Field(default=None, ge=0, le=100)
    reviewed_trusted_only: bool = False


class SearchRequest(BaseModel):
    query: str = ""
    filters: dict[str, Any] = Field(default_factory=dict)
    structured_query: ExperimentQuery | None = None
    limit: int = Field(default=10, ge=1, le=100)


class SearchResult(BaseModel):
    dataset_id: UUID | str
    score: float = Field(ge=0.0)
    why_matched: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    inferred_concepts: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    missing_metadata_warnings: list[str] = Field(default_factory=list)
    reusable_reason: str | None = None
    dataset_card_preview: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    parsed_query: dict[str, Any] = Field(default_factory=dict)
    results: list[SearchResult] = Field(default_factory=list)


# Dataset Comparison Schemas
class DatasetCompareRequest(BaseModel):
    """Request to compare multiple datasets."""

    dataset_ids: list[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="List of 2-5 dataset IDs to compare",
    )


class DatasetComparisonItemRead(BaseModel):
    """Comparison data for a single dataset."""

    dataset_id: str
    title: str
    source: str
    source_id: str
    url: str | None = None
    doi: str | None = None
    license: str | None = None

    task_labels: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    species: list[str] = Field(default_factory=list)
    brain_regions: list[str] = Field(default_factory=list)
    behavior_labels: list[str] = Field(default_factory=list)
    data_standards: list[str] = Field(default_factory=list)

    has_trials: bool = False
    has_events: bool = False
    has_behavior: bool = False
    subject_count: int | None = None
    session_count: int | None = None

    linked_paper_count: int = 0
    linked_papers: list[dict[str, Any]] = Field(default_factory=list)

    analysis_readiness_score: int = 0
    strengths: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    missing_metadata: list[str] = Field(default_factory=list)

    available_notebook_templates: list[str] = Field(default_factory=list)
    suggested_analyses: list[str] = Field(default_factory=list)
    matched_recipes: list[dict[str, Any]] = Field(default_factory=list)

    qa_status: str = "auto_generated"


class FieldComparisonRead(BaseModel):
    """Comparison of a single field across datasets."""

    field_name: str
    field_label: str
    values: dict[str, Any] = Field(default_factory=dict)
    all_same: bool = False
    union_values: list[Any] = Field(default_factory=list)
    intersection_values: list[Any] = Field(default_factory=list)


class ComparisonResultRead(BaseModel):
    """Complete comparison result for multiple datasets."""

    dataset_ids: list[str]
    datasets: list[DatasetComparisonItemRead]
    field_comparisons: list[FieldComparisonRead] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime
