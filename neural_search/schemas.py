"""Pydantic schemas for API, extraction, search, and generation outputs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


def _non_empty(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("must not be empty")
    return cleaned


def _normalized_identifier(value: str) -> str:
    cleaned = _non_empty(value).strip().lower().replace("-", "_").replace(" ", "_")
    return "_".join(part for part in cleaned.split("_") if part)


class EvidenceLabel(BaseModel):
    """A normalized scientific label with extraction provenance."""

    id: str
    label: str
    label_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: str | None = None
    source_field: str | None = None
    source_value: str | None = None
    extractor_name: str = "neural_search.rule_extractor"
    extractor_version: str = "v0.3.0"

    @field_validator("id", "label", "extractor_name", "extractor_version")
    @classmethod
    def required_text(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("label_type")
    @classmethod
    def normalize_label_type(cls, value: str) -> str:
        return _normalized_identifier(value)


class UsabilityFlags(BaseModel):
    """Conservative metadata-derived usability hints for a normalized dataset."""

    has_trials: bool | None = None
    has_behavior: bool | None = None
    has_neural_data: bool | None = None
    has_continuous_behavior: bool | None = None
    has_event_timestamps: bool | None = None
    has_raw_data: bool | None = None
    has_processed_data: bool | None = None
    has_standard_format: bool | None = None


class AnalysisAffordance(BaseModel):
    """Evidence-backed estimate of whether a dataset supports an analysis."""

    analysis_id: str
    support_level: Literal["high", "medium", "low", "unsupported", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    required_fields_present: list[str] = Field(default_factory=list)
    helpful_fields_present: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    detector_name: str = "rule_based_analysis_affordance_detector"
    detector_version: str = "v0.3.0"

    @field_validator("analysis_id", "detector_name", "detector_version")
    @classmethod
    def required_text(cls, value: str) -> str:
        return _non_empty(value)


class NormalizedDatasetRecord(BaseModel):
    """Source-normalized dataset record for provenance-aware corpus ingestion."""

    dataset_id: str
    source: str
    source_id: str
    title: str
    description: str | None = None
    url: str | None = None
    raw_payload_path: str | None = None
    species: list[EvidenceLabel] = Field(default_factory=list)
    modalities: list[EvidenceLabel] = Field(default_factory=list)
    brain_regions: list[EvidenceLabel] = Field(default_factory=list)
    tasks: list[EvidenceLabel] = Field(default_factory=list)
    behavioral_events: list[EvidenceLabel] = Field(default_factory=list)
    analysis_goals: list[EvidenceLabel] = Field(default_factory=list)
    data_standards: list[EvidenceLabel] = Field(default_factory=list)
    file_formats: list[EvidenceLabel] = Field(default_factory=list)
    linked_papers: list[str] = Field(default_factory=list)
    usability_flags: UsabilityFlags = Field(default_factory=UsabilityFlags)
    missing_fields: list[str] = Field(default_factory=list)
    analysis_affordances: list[AnalysisAffordance] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    extractor_version: str = "v0.3.0"

    @field_validator("dataset_id", "source", "source_id", "title", "extractor_version")
    @classmethod
    def required_text(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        return _normalized_identifier(value)


class NormalizedPaperRecord(BaseModel):
    """Source-normalized paper record with extracted provenance labels."""

    paper_id: str
    source: str
    source_id: str
    title: str
    abstract: str | None = None
    doi: str | None = None
    url: str | None = None
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    linked_datasets: list[str] = Field(default_factory=list)
    extracted_labels: list[EvidenceLabel] = Field(default_factory=list)
    raw_payload_path: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    extractor_version: str = "v0.3.0"

    @field_validator("paper_id", "source", "source_id", "title", "extractor_version")
    @classmethod
    def required_text(cls, value: str) -> str:
        return _non_empty(value)

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        return _normalized_identifier(value)


class ScoreBreakdown(BaseModel):
    """Interpretable retrieval score heads, normalized to 0..1."""

    lexical_score: float = Field(default=0.0, ge=0.0, le=1.0)
    ontology_score: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    field_semantic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    graph_score: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    usability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    analysis_fit_score: float = Field(default=0.0, ge=0.0, le=1.0)
    negative_constraint_score: float = Field(default=0.0, ge=0.0, le=1.0)
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)


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
    missing_requirements: list[str] = Field(default_factory=list)
    negative_constraint_matches: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    explanation: str | None = None
    reusable_reason: str | None = None
    dataset_card_preview: dict[str, Any] = Field(default_factory=dict)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    graph_context: dict[str, Any] | None = None
    linked_papers: list[dict[str, Any]] = Field(default_factory=list)
    filtered_constraints: list[dict[str, Any]] = Field(default_factory=list)
    missing_metadata: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    parsed_query: dict[str, Any] = Field(default_factory=dict)
    results: list[SearchResult] = Field(default_factory=list)
    filtered_constraints: list[dict[str, Any]] = Field(default_factory=list)


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
