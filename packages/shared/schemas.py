"""Core Pydantic schemas for Neural Search."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class DatasetSource(str, Enum):
    """Source repository for a dataset."""

    DANDI = "dandi"
    OPENNEURO = "openneuro"
    OTHER = "other"


class DataStandard(str, Enum):
    """Data standard format."""

    NWB = "nwb"
    BIDS = "bids"
    OTHER = "other"


class ExtractionLabel(BaseModel):
    """A single extracted label with provenance."""

    label: str = Field(..., description="The extracted label value")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score 0-1"
    )
    evidence: str = Field(..., description="Evidence supporting this label")
    source_span: Optional[str] = Field(
        None, description="Source text span if available"
    )
    extractor: str = Field(
        default="deterministic", description="Extraction method used"
    )


class ExtractionResult(BaseModel):
    """Complete extraction results for a dataset."""

    dataset_id: str
    task_labels: list[ExtractionLabel] = Field(default_factory=list)
    behavior_labels: list[ExtractionLabel] = Field(default_factory=list)
    modality_labels: list[ExtractionLabel] = Field(default_factory=list)
    region_labels: list[ExtractionLabel] = Field(default_factory=list)
    species_labels: list[ExtractionLabel] = Field(default_factory=list)
    data_standard: Optional[DataStandard] = None
    analysis_affordances: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    extracted_at: datetime = Field(default_factory=datetime.utcnow)


class DatasetAsset(BaseModel):
    """An asset (file) within a dataset."""

    path: str
    size_bytes: Optional[int] = None
    content_type: Optional[str] = None
    is_nwb: bool = False
    is_bids: bool = False


class DatasetRecord(BaseModel):
    """Normalized dataset record from any source."""

    id: str = Field(..., description="Unique dataset identifier")
    source: DatasetSource = Field(..., description="Source repository")
    source_id: str = Field(..., description="ID in the source repository")
    title: str = Field(..., description="Dataset title")
    description: Optional[str] = Field(None, description="Dataset description")
    contributors: list[str] = Field(default_factory=list)
    species: list[str] = Field(default_factory=list)
    license: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    version: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Extracted metadata
    data_standard: Optional[DataStandard] = None
    modalities: list[str] = Field(default_factory=list)
    brain_regions: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)

    # Raw metadata for provenance
    raw_metadata: Optional[dict[str, Any]] = Field(
        None, description="Original source metadata"
    )

    # Assets
    assets: list[DatasetAsset] = Field(default_factory=list)
    nwb_count: int = 0
    total_size_bytes: Optional[int] = None


class PaperRecord(BaseModel):
    """A paper record from OpenAlex or other sources."""

    id: str
    source: str = "openalex"
    source_id: str
    title: str
    abstract: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    doi: Optional[str] = None
    publication_date: Optional[datetime] = None
    journal: Optional[str] = None
    url: Optional[str] = None
    citation_count: int = 0

    # Links to datasets
    linked_dataset_ids: list[str] = Field(default_factory=list)


class AnalysisReadiness(BaseModel):
    """Analysis readiness assessment for a dataset."""

    score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall readiness score"
    )
    strengths: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    missing_metadata: list[str] = Field(default_factory=list)
    suggested_analyses: list[str] = Field(default_factory=list)


class DatasetCard(BaseModel):
    """Generated dataset card."""

    dataset_id: str
    title: str
    summary: str

    # Core metadata
    source: DatasetSource
    data_standard: Optional[DataStandard] = None
    species: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    brain_regions: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)

    # Extracted labels with provenance
    extraction: Optional[ExtractionResult] = None

    # Readiness assessment
    readiness: Optional[AnalysisReadiness] = None

    # Links
    url: Optional[str] = None
    doi: Optional[str] = None
    related_papers: list[str] = Field(default_factory=list)

    # Rendered content
    markdown: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class SearchQuery(BaseModel):
    """Search query parameters."""

    query: str = Field(..., description="Search query text")
    task_filter: Optional[list[str]] = None
    modality_filter: Optional[list[str]] = None
    species_filter: Optional[list[str]] = None
    source_filter: Optional[list[DatasetSource]] = None
    min_readiness: Optional[float] = Field(None, ge=0.0, le=1.0)
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    include_vector_search: bool = True


class SearchResultItem(BaseModel):
    """A single search result."""

    dataset: DatasetRecord
    score: float = Field(..., description="Relevance score")
    why_matched: list[str] = Field(
        default_factory=list, description="Explanation of match"
    )
    warnings: list[str] = Field(default_factory=list)
    suggested_next_actions: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Search results response."""

    query: str
    total_count: int
    results: list[SearchResultItem]
    facets: Optional[dict[str, dict[str, int]]] = None
    search_time_ms: float
