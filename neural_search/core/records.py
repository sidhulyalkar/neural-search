"""Canonical Scientific Record Model.

This module defines the foundational data model for all searchable scientific objects.
The model distinguishes between different metadata layers:

1. Raw: Original source metadata, preserved verbatim
2. Normalized: Standardized labels and identifiers
3. Extracted: Labels extracted from text with confidence
4. Inferred: Labels derived from context or relationships
5. Learned: Embeddings and similarity-derived attributes
6. Graph: Relationships from the knowledge graph

Every label/entity includes provenance and confidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ScientificRecordType(StrEnum):
    """Types of scientific records in the system."""

    DATASET = "dataset"
    PAPER = "paper"
    ANALYSIS_METHOD = "analysis_method"
    BEHAVIORAL_TASK = "behavioral_task"
    BRAIN_REGION = "brain_region"
    MODALITY = "modality"
    SPECIES = "species"
    EXPERIMENTAL_DESIGN = "experimental_design"
    CLAIM = "claim"
    MODEL_ARCHITECTURE = "model_architecture"
    FILE_FORMAT = "file_format"
    INSTITUTION = "institution"
    AUTHOR = "author"


class EntityType(StrEnum):
    """Types of scientific entities that can be extracted."""

    SPECIES = "species"
    STRAIN = "strain"
    MODALITY = "modality"
    SIGNAL_TYPE = "signal_type"
    BRAIN_REGION = "brain_region"
    CELL_TYPE = "cell_type"
    BEHAVIORAL_TASK = "behavioral_task"
    STIMULUS_TYPE = "stimulus_type"
    DISEASE_MODEL = "disease_model"
    PERTURBATION = "perturbation"
    RECORDING_TECHNOLOGY = "recording_technology"
    ANALYSIS_METHOD = "analysis_method"
    FILE_FORMAT = "file_format"
    MODEL_ARCHITECTURE = "model_architecture"
    METRIC = "metric"
    CLAIM = "claim"
    LATENT_CONSTRUCT = "latent_construct"
    DATASET_ACCESSION = "dataset_accession"
    DOI = "doi"
    SOFTWARE_TOOL = "software_tool"


class MetadataLayer(StrEnum):
    """The layer from which metadata originated."""

    RAW = "raw"              # Original source metadata
    NORMALIZED = "normalized" # Standardized after ingestion
    EXTRACTED = "extracted"   # Extracted from text
    INFERRED = "inferred"     # Derived from context/relationships
    LEARNED = "learned"       # From embeddings/ML
    GRAPH = "graph"           # From knowledge graph
    HUMAN = "human"           # Human-annotated/verified


class ExtractionProvenance(BaseModel):
    """Provenance information for an extracted or inferred entity."""

    source_type: str                    # e.g., "metadata", "abstract", "methods"
    source_field: str | None = None     # e.g., "description", "title"
    source_text: str | None = None      # The actual text that was analyzed
    evidence_text: str | None = None    # Snippet showing the evidence
    extractor_name: str = "unknown"
    extractor_version: str = "v0.0.0"
    extraction_method: str = "rule"     # "rule", "embedding", "llm", "human"
    extracted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ScientificEntity(BaseModel):
    """A scientific entity with full provenance and confidence tracking.

    This is the fundamental unit for representing extracted knowledge:
    species, modalities, tasks, brain regions, etc.
    """

    id: str                             # Stable identifier (e.g., "species:mus_musculus")
    label: str                          # Human-readable label
    entity_type: EntityType
    layer: MetadataLayer                # Which layer this came from
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    # Provenance
    provenance: ExtractionProvenance | None = None

    # Semantic
    canonical_id: str | None = None     # Link to ontology term if available
    aliases: list[str] = Field(default_factory=list)

    # Quality
    verified: bool = False              # Has a human verified this?
    verification_notes: str | None = None

    @field_validator("id", "label")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    def to_evidence_label(self) -> dict[str, Any]:
        """Convert to legacy EvidenceLabel format for compatibility."""
        return {
            "id": self.id,
            "label": self.label,
            "label_type": self.entity_type.value,
            "confidence": self.confidence,
            "evidence_text": self.provenance.evidence_text if self.provenance else None,
            "source_field": self.provenance.source_field if self.provenance else None,
            "extractor_name": self.provenance.extractor_name if self.provenance else "unknown",
            "extractor_version": self.provenance.extractor_version if self.provenance else "v0.0.0",
        }


class AnalysisAffordanceV2(BaseModel):
    """Enhanced analysis affordance with requirements and provenance.

    An affordance describes what analyses a dataset can support.
    """

    analysis_id: str                    # e.g., "spike_sorting", "latent_state_modeling"
    analysis_label: str                 # Human-readable name

    # Support level
    support_level: Literal["high", "medium", "low", "unsupported", "unknown"] = "unknown"
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    # Requirements
    required_fields_present: list[str] = Field(default_factory=list)
    required_fields_missing: list[str] = Field(default_factory=list)
    helpful_fields_present: list[str] = Field(default_factory=list)

    # Evidence
    evidence: list[str] = Field(default_factory=list)
    provenance: ExtractionProvenance | None = None

    # Metadata
    min_samples_required: int | None = None
    recommended_preprocessing: list[str] = Field(default_factory=list)
    compatible_tools: list[str] = Field(default_factory=list)


class EmbeddingStatus(BaseModel):
    """Status of embeddings for a record."""

    has_title_embedding: bool = False
    has_description_embedding: bool = False
    has_methods_embedding: bool = False
    has_combined_embedding: bool = False
    embedding_model: str | None = None
    embedding_version: str | None = None
    last_embedded_at: str | None = None


class GraphStatus(BaseModel):
    """Status of graph relationships for a record."""

    in_graph: bool = False
    node_id: str | None = None
    edge_count: int = 0
    linked_papers: int = 0
    linked_datasets: int = 0
    last_graph_update: str | None = None


class QualityScore(BaseModel):
    """Quality and completeness assessment for a record."""

    overall_score: float = Field(ge=0.0, le=1.0, default=0.0)
    metadata_completeness: float = Field(ge=0.0, le=1.0, default=0.0)
    provenance_strength: float = Field(ge=0.0, le=1.0, default=0.0)
    label_confidence_avg: float = Field(ge=0.0, le=1.0, default=0.0)
    missing_critical_fields: list[str] = Field(default_factory=list)
    missing_optional_fields: list[str] = Field(default_factory=list)
    qa_status: Literal["auto_generated", "needs_review", "reviewed", "trusted", "rejected"] = "auto_generated"
    reviewer_notes: str | None = None


class ScientificRecord(BaseModel):
    """Canonical representation for all searchable scientific objects.

    This is the unified model that every dataset, paper, method, etc.
    is converted to for indexing and retrieval. It maintains clear
    separation between metadata layers and tracks provenance throughout.

    Design principles:
    1. Stable ID: Every record has a globally unique, deterministic ID
    2. Layered metadata: Raw vs. normalized vs. inferred are distinct
    3. Provenance tracking: Every label knows where it came from
    4. Confidence scoring: Uncertainty is quantified throughout
    5. Embedding-ready: Fields are structured for semantic indexing
    6. Graph-aware: Tracks relationship status
    """

    # Identity
    record_id: str                      # e.g., "dataset:dandi:000026"
    record_type: ScientificRecordType
    source: str                         # e.g., "dandi", "openalex", "curated"
    source_id: str                      # Original ID from source

    # Core text fields (embedding targets)
    title: str
    description: str | None = None
    abstract: str | None = None          # For papers
    methods_summary: str | None = None   # Extracted or authored methods
    scientific_summary: str | None = None # Generated scientific summary

    # URLs and identifiers
    url: str | None = None
    doi: str | None = None
    accession_id: str | None = None

    # Scientific entities by type (all with provenance)
    species: list[ScientificEntity] = Field(default_factory=list)
    modalities: list[ScientificEntity] = Field(default_factory=list)
    brain_regions: list[ScientificEntity] = Field(default_factory=list)
    tasks: list[ScientificEntity] = Field(default_factory=list)
    behavioral_events: list[ScientificEntity] = Field(default_factory=list)
    cell_types: list[ScientificEntity] = Field(default_factory=list)
    recording_technologies: list[ScientificEntity] = Field(default_factory=list)
    analysis_methods: list[ScientificEntity] = Field(default_factory=list)
    file_formats: list[ScientificEntity] = Field(default_factory=list)
    software_tools: list[ScientificEntity] = Field(default_factory=list)

    # Analysis affordances
    analysis_affordances: list[AnalysisAffordanceV2] = Field(default_factory=list)

    # Usability signals (dataset-specific)
    has_trials: bool | None = None
    has_behavior: bool | None = None
    has_neural_data: bool | None = None
    has_continuous_behavior: bool | None = None
    has_event_timestamps: bool | None = None
    has_raw_data: bool | None = None
    has_processed_data: bool | None = None
    trial_count: int | None = None
    subject_count: int | None = None
    session_count: int | None = None

    # Paper-specific
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    journal: str | None = None
    citations: int | None = None

    # Linked records
    linked_papers: list[str] = Field(default_factory=list)
    linked_datasets: list[str] = Field(default_factory=list)

    # Metadata layers - preserve raw and track transformations
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    raw_metadata_path: str | None = None

    # Status tracking
    embedding_status: EmbeddingStatus = Field(default_factory=EmbeddingStatus)
    graph_status: GraphStatus = Field(default_factory=GraphStatus)
    quality: QualityScore = Field(default_factory=QualityScore)

    # Timestamps
    source_created_at: str | None = None
    source_updated_at: str | None = None
    ingested_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated_at: str | None = None

    @field_validator("record_id", "source", "source_id", "title")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    def get_entities_by_type(self, entity_type: EntityType) -> list[ScientificEntity]:
        """Get all entities of a given type."""
        type_mapping = {
            EntityType.SPECIES: self.species,
            EntityType.MODALITY: self.modalities,
            EntityType.BRAIN_REGION: self.brain_regions,
            EntityType.BEHAVIORAL_TASK: self.tasks,
            EntityType.CELL_TYPE: self.cell_types,
            EntityType.RECORDING_TECHNOLOGY: self.recording_technologies,
            EntityType.ANALYSIS_METHOD: self.analysis_methods,
            EntityType.FILE_FORMAT: self.file_formats,
            EntityType.SOFTWARE_TOOL: self.software_tools,
        }
        return type_mapping.get(entity_type, [])

    def get_all_entities(self) -> list[ScientificEntity]:
        """Get all scientific entities across all types."""
        return [
            *self.species,
            *self.modalities,
            *self.brain_regions,
            *self.tasks,
            *self.behavioral_events,
            *self.cell_types,
            *self.recording_technologies,
            *self.analysis_methods,
            *self.file_formats,
            *self.software_tools,
        ]

    def get_entities_by_layer(self, layer: MetadataLayer) -> list[ScientificEntity]:
        """Get all entities from a specific metadata layer."""
        return [e for e in self.get_all_entities() if e.layer == layer]

    def get_high_confidence_entities(self, threshold: float = 0.8) -> list[ScientificEntity]:
        """Get entities with confidence above threshold."""
        return [e for e in self.get_all_entities() if e.confidence >= threshold]

    def get_text_for_embedding(self, fields: list[str] | None = None) -> str:
        """Combine text fields for embedding generation."""
        if fields is None:
            fields = ["title", "description", "abstract", "methods_summary", "scientific_summary"]

        parts = []
        for field in fields:
            value = getattr(self, field, None)
            if value:
                parts.append(str(value))

        return " ".join(parts)

    def compute_quality_score(self) -> QualityScore:
        """Compute a quality/completeness score for this record."""
        critical_fields = ["title", "description", "species", "modalities"]
        optional_fields = ["brain_regions", "tasks", "doi", "url"]

        missing_critical = []
        missing_optional = []

        for field in critical_fields:
            value = getattr(self, field, None)
            if not value or (isinstance(value, list) and len(value) == 0):
                missing_critical.append(field)

        for field in optional_fields:
            value = getattr(self, field, None)
            if not value or (isinstance(value, list) and len(value) == 0):
                missing_optional.append(field)

        # Compute scores
        completeness = 1.0 - (len(missing_critical) * 0.15 + len(missing_optional) * 0.05)
        completeness = max(0.0, min(1.0, completeness))

        # Provenance strength based on linked papers and verified entities
        provenance = 0.0
        if self.linked_papers:
            provenance += 0.3
        if self.doi:
            provenance += 0.2
        verified_entities = [e for e in self.get_all_entities() if e.verified]
        if verified_entities:
            provenance += 0.3 * min(len(verified_entities) / 5, 1.0)
        provenance = min(1.0, provenance)

        # Average confidence
        all_entities = self.get_all_entities()
        avg_confidence = (
            sum(e.confidence for e in all_entities) / len(all_entities)
            if all_entities else 0.5
        )

        overall = (completeness * 0.4 + provenance * 0.3 + avg_confidence * 0.3)

        return QualityScore(
            overall_score=round(overall, 3),
            metadata_completeness=round(completeness, 3),
            provenance_strength=round(provenance, 3),
            label_confidence_avg=round(avg_confidence, 3),
            missing_critical_fields=missing_critical,
            missing_optional_fields=missing_optional,
            qa_status=self.quality.qa_status,
            reviewer_notes=self.quality.reviewer_notes,
        )


def make_scientific_record_id(record_type: ScientificRecordType, source: str, source_id: str) -> str:
    """Create a stable record ID."""
    type_prefix = record_type.value
    source_clean = source.lower().replace("-", "_").replace(" ", "_")
    source_id_clean = source_id.replace(" ", "_")
    return f"{type_prefix}:{source_clean}:{source_id_clean}"


def scientific_entity_from_label(
    label: str,
    entity_type: EntityType,
    layer: MetadataLayer = MetadataLayer.NORMALIZED,
    confidence: float = 1.0,
    provenance: ExtractionProvenance | None = None,
) -> ScientificEntity:
    """Create a ScientificEntity from a simple label string."""
    label_clean = label.strip().lower().replace(" ", "_")
    entity_id = f"{entity_type.value}:{label_clean}"

    return ScientificEntity(
        id=entity_id,
        label=label,
        entity_type=entity_type,
        layer=layer,
        confidence=confidence,
        provenance=provenance,
    )
