"""DatasetCardV1 and CorpusSnapshot schemas.

These schemas implement the canonical, auditable data objects recommended for
provenance-aware scientific dataset retrieval:

1. DatasetCardV1: Canonical dataset card that every retrieval signal can use.
   Embedding search, BM25, graph construction, affordance detection, and
   explanations should share this canonical view.

2. CorpusSnapshot: Versioned corpus metadata for reproducibility.
   Records source counts, adapter versions, and content hashes.

3. ProvenanceEvidence: Evidence backing for knowledge graph edges.

4. AffordanceRequirement: Required/optional features for analysis affordances.

Design principles:
- Every field has clear provenance
- Corpus snapshots are deterministic and hashable
- Evidence-backed relationships (no edge without evidence)
- Compatible with existing NormalizedDatasetRecord for migration
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from neural_search.core.records import (
    AnalysisAffordanceV2,
    ExtractionProvenance,
    MetadataLayer,
    QualityScore,
    ScientificEntity,
)


class DataStandard(StrEnum):
    """Supported data standards."""

    NWB = "nwb"
    BIDS = "bids"
    NIFTI = "nifti"
    DICOM = "dicom"
    HDF5 = "hdf5"
    ZARR = "zarr"
    OMETIFF = "ometiff"
    CSV = "csv"
    TSV = "tsv"
    OTHER = "other"


class DatasetCardV1(BaseModel):
    """Canonical dataset card for retrieval.

    This is the unified representation used by all retrieval components:
    - BM25 lexical search
    - Dense embedding search
    - Knowledge graph construction
    - Affordance detection
    - Result explanation

    Every dataset is converted to this format before indexing.
    """

    # Identity
    dataset_id: str                      # e.g., "dandi:000026"
    source: str                          # "dandi", "openneuro", "allen", etc.
    source_id: str                       # Original ID from source
    source_url: str | None = None        # Direct link to source
    version: str | None = None           # Dataset version if available

    # Core text fields
    title: str
    description: str | None = None
    abstract: str | None = None          # If from associated paper
    methods_summary: str | None = None   # Experimental methods

    # License and access
    license: str | None = None
    access_type: Literal["open", "restricted", "embargoed", "unknown"] = "unknown"

    # Scientific classification (all with provenance via ScientificEntity)
    organism: list[str] = Field(default_factory=list)   # High-level organism
    species: list[str] = Field(default_factory=list)    # Specific species
    strain: list[str] = Field(default_factory=list)     # Strain if applicable
    modality: list[str] = Field(default_factory=list)   # Recording modalities
    brain_region: list[str] = Field(default_factory=list)
    task: list[str] = Field(default_factory=list)       # Behavioral tasks
    stimuli: list[str] = Field(default_factory=list)    # Stimulus types
    behavioral_events: list[str] = Field(default_factory=list)
    cell_types: list[str] = Field(default_factory=list)

    # Data standards and formats
    data_standards: list[str] = Field(default_factory=list)  # NWB, BIDS, etc.
    file_modalities: list[str] = Field(default_factory=list)  # nwb, edf, nifti, etc.

    # Quantitative metadata
    n_subjects: int | None = None
    n_sessions: int | None = None
    n_trials: int | None = None
    n_units: int | None = None           # For electrophysiology
    n_rois: int | None = None            # For imaging
    n_channels: int | None = None        # For EEG/MEG
    duration_hours: float | None = None  # Total recording duration
    file_count: int | None = None
    size_bytes: int | None = None

    # Linked publications
    linked_publications: list[str] = Field(default_factory=list)  # Paper titles
    linked_dois: list[str] = Field(default_factory=list)          # DOIs

    # Analysis affordances
    analysis_affordances: list[str] = Field(default_factory=list)

    # Usability flags
    has_trials: bool | None = None
    has_behavior: bool | None = None
    has_neural_data: bool | None = None
    has_continuous_behavior: bool | None = None
    has_event_timestamps: bool | None = None
    has_raw_data: bool | None = None
    has_processed_data: bool | None = None

    # Quality and provenance
    quality_flags: list[str] = Field(default_factory=list)  # Warnings, issues
    provenance: list[dict[str, Any]] = Field(default_factory=list)  # Extraction chain

    # Generated text card for embedding
    text_card: str = ""

    # Detailed entities with provenance (for advanced use)
    entities: list[ScientificEntity] = Field(default_factory=list)

    # Full affordance details (for validation)
    affordance_details: list[AnalysisAffordanceV2] = Field(default_factory=list)

    # Quality assessment
    quality_score: QualityScore | None = None

    # Timestamps
    source_created_at: str | None = None
    source_updated_at: str | None = None
    card_created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    card_version: str = "v1"

    @field_validator("dataset_id", "source", "source_id", "title")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    def generate_text_card(self) -> str:
        """Generate a text representation for embedding and retrieval.

        This combines all searchable fields into a single text block
        suitable for dense retrieval.
        """
        parts = []

        # Title is always included
        parts.append(f"Title: {self.title}")

        # Description/abstract
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")
        if self.methods_summary:
            parts.append(f"Methods: {self.methods_summary}")

        # Scientific metadata
        if self.species:
            parts.append(f"Species: {', '.join(self.species)}")
        if self.modality:
            parts.append(f"Modality: {', '.join(self.modality)}")
        if self.brain_region:
            parts.append(f"Brain regions: {', '.join(self.brain_region)}")
        if self.task:
            parts.append(f"Tasks: {', '.join(self.task)}")
        if self.stimuli:
            parts.append(f"Stimuli: {', '.join(self.stimuli)}")
        if self.behavioral_events:
            parts.append(f"Behavioral events: {', '.join(self.behavioral_events)}")

        # Data info
        if self.data_standards:
            parts.append(f"Data standards: {', '.join(self.data_standards)}")
        if self.n_subjects:
            parts.append(f"Subjects: {self.n_subjects}")
        if self.n_sessions:
            parts.append(f"Sessions: {self.n_sessions}")

        # Analysis
        if self.analysis_affordances:
            parts.append(f"Analysis affordances: {', '.join(self.analysis_affordances)}")

        # Publications
        if self.linked_publications:
            parts.append(f"Publications: {'; '.join(self.linked_publications[:3])}")

        return "\n".join(parts)

    def update_text_card(self) -> None:
        """Update the text_card field."""
        self.text_card = self.generate_text_card()

    def compute_hash(self) -> str:
        """Compute a deterministic hash of the card content.

        Used for detecting changes in corpus snapshots.
        """
        # Create a canonical representation
        canonical = {
            "dataset_id": self.dataset_id,
            "source": self.source,
            "source_id": self.source_id,
            "title": self.title,
            "description": self.description,
            "species": sorted(self.species),
            "modality": sorted(self.modality),
            "brain_region": sorted(self.brain_region),
            "task": sorted(self.task),
            "data_standards": sorted(self.data_standards),
            "analysis_affordances": sorted(self.analysis_affordances),
            "card_version": self.card_version,
        }
        json_str = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


class SourceSnapshot(BaseModel):
    """Metadata about a data source at ingestion time."""

    source_name: str                     # e.g., "dandi", "openneuro"
    adapter_name: str                    # e.g., "DandiAdapter"
    adapter_version: str                 # e.g., "v0.7.3"
    retrieval_date: str                  # ISO timestamp
    dataset_count: int
    byte_count: int | None = None        # Total size if available
    api_version: str | None = None       # Source API version if known
    notes: str | None = None


class CorpusSnapshot(BaseModel):
    """Versioned corpus metadata for reproducibility.

    Every corpus export includes a snapshot that records:
    - What sources were used
    - When they were accessed
    - How many records from each
    - A hash of the complete corpus

    This enables reproducible benchmarks and change tracking.
    """

    snapshot_id: str                     # e.g., "corpus_20260527_abc123"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Repository info
    repo_commit: str | None = None       # Git commit if available
    branch: str | None = None

    # Source details
    source_counts: dict[str, int] = Field(default_factory=dict)  # source -> count
    sources: list[SourceSnapshot] = Field(default_factory=list)
    total_records: int = 0

    # Content hashes
    records_hash: str = ""               # Hash of all record hashes
    cards_hash: str = ""                 # Hash of all card hashes

    # Adapter versions
    adapters: dict[str, str] = Field(default_factory=dict)  # adapter -> version

    # Configuration
    config_hash: str | None = None       # Hash of config used

    # Notes
    notes: str | None = None

    @staticmethod
    def generate_snapshot_id() -> str:
        """Generate a unique snapshot ID."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.sha256(
            datetime.now(UTC).isoformat().encode()
        ).hexdigest()[:8]
        return f"corpus_{timestamp}_{random_suffix}"

    def compute_records_hash(self, card_hashes: list[str]) -> str:
        """Compute a hash from all individual card hashes."""
        # Sort for determinism
        sorted_hashes = sorted(card_hashes)
        combined = "|".join(sorted_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


class ProvenanceEvidence(BaseModel):
    """Evidence supporting a knowledge graph edge.

    Every edge in the knowledge graph must have evidence.
    No evidence = no edge.
    """

    evidence_type: Literal[
        "structured_metadata",  # Direct from source metadata
        "text_span",            # Extracted from text
        "doi_relation",         # From DOI/citation graph
        "file_schema",          # From NWB/BIDS structure
        "content_signature",    # From data content analysis
        "manual_label",         # Human annotation
        "inferred",             # Derived from other evidence
    ]
    source: str                          # Where the evidence came from
    field_path: str | None = None        # JSON path in source if applicable
    text: str | None = None              # Evidence text if from text
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    extractor: str = "unknown"
    extractor_version: str = "v0.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ProvenanceEdge(BaseModel):
    """An evidence-backed knowledge graph edge.

    Every relationship in the graph must include:
    1. Source and target identifiers
    2. Edge type
    3. Confidence score
    4. Evidence list (at least one)
    5. Review status
    6. Corpus snapshot reference
    """

    edge_id: str                         # Unique edge identifier
    source_id: str                       # e.g., "dataset:dandi:000026"
    target_id: str                       # e.g., "task:reversal_learning"
    edge_type: str                       # e.g., "has_task", "described_by_paper"

    # Scoring
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    weight: float = Field(ge=0.0, le=1.0, default=1.0)

    # Evidence (required - no edge without evidence)
    evidence: list[ProvenanceEvidence] = Field(min_length=1)

    # Review status
    review_status: Literal[
        "unreviewed",
        "machine_validated",
        "human_validated",
        "rejected",
    ] = "unreviewed"

    # Provenance
    corpus_snapshot_id: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str | None = None

    @staticmethod
    def generate_edge_id(source_id: str, target_id: str, edge_type: str) -> str:
        """Generate a deterministic edge ID."""
        combined = f"{source_id}|{edge_type}|{target_id}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def aggregate_confidence(self) -> float:
        """Compute aggregate confidence from evidence."""
        if not self.evidence:
            return 0.0
        # Weighted average with diminishing returns for additional evidence
        confidences = sorted([e.confidence for e in self.evidence], reverse=True)
        total = 0.0
        weight = 1.0
        for conf in confidences:
            total += conf * weight
            weight *= 0.5  # Diminishing returns
        return min(1.0, total)


class AffordanceRequirement(BaseModel):
    """Requirements for an analysis affordance.

    Specifies what features a dataset must have to support an analysis.
    Used for validating affordance predictions against actual data.
    """

    affordance_id: str                   # e.g., "q_learning"
    label: str                           # "Q-learning model fitting"

    # Required features - all must be present
    required_features: list[str] = Field(default_factory=list)
    # e.g., ["trial_table", "ordered_trials", "choice_sequence", "reward_signal"]

    # Optional but helpful features
    optional_features: list[str] = Field(default_factory=list)
    # e.g., ["reaction_time", "stimulus_identity", "block_label"]

    # Features that rule out this affordance
    negative_conditions: list[str] = Field(default_factory=list)
    # e.g., ["only_summary_statistics", "no_trialwise_behavior"]

    # Validation methods
    validation_methods: list[str] = Field(default_factory=list)
    # e.g., ["nwb_trials_column_check", "bids_events_column_check"]

    # Minimum data requirements
    min_trials: int | None = None
    min_subjects: int | None = None
    min_sessions: int | None = None

    # Confidence rules
    confidence_rules: dict[str, str] = Field(default_factory=lambda: {
        "high": "all required features found in structured files",
        "medium": "required features inferred from metadata but not verified",
        "low": "only textual evidence",
    })

    # Metadata
    description: str | None = None
    example_use_cases: list[str] = Field(default_factory=list)
    related_methods: list[str] = Field(default_factory=list)


class AffordanceValidationResult(BaseModel):
    """Result of validating an affordance against a dataset."""

    dataset_id: str
    affordance_id: str
    supported: bool
    support_level: Literal["high", "medium", "low", "unsupported", "unknown"] = "unknown"
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    # Feature analysis
    found_required_features: list[str] = Field(default_factory=list)
    missing_required_features: list[str] = Field(default_factory=list)
    found_optional_features: list[str] = Field(default_factory=list)
    negative_conditions_found: list[str] = Field(default_factory=list)

    # Evidence
    evidence: list[ProvenanceEvidence] = Field(default_factory=list)

    # Validation details
    validation_method: str | None = None
    validation_notes: str | None = None
    validated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


# Factory functions

def create_dataset_card_from_normalized(
    record: Any,  # NormalizedDatasetRecord
    generate_text: bool = True,
) -> DatasetCardV1:
    """Create a DatasetCardV1 from a NormalizedDatasetRecord.

    This provides a migration path from the legacy schema.
    """
    def extract_labels(labels: list) -> list[str]:
        """Extract label strings from EvidenceLabel objects."""
        if not labels:
            return []
        return [
            getattr(label, "label", label) if hasattr(label, "label") else str(label)
            for label in labels
        ]

    def extract_ids(labels: list) -> list[str]:
        """Extract IDs from EvidenceLabel objects."""
        if not labels:
            return []
        return [
            getattr(label, "id", label) if hasattr(label, "id") else str(label)
            for label in labels
        ]

    # Build provenance from evidence labels
    provenance = []
    for field_name in ["species", "modalities", "brain_regions", "tasks"]:
        labels = getattr(record, field_name, [])
        for label in labels:
            if hasattr(label, "evidence_text") and label.evidence_text:
                provenance.append({
                    "field": field_name,
                    "label": getattr(label, "label", str(label)),
                    "evidence": label.evidence_text,
                    "source": getattr(label, "source_field", "unknown"),
                })

    # Get usability flags
    usability = getattr(record, "usability", None)

    card = DatasetCardV1(
        dataset_id=record.dataset_id,
        source=record.source,
        source_id=record.source_id,
        source_url=getattr(record, "url", None),
        title=record.title,
        description=getattr(record, "description", None),
        species=extract_labels(getattr(record, "species", [])),
        modality=extract_labels(getattr(record, "modalities", [])),
        brain_region=extract_labels(getattr(record, "brain_regions", [])),
        task=extract_labels(getattr(record, "tasks", [])),
        behavioral_events=extract_labels(getattr(record, "behavioral_events", [])),
        data_standards=extract_labels(getattr(record, "data_standards", [])),
        file_modalities=extract_labels(getattr(record, "file_formats", [])),
        linked_publications=[
            getattr(p, "title", str(p))
            for p in getattr(record, "linked_papers", [])
            if p
        ],
        linked_dois=[
            getattr(p, "doi", None)
            for p in getattr(record, "linked_papers", [])
            if p and getattr(p, "doi", None)
        ],
        analysis_affordances=[
            a.analysis_id if hasattr(a, "analysis_id") else str(a)
            for a in getattr(record, "analysis_affordances", [])
        ],
        has_trials=usability.has_trials if usability else None,
        has_behavior=usability.has_behavior if usability else None,
        has_neural_data=usability.has_neural_data if usability else None,
        has_continuous_behavior=usability.has_continuous_behavior if usability else None,
        has_event_timestamps=usability.has_event_timestamps if usability else None,
        has_raw_data=usability.has_raw_data if usability else None,
        has_processed_data=usability.has_processed_data if usability else None,
        provenance=provenance,
    )

    if generate_text:
        card.update_text_card()

    return card


def create_corpus_snapshot(
    cards: list[DatasetCardV1],
    sources: list[SourceSnapshot] | None = None,
    repo_commit: str | None = None,
    notes: str | None = None,
) -> CorpusSnapshot:
    """Create a CorpusSnapshot from a list of DatasetCardV1 objects."""
    # Count by source
    source_counts: dict[str, int] = {}
    for card in cards:
        source_counts[card.source] = source_counts.get(card.source, 0) + 1

    # Compute hashes
    card_hashes = [card.compute_hash() for card in cards]

    snapshot = CorpusSnapshot(
        snapshot_id=CorpusSnapshot.generate_snapshot_id(),
        repo_commit=repo_commit,
        source_counts=source_counts,
        sources=sources or [],
        total_records=len(cards),
    )

    snapshot.records_hash = snapshot.compute_records_hash(card_hashes)
    snapshot.notes = notes

    return snapshot
