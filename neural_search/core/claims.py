"""Reusability claims: provenance-backed atomic assertions.

This module defines the ReusabilityClaim schema, which is the atomic unit
for all evidence-backed assertions in Neural Search. Every label, relationship,
and affordance assessment should be backed by one or more claims.

A claim answers: "Why do we believe X about dataset Y?"

Key principles:
- Every claim has a source (archive metadata, paper, file inspection, etc.)
- Every claim has confidence (how sure are we?)
- Every claim has provenance (what extractor/human made this claim?)
- Claims can be reviewed and accepted/rejected
- Graph edges and search explanations cite claim IDs

Example claim:
    ReusabilityClaim(
        claim_id="claim:dandi:000026:has_task:delay_discounting:a1b2c3",
        subject_id="dandi:000026",
        predicate="has_task",
        object_id="task:delay_discounting",
        object_label="Delay discounting",
        source_type=EvidenceSourceType.PAPER_METHODS,
        source_ref="doi:10.1234/example",
        evidence_text="subjects chose between immediate and delayed rewards",
        extractor="rule.delay_discounting.v2",
        confidence=0.91,
        review_status=ReviewStatus.UNREVIEWED,
    )
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class EvidenceSourceType(StrEnum):
    """Source type for a claim's evidence.

    Ordered roughly by confidence/reliability.
    """

    FILE_INSPECTION = "file_inspection"      # Direct file/NWB inspection (highest)
    ARCHIVE_METADATA = "archive_metadata"    # Official archive metadata (DANDI, OpenNeuro)
    PAPER_METHODS = "paper_methods"          # Methods section of linked paper
    PAPER_ABSTRACT = "paper_abstract"        # Abstract of linked paper
    README = "readme"                        # Dataset README or documentation
    CODE = "code"                            # Code/scripts in dataset
    HUMAN_REVIEW = "human_review"            # Expert human annotation
    INFERRED_ONTOLOGY = "inferred_ontology"  # Inferred via ontology synonyms
    INFERRED_EMBEDDING = "inferred_embedding"  # Inferred via embedding similarity
    BROAD_TAXONOMY = "broad_taxonomy"        # Broad taxonomic inference (lowest)


# Default confidence by source type
SOURCE_CONFIDENCE_DEFAULTS: dict[EvidenceSourceType, float] = {
    EvidenceSourceType.FILE_INSPECTION: 0.95,
    EvidenceSourceType.ARCHIVE_METADATA: 0.90,
    EvidenceSourceType.PAPER_METHODS: 0.85,
    EvidenceSourceType.README: 0.75,
    EvidenceSourceType.CODE: 0.75,
    EvidenceSourceType.PAPER_ABSTRACT: 0.65,
    EvidenceSourceType.HUMAN_REVIEW: 0.95,
    EvidenceSourceType.INFERRED_ONTOLOGY: 0.55,
    EvidenceSourceType.INFERRED_EMBEDDING: 0.45,
    EvidenceSourceType.BROAD_TAXONOMY: 0.35,
}


class ReviewStatus(StrEnum):
    """Review status for a claim."""

    UNREVIEWED = "unreviewed"    # Not yet reviewed
    TRUSTED = "trusted"          # Reviewed and accepted
    REJECTED = "rejected"        # Reviewed and rejected
    NEEDS_REVIEW = "needs_review"  # Flagged for expert review


# Standard predicates for dataset claims
class ClaimPredicate(StrEnum):
    """Standard predicates for reusability claims."""

    # Scientific classification
    HAS_TASK = "has_task"
    HAS_MODALITY = "has_modality"
    HAS_BRAIN_REGION = "has_brain_region"
    HAS_SPECIES = "has_species"
    HAS_CELL_TYPE = "has_cell_type"
    HAS_STIMULUS = "has_stimulus"
    HAS_BEHAVIORAL_EVENT = "has_behavioral_event"

    # Data structure
    HAS_TRIAL_STRUCTURE = "has_trial_structure"
    HAS_VARIABLE = "has_variable"
    HAS_DATA_STANDARD = "has_data_standard"

    # Analysis support
    SUPPORTS_AFFORDANCE = "supports_affordance"
    HAS_REQUIRED_FEATURE = "has_required_feature"
    MISSING_REQUIRED_FEATURE = "missing_required_feature"
    HAS_HARD_BLOCKER = "has_hard_blocker"

    # Relationships
    LINKED_TO_PAPER = "linked_to_paper"
    LINKED_TO_DATASET = "linked_to_dataset"
    DERIVED_FROM = "derived_from"

    # Quality
    HAS_QUALITY_FLAG = "has_quality_flag"
    HAS_WARNING = "has_warning"


class ReusabilityClaim(BaseModel):
    """A provenance-backed assertion about a dataset.

    This is the atomic unit for all evidence-backed information in Neural Search.
    Every label, relationship, and affordance assessment should trace back to claims.
    """

    # Identity
    claim_id: str = Field(description="Unique claim identifier")

    # Subject-Predicate-Object triple
    subject_id: str = Field(description="Dataset or entity ID (e.g., 'dandi:000026')")
    predicate: str = Field(description="Relationship type (e.g., 'has_task')")
    object_id: str = Field(description="Object ID (e.g., 'task:delay_discounting')")
    object_label: str | None = Field(
        default=None,
        description="Human-readable label for the object"
    )

    # Evidence source
    source_type: EvidenceSourceType = Field(
        description="Type of evidence source"
    )
    source_ref: str | None = Field(
        default=None,
        description="Reference to source (DOI, file path, etc.)"
    )
    evidence_text: str | None = Field(
        default=None,
        description="Quoted or summarized evidence text"
    )

    # Extraction provenance
    extractor: str | None = Field(
        default=None,
        description="Extractor/rule that generated this claim"
    )
    extractor_version: str | None = Field(
        default=None,
        description="Version of the extractor"
    )

    # Confidence and review
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0)"
    )
    review_status: ReviewStatus = Field(
        default=ReviewStatus.UNREVIEWED,
        description="Review status"
    )
    reviewed_by: str | None = Field(
        default=None,
        description="Reviewer ID if reviewed"
    )
    reviewed_at: str | None = Field(
        default=None,
        description="Review timestamp if reviewed"
    )

    # Timestamps
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str | None = None

    # Optional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    @field_validator("claim_id", "subject_id", "predicate", "object_id")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        """Ensure required string fields are not empty."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> "ReusabilityClaim":
        """Deserialize from JSONL format."""
        return cls.model_validate_json(line)

    def with_review(
        self,
        status: ReviewStatus,
        reviewer: str,
    ) -> "ReusabilityClaim":
        """Return a new claim with updated review status."""
        return self.model_copy(
            update={
                "review_status": status,
                "reviewed_by": reviewer,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )


def make_claim_id(
    subject_id: str,
    predicate: str,
    object_id: str,
    source_ref: str | None = None,
) -> str:
    """Generate a stable claim ID from its components.

    The ID is deterministic for the same inputs, enabling deduplication
    and stable fixture generation.
    """
    parts = [subject_id, predicate, object_id]
    if source_ref:
        parts.append(source_ref)

    key = "|".join(parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]

    # Clean subject_id for readable prefix
    clean_subject = subject_id.replace(":", "_").replace("/", "_")[:20]
    clean_predicate = predicate.replace(":", "_")[:15]

    return f"claim:{clean_subject}:{clean_predicate}:{digest}"


def create_claim(
    subject_id: str,
    predicate: str,
    object_id: str,
    source_type: EvidenceSourceType,
    *,
    object_label: str | None = None,
    source_ref: str | None = None,
    evidence_text: str | None = None,
    extractor: str | None = None,
    confidence: float | None = None,
) -> ReusabilityClaim:
    """Factory function to create a claim with sensible defaults.

    If confidence is not provided, uses the default for the source type.
    """
    if confidence is None:
        confidence = SOURCE_CONFIDENCE_DEFAULTS.get(source_type, 0.5)

    claim_id = make_claim_id(subject_id, predicate, object_id, source_ref)

    return ReusabilityClaim(
        claim_id=claim_id,
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        object_label=object_label,
        source_type=source_type,
        source_ref=source_ref,
        evidence_text=evidence_text,
        extractor=extractor,
        confidence=confidence,
    )


class ClaimStore:
    """In-memory store for claims with JSONL persistence.

    Supports:
    - Adding/removing claims
    - Querying by subject, predicate, or object
    - Loading/saving to JSONL files
    - Claim deduplication by ID
    """

    def __init__(self) -> None:
        self._claims: dict[str, ReusabilityClaim] = {}

    def add(self, claim: ReusabilityClaim) -> None:
        """Add a claim to the store."""
        self._claims[claim.claim_id] = claim

    def add_many(self, claims: list[ReusabilityClaim]) -> None:
        """Add multiple claims."""
        for claim in claims:
            self.add(claim)

    def get(self, claim_id: str) -> ReusabilityClaim | None:
        """Get a claim by ID."""
        return self._claims.get(claim_id)

    def remove(self, claim_id: str) -> bool:
        """Remove a claim by ID. Returns True if removed."""
        if claim_id in self._claims:
            del self._claims[claim_id]
            return True
        return False

    def all_claims(self) -> list[ReusabilityClaim]:
        """Get all claims."""
        return list(self._claims.values())

    def query_by_subject(self, subject_id: str) -> list[ReusabilityClaim]:
        """Get all claims about a subject."""
        return [c for c in self._claims.values() if c.subject_id == subject_id]

    def query_by_predicate(self, predicate: str) -> list[ReusabilityClaim]:
        """Get all claims with a predicate."""
        return [c for c in self._claims.values() if c.predicate == predicate]

    def query_by_object(self, object_id: str) -> list[ReusabilityClaim]:
        """Get all claims with an object."""
        return [c for c in self._claims.values() if c.object_id == object_id]

    def query(
        self,
        subject_id: str | None = None,
        predicate: str | None = None,
        object_id: str | None = None,
        source_type: EvidenceSourceType | None = None,
        min_confidence: float | None = None,
        review_status: ReviewStatus | None = None,
    ) -> list[ReusabilityClaim]:
        """Query claims with optional filters."""
        results = list(self._claims.values())

        if subject_id is not None:
            results = [c for c in results if c.subject_id == subject_id]
        if predicate is not None:
            results = [c for c in results if c.predicate == predicate]
        if object_id is not None:
            results = [c for c in results if c.object_id == object_id]
        if source_type is not None:
            results = [c for c in results if c.source_type == source_type]
        if min_confidence is not None:
            results = [c for c in results if c.confidence >= min_confidence]
        if review_status is not None:
            results = [c for c in results if c.review_status == review_status]

        return results

    def save_jsonl(self, path: str | Path) -> int:
        """Save all claims to JSONL file. Returns count."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for claim in self._claims.values():
                f.write(claim.to_jsonl() + "\n")

        return len(self._claims)

    def load_jsonl(self, path: str | Path) -> int:
        """Load claims from JSONL file. Returns count loaded."""
        path = Path(path)
        if not path.exists():
            return 0

        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    claim = ReusabilityClaim.from_jsonl(line)
                    self.add(claim)
                    count += 1

        return count

    def __len__(self) -> int:
        return len(self._claims)

    def __contains__(self, claim_id: str) -> bool:
        return claim_id in self._claims


# Convenience functions for common claim types


def claim_has_task(
    dataset_id: str,
    task_id: str,
    task_label: str,
    source_type: EvidenceSourceType,
    **kwargs: Any,
) -> ReusabilityClaim:
    """Create a has_task claim."""
    return create_claim(
        subject_id=dataset_id,
        predicate=ClaimPredicate.HAS_TASK,
        object_id=f"task:{task_id}",
        object_label=task_label,
        source_type=source_type,
        **kwargs,
    )


def claim_has_modality(
    dataset_id: str,
    modality_id: str,
    modality_label: str,
    source_type: EvidenceSourceType,
    **kwargs: Any,
) -> ReusabilityClaim:
    """Create a has_modality claim."""
    return create_claim(
        subject_id=dataset_id,
        predicate=ClaimPredicate.HAS_MODALITY,
        object_id=f"modality:{modality_id}",
        object_label=modality_label,
        source_type=source_type,
        **kwargs,
    )


def claim_supports_affordance(
    dataset_id: str,
    affordance_id: str,
    affordance_label: str,
    source_type: EvidenceSourceType,
    **kwargs: Any,
) -> ReusabilityClaim:
    """Create a supports_affordance claim."""
    return create_claim(
        subject_id=dataset_id,
        predicate=ClaimPredicate.SUPPORTS_AFFORDANCE,
        object_id=f"affordance:{affordance_id}",
        object_label=affordance_label,
        source_type=source_type,
        **kwargs,
    )


def claim_has_variable(
    dataset_id: str,
    variable_name: str,
    source_type: EvidenceSourceType,
    **kwargs: Any,
) -> ReusabilityClaim:
    """Create a has_variable claim for dataset structure."""
    return create_claim(
        subject_id=dataset_id,
        predicate=ClaimPredicate.HAS_VARIABLE,
        object_id=f"variable:{variable_name}",
        object_label=variable_name,
        source_type=source_type,
        **kwargs,
    )


def claim_linked_to_paper(
    dataset_id: str,
    paper_doi: str,
    paper_title: str | None = None,
    source_type: EvidenceSourceType = EvidenceSourceType.ARCHIVE_METADATA,
    **kwargs: Any,
) -> ReusabilityClaim:
    """Create a linked_to_paper claim."""
    return create_claim(
        subject_id=dataset_id,
        predicate=ClaimPredicate.LINKED_TO_PAPER,
        object_id=f"paper:{paper_doi}",
        object_label=paper_title,
        source_type=source_type,
        **kwargs,
    )
