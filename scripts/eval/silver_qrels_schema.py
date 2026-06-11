"""Pydantic schemas for silver (machine-generated) qrels — Neural Search Benchmark v0.7.

Silver qrels are machine-generated weak labels produced by rule-based labeling
functions, affordance probes, concept-memory signals, and optional LLM judges.

IMPORTANT TERMINOLOGY:
  silver_qrels   — machine-generated labels (this module)
  gold_qrels     — human-reviewed / adjudicated labels (artifacts/qrels.jsonl)
  review_queue   — examples selected for human review
  consensus_qrels — labels accepted by agreement across multiple weak labelers
  disagreement_qrels — labels requiring human adjudication

Silver labels MUST NOT be reported as expert-validated results in the whitepaper.
Use the SILVER_EVAL_WATERMARK constant in every report header.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_TYPE_SILVER = "silver"
LABEL_TYPE_GOLD = "gold"
VALID_LABEL_TYPES = frozenset({LABEL_TYPE_SILVER, LABEL_TYPE_GOLD})

VALID_LABELER_SOURCES = frozenset(
    {
        "rules",
        "metadata",
        "concept_memory",
        "affordance_probe",
        "llm_judge",
    }
)

# Every silver-label evaluation report MUST begin with this watermark.
SILVER_EVAL_WATERMARK = (
    "SILVER LABEL DIAGNOSTIC — NOT EXPERT VALIDATION\n"
    "These labels are machine-generated and have not been reviewed by human annotators.\n"
    "Do not report these metrics as final scientific results or include them in the whitepaper."
)


# ---------------------------------------------------------------------------
# Core silver models
# ---------------------------------------------------------------------------


class LabelingFunctionVote(BaseModel):
    """A single labeling-function vote on one (query, dataset) pair."""

    source: str  # e.g. "rules", "concept_memory", "affordance_probe"
    vote: int | None = None  # 0-3 or None (abstain)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)

    @field_validator("vote", mode="before")
    @classmethod
    def validate_vote(cls, v: int | None) -> int | None:
        if v is not None and v not in (0, 1, 2, 3):
            raise ValueError(f"vote must be 0–3 or None (abstain), got {v!r}")
        return v


class SilverLabelEvidence(BaseModel):
    """One piece of evidence supporting a silver relevance label."""

    field: str   # corpus field that contributed, e.g. "modalities"
    value: str   # raw value, e.g. "calcium_imaging"
    signal: str  # signal name, e.g. "modality_match"


class SilverQrelsEntry(BaseModel):
    """A single machine-generated (silver) relevance judgement.

    Designed to be stored alongside—but never confused with—QrelsEntryV1.
    The ``label_type`` field is always "silver" and is validated accordingly.
    """

    query_id: str
    dataset_id: str
    label_type: str = Field(default=LABEL_TYPE_SILVER)
    relevance: int = Field(ge=0, le=3)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    label_sources: list[str] = Field(default_factory=list)
    votes: dict[str, int | None] = Field(default_factory=dict)
    hard_negative_violation: bool = False
    missing_metadata: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    needs_human_review: bool = False
    review_priority: float = Field(default=0.0, ge=0.0, le=1.0)
    all_votes: list[LabelingFunctionVote] = Field(default_factory=list)
    seed: int | None = None
    schema_version: str = "0.7"

    @field_validator("label_type")
    @classmethod
    def must_be_silver(cls, v: str) -> str:
        if v != LABEL_TYPE_SILVER:
            raise ValueError(
                f"SilverQrelsEntry.label_type must be {LABEL_TYPE_SILVER!r}, got {v!r}"
            )
        return v

    @field_validator("relevance")
    @classmethod
    def validate_relevance(cls, v: int) -> int:
        if v not in (0, 1, 2, 3):
            raise ValueError(f"relevance must be 0–3, got {v}")
        return v


class SilverQrelsSummary(BaseModel):
    """Aggregate summary statistics for a generated silver qrels file."""

    total_labels: int = 0
    by_relevance: dict[str, int] = Field(default_factory=dict)
    confidence_mean: float = 0.0
    confidence_below_0_5: int = 0
    needs_human_review_count: int = 0
    hard_negative_violation_count: int = 0
    abstention_by_labeler: dict[str, int] = Field(default_factory=dict)
    disagreement_rate: float = 0.0
    per_intent_coverage: dict[str, int] = Field(default_factory=dict)
    per_source_coverage: dict[str, int] = Field(default_factory=dict)
    per_modality_coverage: dict[str, int] = Field(default_factory=dict)
    high_confidence_positives: list[dict[str, Any]] = Field(default_factory=list)
    high_confidence_negatives: list[dict[str, Any]] = Field(default_factory=list)
    examples_needing_review: list[dict[str, Any]] = Field(default_factory=list)
    watermark: str = Field(default=SILVER_EVAL_WATERMARK)
    seed: int | None = None


class ReviewQueueEntry(BaseModel):
    """A single entry in the human-review queue with full annotation context."""

    query_id: str
    dataset_id: str
    query_text: str = ""
    query_intent: str = ""
    dataset_title: str = ""
    dataset_source: str = ""
    silver_relevance: int = Field(ge=0, le=3)
    silver_confidence: float = Field(ge=0.0, le=1.0)
    disagreement_summary: str = ""
    why_selected: str = ""
    annotation_priority: float = Field(default=0.0, ge=0.0, le=1.0)
    fields_needed: list[str] = Field(default_factory=list)
    hard_negative_violation: bool = False
    missing_metadata: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    all_votes: list[LabelingFunctionVote] = Field(default_factory=list)
    schema_version: str = "0.7"
