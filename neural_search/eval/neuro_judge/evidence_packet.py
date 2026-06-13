"""Pydantic schemas for neuro_judge evidence packets and judgments.

EvidencePacket   — input to the judge: all retrieved evidence for one pair
NeuroJudgment    — output of a single judge invocation
ConsensusResult  — outcome of multi-judge consensus
ConflictRecord   — pair that requires human arbitration
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEURO_JUDGE_WATERMARK = (
    "PRELIMINARY NEURO-JUDGE EVALUATION — RAG-GROUNDED LLM LABELS, NOT PURE HUMAN GOLD\n"
    "These labels are produced by an LLM judge with retrieval-augmented evidence.\n"
    "They have NOT been reviewed by domain experts and must not be reported as final results."
)

PROMPT_VERSION_DEFAULT = "v1"

# Allowed label provenance strings (ordered by trust level)
LabelProvenance = Literal[
    "neuro_judge",
    "neuro_judge_rag",
    "neuro_judge_consensus",
    "expert_audited_consensus",
    "human_gold",
]

VALID_LABEL_PROVENANCES = frozenset(
    {
        "neuro_judge",
        "neuro_judge_rag",
        "neuro_judge_consensus",
        "expert_audited_consensus",
        "human_gold",
    }
)

# Human gold is reserved — the judge pipeline must never emit it.
_RESERVED_PROVENANCES = frozenset({"human_gold"})


# ---------------------------------------------------------------------------
# Evidence packet sub-models
# ---------------------------------------------------------------------------


class LinkedPaper(BaseModel):
    """A paper linked to the dataset."""

    title: str = ""
    abstract: str = ""
    doi: str = ""
    year: int | None = None


class AffordanceMatch(BaseModel):
    """One matched or unmatched analysis affordance."""

    affordance: str
    matched: bool
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    missing_requirements: list[str] = Field(default_factory=list)
    rationale: str = ""


# ---------------------------------------------------------------------------
# EvidencePacket
# ---------------------------------------------------------------------------


class EvidencePacket(BaseModel):
    """All retrieved evidence for one (query, dataset) pair.

    Built by evidence_retriever.build_evidence_packet() before judging.
    """

    # — query side —
    query_id: str
    query_text: str
    query_intent: str = ""
    hard_negatives: list[str] = Field(default_factory=list)
    expected_species: list[str] = Field(default_factory=list)
    expected_modalities: list[str] = Field(default_factory=list)
    expected_brain_regions: list[str] = Field(default_factory=list)
    expected_tasks: list[str] = Field(default_factory=list)
    expected_analysis_affordances: list[str] = Field(default_factory=list)

    # — dataset side —
    dataset_id: str
    title: str = ""
    source_archive: str = ""
    source_url: str = ""
    description: str = ""
    dataset_modalities: list[str] = Field(default_factory=list)
    dataset_species: list[str] = Field(default_factory=list)
    dataset_brain_regions: list[str] = Field(default_factory=list)
    dataset_tasks: list[str] = Field(default_factory=list)
    data_standards: list[str] = Field(default_factory=list)
    license: str = ""

    # — derived evidence —
    linked_papers: list[LinkedPaper] = Field(default_factory=list)
    affordance_matches: list[AffordanceMatch] = Field(default_factory=list)
    concept_explanation_summary: str = ""
    matched_concept_names: list[str] = Field(default_factory=list)
    concept_missing_evidence: list[str] = Field(default_factory=list)
    concept_hard_negative_conflicts: list[str] = Field(default_factory=list)

    # — raw data signals —
    has_raw_data: bool | None = None
    has_processed_data: bool | None = None
    file_format_evidence: list[str] = Field(default_factory=list)

    # — warnings —
    known_failure_warnings: list[str] = Field(default_factory=list)

    # — meta —
    schema_version: str = "1.0"

    # ---------------------------------------------------------------------------

    def packet_hash(self) -> str:
        """SHA-256 of the canonical JSON representation (stable across runs)."""
        canonical = self.model_dump(mode="json", exclude={"schema_version"})
        blob = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# NeuroJudgment
# ---------------------------------------------------------------------------


class NeuroJudgment(BaseModel):
    """Output of one judge invocation for one (query, dataset) pair."""

    query_id: str
    dataset_id: str

    # — core label —
    label: int = Field(ge=0, le=3)
    confidence: float = Field(ge=0.0, le=1.0)

    # — explanation —
    rationale_short: str = ""
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    matched_dimensions: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)

    # — flags —
    hard_negative_detected: bool = False
    evidence_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    required_dimensions_present: list[str] = Field(default_factory=list)
    required_dimensions_missing: list[str] = Field(default_factory=list)
    abstain_recommended: bool = False
    abstain_reason: str | None = None

    # — provenance —
    judge_model: str = ""
    prompt_version: str = PROMPT_VERSION_DEFAULT
    evidence_packet_hash: str = ""
    label_provenance: str = "neuro_judge"

    @field_validator("label_provenance")
    @classmethod
    def _no_human_gold(cls, v: str) -> str:
        if v in _RESERVED_PROVENANCES:
            raise ValueError(
                f"label_provenance '{v}' is reserved for human annotation "
                "and must not be emitted by the judge pipeline."
            )
        if v not in VALID_LABEL_PROVENANCES:
            raise ValueError(f"Unknown label_provenance: {v!r}")
        return v

    @field_validator("label", mode="before")
    @classmethod
    def _coerce_label(cls, v: Any) -> int:
        if v is None:
            raise ValueError("label must be an integer 0–3, not None")
        return int(v)


# ---------------------------------------------------------------------------
# ConsensusResult
# ---------------------------------------------------------------------------


class ConsensusResult(BaseModel):
    """Consensus label produced from multiple judge outputs."""

    query_id: str
    dataset_id: str

    label: int = Field(ge=0, le=3)
    confidence: float = Field(ge=0.0, le=1.0)
    label_provenance: str = "neuro_judge_consensus"
    judge_count: int = 1
    exact_agreement: bool = False
    minor_disagreement: bool = False
    hard_negative_detected: bool = False
    evidence_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    required_dimensions_present: list[str] = Field(default_factory=list)
    required_dimensions_missing: list[str] = Field(default_factory=list)
    abstain_recommended: bool = False
    abstain_reason: str | None = None

    rationale_short: str = ""
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    matched_dimensions: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)

    judge_models: list[str] = Field(default_factory=list)
    prompt_version: str = PROMPT_VERSION_DEFAULT
    evidence_packet_hash: str = ""
    watermark: str = NEURO_JUDGE_WATERMARK

    @field_validator("label_provenance")
    @classmethod
    def _valid_provenance(cls, v: str) -> str:
        if v in _RESERVED_PROVENANCES:
            raise ValueError(f"label_provenance '{v}' is reserved")
        if v not in VALID_LABEL_PROVENANCES:
            raise ValueError(f"Unknown label_provenance: {v!r}")
        return v


# ---------------------------------------------------------------------------
# ConflictRecord
# ---------------------------------------------------------------------------


class ConflictRecord(BaseModel):
    """A pair that could not reach consensus and needs human arbitration."""

    query_id: str
    dataset_id: str

    judgments: list[NeuroJudgment]
    conflict_reason: str  # e.g. "label_diff_gte_2", "hn_detection_differs", etc.
    ndcg_impact: float = 0.0
    priority: float = 0.0  # higher = route to human first

    evidence_packet_hash: str = ""
    watermark: str = NEURO_JUDGE_WATERMARK

    @model_validator(mode="after")
    def _set_priority(self) -> ConflictRecord:
        self.priority = self.ndcg_impact + (1.0 if "hn" in self.conflict_reason else 0.0)
        return self
