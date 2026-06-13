"""Pydantic schemas for the Graph-Indexed Concept Memory module."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from neural_search.field_state.concept_memory.artifact_utils import artifact_timestamp

VALID_CONCEPT_TYPES = frozenset({
    "method",
    "dataset",
    "paper",
    "claim",
    "benchmark_gap",
    "opportunity",
    "task",
    "modality",
    "brain_region",
    "species",
    "tool",
    "model",
    "metric",
    "failure_mode",
    "experimental_protocol",
    "analysis_affordance",
    "open_problem",
    "neuroscience_concept",
})

VALID_RELATION_TYPES = frozenset({
    "supports",
    "contradicts",
    "mentions",
    "uses_method",
    "uses_dataset",
    "has_modality",
    "has_task",
    "has_brain_region",
    "has_species",
    "evaluated_by",
    "requires_metric",
    "has_failure_mode",
    "enables_analysis",
    "linked_to_opportunity",
    "linked_to_benchmark_gap",
    "derived_from_note",
    "derived_from_artifact",
})

VALID_EVIDENCE_STRENGTHS = frozenset({"none", "weak", "moderate", "strong"})


class ConceptNode(BaseModel):
    concept_id: str
    canonical_name: str
    concept_type: str
    aliases: list[str] = []
    description: str | None = None
    field: str = "neuroscience_dataset_reuse"
    source_ids: list[str] = []
    source_note_paths: list[str] = []
    source_artifacts: list[str] = []
    evidence_count: int = 0
    claim_count: int = 0
    dataset_count: int = 0
    paper_count: int = 0
    review_status: str = "unreviewed"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = []
    metadata: dict[str, Any] = {}

    @field_validator("concept_id", "canonical_name")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @field_validator("concept_type")
    @classmethod
    def valid_concept_type(cls, value: str) -> str:
        if value not in VALID_CONCEPT_TYPES:
            raise ValueError(f"concept_type must be one of {sorted(VALID_CONCEPT_TYPES)}")
        return value

    def to_jsonl(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> ConceptNode:
        return cls.model_validate_json(line)


class EvidenceLink(BaseModel):
    evidence_id: str
    source_concept_id: str
    target_concept_id: str | None = None
    evidence_type: str
    relation_type: str
    evidence_text: str | None = None
    evidence_source_id: str | None = None
    source_artifact: str | None = None
    source_note_path: str | None = None
    source_repository: str | None = None
    source_record_id: str | None = None
    source_field: str | None = None
    extractor_name: str | None = None
    extractor_version: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    review_status: str = "unreviewed"
    created_at: str = Field(default_factory=artifact_timestamp)
    metadata: dict[str, Any] = {}

    @field_validator("evidence_id", "source_concept_id")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @field_validator("relation_type")
    @classmethod
    def valid_relation_type(cls, value: str) -> str:
        if value not in VALID_RELATION_TYPES:
            raise ValueError(f"relation_type must be one of {sorted(VALID_RELATION_TYPES)}")
        return value

    def to_jsonl(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> EvidenceLink:
        return cls.model_validate_json(line)


class ConceptEmbeddingRecord(BaseModel):
    concept_id: str
    text: str
    embedding_model: str = "none"
    embedding: list[float] | None = None
    embedding_path: str | None = None
    source_hash: str = ""
    created_at: str = Field(default_factory=artifact_timestamp)
    metadata: dict[str, Any] = {}

    def to_jsonl(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> ConceptEmbeddingRecord:
        return cls.model_validate_json(line)


class ConceptBasis(BaseModel):
    concept_id: str
    canonical_name: str
    concept_type: str
    summary: str = ""
    supporting_claim_ids: list[str] = []
    supporting_dataset_ids: list[str] = []
    supporting_paper_ids: list[str] = []
    supporting_note_paths: list[str] = []
    related_opportunity_ids: list[str] = []
    related_benchmark_gap_ids: list[str] = []
    evidence_links: list[str] = []
    supporting_count: int = 0
    contradicting_count: int = 0
    neutral_or_metadata_count: int = 0
    missing_count: int = 0
    reviewed_supporting_count: int = 0
    reviewed_contradicting_count: int = 0
    reviewed_neutral_or_metadata_count: int = 0
    contradicting_evidence_links: list[str] = []
    metadata_evidence_links: list[str] = []
    evidence_strength: str = "none"
    uncertainty_notes: list[str] = []
    next_validation_actions: list[str] = []
    metadata: dict[str, Any] = {}

    @field_validator("concept_id", "canonical_name")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @field_validator("evidence_strength")
    @classmethod
    def valid_evidence_strength(cls, value: str) -> str:
        if value not in VALID_EVIDENCE_STRENGTHS:
            raise ValueError(f"evidence_strength must be one of {sorted(VALID_EVIDENCE_STRENGTHS)}")
        return value

    def to_jsonl(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> ConceptBasis:
        return cls.model_validate_json(line)


class ConceptSearchResult(BaseModel):
    concept_id: str
    canonical_name: str
    concept_type: str
    score: float
    lexical_score: float | None = None
    embedding_score: float | None = None
    graph_boost: float | None = None
    graph_boost_raw: float | None = None
    graph_boost_degree_normalized: float | None = None
    missingness_penalty: float = 0.0
    final_score: float | None = None
    matched_terms: list[str] = []
    matched_concepts: list[str] = []
    warnings: list[str] = []
    evidence_count: int = 0
    top_evidence: list[str] = []
    related_claims: list[str] = []
    related_datasets: list[str] = []
    source_note_paths: list[str] = []


class MatchedConceptInfo(BaseModel):
    concept_id: str
    canonical_name: str
    concept_type: str
    match_score: float
    evidence_texts: list[str] = []
    relation_types: list[str] = []


class ScoreBreakdown(BaseModel):
    base_score: float
    concept_boost: float
    evidence_boost: float
    hard_negative_penalty: float
    final_score: float
    concept_boost_scale: float = 0.3
    evidence_boost_scale: float = 0.1
    matched_concept_count: int = 0


class ConceptRerankedResult(BaseModel):
    dataset_id: str
    dataset_title: str
    base_score: float
    concept_boost: float
    evidence_boost: float
    hard_negative_penalty: float
    final_score: float
    matched_concepts: list[MatchedConceptInfo] = []
    explanation_summary: str = ""

    def score_breakdown(self) -> ScoreBreakdown:
        return ScoreBreakdown(
            base_score=self.base_score,
            concept_boost=self.concept_boost,
            evidence_boost=self.evidence_boost,
            hard_negative_penalty=self.hard_negative_penalty,
            final_score=self.final_score,
            matched_concept_count=len(self.matched_concepts),
        )

    def to_jsonl(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> ConceptRerankedResult:
        return cls.model_validate_json(line)


class ConceptExplanation(BaseModel):
    dataset_id: str
    dataset_title: str
    query: str
    matched_concepts: list[MatchedConceptInfo] = []
    score_breakdown: ScoreBreakdown | None = None
    missing_evidence: list[str] = []
    hard_negative_conflicts: list[str] = []
    explanation_markdown: str = ""

    def to_jsonl(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> ConceptExplanation:
        return cls.model_validate_json(line)
