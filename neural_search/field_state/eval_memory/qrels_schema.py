"""Schemas for evaluation memory and qrels adjudication."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

Score = Literal[0, 1, 2, 3]
ReviewStatus = Literal["unreviewed", "reviewed", "needs_adjudication", "adjudicated", "rejected"]
AdjudicationStatus = Literal[
    "single_review",
    "agreement",
    "adjudicated",
    "needs_adjudication",
    "rejected",
]


def stable_qrels_candidate_id(query_id: str, dataset_id: str) -> str:
    """Return a stable qrels candidate ID."""
    raw = f"{query_id}:{dataset_id}"
    if len(raw) <= 90 and "/" not in raw and " " not in raw:
        return f"qrels_candidate:{raw}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"qrels_candidate:{digest}"


class QrelsCandidate(BaseModel):
    """Candidate query-dataset pair for human qrels review."""

    id: str
    query_id: str
    query_text: str
    query_intent: str | None = None
    dataset_id: str
    dataset_title: str
    dataset_source: str | None = None
    dataset_description: str | None = None
    rank: int | None = Field(default=None, ge=0)
    retrieval_score: float | None = None
    retrieval_method: str | None = None
    hard_negative_reason: str | None = None
    expected_relevance_hint: str | None = None
    field: str = "neuroscience_dataset_reuse"
    source_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "0.3"

    @field_validator("id", "query_id", "query_text", "dataset_id", "dataset_title")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @model_validator(mode="after")
    def ensure_stable_id(self) -> Self:
        if not self.id:
            self.id = stable_qrels_candidate_id(self.query_id, self.dataset_id)
        return self

    def to_jsonl(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> QrelsCandidate:
        return cls.model_validate_json(line)


class QrelsReview(BaseModel):
    """Human review overlay for one qrels candidate."""

    candidate_id: str
    query_id: str
    dataset_id: str
    annotator_id: str | None = None
    relevance_score: Score | None = None
    usefulness_score: Score | None = None
    hard_negative_violation: bool | None = None
    label_confidence: str | None = None
    review_status: ReviewStatus = "unreviewed"
    rationale: str | None = None
    reviewer_notes: str | None = None
    reviewed_at: str | None = None
    source_note_path: str | None = None
    adjudicated_relevance_score: Score | None = None
    adjudicated_usefulness_score: Score | None = None
    adjudicated_hard_negative_violation: bool | None = None
    adjudicator_notes: str | None = None
    schema_version: str = "0.3"

    @field_validator("candidate_id", "query_id", "dataset_id")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    def to_jsonl(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> QrelsReview:
        return cls.model_validate_json(line)


class AdjudicatedQrel(BaseModel):
    """Final qrels record derived from human review/adjudication."""

    candidate_id: str
    query_id: str
    dataset_id: str
    final_relevance_score: Score
    final_usefulness_score: Score | None = None
    final_hard_negative_violation: bool = False
    adjudication_status: AdjudicationStatus
    annotator_scores: list[dict[str, Any]] = Field(default_factory=list)
    disagreement: bool = False
    adjudicator_notes: str | None = None
    source_review_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = "0.3"

    @field_validator("candidate_id", "query_id", "dataset_id")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    def to_jsonl(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> AdjudicatedQrel:
        return cls.model_validate_json(line)
