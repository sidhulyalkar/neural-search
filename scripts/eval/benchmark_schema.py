"""Shared Pydantic models for Neural Search Benchmark v1.

Accepts both v1 (query_text, must_have, hard_negatives) and legacy field names
(query, required_evidence, known_failure_modes) for backward compatibility with
the 15 existing queries in artifacts/benchmark_queries.jsonl.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Canonical intent values + legacy aliases
# ---------------------------------------------------------------------------

VALID_INTENTS = frozenset(
    {
        "EXACT_LOOKUP",
        "REPLICATION",
        "PIPELINE_REUSE",
        "CROSS_DATASET_COMPARISON",
        "META_ANALYSIS",
        "METHOD_TRANSFER",
        "EXPLORATION",
        # legacy aliases used in artifacts/benchmark_queries.jsonl
        "MODEL_VALIDATION",
        "REANALYSIS_FEASIBILITY",
    }
)

CANONICAL_INTENTS = frozenset(
    {
        "EXACT_LOOKUP",
        "REPLICATION",
        "PIPELINE_REUSE",
        "CROSS_DATASET_COMPARISON",
        "META_ANALYSIS",
        "METHOD_TRANSFER",
        "EXPLORATION",
    }
)

_INTENT_ALIASES: dict[str, str] = {
    "MODEL_VALIDATION": "REPLICATION",
    "REANALYSIS_FEASIBILITY": "PIPELINE_REUSE",
}

VALID_SPLITS = frozenset({"development", "held_out_test", "smoke"})


# ---------------------------------------------------------------------------
# BenchmarkQueryV1
# ---------------------------------------------------------------------------


class BenchmarkQueryV1(BaseModel):
    """A single benchmark query record (v1 schema).

    Accepts both v1 field names and legacy aliases.
    """

    query_id: str
    query_text: str
    intent: str
    scientific_goal: str
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    hard_negatives: list[str] = Field(default_factory=list)
    expected_modalities: list[str] = Field(default_factory=list)
    expected_species: list[str] = Field(default_factory=list)
    expected_tasks: list[str] = Field(default_factory=list)
    expected_brain_regions: list[str] = Field(default_factory=list)
    split: str = "development"
    notes: str = ""

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        if v not in VALID_INTENTS:
            raise ValueError(
                f"intent {v!r} not in allowed values: {sorted(VALID_INTENTS)}"
            )
        return v

    @field_validator("split")
    @classmethod
    def validate_split(cls, v: str) -> str:
        if v not in VALID_SPLITS:
            raise ValueError(f"split {v!r} not in {sorted(VALID_SPLITS)}")
        return v

    @model_validator(mode="before")
    @classmethod
    def apply_aliases(cls, data: dict) -> dict:
        """Map legacy field names to v1 canonical names."""
        if isinstance(data, dict):
            # query → query_text
            if "query_text" not in data and "query" in data:
                data["query_text"] = data.pop("query")
            # required_evidence → must_have
            if "must_have" not in data and "required_evidence" in data:
                data["must_have"] = data.pop("required_evidence")
            # known_failure_modes → hard_negatives
            if "hard_negatives" not in data and "known_failure_modes" in data:
                data["hard_negatives"] = data.pop("known_failure_modes")
        return data

    def canonical_intent(self) -> str:
        """Return the canonical intent name, resolving legacy aliases."""
        return _INTENT_ALIASES.get(self.intent, self.intent)


# ---------------------------------------------------------------------------
# QrelsEntryV1
# ---------------------------------------------------------------------------


class QrelsEntryV1(BaseModel):
    """A single relevance judgement record (qrels v1 schema)."""

    query_id: str
    dataset_id: str
    relevance: int = Field(ge=0, le=3)
    label: str = ""
    rationale: str = ""
    hard_negative_violation: bool = False
    missing_metadata: list[str] = Field(default_factory=list)
    annotator_id: str = ""
    timestamp: str = ""
    adjudicated: bool = False
    adjudication_notes: str = ""

    @field_validator("relevance")
    @classmethod
    def validate_relevance(cls, v: int) -> int:
        if v not in (0, 1, 2, 3):
            raise ValueError(f"relevance must be 0-3, got {v}")
        return v

    def requires_rationale(self) -> bool:
        """Returns True when spec requires a non-empty rationale."""
        return self.relevance in (0, 3)


# ---------------------------------------------------------------------------
# ValidationResult (returned by validators, not persisted)
# ---------------------------------------------------------------------------


class ValidationError(BaseModel):
    field: str
    message: str
    record_id: str = ""


class ValidationResult(BaseModel):
    ok: bool
    errors: list[ValidationError] = Field(default_factory=list)
    warnings: list[ValidationError] = Field(default_factory=list)
    record_count: int = 0

    def add_error(self, field: str, message: str, record_id: str = "") -> None:
        self.errors.append(
            ValidationError(field=field, message=message, record_id=record_id)
        )
        self.ok = False

    def add_warning(self, field: str, message: str, record_id: str = "") -> None:
        self.warnings.append(
            ValidationError(field=field, message=message, record_id=record_id)
        )
