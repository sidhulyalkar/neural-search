"""Versioned agent-facing API contract models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GraphContextV1(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "linked_papers": [{"paper_id": "paper:openalex:W2963345511"}],
                    "analysis_affordances": ["event_aligned_analysis"],
                    "matched_query_context": {"modalities": ["neuropixels"]},
                }
            ]
        }
    )

    linked_papers: list[dict[str, Any]] = Field(default_factory=list)
    analysis_affordances: list[str] = Field(default_factory=list)
    matched_query_context: dict[str, Any] = Field(default_factory=dict)


class LinkedPaperV1(BaseModel):
    paper_id: str
    title: str | None = None
    doi: str | None = None
    url: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SearchResultV1(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "dataset_id": "dataset:dandi:000026",
                    "score": 81.2,
                    "score_breakdown": {"graph_score": 0.4, "field_semantic_score": 0.3},
                    "graph_context": {
                        "analysis_affordances": ["event_aligned_analysis"],
                        "linked_papers": [{"paper_id": "paper:openalex:W2963345511"}],
                    },
                    "filtered_constraints": [],
                    "missing_metadata": ["license"],
                }
            ]
        }
    )

    dataset_id: str
    score: float
    why_matched: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    graph_context: GraphContextV1 | None = None
    linked_papers: list[LinkedPaperV1] = Field(default_factory=list)
    filtered_constraints: list[dict[str, Any]] = Field(default_factory=list)
    missing_metadata: list[str] = Field(default_factory=list)


class SearchResponseV1(BaseModel):
    schema_version: Literal["v1"] = "v1"
    query: str
    parsed_query: dict[str, Any] = Field(default_factory=dict)
    filtered_constraints: list[dict[str, Any]] = Field(default_factory=list)
    results: list[SearchResultV1] = Field(default_factory=list)


class DatasetCardV1(BaseModel):
    schema_version: Literal["v1"] = "v1"
    dataset_id: str
    title: str
    missing_metadata: list[str] = Field(default_factory=list)
    analysis_affordances: list[str] = Field(default_factory=list)
    linked_papers: list[LinkedPaperV1] = Field(default_factory=list)
    graph_context: GraphContextV1 | None = None


class WorkflowOutputV1(BaseModel):
    schema_version: Literal["v1"] = "v1"
    workflow_name: str
    query: str | None = None
    results: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BenchmarkAuditV1(BaseModel):
    schema_version: Literal["v1"] = "v1"
    suite: str
    failed_queries: list[str] = Field(default_factory=list)
    hard_negative_violations: int = 0
    recommendations: list[str] = Field(default_factory=list)
