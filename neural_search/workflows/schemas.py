"""Stable schemas for agent-facing research workflows."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetDiscoveryResult(BaseModel):
    """One dataset result prepared for agent consumption."""

    dataset_id: str
    title: str | None = None
    source: str | None = None
    source_id: str | None = None
    score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    why_matched: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    inferred_concepts: list[str] = Field(default_factory=list)
    missing_metadata: list[str] = Field(default_factory=list)
    linked_papers: list[dict[str, Any]] = Field(default_factory=list)
    graph_context: dict[str, Any] = Field(default_factory=dict)
    evidence_snippets: list[str] = Field(default_factory=list)
    reusable_reason: str | None = None


class DatasetDiscoveryWorkflowResponse(BaseModel):
    """Agent-facing dataset discovery workflow output."""

    workflow: str = "dataset_discovery"
    query: str
    parsed_query: dict[str, Any] = Field(default_factory=dict)
    total_count: int
    filtered_constraints: list[dict[str, Any]] = Field(default_factory=list)
    results: list[DatasetDiscoveryResult] = Field(default_factory=list)


class BenchmarkAuditIssue(BaseModel):
    """One benchmark query that needs scientific or retrieval attention."""

    query_id: str
    query: str
    failure_types: list[str] = Field(default_factory=list)
    why_failed: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    hard_negative_violations: list[str] = Field(default_factory=list)
    missed_expected_datasets: list[str] = Field(default_factory=list)
    top_false_positives: list[str] = Field(default_factory=list)
    precision_at_5: float = 0.0
    label_recall_at_10: float = 0.0


class BenchmarkAuditWorkflowResponse(BaseModel):
    """Agent-facing benchmark audit summary."""

    workflow: str = "benchmark_audit"
    suite: str
    total_queries: int
    failed_query_count: int
    hard_negative_violation_count: int
    filtered_constraint_count: int
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    issues: list[BenchmarkAuditIssue] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
