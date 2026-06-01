"""Benchmark audit workflow helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from neural_search.workflows.schemas import (
    BenchmarkAuditIssue,
    BenchmarkAuditWorkflowResponse,
)


def load_and_audit_benchmark_report(path: str | Path) -> BenchmarkAuditWorkflowResponse:
    """Load a benchmark JSON report and summarize actionable failures."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return audit_benchmark_report(payload)


def audit_benchmark_report(report: Any) -> BenchmarkAuditWorkflowResponse:
    """Summarize benchmark failures for agent and developer workflows."""

    payload = _as_mapping(report)
    queries = [
        _as_mapping(query)
        for query in payload.get("queries", [])
        if isinstance(_as_mapping(query), Mapping)
    ]
    issues = [
        _issue_from_query(query)
        for query in queries
        if _query_needs_attention(query)
    ]
    hard_negative_violation_count = sum(
        len(query.get("hard_negative_violations", []) or [])
        for query in queries
    )
    filtered_constraint_count = sum(
        len((query.get("parsed_query", {}) or {}).get("filtered_negative_constraints", []) or [])
        for query in queries
    )

    return BenchmarkAuditWorkflowResponse(
        suite=str(payload.get("suite", "unknown")),
        total_queries=int(payload.get("total_queries", len(queries)) or 0),
        failed_query_count=len(issues),
        hard_negative_violation_count=hard_negative_violation_count,
        filtered_constraint_count=filtered_constraint_count,
        aggregate_metrics=_aggregate_metrics(payload),
        issues=issues,
        recommendations=list(payload.get("recommendations", []) or []),
    )


def _as_mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _aggregate_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    metric_names = [
        "mean_precision_at_1",
        "mean_precision_at_3",
        "mean_precision_at_5",
        "mean_precision_at_10",
        "mean_recall_at_5",
        "mean_recall_at_10",
        "mean_label_recall_at_10",
        "mean_mrr",
        "mean_ndcg_at_10",
        "mean_task_match_rate",
        "mean_modality_match_rate",
        "mean_behavior_match_rate",
    ]
    return {
        name: float(payload.get(name, 0.0) or 0.0)
        for name in metric_names
        if name in payload
    }


def _query_needs_attention(query: Mapping[str, Any]) -> bool:
    return bool(
        query.get("why_failed")
        or query.get("hard_negative_violations")
        or query.get("missed_expected_datasets")
        or query.get("top_false_positives")
        or float(query.get("label_recall_at_10", 1.0) or 0.0) < 1.0
    )


def _issue_from_query(query: Mapping[str, Any]) -> BenchmarkAuditIssue:
    return BenchmarkAuditIssue(
        query_id=str(query.get("query_id", "")),
        query=str(query.get("query", "")),
        failure_types=_failure_types(query),
        why_failed=list(query.get("why_failed", []) or []),
        warnings=list(query.get("warnings", []) or []),
        hard_negative_violations=list(query.get("hard_negative_violations", []) or []),
        missed_expected_datasets=list(query.get("missed_expected_datasets", []) or []),
        top_false_positives=list(query.get("top_false_positives", []) or []),
        precision_at_5=float(query.get("precision_at_5", 0.0) or 0.0),
        label_recall_at_10=float(query.get("label_recall_at_10", 0.0) or 0.0),
    )


def _failure_types(query: Mapping[str, Any]) -> list[str]:
    failure_types: list[str] = []
    if query.get("hard_negative_violations"):
        failure_types.append("constraints")
    if query.get("missed_expected_datasets"):
        failure_types.append("dataset_recall")
    if query.get("top_false_positives"):
        failure_types.append("precision")
    missing_label_fields = [
        "missing_expected_tasks",
        "missing_expected_modalities",
        "missing_expected_behaviors",
        "missing_expected_regions",
        "missing_expected_species",
        "missing_expected_analysis",
    ]
    if any(query.get(field) for field in missing_label_fields):
        failure_types.append("label_recall")
    if query.get("why_failed") and not failure_types:
        failure_types.append("benchmark_threshold")
    if float(query.get("label_recall_at_10", 1.0) or 0.0) < 1.0:
        failure_types.append("label_recall")
    return sorted(dict.fromkeys(failure_types))
