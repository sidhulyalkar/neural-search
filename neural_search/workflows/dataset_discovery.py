"""Dataset discovery workflow built on the canonical search path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.search import search_datasets
from neural_search.workflows.schemas import (
    DatasetDiscoveryResult,
    DatasetDiscoveryWorkflowResponse,
)


def run_dataset_discovery(
    query: str,
    *,
    filters: Mapping[str, Any] | None = None,
    structured_query: Mapping[str, Any] | None = None,
    datasets: Sequence[Mapping[str, Any]] | None = None,
    limit: int = 10,
    retrieval_config: Mapping[str, Any] | None = None,
) -> DatasetDiscoveryWorkflowResponse:
    """Run dataset discovery and return an agent-facing, auditable response."""

    records = list(datasets) if datasets is not None else build_demo_seed()
    response = search_datasets(
        query=query,
        filters=filters,
        structured_query=structured_query,
        datasets=records,
        limit=limit,
        retrieval_config=retrieval_config,
    )
    lookup = _record_lookup(records)
    results = [
        _workflow_result(result, lookup.get(str(result.dataset_id), {}))
        for result in response.results
    ]
    filtered_constraints = response.parsed_query.get("filtered_negative_constraints", [])
    return DatasetDiscoveryWorkflowResponse(
        query=response.query,
        parsed_query=response.parsed_query,
        total_count=len(results),
        filtered_constraints=filtered_constraints
        if isinstance(filtered_constraints, list)
        else [],
        results=results,
    )


def _record_lookup(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}
    for record in records:
        dataset = record.get("dataset", record)
        if not isinstance(dataset, Mapping):
            continue
        for key in ("id", "source_id"):
            value = dataset.get(key)
            if value is not None:
                lookup[str(value)] = record
    return lookup


def _workflow_result(result: Any, record: Mapping[str, Any]) -> DatasetDiscoveryResult:
    dataset = record.get("dataset", record)
    if not isinstance(dataset, Mapping):
        dataset = {}
    preview = result.dataset_card_preview or {}
    graph_context = preview.get("graph_context", {})
    missing_metadata = list(
        dict.fromkeys(
            [
                *result.missing_requirements,
                *result.missing_metadata_warnings,
            ]
        )
    )
    return DatasetDiscoveryResult(
        dataset_id=str(result.dataset_id),
        title=_optional_string(dataset.get("title")),
        source=_optional_string(dataset.get("source")),
        source_id=_optional_string(dataset.get("source_id")),
        score=result.score,
        score_breakdown=result.score_breakdown,
        why_matched=result.why_matched,
        warnings=result.warnings,
        matched_terms=result.matched_terms,
        inferred_concepts=result.inferred_concepts,
        missing_metadata=missing_metadata,
        linked_papers=list(record.get("papers", [])),
        graph_context=graph_context if isinstance(graph_context, dict) else {},
        evidence_snippets=result.evidence_snippets,
        reusable_reason=result.reusable_reason,
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
