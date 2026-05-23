"""Structured experiment query helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neural_search.schemas import ExperimentQuery

STRUCTURED_FILTER_FIELD_MAP = {
    "task": "tasks",
    "behavior": "behaviors",
    "modality": "modalities",
    "species": "species",
    "brain_region": "brain_regions",
    "data_standard": "data_standards",
    "source_archive": "source",
}


def normalize_structured_query(value: ExperimentQuery | Mapping[str, Any] | None) -> ExperimentQuery:
    """Convert API/dict structured query input into a validated model."""

    if value is None:
        return ExperimentQuery()
    if isinstance(value, ExperimentQuery):
        return value
    return ExperimentQuery.model_validate(value)


def structured_query_to_filters(
    value: ExperimentQuery | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Convert experiment-builder fields into retrieval filters."""

    query = normalize_structured_query(value)
    filters: dict[str, Any] = {}
    for source_field, filter_field in STRUCTURED_FILTER_FIELD_MAP.items():
        selected = _clean_list(getattr(query, source_field))
        if selected:
            filters[filter_field] = selected
    if query.min_analysis_readiness_score is not None:
        filters["min_analysis_readiness_score"] = query.min_analysis_readiness_score
    if query.reviewed_trusted_only:
        filters["qa_status"] = ["reviewed", "trusted"]
    return filters


def structured_query_to_text(value: ExperimentQuery | Mapping[str, Any] | None) -> str:
    """Render structured fields as a natural-language query fragment."""

    query = normalize_structured_query(value)
    parts: list[str] = []
    _append(parts, "task", query.task)
    _append(parts, "behavior", query.behavior)
    _append(parts, "modality", query.modality)
    _append(parts, "species", query.species)
    _append(parts, "brain region", query.brain_region)
    _append(parts, "data standard", query.data_standard)
    _append(parts, "source archive", query.source_archive)
    _append(parts, "analysis goal", query.analysis_goal)
    if query.min_analysis_readiness_score is not None:
        parts.append(f"minimum analysis readiness {query.min_analysis_readiness_score}")
    if query.reviewed_trusted_only:
        parts.append("reviewed or trusted dataset card")
    return "; ".join(parts)


def combine_query_and_structured_text(
    query_text: str,
    value: ExperimentQuery | Mapping[str, Any] | None,
) -> str:
    """Combine free text with builder-derived natural-language terms."""

    structured_text = structured_query_to_text(value)
    if query_text and structured_text:
        return f"{query_text.strip()}; {structured_text}"
    return query_text.strip() or structured_text


def merge_filters(
    filters: Mapping[str, Any] | None,
    structured_query: ExperimentQuery | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge explicit API filters with structured-query filters."""

    merged = dict(filters or {})
    for key, value in structured_query_to_filters(structured_query).items():
        if key in merged and isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = sorted({str(item) for item in merged[key]} & {str(item) for item in value})
        else:
            merged[key] = value
    return merged


def _append(parts: list[str], label: str, values: list[str]) -> None:
    cleaned = _clean_list(values)
    if cleaned:
        parts.append(f"{label}: {', '.join(cleaned)}")


def _clean_list(values: list[str]) -> list[str]:
    return [str(value).strip() for value in values if str(value).strip()]
