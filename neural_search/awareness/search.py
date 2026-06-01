"""Awareness-aware retrieval bridge built on the existing search API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from neural_search.awareness.scoring import score_dataset_awareness
from neural_search.awareness.taxonomy import infer_query_awareness
from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.schemas import SearchResponse, SearchResult
from neural_search.search.core import search_datasets


def _get_value(obj: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def _record_dataset(record: Any) -> Any:
    if isinstance(record, Mapping) and "dataset" in record:
        return record["dataset"]
    return record


def _dataset_lookup(records: Sequence[Any]) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    for record in records:
        dataset = _record_dataset(record)
        for key in ("dataset_id", "id", "source_id"):
            value = _get_value(dataset, key, None)
            if value is not None:
                lookup[str(value)] = dataset
    return lookup


def _append_unique(values: list[str], candidate: str) -> None:
    if candidate and candidate not in values:
        values.append(candidate)


def _annotate_result(result: SearchResult, dataset: Any, query: str) -> float:
    query_awareness = infer_query_awareness(query)
    awareness_score = score_dataset_awareness(dataset, query_awareness)
    score_payload = awareness_score.model_dump()

    result.score_breakdown["awareness_score"] = round(awareness_score.score, 3)
    result.dataset_card_preview["data_form_awareness"] = score_payload

    if awareness_score.matched_data_forms:
        _append_unique(
            result.why_matched,
            "Data forms matched: " + ", ".join(awareness_score.matched_data_forms),
        )
    if awareness_score.matched_analysis_families:
        _append_unique(
            result.why_matched,
            "Analysis families matched: "
            + ", ".join(awareness_score.matched_analysis_families),
        )
    if awareness_score.cross_modal_opportunities:
        _append_unique(
            result.why_matched,
            "Cross-modal opportunities: "
            + ", ".join(awareness_score.cross_modal_opportunities[:4]),
        )
    for warning in awareness_score.warnings:
        _append_unique(result.warnings, f"Awareness warning: {warning}")
    return awareness_score.score


def search_datasets_with_awareness(
    query: str,
    filters: Mapping[str, Any] | None = None,
    structured_query: Mapping[str, Any] | None = None,
    datasets: Sequence[Mapping[str, Any]] | None = None,
    limit: int = 10,
    retrieval_config: Mapping[str, Any] | None = None,
    *,
    awareness_weight: float = 0.12,
    rerank: bool = False,
) -> SearchResponse:
    """Search datasets and annotate results with data-form awareness.

    The function preserves the public search response schema. Awareness is added
    under existing extension points: ``parsed_query``, ``score_breakdown``,
    ``dataset_card_preview``, ``why_matched``, and ``warnings``.
    """

    query_awareness = infer_query_awareness(query)
    search_limit = max(limit, limit * 2) if rerank else limit
    response = search_datasets(
        query=query,
        filters=filters,
        structured_query=structured_query,
        datasets=datasets,
        limit=search_limit,
        retrieval_config=retrieval_config,
    )
    records = list(datasets) if datasets is not None else build_demo_seed()
    by_id = _dataset_lookup(records)
    response.parsed_query["query_awareness"] = query_awareness.model_dump()

    bounded_weight = max(0.0, min(float(awareness_weight), 1.0))
    for result in response.results:
        dataset = by_id.get(str(result.dataset_id))
        if dataset is None:
            result.score_breakdown["awareness_score"] = 0.0
            continue
        awareness_score = _annotate_result(result, dataset, response.query)
        if rerank:
            base_score = float(result.score_breakdown.get("final_score", result.score / 100.0))
            final_score = (
                (1.0 - bounded_weight) * base_score
                + bounded_weight * awareness_score
            )
            final_score = max(0.0, min(final_score, 1.0))
            result.score = round(final_score * 100, 2)
            result.score_breakdown["final_score"] = round(final_score, 3)
            result.score_breakdown["awareness_rerank_weight"] = round(bounded_weight, 3)

    if rerank:
        response.results.sort(key=lambda item: item.score, reverse=True)
    response.results = response.results[:limit]
    return response
