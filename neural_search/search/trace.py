"""Search trace capture for debugging retrieval behavior."""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from neural_search.search.core import parse_query, search_datasets


class SearchTraceResult(BaseModel):
    dataset_id: str
    rank: int
    score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    why_matched: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    negative_constraint_matches: list[str] = Field(default_factory=list)


class SearchTrace(BaseModel):
    query: str
    generated_at: str
    parsed_query: dict[str, Any] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)
    filtered_constraints: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_config_summary: dict[str, Any] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    results: list[SearchTraceResult] = Field(default_factory=list)


def capture_search_trace(
    query: str,
    *,
    filters: dict[str, Any] | None = None,
    structured_query: dict[str, Any] | None = None,
    datasets: list[dict[str, Any]] | None = None,
    limit: int = 10,
    retrieval_config: dict[str, Any] | None = None,
) -> SearchTrace:
    """Run a search and return a serializable trace object."""

    parse_start = perf_counter()
    parsed = parse_query(query, retrieval_config)
    parse_ms = (perf_counter() - parse_start) * 1000

    search_start = perf_counter()
    response = search_datasets(
        query=query,
        filters=filters or {},
        structured_query=structured_query,
        datasets=datasets,
        limit=limit,
        retrieval_config=retrieval_config,
    )
    search_ms = (perf_counter() - search_start) * 1000

    graph_config = (retrieval_config or {}).get("graph", {})
    field_config = (retrieval_config or {}).get("field_embeddings", {})
    return SearchTrace(
        query=response.query,
        generated_at=datetime.now(UTC).isoformat(),
        parsed_query=response.parsed_query or parsed,
        filters=filters or {},
        filtered_constraints=response.parsed_query.get("filtered_negative_constraints", []),
        retrieval_config_summary={
            "graph_enabled": bool(graph_config.get("enabled")) if isinstance(graph_config, dict) else False,
            "field_embeddings_enabled": bool(field_config.get("enabled"))
            if isinstance(field_config, dict)
            else False,
        },
        timings_ms={
            "parse": round(parse_ms, 3),
            "search": round(search_ms, 3),
        },
        results=[
            SearchTraceResult(
                dataset_id=str(result.dataset_id),
                rank=index,
                score=result.score,
                score_breakdown=result.score_breakdown,
                why_matched=result.why_matched,
                warnings=result.warnings,
                negative_constraint_matches=result.negative_constraint_matches,
            )
            for index, result in enumerate(response.results, 1)
        ],
    )
