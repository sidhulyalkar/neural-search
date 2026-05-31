"""Search trace capture for debugging retrieval behavior."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
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
        generated_at=datetime.now(timezone.utc).isoformat(),
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


def write_search_trace(trace: SearchTrace, path: str | Path) -> Path:
    """Write a search trace as deterministic JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(trace.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a Neural Search trace as JSON.")
    parser.add_argument("query", help="Search query to trace")
    parser.add_argument("--out", required=True, help="Output trace JSON path")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--retrieval-config-json",
        help="Optional JSON object merged into retrieval configuration.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    retrieval_config = (
        json.loads(args.retrieval_config_json)
        if args.retrieval_config_json
        else None
    )
    if retrieval_config is not None and not isinstance(retrieval_config, dict):
        raise ValueError("--retrieval-config-json must decode to an object")
    trace = capture_search_trace(
        args.query,
        limit=args.limit,
        retrieval_config=retrieval_config,
    )
    output = write_search_trace(trace, args.out)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
