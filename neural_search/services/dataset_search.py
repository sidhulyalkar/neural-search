"""Application service for experiment-aware dataset retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from neural_search.search import search_datasets


class DatasetSearchService:
    """Transport-independent orchestration around the core retrieval engine."""

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        structured_query: Mapping[str, Any] | None = None,
        datasets: Sequence[Mapping[str, Any]] | None = None,
        limit: int = 10,
        retrieval_config: Mapping[str, Any] | None = None,
    ) -> Any:
        if not query.strip() and not structured_query:
            raise ValueError(
                "Enter a free-text experiment description or provide a structured query."
            )
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        return search_datasets(
            query=query,
            filters=filters or {},
            structured_query=structured_query,
            datasets=datasets,
            limit=limit,
            retrieval_config=retrieval_config,
        )
