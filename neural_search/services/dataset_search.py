"""Application service for experiment-aware dataset retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from neural_search.search import search_datasets
from neural_search.services.corpus_access import CorpusAccessService


class DatasetSearchService:
    """Transport-independent orchestration around the core retrieval engine."""

    def __init__(self, *, corpus_service: CorpusAccessService | None = None) -> None:
        self.corpus_service = corpus_service or CorpusAccessService()

    def search_with_context(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        structured_query: Mapping[str, Any] | None = None,
        datasets: Sequence[Mapping[str, Any]] | None = None,
        limit: int = 10,
        retrieval_config: Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Run search and return explicit scientific-runtime context."""

        if not query.strip() and not structured_query:
            raise ValueError(
                "Enter a free-text experiment description or provide a structured query."
            )
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")

        if datasets is None:
            active_datasets, corpus_source = self.corpus_service.load()
        else:
            active_datasets = datasets
            corpus_source = "caller_supplied"

        response = search_datasets(
            query=query,
            filters=filters or {},
            structured_query=structured_query,
            datasets=active_datasets,
            limit=limit,
            retrieval_config=retrieval_config,
        )
        return response, {
            "corpus_source": corpus_source,
            "execution_profile": self.corpus_service.profile,
            "dataset_count": len(active_datasets),
            "demo_fallback": corpus_source == "demo_fallback",
        }

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
        response, _ = self.search_with_context(
            query,
            filters=filters,
            structured_query=structured_query,
            datasets=datasets,
            limit=limit,
            retrieval_config=retrieval_config,
        )
        return response
