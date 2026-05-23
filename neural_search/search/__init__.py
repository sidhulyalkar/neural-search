"""Explainable in-memory search skeleton."""

from neural_search.search.core import (
    load_retrieval_config,
    parse_query,
    score_dataset_against_query,
    search_datasets,
)

__all__ = [
    "load_retrieval_config",
    "parse_query",
    "score_dataset_against_query",
    "search_datasets",
]
