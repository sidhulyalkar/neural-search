"""Neuroscience data-form awareness helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AwarenessScore",
    "DataForm",
    "DatasetAwareness",
    "QueryAwareness",
    "detect_data_forms",
    "infer_query_awareness",
    "score_dataset_awareness",
    "search_datasets_with_awareness",
]


def __getattr__(name: str) -> Any:
    if name in {"DataForm", "QueryAwareness", "detect_data_forms", "infer_query_awareness"}:
        from neural_search.awareness import taxonomy

        return getattr(taxonomy, name)
    if name in {"AwarenessScore", "DatasetAwareness", "score_dataset_awareness"}:
        from neural_search.awareness import scoring

        return getattr(scoring, name)
    if name == "search_datasets_with_awareness":
        from neural_search.awareness import search

        return getattr(search, name)
    raise AttributeError(name)
