"""Report generation utilities for Neural Search."""

from __future__ import annotations

__all__ = [
    "compile_dataset_report",
    "generate_markdown_report",
    "generate_json_report",
    "main",
]


def __getattr__(name: str):
    if name in __all__:
        from neural_search.reports import dataset_compilation

        return getattr(dataset_compilation, name)
    raise AttributeError(name)
