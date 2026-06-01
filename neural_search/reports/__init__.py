"""Report generation utilities for Neural Search."""

from __future__ import annotations

__all__ = [
    "build_scientific_readiness_report",
    "compile_dataset_report",
    "render_scientific_readiness_markdown",
    "generate_markdown_report",
    "generate_json_report",
    "write_scientific_readiness_reports",
    "main",
]


def __getattr__(name: str):
    readiness_exports = {
        "build_scientific_readiness_report",
        "render_scientific_readiness_markdown",
        "write_scientific_readiness_reports",
    }
    if name in readiness_exports:
        from neural_search.reports import scientific_readiness

        return getattr(scientific_readiness, name)
    if name in __all__:
        from neural_search.reports import dataset_compilation

        return getattr(dataset_compilation, name)
    raise AttributeError(name)
