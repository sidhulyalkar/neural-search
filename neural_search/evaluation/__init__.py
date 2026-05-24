"""Evaluation module for benchmark queries and metrics."""

from importlib import import_module
from typing import Any

from neural_search.evaluation.benchmark import (
    BenchmarkQuery,
    BenchmarkResult,
    EvaluationSummary,
    load_benchmark_queries,
    run_benchmark,
)

_DETAILED_EXPORTS = {
    "EvaluationReport",
    "QueryEvaluation",
    "evaluate_query",
    "generate_json_report",
    "generate_markdown_report",
    "run_full_benchmark",
    "write_reports",
}

_ABLATION_EXPORTS = {
    "AblationReport",
    "AblationResult",
    "run_ablation",
    "generate_ablation_markdown",
    "generate_ablation_json",
    "write_ablation_reports",
    "ABLATION_MODES",
}


def __getattr__(name: str) -> Any:
    if name in _DETAILED_EXPORTS:
        module = import_module("neural_search.evaluation.run_benchmark")
        return getattr(module, name)
    if name in _ABLATION_EXPORTS:
        module = import_module("neural_search.evaluation.run_ablation")
        return getattr(module, name)
    raise AttributeError(f"module 'neural_search.evaluation' has no attribute {name!r}")

__all__ = [
    # Legacy benchmark API
    "BenchmarkQuery",
    "BenchmarkResult",
    "EvaluationSummary",
    "load_benchmark_queries",
    "run_benchmark",
    # New detailed benchmark API
    "EvaluationReport",
    "QueryEvaluation",
    "evaluate_query",
    "generate_json_report",
    "generate_markdown_report",
    "run_full_benchmark",
    "write_reports",
    # Ablation API
    "ABLATION_MODES",
    "AblationReport",
    "AblationResult",
    "generate_ablation_json",
    "generate_ablation_markdown",
    "run_ablation",
    "write_ablation_reports",
]
