"""Benchmark evaluation for search quality."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from neural_search.search import search_datasets


DEFAULT_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "eval" / "benchmark_queries.yaml"
)


@dataclass
class BenchmarkQuery:
    """A benchmark query with expected results."""

    id: str
    query: str
    expected_tasks: list[str] = field(default_factory=list)
    expected_behaviors: list[str] = field(default_factory=list)
    expected_modalities_any: list[str] = field(default_factory=list)
    expected_analysis_any: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark query."""

    query_id: str
    query: str
    returned_ids: list[str]
    expected_ids: list[str]
    precision_at_k: float
    recall: float
    matched_tasks: list[str]
    matched_behaviors: list[str]
    warnings: list[str] = field(default_factory=list)


@dataclass
class EvaluationSummary:
    """Summary of a complete benchmark run."""

    results: list[BenchmarkResult]
    mean_precision: float
    mean_recall: float
    total_queries: int
    queries_with_results: int


def load_benchmark_queries(
    path: str | Path = DEFAULT_BENCHMARK_PATH,
) -> list[BenchmarkQuery]:
    """
    Load benchmark queries from YAML file.

    Args:
        path: Path to benchmark queries YAML.

    Returns:
        List of BenchmarkQuery objects.
    """
    path = Path(path)
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    queries = []
    for item in data.get("benchmark_queries", []):
        queries.append(
            BenchmarkQuery(
                id=item.get("id", ""),
                query=item.get("query", ""),
                expected_tasks=item.get("expected_tasks", []),
                expected_behaviors=item.get("expected_behaviors", []),
                expected_modalities_any=item.get("expected_modalities_any", []),
                expected_analysis_any=item.get("expected_analysis_any", []),
                notes=item.get("notes"),
            )
        )

    return queries


def run_benchmark(
    benchmark_queries: list[BenchmarkQuery] | None = None,
    datasets: list[dict[str, Any]] | None = None,
    k: int = 5,
) -> EvaluationSummary:
    """
    Run benchmark evaluation.

    Args:
        benchmark_queries: Queries to evaluate (loads default if None).
        datasets: Datasets to search (uses demo seed if None).
        k: Number of results to consider for precision@k.

    Returns:
        EvaluationSummary with all results.
    """
    if benchmark_queries is None:
        benchmark_queries = load_benchmark_queries()

    results: list[BenchmarkResult] = []

    for query in benchmark_queries:
        # Run search
        response = search_datasets(
            query=query.query,
            filters={},
            datasets=datasets,
            limit=k,
        )

        # Extract returned dataset IDs
        returned_ids = [r.dataset_id for r in response.results]

        # For now, use expected tasks/behaviors as proxy for expected IDs
        # In a real system, you'd have human-annotated relevance judgments
        expected_ids = query.expected_tasks + query.expected_behaviors

        # Calculate metrics
        matched_tasks = []
        matched_behaviors = []
        for result in response.results:
            # Check if result matches expected criteria
            for reason in result.why_matched:
                if "Task matched" in reason:
                    task = reason.replace("Task matched: ", "")
                    if task in query.expected_tasks:
                        matched_tasks.append(task)
                if "Behavior matched" in reason:
                    behavior = reason.replace("Behavior matched: ", "")
                    if behavior in query.expected_behaviors:
                        matched_behaviors.append(behavior)

        # Precision: how many returned results are relevant
        relevant_returned = len(set(matched_tasks + matched_behaviors))
        precision = relevant_returned / k if k > 0 else 0.0

        # Recall: how many expected items were found
        expected_count = len(query.expected_tasks) + len(query.expected_behaviors)
        recall = relevant_returned / expected_count if expected_count > 0 else 0.0

        warnings = []
        if not response.results:
            warnings.append("No results returned")
        if not matched_tasks and query.expected_tasks:
            warnings.append(f"Expected tasks not found: {query.expected_tasks}")
        if not matched_behaviors and query.expected_behaviors:
            warnings.append(f"Expected behaviors not found: {query.expected_behaviors}")

        results.append(
            BenchmarkResult(
                query_id=query.id,
                query=query.query,
                returned_ids=returned_ids,
                expected_ids=expected_ids,
                precision_at_k=precision,
                recall=recall,
                matched_tasks=matched_tasks,
                matched_behaviors=matched_behaviors,
                warnings=warnings,
            )
        )

    # Compute summary
    precisions = [r.precision_at_k for r in results]
    recalls = [r.recall for r in results]
    queries_with_results = sum(1 for r in results if r.returned_ids)

    return EvaluationSummary(
        results=results,
        mean_precision=sum(precisions) / len(precisions) if precisions else 0.0,
        mean_recall=sum(recalls) / len(recalls) if recalls else 0.0,
        total_queries=len(results),
        queries_with_results=queries_with_results,
    )
