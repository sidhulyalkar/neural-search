"""Search intelligence planning and coverage analysis helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CoverageGap",
    "SearchCoveragePlan",
    "SearchIntelligencePlan",
    "apply_search_intelligence_config",
    "build_search_coverage_plan",
    "build_benchmark_query_seeds",
    "build_review_queue",
    "load_relevance_judgments",
    "plan_search_intelligence",
    "search_datasets_with_intelligence",
    "summarize_relevance_judgments",
    "write_search_coverage_plan",
    "write_review_queue",
]


def __getattr__(name: str) -> Any:
    if name in {
        "CoverageGap",
        "SearchCoveragePlan",
        "build_benchmark_query_seeds",
        "build_search_coverage_plan",
        "write_search_coverage_plan",
    }:
        from neural_search.intelligence import coverage

        return getattr(coverage, name)
    if name in {
        "apply_search_intelligence_config",
        "search_datasets_with_intelligence",
    }:
        from neural_search.intelligence import integration

        return getattr(integration, name)
    if name in {"SearchIntelligencePlan", "plan_search_intelligence"}:
        from neural_search.intelligence import planner

        return getattr(planner, name)
    if name in {
        "build_review_queue",
        "load_relevance_judgments",
        "summarize_relevance_judgments",
        "write_review_queue",
    }:
        from neural_search.intelligence import review

        return getattr(review, name)
    raise AttributeError(name)
