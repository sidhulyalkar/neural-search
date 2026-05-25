"""Search intelligence planning and coverage analysis helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CoverageGap",
    "SearchCoveragePlan",
    "SearchIntelligencePlan",
    "build_search_coverage_plan",
    "build_benchmark_query_seeds",
    "plan_search_intelligence",
    "write_search_coverage_plan",
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
    if name in {"SearchIntelligencePlan", "plan_search_intelligence"}:
        from neural_search.intelligence import planner

        return getattr(planner, name)
    raise AttributeError(name)
