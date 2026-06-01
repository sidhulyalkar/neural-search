"""Search intelligence planning and coverage analysis helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CoverageGap",
    "SearchCoveragePlan",
    "SearchIntelligencePlan",
    "EvaluationQuery",
    "PromotionGateReport",
    "CalibrationReport",
    "CorpusKnowledgeExpansionPlan",
    "ExpansionTask",
    "apply_search_intelligence_config",
    "build_search_coverage_plan",
    "build_benchmark_query_seeds",
    "build_corpus_knowledge_expansion_plan",
    "build_review_queue",
    "evaluate_promotion_gates",
    "evaluate_query_plan",
    "calibrate_scores_against_judgments",
    "load_search_records_from_normalized",
    "load_relevance_judgments",
    "plan_search_intelligence",
    "run_query_plan_evaluation",
    "search_datasets_with_intelligence",
    "summarize_human_labels_by_intent",
    "summarize_relevance_judgments",
    "write_promotion_gate_report",
    "write_calibration_report",
    "write_corpus_knowledge_expansion_plan",
    "write_query_plan_evaluation_report",
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
        "CorpusKnowledgeExpansionPlan",
        "ExpansionTask",
        "build_corpus_knowledge_expansion_plan",
        "write_corpus_knowledge_expansion_plan",
    }:
        from neural_search.intelligence import expansion

        return getattr(expansion, name)
    if name in {
        "apply_search_intelligence_config",
        "search_datasets_with_intelligence",
    }:
        from neural_search.intelligence import integration

        return getattr(integration, name)
    if name in {
        "EvaluationQuery",
        "evaluate_query_plan",
        "load_search_records_from_normalized",
        "run_query_plan_evaluation",
        "write_query_plan_evaluation_report",
    }:
        from neural_search.intelligence import evaluation

        return getattr(evaluation, name)
    if name in {
        "CalibrationReport",
        "calibrate_scores_against_judgments",
        "write_calibration_report",
    }:
        from neural_search.intelligence import calibration

        return getattr(calibration, name)
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
    if name in {
        "PromotionGateReport",
        "evaluate_promotion_gates",
        "summarize_human_labels_by_intent",
        "write_promotion_gate_report",
    }:
        from neural_search.intelligence import promotion

        return getattr(promotion, name)
    raise AttributeError(name)
