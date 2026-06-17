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
    "ComponentImpact",
    "run_ablation",
    "run_component_ablation",
    "compute_component_impact",
    "generate_ablation_markdown",
    "generate_ablation_json",
    "write_ablation_reports",
    "ABLATION_MODES",
    "COMPONENT_ABLATION_MODES",
}

_CALIBRATION_EXPORTS = {
    "CalibrationConfig",
    "CalibrationCurve",
    "CalibrationResult",
    "ReliabilityBin",
    "calibrate_from_labels",
    "compute_brier_score",
    "compute_calibration_metrics",
    "compute_ece",
    "compute_reliability_bins",
    "explain_calibration",
}

_LADDER_EXPORTS = {
    "BaselineLadderReport",
    "LadderResult",
    "LADDER_MODES",
    "run_baseline_ladder",
    "generate_ladder_markdown",
    "generate_ladder_json",
    "write_ladder_reports",
}

_LINKAGE_EXPORTS = {
    "AnnotatorLabel",
    "DatasetPair",
    "LinkageBenchmark",
    "LinkageMetrics",
    "LinkageType",
    "RelatednessScore",
    "create_sample_benchmark",
    "evaluate_linkage",
    "load_benchmark",
    "save_benchmark",
}

_RELATEDNESS_EXPORTS = {
    "RelatednessScorer",
    "RelatednessScore",
    "RelatednessDimension",
    "DimensionScore",
    "score_dataset_pair",
    "compute_corpus_relatedness_matrix",
    "auto_label_linkage_benchmark",
}


def __getattr__(name: str) -> Any:
    if name in _DETAILED_EXPORTS:
        module = import_module("neural_search.evaluation.run_benchmark")
        return getattr(module, name)
    if name in _ABLATION_EXPORTS:
        module = import_module("neural_search.evaluation.run_ablation")
        return getattr(module, name)
    if name in _CALIBRATION_EXPORTS:
        module = import_module("neural_search.evaluation.calibration")
        return getattr(module, name)
    if name in _LADDER_EXPORTS:
        module = import_module("neural_search.evaluation.run_baseline_ladder")
        return getattr(module, name)
    if name in _LINKAGE_EXPORTS:
        module = import_module("neural_search.evaluation.dataset_linkage")
        return getattr(module, name)
    if name in _RELATEDNESS_EXPORTS:
        module = import_module("neural_search.evaluation.relatedness_scorer")
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
    "COMPONENT_ABLATION_MODES",
    "AblationReport",
    "AblationResult",
    "ComponentImpact",
    "compute_component_impact",
    "generate_ablation_json",
    "generate_ablation_markdown",
    "run_ablation",
    "run_component_ablation",
    "write_ablation_reports",
    # Calibration API
    "CalibrationConfig",
    "CalibrationCurve",
    "CalibrationResult",
    "ReliabilityBin",
    "calibrate_from_labels",
    "compute_brier_score",
    "compute_calibration_metrics",
    "compute_ece",
    "compute_reliability_bins",
    "explain_calibration",
    # Baseline Ladder API
    "BaselineLadderReport",
    "LadderResult",
    "LADDER_MODES",
    "run_baseline_ladder",
    "generate_ladder_markdown",
    "generate_ladder_json",
    "write_ladder_reports",
    # Dataset Linkage API
    "AnnotatorLabel",
    "DatasetPair",
    "LinkageBenchmark",
    "LinkageMetrics",
    "LinkageType",
    "RelatednessScore",
    "create_sample_benchmark",
    "evaluate_linkage",
    "load_benchmark",
    "save_benchmark",
    # Relatedness Scorer API
    "RelatednessScorer",
    "RelatednessDimension",
    "DimensionScore",
    "score_dataset_pair",
    "compute_corpus_relatedness_matrix",
    "auto_label_linkage_benchmark",
]
