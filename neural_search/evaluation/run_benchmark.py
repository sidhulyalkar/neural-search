"""Benchmark evaluation runner with detailed reporting.

Runs benchmark queries against the search system and generates
comprehensive evaluation reports with precision, recall, and match rates.

Usage:
    python -m neural_search.evaluation.run_benchmark
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.normalized import load_normalized_records
from neural_search.ontology import normalize_text
from neural_search.schemas import NormalizedDatasetRecord
from neural_search.search import search_datasets

EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "eval"
BENCHMARK_PATH = EVAL_DIR / "benchmark_queries.yaml"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "eval" / "results"
SUITE_PATHS = {
    "demo_v02": EVAL_DIR / "benchmark_queries_demo_v02.yaml",
    "real_corpus": EVAL_DIR / "benchmark_queries_real_corpus.yaml",
    "real_v07": EVAL_DIR / "benchmark_queries_real_v07.yaml",
    "adversarial": EVAL_DIR / "benchmark_queries_adversarial.yaml",
}
SUITE_CHOICES = (*SUITE_PATHS.keys(), "all")


@dataclass
class BenchmarkQuery:
    """A benchmark query with expected labels."""

    id: str
    query: str
    expected_dataset_ids: list[str] = field(default_factory=list)
    expected_tasks: list[str] = field(default_factory=list)
    expected_behaviors: list[str] = field(default_factory=list)
    expected_modalities_any: list[str] = field(default_factory=list)
    expected_regions_any: list[str] = field(default_factory=list)
    expected_species: list[str] = field(default_factory=list)
    expected_data_standards: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    expected_analysis_any: list[str] = field(default_factory=list)
    hard_negative_dataset_ids: list[str] = field(default_factory=list)
    hard_negative_modalities: list[str] = field(default_factory=list)
    hard_negative_species: list[str] = field(default_factory=list)
    analysis_intent: str | None = None
    minimum_precision_at_5: float = 0.0
    minimum_label_recall_at_10: float = 0.0
    notes: str | None = None


@dataclass
class QueryEvaluation:
    """Evaluation results for a single query."""

    query_id: str
    query: str
    num_results: int
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    label_recall_at_10: float
    mrr: float
    ndcg_at_10: float
    task_match_rate: float
    modality_match_rate: float
    behavior_match_rate: float
    matched_tasks: list[str]
    matched_modalities: list[str]
    matched_behaviors: list[str]
    matched_regions: list[str]
    matched_species: list[str]
    matched_data_standards: list[str]
    matched_sources: list[str]
    matched_analysis: list[str]
    missing_expected_tasks: list[str]
    missing_expected_modalities: list[str]
    missing_expected_behaviors: list[str]
    missing_expected_regions: list[str]
    missing_expected_species: list[str]
    missing_expected_data_standards: list[str]
    missing_expected_sources: list[str]
    missing_expected_analysis: list[str]
    expected_dataset_ids: list[str]
    missed_expected_datasets: list[str]
    hard_negative_violations: list[str]
    top_false_positives: list[str]
    why_failed: list[str]
    top_results: list[dict[str, Any]]
    warnings: list[str]
    parsed_query: dict[str, Any]


@dataclass
class EvaluationReport:
    """Complete evaluation report across all benchmark queries."""

    generated_at: str
    total_queries: int
    queries_with_results: int
    mean_precision_at_1: float
    mean_precision_at_3: float
    mean_precision_at_5: float
    mean_precision_at_10: float
    mean_recall_at_5: float
    mean_recall_at_10: float
    mean_label_recall_at_10: float
    mean_mrr: float
    mean_ndcg_at_10: float
    mean_task_match_rate: float
    mean_modality_match_rate: float
    mean_behavior_match_rate: float
    queries: list[QueryEvaluation]
    summary_warnings: list[str]
    recommendations: list[str]
    suite: str = "demo_v02"


def load_benchmark_queries(path: Path | None = None) -> list[BenchmarkQuery]:
    """Load benchmark queries from YAML file."""
    benchmark_path = path or BENCHMARK_PATH
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    with benchmark_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    queries = []
    for item in data.get("benchmark_queries", []):
        queries.append(
            BenchmarkQuery(
                id=item.get("id", ""),
                query=item.get("query", ""),
                expected_dataset_ids=item.get("expected_dataset_ids", []),
                expected_tasks=item.get("expected_tasks", []),
                expected_behaviors=item.get("expected_behaviors", []),
                expected_modalities_any=item.get("expected_modalities_any", []),
                expected_regions_any=item.get("expected_regions_any", []),
                expected_species=item.get("expected_species", []),
                expected_data_standards=item.get("expected_data_standards", []),
                expected_sources=item.get("expected_sources", []),
                expected_analysis_any=item.get("expected_analysis_any", []),
                hard_negative_dataset_ids=item.get("hard_negative_dataset_ids", []),
                hard_negative_modalities=item.get("hard_negative_modalities", []),
                hard_negative_species=item.get("hard_negative_species", []),
                analysis_intent=item.get("analysis_intent"),
                minimum_precision_at_5=float(item.get("minimum_precision_at_5", 0.0) or 0.0),
                minimum_label_recall_at_10=float(
                    item.get("minimum_label_recall_at_10", 0.0) or 0.0
                ),
                notes=item.get("notes"),
            )
        )
    return queries


def benchmark_path_for_suite(suite: str) -> Path:
    """Resolve a named benchmark suite to a query YAML path."""

    if suite not in SUITE_PATHS:
        raise ValueError(f"Unknown benchmark suite: {suite}")
    return SUITE_PATHS[suite]


def output_dir_for_suite(suite: str) -> Path:
    """Return the suite-specific default result directory."""

    return RESULTS_DIR / suite


def _labels_as_strings(labels: list[Any]) -> list[str]:
    return [str(getattr(label, "label", label)) for label in labels]


def _real_v07_search_records() -> list[dict[str, Any]]:
    records_path = Path(__file__).resolve().parents[2] / "data" / "corpus" / "normalized" / "real_v07.datasets.jsonl"
    if not records_path.exists():
        return []
    normalized = load_normalized_records(records_path)
    datasets = [record for record in normalized if isinstance(record, NormalizedDatasetRecord)]
    search_records: list[dict[str, Any]] = []
    for dataset in datasets:
        analyses = [item.analysis_id for item in dataset.analysis_affordances]
        search_records.append(
            {
                "dataset": {
                    "id": dataset.dataset_id,
                    "source": dataset.source,
                    "source_id": dataset.source_id,
                    "title": dataset.title,
                    "description": dataset.description or "",
                    "url": dataset.url,
                    "species": _labels_as_strings(dataset.species),
                    "modalities": _labels_as_strings(dataset.modalities),
                    "brain_regions": _labels_as_strings(dataset.brain_regions),
                    "tasks": _labels_as_strings(dataset.tasks),
                    "behaviors": _labels_as_strings(dataset.behavioral_events),
                    "data_standards": _labels_as_strings(dataset.data_standards),
                    "analysis_affordances": analyses,
                    "has_trials": bool(dataset.usability_flags.has_trials),
                    "has_behavior": bool(dataset.usability_flags.has_behavior),
                    "has_raw_data": bool(dataset.usability_flags.has_raw_data),
                    "has_processed_data": bool(dataset.usability_flags.has_processed_data),
                    "linked_paper_ids": dataset.linked_papers,
                    "metadata_json": {
                        "missing_fields": dataset.missing_fields,
                        "analysis_affordances": analyses,
                    },
                },
                "card": {
                    "summary": dataset.description or dataset.title,
                    "why_relevant": [
                        f"{dataset.source} fixture record",
                        "File-inspection claims are available",
                    ],
                    "analysis_readiness": {
                        "score": 85 if dataset.usability_flags.has_trials else 65,
                    },
                    "suggested_analyses": analyses,
                    "missing_fields": dataset.missing_fields,
                    "scientific_labels": {
                        "tasks": [
                            {"id": label.label, "label": label.label}
                            for label in dataset.tasks
                        ],
                        "modalities": [
                            {"id": label.label, "label": label.label}
                            for label in dataset.modalities
                        ],
                        "behaviors": [
                            {"id": label.label, "label": label.label}
                            for label in dataset.behavioral_events
                        ],
                        "brain_regions": [
                            {"id": label.label, "label": label.label}
                            for label in dataset.brain_regions
                        ],
                        "species": [
                            {"id": label.label, "label": label.label}
                            for label in dataset.species
                        ],
                    },
                },
            }
        )
    return search_records


def _datasets_for_suite(suite: str) -> list[dict[str, Any]]:
    if suite == "real_v07":
        records = _real_v07_search_records()
        if records:
            return records
    return build_demo_seed()


def _normalize_list(values: list[str]) -> set[str]:
    """Normalize a list of labels for comparison."""
    return {normalize_text(v) for v in values}


def _extract_result_labels(
    results: list[dict[str, Any]],
    datasets: list[dict[str, Any]] | None = None,
) -> dict[str, set[str]]:
    """Extract all labels from search results and their underlying datasets."""
    tasks: set[str] = set()
    modalities: set[str] = set()
    behaviors: set[str] = set()
    regions: set[str] = set()
    species: set[str] = set()
    data_standards: set[str] = set()
    sources: set[str] = set()
    analysis: set[str] = set()

    # Build lookup for dataset metadata
    dataset_lookup: dict[str, dict[str, Any]] = {}
    if datasets:
        for record in datasets:
            ds = record.get("dataset", record)
            ds_id = ds.get("id", ds.get("source_id", ""))
            dataset_lookup[str(ds_id)] = ds
            source_id = ds.get("source_id")
            if source_id:
                dataset_lookup[str(source_id)] = ds

    for result in results:
        why_matched = result.get("why_matched", [])

        # Extract from why_matched reasons
        for reason in why_matched:
            if reason.startswith("Task matched:"):
                tasks.add(normalize_text(reason.replace("Task matched:", "").strip()))
            elif reason.startswith("Modality matched:"):
                modalities.add(normalize_text(reason.replace("Modality matched:", "").strip()))
            elif reason.startswith("Behavior matched:"):
                behaviors.add(normalize_text(reason.replace("Behavior matched:", "").strip()))
            elif reason.startswith("Brain region matched:"):
                regions.add(normalize_text(reason.replace("Brain region matched:", "").strip()))
            elif reason.startswith("Species matched:"):
                species.add(normalize_text(reason.replace("Species matched:", "").strip()))
            elif reason.startswith("Analysis matched:"):
                analysis.add(normalize_text(reason.replace("Analysis matched:", "").strip()))

        # Also extract from underlying dataset metadata
        dataset_id = str(result.get("dataset_id", ""))
        if dataset_id in dataset_lookup:
            ds = dataset_lookup[dataset_id]
            for task in ds.get("tasks", []):
                tasks.add(normalize_text(task))
            for mod in ds.get("modalities", []):
                modalities.add(normalize_text(mod))
            for beh in ds.get("behaviors", []):
                behaviors.add(normalize_text(beh))
            for region in ds.get("brain_regions", []):
                regions.add(normalize_text(region))
            for item in ds.get("species", []):
                species.add(normalize_text(item))
            for standard in ds.get("data_standards", []):
                data_standards.add(normalize_text(standard))
            source = ds.get("source")
            if source:
                sources.add(normalize_text(source))
            preview = result.get("dataset_card_preview", {})
            for item in preview.get("suggested_analyses", []) if isinstance(preview, dict) else []:
                analysis.add(normalize_text(item))

    return {
        "tasks": tasks,
        "modalities": modalities,
        "behaviors": behaviors,
        "regions": regions,
        "species": species,
        "data_standards": data_standards,
        "sources": sources,
        "analysis": analysis,
    }


def _dataset_lookup(datasets: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for record in datasets or []:
        ds = record.get("dataset", record)
        for key in ("id", "source_id"):
            value = ds.get(key)
            if value:
                lookup[str(value)] = ds
    return lookup


def _result_relevance(
    result_id: str,
    dataset: dict[str, Any] | None,
    query: BenchmarkQuery,
) -> float:
    expected_ids = set(query.expected_dataset_ids)
    hard_negative_ids = set(query.hard_negative_dataset_ids)
    if result_id in hard_negative_ids:
        return 0.0
    if expected_ids:
        return 1.0 if result_id in expected_ids else 0.0
    if dataset is None:
        return 0.0

    hard_negative_modalities = _normalize_list(query.hard_negative_modalities)
    hard_negative_species = _normalize_list(query.hard_negative_species)
    if hard_negative_modalities & _normalize_list(dataset.get("modalities", [])):
        return 0.0
    if hard_negative_species & _normalize_list(dataset.get("species", [])):
        return 0.0

    expected_groups = [
        (_normalize_list(query.expected_tasks), _normalize_list(dataset.get("tasks", []))),
        (_normalize_list(query.expected_behaviors), _normalize_list(dataset.get("behaviors", []))),
        (
            _normalize_list(query.expected_modalities_any),
            _normalize_list(dataset.get("modalities", [])),
        ),
        (
            _normalize_list(query.expected_regions_any),
            _normalize_list(dataset.get("brain_regions", [])),
        ),
        (_normalize_list(query.expected_species), _normalize_list(dataset.get("species", []))),
        (
            _normalize_list(query.expected_data_standards),
            _normalize_list(dataset.get("data_standards", [])),
        ),
        (_normalize_list(query.expected_sources), _normalize_list([dataset.get("source", "")])),
    ]
    active_groups = [(expected, actual) for expected, actual in expected_groups if expected]
    if not active_groups:
        return 1.0 if result_id not in hard_negative_ids else 0.0
    matched_groups = sum(1 for expected, actual in active_groups if expected & actual)
    return matched_groups / len(active_groups)


def _precision_at(relevance: list[float], k: int) -> float:
    top = relevance[:k]
    if not top:
        return 0.0
    return sum(1 for value in top if value > 0) / len(top)


def _recall_at_ids(returned_ids: list[str], expected_ids: list[str], k: int) -> float:
    if not expected_ids:
        return 0.0
    return len(set(returned_ids[:k]) & set(expected_ids)) / len(set(expected_ids))


def _mrr(relevance: list[float]) -> float:
    for index, value in enumerate(relevance, 1):
        if value > 0:
            return 1.0 / index
    return 0.0


def _ndcg(relevance: list[float], k: int) -> float:
    def dcg(values: list[float]) -> float:
        import math

        return sum(value / math.log2(index + 2) for index, value in enumerate(values))

    observed = relevance[:k]
    ideal = sorted(relevance, reverse=True)[:k]
    ideal_score = dcg(ideal)
    if ideal_score == 0:
        return 0.0
    return dcg(observed) / ideal_score


def _hard_negative_violations(
    returned_ids: list[str],
    lookup: dict[str, dict[str, Any]],
    query: BenchmarkQuery,
) -> list[str]:
    violations: list[str] = []
    hard_negative_ids = set(query.hard_negative_dataset_ids)
    hard_negative_modalities = _normalize_list(query.hard_negative_modalities)
    hard_negative_species = _normalize_list(query.hard_negative_species)
    for result_id in returned_ids[:10]:
        dataset = lookup.get(result_id, {})
        if result_id in hard_negative_ids:
            violations.append(f"{result_id}: hard-negative dataset returned")
        modalities = _normalize_list(dataset.get("modalities", []))
        species = _normalize_list(dataset.get("species", []))
        if hard_negative_modalities & modalities:
            violations.append(
                f"{result_id}: hard-negative modality "
                f"{sorted(hard_negative_modalities & modalities)}"
            )
        if hard_negative_species & species:
            violations.append(
                f"{result_id}: hard-negative species {sorted(hard_negative_species & species)}"
            )
    return violations


def evaluate_query(
    query: BenchmarkQuery,
    datasets: list[dict[str, Any]] | None = None,
    k: int = 10,
    retrieval_config: dict[str, Any] | None = None,
) -> QueryEvaluation:
    """Evaluate a single benchmark query."""
    response = search_datasets(
        query=query.query,
        filters={},
        datasets=datasets,
        limit=k,
        retrieval_config=retrieval_config,
    )

    results = [
        {
            "dataset_id": r.dataset_id,
            "score": r.score,
            "why_matched": r.why_matched,
            "warnings": r.warnings,
            "dataset_card_preview": r.dataset_card_preview,
            "score_breakdown": r.score_breakdown,
        }
        for r in response.results
    ]
    returned_ids = [str(result["dataset_id"]) for result in results]
    lookup = _dataset_lookup(datasets)

    expected_ids = [str(item) for item in query.expected_dataset_ids]
    expected_tasks = _normalize_list(query.expected_tasks)
    expected_modalities = _normalize_list(query.expected_modalities_any)
    expected_behaviors = _normalize_list(query.expected_behaviors)
    expected_regions = _normalize_list(query.expected_regions_any)
    expected_species = _normalize_list(query.expected_species)
    expected_data_standards = _normalize_list(query.expected_data_standards)
    expected_sources = _normalize_list(query.expected_sources)
    expected_analysis = _normalize_list(query.expected_analysis_any)

    result_labels = _extract_result_labels(results, datasets)
    matched_tasks = sorted(expected_tasks & result_labels["tasks"])
    matched_modalities = sorted(expected_modalities & result_labels["modalities"])
    matched_behaviors = sorted(expected_behaviors & result_labels["behaviors"])
    matched_regions = sorted(expected_regions & result_labels["regions"])
    matched_species = sorted(expected_species & result_labels["species"])
    matched_data_standards = sorted(expected_data_standards & result_labels["data_standards"])
    matched_sources = sorted(expected_sources & result_labels["sources"])
    matched_analysis = sorted(expected_analysis & result_labels["analysis"])

    missing_tasks = sorted(expected_tasks - result_labels["tasks"])
    missing_modalities = sorted(expected_modalities - result_labels["modalities"])
    missing_behaviors = sorted(expected_behaviors - result_labels["behaviors"])
    missing_regions = sorted(expected_regions - result_labels["regions"])
    missing_species = sorted(expected_species - result_labels["species"])
    missing_data_standards = sorted(expected_data_standards - result_labels["data_standards"])
    missing_sources = sorted(expected_sources - result_labels["sources"])
    missing_analysis = sorted(expected_analysis - result_labels["analysis"])

    task_match_rate = len(matched_tasks) / len(expected_tasks) if expected_tasks else 1.0
    modality_match_rate = len(matched_modalities) / len(expected_modalities) if expected_modalities else 1.0
    behavior_match_rate = len(matched_behaviors) / len(expected_behaviors) if expected_behaviors else 1.0

    all_expected = (
        expected_tasks
        | expected_modalities
        | expected_behaviors
        | expected_regions
        | expected_species
        | expected_data_standards
        | expected_sources
        | expected_analysis
    )
    all_matched = (
        set(matched_tasks)
        | set(matched_modalities)
        | set(matched_behaviors)
        | set(matched_regions)
        | set(matched_species)
        | set(matched_data_standards)
        | set(matched_sources)
        | set(matched_analysis)
    )
    label_recall = len(all_matched) / len(all_expected) if all_expected else 1.0

    relevance = [
        _result_relevance(result_id, lookup.get(result_id), query)
        for result_id in returned_ids
    ]
    precision_at_1 = _precision_at(relevance, 1)
    precision_at_3 = _precision_at(relevance, 3)
    precision_at_5 = _precision_at(relevance, 5)
    precision_at_10 = _precision_at(relevance, 10)
    recall_at_5 = _recall_at_ids(returned_ids, expected_ids, 5)
    recall_at_10 = _recall_at_ids(returned_ids, expected_ids, 10)
    hard_negative_violations = _hard_negative_violations(returned_ids, lookup, query)
    missed_expected_datasets = sorted(set(expected_ids) - set(returned_ids[:10]))
    top_false_positives = [
        result_id
        for result_id, value in zip(returned_ids[:10], relevance[:10], strict=False)
        if value == 0
    ][:5]

    warnings = []
    if not results:
        warnings.append("No results returned")
    if missing_tasks:
        warnings.append(f"Expected tasks not found: {missing_tasks}")
    if missing_modalities:
        warnings.append(f"Expected modalities not found: {missing_modalities}")
    if missing_behaviors:
        warnings.append(f"Expected behaviors not found: {missing_behaviors}")
    if missing_regions:
        warnings.append(f"Expected regions not found: {missing_regions}")
    if missing_species:
        warnings.append(f"Expected species not found: {missing_species}")
    if missing_data_standards:
        warnings.append(f"Expected data standards not found: {missing_data_standards}")
    if missing_sources:
        warnings.append(f"Expected sources not found: {missing_sources}")
    if missing_analysis:
        warnings.append(f"Expected analyses not found: {missing_analysis}")
    if missed_expected_datasets:
        warnings.append(f"Expected datasets not returned: {missed_expected_datasets}")
    warnings.extend(hard_negative_violations)

    for r in results:
        warnings.extend(r.get("warnings", []))

    why_failed: list[str] = []
    if precision_at_5 < query.minimum_precision_at_5:
        why_failed.append(
            f"Precision@5 {precision_at_5:.1%} below minimum "
            f"{query.minimum_precision_at_5:.1%}"
        )
    if label_recall < query.minimum_label_recall_at_10:
        why_failed.append(
            f"Label recall@10 {label_recall:.1%} below minimum "
            f"{query.minimum_label_recall_at_10:.1%}"
        )
    if missed_expected_datasets:
        why_failed.append(f"Missed expected datasets: {missed_expected_datasets}")
    if hard_negative_violations:
        why_failed.append(f"Hard-negative violations: {hard_negative_violations}")
    if not results:
        why_failed.append("No results returned")

    return QueryEvaluation(
        query_id=query.id,
        query=query.query,
        num_results=len(results),
        precision_at_1=round(precision_at_1, 3),
        precision_at_3=round(precision_at_3, 3),
        precision_at_5=round(precision_at_5, 3),
        precision_at_10=round(precision_at_10, 3),
        recall_at_5=round(recall_at_5, 3),
        recall_at_10=round(recall_at_10, 3),
        label_recall_at_10=round(label_recall, 3),
        mrr=round(_mrr(relevance), 3),
        ndcg_at_10=round(_ndcg(relevance, 10), 3),
        task_match_rate=round(task_match_rate, 3),
        modality_match_rate=round(modality_match_rate, 3),
        behavior_match_rate=round(behavior_match_rate, 3),
        matched_tasks=matched_tasks,
        matched_modalities=matched_modalities,
        matched_behaviors=matched_behaviors,
        matched_regions=matched_regions,
        matched_species=matched_species,
        matched_data_standards=matched_data_standards,
        matched_sources=matched_sources,
        matched_analysis=matched_analysis,
        missing_expected_tasks=missing_tasks,
        missing_expected_modalities=missing_modalities,
        missing_expected_behaviors=missing_behaviors,
        missing_expected_regions=missing_regions,
        missing_expected_species=missing_species,
        missing_expected_data_standards=missing_data_standards,
        missing_expected_sources=missing_sources,
        missing_expected_analysis=missing_analysis,
        expected_dataset_ids=expected_ids,
        missed_expected_datasets=missed_expected_datasets,
        hard_negative_violations=hard_negative_violations,
        top_false_positives=top_false_positives,
        why_failed=why_failed,
        top_results=results[:5],
        warnings=list(set(warnings)),
        parsed_query=response.parsed_query,
    )


def run_full_benchmark(
    benchmark_path: Path | None = None,
    datasets: list[dict[str, Any]] | None = None,
    suite: str = "demo_v02",
    retrieval_config: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Run complete benchmark evaluation."""
    queries = load_benchmark_queries(benchmark_path)
    if datasets is None:
        datasets = _datasets_for_suite(suite)

    evaluations = [evaluate_query(q, datasets, retrieval_config=retrieval_config) for q in queries]

    queries_with_results = sum(1 for e in evaluations if e.num_results > 0)

    mean_p1 = sum(e.precision_at_1 for e in evaluations) / len(evaluations) if evaluations else 0
    mean_p3 = sum(e.precision_at_3 for e in evaluations) / len(evaluations) if evaluations else 0
    mean_p5 = sum(e.precision_at_5 for e in evaluations) / len(evaluations) if evaluations else 0
    mean_p10 = sum(e.precision_at_10 for e in evaluations) / len(evaluations) if evaluations else 0
    mean_r5 = sum(e.recall_at_5 for e in evaluations) / len(evaluations) if evaluations else 0
    mean_r10 = sum(e.recall_at_10 for e in evaluations) / len(evaluations) if evaluations else 0
    mean_recall = sum(e.label_recall_at_10 for e in evaluations) / len(evaluations) if evaluations else 0
    mean_mrr = sum(e.mrr for e in evaluations) / len(evaluations) if evaluations else 0
    mean_ndcg = sum(e.ndcg_at_10 for e in evaluations) / len(evaluations) if evaluations else 0
    mean_task = sum(e.task_match_rate for e in evaluations) / len(evaluations) if evaluations else 0
    mean_mod = sum(e.modality_match_rate for e in evaluations) / len(evaluations) if evaluations else 0
    mean_beh = sum(e.behavior_match_rate for e in evaluations) / len(evaluations) if evaluations else 0

    all_missing_tasks: Counter[str] = Counter()
    all_missing_modalities: Counter[str] = Counter()
    all_missing_behaviors: Counter[str] = Counter()

    for e in evaluations:
        all_missing_tasks.update(e.missing_expected_tasks)
        all_missing_modalities.update(e.missing_expected_modalities)
        all_missing_behaviors.update(e.missing_expected_behaviors)

    summary_warnings = []
    if mean_task < 0.5:
        summary_warnings.append(f"Low task match rate ({mean_task:.1%})")
    if mean_mod < 0.5:
        summary_warnings.append(f"Low modality match rate ({mean_mod:.1%})")
    if mean_beh < 0.5:
        summary_warnings.append(f"Low behavior match rate ({mean_beh:.1%})")
    if queries_with_results < len(queries):
        summary_warnings.append(
            f"{len(queries) - queries_with_results} queries returned no results"
        )

    recommendations = []
    if all_missing_tasks:
        top_missing = [t for t, _ in all_missing_tasks.most_common(3)]
        recommendations.append(f"Add ontology coverage for tasks: {top_missing}")
    if all_missing_modalities:
        top_missing = [m for m, _ in all_missing_modalities.most_common(3)]
        recommendations.append(f"Add synonym expansion for modalities: {top_missing}")
    if all_missing_behaviors:
        top_missing = [b for b, _ in all_missing_behaviors.most_common(3)]
        recommendations.append(f"Add synonym expansion for behaviors: {top_missing}")
    if mean_p5 < 0.6:
        recommendations.append("Consider adjusting scoring weights for better precision")
    if mean_recall < 0.5:
        recommendations.append("Expand ontology synonyms for better recall")

    return EvaluationReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_queries=len(queries),
        queries_with_results=queries_with_results,
        mean_precision_at_1=round(mean_p1, 3),
        mean_precision_at_3=round(mean_p3, 3),
        mean_precision_at_5=round(mean_p5, 3),
        mean_precision_at_10=round(mean_p10, 3),
        mean_recall_at_5=round(mean_r5, 3),
        mean_recall_at_10=round(mean_r10, 3),
        mean_label_recall_at_10=round(mean_recall, 3),
        mean_mrr=round(mean_mrr, 3),
        mean_ndcg_at_10=round(mean_ndcg, 3),
        mean_task_match_rate=round(mean_task, 3),
        mean_modality_match_rate=round(mean_mod, 3),
        mean_behavior_match_rate=round(mean_beh, 3),
        queries=evaluations,
        summary_warnings=summary_warnings,
        recommendations=recommendations,
        suite=suite,
    )


def generate_markdown_report(report: EvaluationReport) -> str:
    """Generate Markdown evaluation report."""
    lines = [
        "# Neural Search Benchmark Evaluation Report",
        "",
        f"Generated: {report.generated_at}",
        f"Suite: {report.suite}",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Queries | {report.total_queries} |",
        f"| Queries with Results | {report.queries_with_results} |",
        f"| Mean Precision@1 | {report.mean_precision_at_1:.1%} |",
        f"| Mean Precision@3 | {report.mean_precision_at_3:.1%} |",
        f"| **Mean Precision@5** | **{report.mean_precision_at_5:.1%}** |",
        f"| Mean Precision@10 | {report.mean_precision_at_10:.1%} |",
        f"| Mean Recall@5 | {report.mean_recall_at_5:.1%} |",
        f"| Mean Recall@10 | {report.mean_recall_at_10:.1%} |",
        f"| **Mean Label Recall@10** | **{report.mean_label_recall_at_10:.1%}** |",
        f"| Mean MRR | {report.mean_mrr:.3f} |",
        f"| Mean NDCG@10 | {report.mean_ndcg_at_10:.3f} |",
        f"| Task Match Rate | {report.mean_task_match_rate:.1%} |",
        f"| Modality Match Rate | {report.mean_modality_match_rate:.1%} |",
        f"| Behavior Match Rate | {report.mean_behavior_match_rate:.1%} |",
        "",
    ]

    if report.summary_warnings:
        lines.extend([
            "## Warnings",
            "",
        ])
        for warning in report.summary_warnings:
            lines.append(f"- {warning}")
        lines.append("")

    if report.recommendations:
        lines.extend([
            "## Recommendations",
            "",
        ])
        for rec in report.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    lines.extend([
        "## Per-Query Results",
        "",
    ])

    for eval_result in report.queries:
        status = "PASS" if not eval_result.why_failed and eval_result.label_recall_at_10 >= 0.5 else "FAIL"
        lines.extend([
            f"### {eval_result.query_id}: {status}",
            "",
            f"**Query:** {eval_result.query}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Results | {eval_result.num_results} |",
            f"| Precision@1 | {eval_result.precision_at_1:.1%} |",
            f"| Precision@3 | {eval_result.precision_at_3:.1%} |",
            f"| Precision@5 | {eval_result.precision_at_5:.1%} |",
            f"| Precision@10 | {eval_result.precision_at_10:.1%} |",
            f"| Recall@5 | {eval_result.recall_at_5:.1%} |",
            f"| Recall@10 | {eval_result.recall_at_10:.1%} |",
            f"| Label Recall@10 | {eval_result.label_recall_at_10:.1%} |",
            f"| MRR | {eval_result.mrr:.3f} |",
            f"| NDCG@10 | {eval_result.ndcg_at_10:.3f} |",
            f"| Task Match | {eval_result.task_match_rate:.1%} |",
            f"| Modality Match | {eval_result.modality_match_rate:.1%} |",
            f"| Behavior Match | {eval_result.behavior_match_rate:.1%} |",
            "",
        ])

        if eval_result.matched_tasks:
            lines.append(f"**Matched tasks:** {', '.join(eval_result.matched_tasks)}")
        if eval_result.matched_modalities:
            lines.append(f"**Matched modalities:** {', '.join(eval_result.matched_modalities)}")
        if eval_result.matched_behaviors:
            lines.append(f"**Matched behaviors:** {', '.join(eval_result.matched_behaviors)}")
        if eval_result.missing_expected_tasks:
            lines.append(f"**Missing tasks:** {', '.join(eval_result.missing_expected_tasks)}")
        if eval_result.missing_expected_modalities:
            lines.append(f"**Missing modalities:** {', '.join(eval_result.missing_expected_modalities)}")
        if eval_result.missing_expected_behaviors:
            lines.append(f"**Missing behaviors:** {', '.join(eval_result.missing_expected_behaviors)}")
        if eval_result.missed_expected_datasets:
            lines.append(f"**Missed datasets:** {', '.join(eval_result.missed_expected_datasets)}")
        if eval_result.hard_negative_violations:
            lines.append(
                "**Hard-negative violations:** "
                + "; ".join(eval_result.hard_negative_violations)
            )

        lines.append("")

        if eval_result.why_failed:
            lines.append("**Why failed:**")
            for reason in eval_result.why_failed:
                lines.append(f"- {reason}")
            lines.append("")

        if eval_result.warnings:
            lines.append("**Warnings:**")
            for w in eval_result.warnings[:5]:
                lines.append(f"- {w}")
            lines.append("")

        lines.append("**Top Results:**")
        lines.append("")
        for i, r in enumerate(eval_result.top_results[:3], 1):
            lines.append(f"{i}. `{r['dataset_id']}` (score: {r['score']})")
            if r.get("why_matched"):
                for reason in r["why_matched"][:3]:
                    lines.append(f"   - {reason}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def generate_json_report(report: EvaluationReport) -> str:
    """Generate JSON evaluation report."""
    data = asdict(report)
    return json.dumps(data, indent=2, default=str)


def _metric_delta(before: dict[str, Any], after: dict[str, Any], key: str) -> float:
    return float(after.get(key, 0) or 0) - float(before.get(key, 0) or 0)


def generate_comparison_markdown(
    before: dict[str, Any],
    after: EvaluationReport,
) -> str:
    """Generate a concise before/after retrieval comparison report."""

    after_dict = asdict(after)
    metric_keys = [
        ("Mean Precision@1", "mean_precision_at_1"),
        ("Mean Precision@3", "mean_precision_at_3"),
        ("Mean Precision@5", "mean_precision_at_5"),
        ("Mean Precision@10", "mean_precision_at_10"),
        ("Mean Recall@5", "mean_recall_at_5"),
        ("Mean Recall@10", "mean_recall_at_10"),
        ("Mean Label Recall@10", "mean_label_recall_at_10"),
        ("Mean MRR", "mean_mrr"),
        ("Mean NDCG@10", "mean_ndcg_at_10"),
        ("Task Match Rate", "mean_task_match_rate"),
        ("Modality Match Rate", "mean_modality_match_rate"),
        ("Behavior Match Rate", "mean_behavior_match_rate"),
    ]
    lines = [
        "# Retrieval Comparison Report",
        "",
        f"Generated: {after.generated_at}",
        "",
        "## Summary",
        "",
        "| Metric | Before | After | Delta |",
        "|--------|--------|-------|-------|",
    ]
    for label, key in metric_keys:
        before_value = float(before.get(key, 0) or 0)
        after_value = float(after_dict.get(key, 0) or 0)
        delta = after_value - before_value
        lines.append(
            f"| {label} | {before_value:.1%} | {after_value:.1%} | {delta:+.1%} |"
        )

    before_queries = {
        item.get("query_id"): item for item in before.get("queries", [])
    }
    lines.extend(["", "## Per-Query Delta", ""])
    lines.append("| Query | Recall Before | Recall After | P@5 Before | P@5 After |")
    lines.append("|-------|---------------|--------------|------------|-----------|")
    for after_query in after_dict.get("queries", []):
        before_query = before_queries.get(after_query.get("query_id"), {})
        lines.append(
            "| {query_id} | {before_recall:.1%} | {after_recall:.1%} | "
            "{before_precision:.1%} | {after_precision:.1%} |".format(
                query_id=after_query.get("query_id", ""),
                before_recall=float(before_query.get("label_recall_at_10", 0) or 0),
                after_recall=float(after_query.get("label_recall_at_10", 0) or 0),
                before_precision=float(before_query.get("precision_at_5", 0) or 0),
                after_precision=float(after_query.get("precision_at_5", 0) or 0),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The upgraded scorer uses configurable weights from `data/config/retrieval.yaml`.",
            "- Paper links contribute confidence only; task, behavior, modality, metadata, and readiness remain the main relevance signals.",
            "- Queries whose expected concepts do not exist in the current demo corpus can still fail despite improved parsing.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_comparison_json(before: dict[str, Any], after: EvaluationReport) -> str:
    """Generate machine-readable before/after comparison details."""

    after_dict = asdict(after)
    metric_keys = [
        "mean_precision_at_1",
        "mean_precision_at_3",
        "mean_precision_at_5",
        "mean_precision_at_10",
        "mean_recall_at_5",
        "mean_recall_at_10",
        "mean_label_recall_at_10",
        "mean_mrr",
        "mean_ndcg_at_10",
        "mean_task_match_rate",
        "mean_modality_match_rate",
        "mean_behavior_match_rate",
    ]
    payload = {
        "generated_at": after.generated_at,
        "before_generated_at": before.get("generated_at"),
        "after_generated_at": after.generated_at,
        "metrics": {
            key: {
                "before": before.get(key),
                "after": after_dict.get(key),
                "delta": round(_metric_delta(before, after_dict, key), 3),
            }
            for key in metric_keys
        },
        "before": before,
        "after": after_dict,
    }
    return json.dumps(payload, indent=2, default=str)


def write_comparison_reports(
    before_report_path: Path,
    after: EvaluationReport,
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Write before/after retrieval comparison reports to disk."""

    before = json.loads(before_report_path.read_text(encoding="utf-8"))
    out = output_dir or RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "retrieval_comparison_report.md"
    json_path = out / "retrieval_comparison_report.json"
    md_path.write_text(generate_comparison_markdown(before, after), encoding="utf-8")
    json_path.write_text(generate_comparison_json(before, after), encoding="utf-8")
    return {"markdown": str(md_path), "json": str(json_path)}


def write_reports(report: EvaluationReport, output_dir: Path | None = None) -> dict[str, str]:
    """Write evaluation reports to files."""
    out = output_dir or output_dir_for_suite(report.suite)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "latest_eval_report.md"
    json_path = out / "latest_eval_report.json"
    suite_json_path = out / "latest.json"

    md_content = generate_markdown_report(report)
    json_content = generate_json_report(report)

    md_path.write_text(md_content, encoding="utf-8")
    json_path.write_text(json_content, encoding="utf-8")
    suite_json_path.write_text(json_content, encoding="utf-8")

    return {"markdown": str(md_path), "json": str(json_path), "latest": str(suite_json_path)}


def run_suite(suite: str, output_dir: Path | None = None) -> dict[str, str] | EvaluationReport:
    """Run one benchmark suite and write reports."""

    report = run_full_benchmark(benchmark_path_for_suite(suite), suite=suite)
    if output_dir is None:
        return write_reports(report)
    return write_reports(report, output_dir / suite)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.evaluation.run_benchmark",
        description="Run benchmark evaluation for Neural Search retrieval.",
    )
    parser.add_argument(
        "--suite",
        choices=SUITE_CHOICES,
        default="demo_v02",
        help="Benchmark suite to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports. Defaults to data/eval/results/.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output JSON to stdout instead of writing files.",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="Optional prior JSON report to compare against.",
    )
    args = parser.parse_args(argv)

    if args.suite == "all":
        all_paths = {
            suite: run_suite(suite, args.output_dir)
            for suite in SUITE_PATHS
        }
        if args.json_only:
            print(json.dumps(all_paths, indent=2, sort_keys=True))
            return 0
        print(json.dumps(all_paths, indent=2, sort_keys=True))
        return 0

    report = run_full_benchmark(benchmark_path_for_suite(args.suite), suite=args.suite)

    if args.json_only:
        print(generate_json_report(report))
        return 0

    suite_output_dir = args.output_dir / args.suite if args.output_dir else None
    paths = write_reports(report, suite_output_dir)
    comparison_paths = None
    if args.compare_to is not None:
        comparison_paths = write_comparison_reports(
            args.compare_to,
            report,
            suite_output_dir,
        )

    print("=" * 70)
    print("NEURAL SEARCH BENCHMARK EVALUATION")
    print("=" * 70)
    print()
    print("SUMMARY METRICS")
    print("-" * 50)
    print(f"  Total Queries:          {report.total_queries}")
    print(f"  Queries with Results:   {report.queries_with_results}")
    print(f"  Mean Precision@1:       {report.mean_precision_at_1:.1%}")
    print(f"  Mean Precision@3:       {report.mean_precision_at_3:.1%}")
    print(f"  Mean Precision@5:       {report.mean_precision_at_5:.1%}")
    print(f"  Mean Precision@10:      {report.mean_precision_at_10:.1%}")
    print(f"  Mean Recall@5:          {report.mean_recall_at_5:.1%}")
    print(f"  Mean Recall@10:         {report.mean_recall_at_10:.1%}")
    print(f"  Mean Label Recall@10:   {report.mean_label_recall_at_10:.1%}")
    print(f"  Mean MRR:               {report.mean_mrr:.3f}")
    print(f"  Mean NDCG@10:           {report.mean_ndcg_at_10:.3f}")
    print(f"  Task Match Rate:        {report.mean_task_match_rate:.1%}")
    print(f"  Modality Match Rate:    {report.mean_modality_match_rate:.1%}")
    print(f"  Behavior Match Rate:    {report.mean_behavior_match_rate:.1%}")
    print()

    if report.summary_warnings:
        print("WARNINGS")
        print("-" * 50)
        for w in report.summary_warnings:
            print(f"  ! {w}")
        print()

    if report.recommendations:
        print("RECOMMENDATIONS")
        print("-" * 50)
        for r in report.recommendations:
            print(f"  > {r}")
        print()

    print("PER-QUERY RESULTS")
    print("-" * 50)
    for e in report.queries:
        status = "PASS" if not e.why_failed and e.label_recall_at_10 >= 0.5 else "FAIL"
        print(f"  [{status}] {e.query_id}: P@5={e.precision_at_5:.0%} R@10={e.label_recall_at_10:.0%}")
    print()

    print("=" * 70)
    print("Reports written to:")
    print(f"  Markdown: {paths['markdown']}")
    print(f"  JSON: {paths['json']}")
    if comparison_paths:
        print(f"  Comparison Markdown: {comparison_paths['markdown']}")
        print(f"  Comparison JSON: {comparison_paths['json']}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
