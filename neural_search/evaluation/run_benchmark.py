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
from neural_search.ontology import normalize_text
from neural_search.search import parse_query, search_datasets


BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "data" / "eval" / "benchmark_queries.yaml"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "eval" / "results"


@dataclass
class BenchmarkQuery:
    """A benchmark query with expected labels."""

    id: str
    query: str
    expected_tasks: list[str] = field(default_factory=list)
    expected_behaviors: list[str] = field(default_factory=list)
    expected_modalities_any: list[str] = field(default_factory=list)
    expected_species: list[str] = field(default_factory=list)
    expected_analysis_any: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class QueryEvaluation:
    """Evaluation results for a single query."""

    query_id: str
    query: str
    num_results: int
    precision_at_5: float
    label_recall_at_10: float
    task_match_rate: float
    modality_match_rate: float
    behavior_match_rate: float
    matched_tasks: list[str]
    matched_modalities: list[str]
    matched_behaviors: list[str]
    missing_expected_tasks: list[str]
    missing_expected_modalities: list[str]
    missing_expected_behaviors: list[str]
    top_results: list[dict[str, Any]]
    warnings: list[str]
    parsed_query: dict[str, Any]


@dataclass
class EvaluationReport:
    """Complete evaluation report across all benchmark queries."""

    generated_at: str
    total_queries: int
    queries_with_results: int
    mean_precision_at_5: float
    mean_label_recall_at_10: float
    mean_task_match_rate: float
    mean_modality_match_rate: float
    mean_behavior_match_rate: float
    queries: list[QueryEvaluation]
    summary_warnings: list[str]
    recommendations: list[str]


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
                expected_tasks=item.get("expected_tasks", []),
                expected_behaviors=item.get("expected_behaviors", []),
                expected_modalities_any=item.get("expected_modalities_any", []),
                expected_species=item.get("expected_species", []),
                expected_analysis_any=item.get("expected_analysis_any", []),
                notes=item.get("notes"),
            )
        )
    return queries


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

    # Build lookup for dataset metadata
    dataset_lookup: dict[str, dict[str, Any]] = {}
    if datasets:
        for record in datasets:
            ds = record.get("dataset", record)
            ds_id = ds.get("id", ds.get("source_id", ""))
            dataset_lookup[str(ds_id)] = ds

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

    return {"tasks": tasks, "modalities": modalities, "behaviors": behaviors}


def evaluate_query(
    query: BenchmarkQuery,
    datasets: list[dict[str, Any]] | None = None,
    k: int = 10,
) -> QueryEvaluation:
    """Evaluate a single benchmark query."""
    response = search_datasets(
        query=query.query,
        filters={},
        datasets=datasets,
        limit=k,
    )

    results = [
        {
            "dataset_id": r.dataset_id,
            "score": r.score,
            "why_matched": r.why_matched,
            "warnings": r.warnings,
            "dataset_card_preview": r.dataset_card_preview,
        }
        for r in response.results
    ]

    expected_tasks = _normalize_list(query.expected_tasks)
    expected_modalities = _normalize_list(query.expected_modalities_any)
    expected_behaviors = _normalize_list(query.expected_behaviors)

    result_labels = _extract_result_labels(results, datasets)
    matched_tasks = sorted(expected_tasks & result_labels["tasks"])
    matched_modalities = sorted(expected_modalities & result_labels["modalities"])
    matched_behaviors = sorted(expected_behaviors & result_labels["behaviors"])

    missing_tasks = sorted(expected_tasks - result_labels["tasks"])
    missing_modalities = sorted(expected_modalities - result_labels["modalities"])
    missing_behaviors = sorted(expected_behaviors - result_labels["behaviors"])

    task_match_rate = len(matched_tasks) / len(expected_tasks) if expected_tasks else 1.0
    modality_match_rate = len(matched_modalities) / len(expected_modalities) if expected_modalities else 1.0
    behavior_match_rate = len(matched_behaviors) / len(expected_behaviors) if expected_behaviors else 1.0

    all_expected = expected_tasks | expected_modalities | expected_behaviors
    all_matched = set(matched_tasks) | set(matched_modalities) | set(matched_behaviors)
    label_recall = len(all_matched) / len(all_expected) if all_expected else 1.0

    relevant_count = 0
    for r in results[:5]:
        why = r.get("why_matched", [])
        if any("matched" in reason.lower() for reason in why):
            relevant_count += 1

    precision_at_5 = relevant_count / min(5, max(len(results), 1))

    warnings = []
    if not results:
        warnings.append("No results returned")
    if missing_tasks:
        warnings.append(f"Expected tasks not found: {missing_tasks}")
    if missing_modalities:
        warnings.append(f"Expected modalities not found: {missing_modalities}")
    if missing_behaviors:
        warnings.append(f"Expected behaviors not found: {missing_behaviors}")

    for r in results:
        warnings.extend(r.get("warnings", []))

    return QueryEvaluation(
        query_id=query.id,
        query=query.query,
        num_results=len(results),
        precision_at_5=round(precision_at_5, 3),
        label_recall_at_10=round(label_recall, 3),
        task_match_rate=round(task_match_rate, 3),
        modality_match_rate=round(modality_match_rate, 3),
        behavior_match_rate=round(behavior_match_rate, 3),
        matched_tasks=matched_tasks,
        matched_modalities=matched_modalities,
        matched_behaviors=matched_behaviors,
        missing_expected_tasks=missing_tasks,
        missing_expected_modalities=missing_modalities,
        missing_expected_behaviors=missing_behaviors,
        top_results=results[:5],
        warnings=list(set(warnings)),
        parsed_query=response.parsed_query,
    )


def run_full_benchmark(
    benchmark_path: Path | None = None,
    datasets: list[dict[str, Any]] | None = None,
) -> EvaluationReport:
    """Run complete benchmark evaluation."""
    queries = load_benchmark_queries(benchmark_path)
    if datasets is None:
        datasets = build_demo_seed()

    evaluations = [evaluate_query(q, datasets) for q in queries]

    queries_with_results = sum(1 for e in evaluations if e.num_results > 0)

    mean_p5 = sum(e.precision_at_5 for e in evaluations) / len(evaluations) if evaluations else 0
    mean_recall = sum(e.label_recall_at_10 for e in evaluations) / len(evaluations) if evaluations else 0
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
        mean_precision_at_5=round(mean_p5, 3),
        mean_label_recall_at_10=round(mean_recall, 3),
        mean_task_match_rate=round(mean_task, 3),
        mean_modality_match_rate=round(mean_mod, 3),
        mean_behavior_match_rate=round(mean_beh, 3),
        queries=evaluations,
        summary_warnings=summary_warnings,
        recommendations=recommendations,
    )


def generate_markdown_report(report: EvaluationReport) -> str:
    """Generate Markdown evaluation report."""
    lines = [
        "# Neural Search Benchmark Evaluation Report",
        "",
        f"Generated: {report.generated_at}",
        "",
        "## Summary Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Queries | {report.total_queries} |",
        f"| Queries with Results | {report.queries_with_results} |",
        f"| **Mean Precision@5** | **{report.mean_precision_at_5:.1%}** |",
        f"| **Mean Label Recall@10** | **{report.mean_label_recall_at_10:.1%}** |",
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
        status = "PASS" if eval_result.label_recall_at_10 >= 0.5 else "FAIL"
        lines.extend([
            f"### {eval_result.query_id}: {status}",
            "",
            f"**Query:** {eval_result.query}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Results | {eval_result.num_results} |",
            f"| Precision@5 | {eval_result.precision_at_5:.1%} |",
            f"| Label Recall@10 | {eval_result.label_recall_at_10:.1%} |",
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
        ("Mean Precision@5", "mean_precision_at_5"),
        ("Mean Label Recall@10", "mean_label_recall_at_10"),
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
        "mean_precision_at_5",
        "mean_label_recall_at_10",
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
    out = output_dir or RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "latest_eval_report.md"
    json_path = out / "latest_eval_report.json"

    md_content = generate_markdown_report(report)
    json_content = generate_json_report(report)

    md_path.write_text(md_content, encoding="utf-8")
    json_path.write_text(json_content, encoding="utf-8")

    return {"markdown": str(md_path), "json": str(json_path)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.evaluation.run_benchmark",
        description="Run benchmark evaluation for Neural Search retrieval.",
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

    report = run_full_benchmark()

    if args.json_only:
        print(generate_json_report(report))
        return 0

    paths = write_reports(report, args.output_dir)
    comparison_paths = None
    if args.compare_to is not None:
        comparison_paths = write_comparison_reports(
            args.compare_to,
            report,
            args.output_dir,
        )

    print("=" * 70)
    print("NEURAL SEARCH BENCHMARK EVALUATION")
    print("=" * 70)
    print()
    print("SUMMARY METRICS")
    print("-" * 50)
    print(f"  Total Queries:          {report.total_queries}")
    print(f"  Queries with Results:   {report.queries_with_results}")
    print(f"  Mean Precision@5:       {report.mean_precision_at_5:.1%}")
    print(f"  Mean Label Recall@10:   {report.mean_label_recall_at_10:.1%}")
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
        status = "PASS" if e.label_recall_at_10 >= 0.5 else "FAIL"
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
