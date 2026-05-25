"""Evaluate planner-aware retrieval before default promotion."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from neural_search.awareness.search import search_datasets_with_awareness
from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.intelligence.integration import search_datasets_with_intelligence
from neural_search.search.core import search_datasets


@dataclass(frozen=True)
class EvaluationQuery:
    """Minimal benchmark query schema for wrapper comparison."""

    id: str
    query: str
    expected_dataset_ids: tuple[str, ...] = ()
    hard_negative_dataset_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class VariantMetrics:
    """Per-variant retrieval metrics for one query."""

    variant: str
    result_ids: tuple[str, ...]
    hit_at_5: float
    mrr: float
    hard_negative_violations: int

    def model_dump(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "result_ids": list(self.result_ids),
            "hit_at_5": self.hit_at_5,
            "mrr": self.mrr,
            "hard_negative_violations": self.hard_negative_violations,
        }


@dataclass(frozen=True)
class QueryPlanEvaluation:
    """Comparison for one query across baseline, awareness, and intelligence."""

    query_id: str
    query: str
    planner_intent: str
    planner_mode: str
    baseline: VariantMetrics
    awareness: VariantMetrics
    intelligence: VariantMetrics
    intelligence_delta: dict[str, float]
    promotion_blocked: bool

    def model_dump(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "planner_intent": self.planner_intent,
            "planner_mode": self.planner_mode,
            "baseline": self.baseline.model_dump(),
            "awareness": self.awareness.model_dump(),
            "intelligence": self.intelligence.model_dump(),
            "intelligence_delta": dict(self.intelligence_delta),
            "promotion_blocked": self.promotion_blocked,
        }


@dataclass(frozen=True)
class QueryPlanEvaluationReport:
    """Aggregated query-plan evaluation report."""

    query_count: int
    promotion_safe: bool
    mean_delta: dict[str, float]
    grouped_by_intent: dict[str, dict[str, Any]]
    queries: tuple[QueryPlanEvaluation, ...]

    def model_dump(self) -> dict[str, Any]:
        return {
            "query_count": self.query_count,
            "promotion_safe": self.promotion_safe,
            "mean_delta": dict(self.mean_delta),
            "grouped_by_intent": self.grouped_by_intent,
            "queries": [query.model_dump() for query in self.queries],
        }


def _load_queries(path: str | Path) -> list[EvaluationQuery]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    queries: list[EvaluationQuery] = []
    for item in payload.get("benchmark_queries", []):
        queries.append(
            EvaluationQuery(
                id=str(item.get("id", "")),
                query=str(item.get("query", "")),
                expected_dataset_ids=tuple(str(value) for value in item.get("expected_dataset_ids", [])),
                hard_negative_dataset_ids=tuple(
                    str(value) for value in item.get("hard_negative_dataset_ids", [])
                ),
            )
        )
    return queries


def _mrr(result_ids: tuple[str, ...], expected_ids: tuple[str, ...]) -> float:
    if not expected_ids:
        return 0.0
    expected = set(expected_ids)
    for rank, dataset_id in enumerate(result_ids, 1):
        if dataset_id in expected:
            return round(1.0 / rank, 4)
    return 0.0


def _hit_at_5(result_ids: tuple[str, ...], expected_ids: tuple[str, ...]) -> float:
    if not expected_ids:
        return 0.0
    return 1.0 if set(result_ids[:5]) & set(expected_ids) else 0.0


def _variant_metrics(
    variant: str,
    response: Any,
    query: EvaluationQuery,
) -> VariantMetrics:
    result_ids = tuple(str(result.dataset_id) for result in response.results)
    hard_negative_ids = set(query.hard_negative_dataset_ids)
    hard_negative_violations = len(set(result_ids[:10]) & hard_negative_ids)
    hard_negative_violations += sum(
        len(result.negative_constraint_matches) for result in response.results
    )
    return VariantMetrics(
        variant=variant,
        result_ids=result_ids,
        hit_at_5=_hit_at_5(result_ids, query.expected_dataset_ids),
        mrr=_mrr(result_ids, query.expected_dataset_ids),
        hard_negative_violations=hard_negative_violations,
    )


def evaluate_query_plan(
    query: EvaluationQuery,
    *,
    datasets: list[dict[str, Any]] | None = None,
    limit: int = 10,
    retrieval_config: dict[str, Any] | None = None,
) -> QueryPlanEvaluation:
    """Run one query through baseline, awareness, and intelligence retrieval."""

    records = datasets if datasets is not None else build_demo_seed()
    baseline_response = search_datasets(
        query.query,
        datasets=records,
        limit=limit,
        retrieval_config=retrieval_config,
    )
    awareness_response = search_datasets_with_awareness(
        query.query,
        datasets=records,
        limit=limit,
        retrieval_config=retrieval_config,
        rerank=True,
    )
    intelligence_response = search_datasets_with_intelligence(
        query.query,
        datasets=records,
        limit=limit,
        retrieval_config=retrieval_config,
        rerank=True,
    )
    baseline = _variant_metrics("baseline", baseline_response, query)
    awareness = _variant_metrics("awareness", awareness_response, query)
    intelligence = _variant_metrics("intelligence", intelligence_response, query)
    plan = intelligence_response.parsed_query.get("search_intelligence_plan", {})
    delta = {
        "hit_at_5": round(intelligence.hit_at_5 - baseline.hit_at_5, 4),
        "mrr": round(intelligence.mrr - baseline.mrr, 4),
        "hard_negative_violations": float(
            intelligence.hard_negative_violations - baseline.hard_negative_violations
        ),
    }
    return QueryPlanEvaluation(
        query_id=query.id,
        query=query.query,
        planner_intent=str(plan.get("intent", "unknown")),
        planner_mode=str(plan.get("mode", "unknown")),
        baseline=baseline,
        awareness=awareness,
        intelligence=intelligence,
        intelligence_delta=delta,
        promotion_blocked=delta["hard_negative_violations"] > 0,
    )


def run_query_plan_evaluation(
    queries: list[EvaluationQuery],
    *,
    datasets: list[dict[str, Any]] | None = None,
    limit: int = 10,
    retrieval_config: dict[str, Any] | None = None,
) -> QueryPlanEvaluationReport:
    """Evaluate a set of queries and summarize planner-promotion readiness."""

    evaluations = tuple(
        evaluate_query_plan(
            query,
            datasets=datasets,
            limit=limit,
            retrieval_config=retrieval_config,
        )
        for query in queries
    )
    grouped: dict[str, list[QueryPlanEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        grouped[evaluation.planner_intent].append(evaluation)

    grouped_summary: dict[str, dict[str, Any]] = {}
    for intent, items in grouped.items():
        grouped_summary[intent] = {
            "query_count": len(items),
            "mean_hit_at_5_delta": round(
                sum(item.intelligence_delta["hit_at_5"] for item in items) / len(items),
                4,
            ),
            "mean_mrr_delta": round(
                sum(item.intelligence_delta["mrr"] for item in items) / len(items),
                4,
            ),
            "hard_negative_violation_delta": int(
                sum(item.intelligence_delta["hard_negative_violations"] for item in items)
            ),
            "promotion_safe": not any(item.promotion_blocked for item in items),
        }

    query_count = len(evaluations)
    mean_delta = {
        "hit_at_5": round(
            sum(item.intelligence_delta["hit_at_5"] for item in evaluations) / query_count,
            4,
        )
        if query_count
        else 0.0,
        "mrr": round(
            sum(item.intelligence_delta["mrr"] for item in evaluations) / query_count,
            4,
        )
        if query_count
        else 0.0,
        "hard_negative_violations": float(
            sum(item.intelligence_delta["hard_negative_violations"] for item in evaluations)
        ),
    }
    return QueryPlanEvaluationReport(
        query_count=query_count,
        promotion_safe=not any(item.promotion_blocked for item in evaluations),
        mean_delta=mean_delta,
        grouped_by_intent=grouped_summary,
        queries=evaluations,
    )


def _markdown(report: QueryPlanEvaluationReport) -> str:
    lines = [
        "# Query Plan Evaluation",
        "",
        f"- Queries: {report.query_count}",
        f"- Promotion safe: {str(report.promotion_safe).lower()}",
        f"- Mean hit@5 delta: {report.mean_delta['hit_at_5']}",
        f"- Mean MRR delta: {report.mean_delta['mrr']}",
        f"- Hard-negative violation delta: {report.mean_delta['hard_negative_violations']}",
        "",
        "## By Intent",
        "",
        "| Intent | Queries | Hit@5 Delta | MRR Delta | Hard Neg Delta | Promotion Safe |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for intent, summary in sorted(report.grouped_by_intent.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    intent,
                    str(summary["query_count"]),
                    str(summary["mean_hit_at_5_delta"]),
                    str(summary["mean_mrr_delta"]),
                    str(summary["hard_negative_violation_delta"]),
                    str(summary["promotion_safe"]).lower(),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Queries", ""])
    for item in report.queries:
        lines.append(
            f"- `{item.query_id}` {item.planner_intent}/{item.planner_mode}: "
            f"hit@5 delta {item.intelligence_delta['hit_at_5']}, "
            f"MRR delta {item.intelligence_delta['mrr']}, "
            f"hard-neg delta {item.intelligence_delta['hard_negative_violations']}"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_query_plan_evaluation_report(
    report: QueryPlanEvaluationReport,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "query_plan_evaluation.json"
    md_path = out / "query_plan_evaluation.md"
    json_path.write_text(
        json.dumps(report.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline, awareness, and intelligence retrieval."
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)

    report = run_query_plan_evaluation(
        _load_queries(args.benchmark),
        limit=args.limit,
    )
    print(
        json.dumps(
            write_query_plan_evaluation_report(report, args.out),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
