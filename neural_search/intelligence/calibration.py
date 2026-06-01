"""Score calibration against human relevance judgments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_search.intelligence.evaluation import (
    EvaluationQuery,
    _load_queries,
    load_search_records_from_normalized,
)
from neural_search.intelligence.integration import search_datasets_with_intelligence
from neural_search.intelligence.review import (
    RelevanceJudgment,
    load_relevance_judgments,
)

POSITIVE_RELEVANCE = {"exact", "highly_relevant", "relevant"}
NEGATIVE_RELEVANCE = {"not_relevant", "hard_negative"}


@dataclass(frozen=True)
class CalibrationItem:
    """One judged query-result score calibration item."""

    query_id: str
    dataset_id: str
    relevance: str
    target: float
    score: float
    rank: int | None
    returned: bool
    planner_intent: str
    planner_data_forms: tuple[str, ...]

    def model_dump(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "dataset_id": self.dataset_id,
            "relevance": self.relevance,
            "target": self.target,
            "score": self.score,
            "rank": self.rank,
            "returned": self.returned,
            "planner_intent": self.planner_intent,
            "planner_data_forms": list(self.planner_data_forms),
        }


@dataclass(frozen=True)
class CalibrationReport:
    """Calibration summary for judged retrieval scores."""

    item_count: int
    positive_count: int
    negative_count: int
    returned_item_count: int
    mean_positive_score: float
    mean_negative_score: float
    score_gap: float
    brier_score: float
    pairwise_auc: float | None
    grouped_by_intent: dict[str, dict[str, Any]]
    grouped_by_data_form: dict[str, dict[str, Any]]
    items: tuple[CalibrationItem, ...]

    def model_dump(self) -> dict[str, Any]:
        return {
            "item_count": self.item_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "returned_item_count": self.returned_item_count,
            "mean_positive_score": self.mean_positive_score,
            "mean_negative_score": self.mean_negative_score,
            "score_gap": self.score_gap,
            "brier_score": self.brier_score,
            "pairwise_auc": self.pairwise_auc,
            "grouped_by_intent": self.grouped_by_intent,
            "grouped_by_data_form": self.grouped_by_data_form,
            "items": [item.model_dump() for item in self.items],
        }


def _target(relevance: str) -> float:
    if relevance in POSITIVE_RELEVANCE:
        return 1.0
    if relevance == "partial":
        return 0.5
    return 0.0


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _pairwise_auc(items: list[CalibrationItem]) -> float | None:
    positives = [item.score for item in items if item.relevance in POSITIVE_RELEVANCE]
    negatives = [item.score for item in items if item.relevance in NEGATIVE_RELEVANCE]
    if not positives or not negatives:
        return None
    wins = 0.0
    comparisons = 0
    for positive in positives:
        for negative in negatives:
            comparisons += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return round(wins / comparisons, 4)


def _summary(items: list[CalibrationItem]) -> dict[str, Any]:
    positives = [item.score for item in items if item.relevance in POSITIVE_RELEVANCE]
    negatives = [item.score for item in items if item.relevance in NEGATIVE_RELEVANCE]
    brier = _mean([(item.score - item.target) ** 2 for item in items])
    return {
        "item_count": len(items),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "returned_item_count": sum(1 for item in items if item.returned),
        "mean_positive_score": _mean(positives),
        "mean_negative_score": _mean(negatives),
        "score_gap": round(_mean(positives) - _mean(negatives), 4),
        "brier_score": brier,
        "pairwise_auc": _pairwise_auc(items),
    }


def calibrate_scores_against_judgments(
    queries: list[EvaluationQuery],
    judgments: list[RelevanceJudgment],
    *,
    datasets: list[dict[str, Any]],
    limit: int = 10,
) -> CalibrationReport:
    """Compare intelligence retrieval scores with judged relevance labels."""

    judgments_by_query: dict[str, list[RelevanceJudgment]] = defaultdict(list)
    for judgment in judgments:
        judgments_by_query[judgment.query_id].append(judgment)

    items: list[CalibrationItem] = []
    for query in queries:
        response = search_datasets_with_intelligence(
            query.query,
            datasets=datasets,
            limit=limit,
            rerank=True,
        )
        plan = response.parsed_query.get("search_intelligence_plan", {})
        data_forms = tuple(
            str(value)
            for value in (
                plan.get("required_data_forms")
                or plan.get("query_awareness", {}).get("requested_data_forms", [])
                or ["unspecified"]
            )
        )
        result_by_id = {
            str(result.dataset_id): (rank, result)
            for rank, result in enumerate(response.results, 1)
        }
        for judgment in judgments_by_query.get(query.id, []):
            match = result_by_id.get(judgment.dataset_id)
            rank = match[0] if match else None
            score = round(float(match[1].score) / 100.0, 4) if match else 0.0
            items.append(
                CalibrationItem(
                    query_id=query.id,
                    dataset_id=judgment.dataset_id,
                    relevance=judgment.relevance,
                    target=_target(judgment.relevance),
                    score=score,
                    rank=rank,
                    returned=match is not None,
                    planner_intent=str(plan.get("intent", "unknown")),
                    planner_data_forms=data_forms,
                )
            )

    grouped_intents: dict[str, list[CalibrationItem]] = defaultdict(list)
    grouped_forms: dict[str, list[CalibrationItem]] = defaultdict(list)
    for item in items:
        grouped_intents[item.planner_intent].append(item)
        for data_form in item.planner_data_forms or ("unspecified",):
            grouped_forms[data_form].append(item)

    summary = _summary(items)
    return CalibrationReport(
        item_count=summary["item_count"],
        positive_count=summary["positive_count"],
        negative_count=summary["negative_count"],
        returned_item_count=summary["returned_item_count"],
        mean_positive_score=summary["mean_positive_score"],
        mean_negative_score=summary["mean_negative_score"],
        score_gap=summary["score_gap"],
        brier_score=summary["brier_score"],
        pairwise_auc=summary["pairwise_auc"],
        grouped_by_intent={
            intent: _summary(group_items)
            for intent, group_items in sorted(grouped_intents.items())
        },
        grouped_by_data_form={
            data_form: _summary(group_items)
            for data_form, group_items in sorted(grouped_forms.items())
        },
        items=tuple(items),
    )


def _markdown(report: CalibrationReport) -> str:
    lines = [
        "# Search Intelligence Score Calibration",
        "",
        f"- Judged items: {report.item_count}",
        f"- Positive judgments: {report.positive_count}",
        f"- Negative judgments: {report.negative_count}",
        f"- Returned judged items: {report.returned_item_count}",
        f"- Mean positive score: {report.mean_positive_score}",
        f"- Mean negative score: {report.mean_negative_score}",
        f"- Score gap: {report.score_gap}",
        f"- Brier score: {report.brier_score}",
        f"- Pairwise AUC: {report.pairwise_auc}",
        "",
        "## By Intent",
        "",
        "| Intent | Items | Pos | Neg | Score Gap | Brier | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for intent, summary in report.grouped_by_intent.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    intent,
                    str(summary["item_count"]),
                    str(summary["positive_count"]),
                    str(summary["negative_count"]),
                    str(summary["score_gap"]),
                    str(summary["brier_score"]),
                    str(summary["pairwise_auc"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## By Data Form",
            "",
            "| Data Form | Items | Pos | Neg | Score Gap | Brier | AUC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for data_form, summary in report.grouped_by_data_form.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    data_form,
                    str(summary["item_count"]),
                    str(summary["positive_count"]),
                    str(summary["negative_count"]),
                    str(summary["score_gap"]),
                    str(summary["brier_score"]),
                    str(summary["pairwise_auc"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_calibration_report(
    report: CalibrationReport,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "score_calibration_report.json"
    md_path = out / "score_calibration_report.md"
    json_path.write_text(
        json.dumps(report.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate intelligence retrieval scores against judgments."
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)

    report = calibrate_scores_against_judgments(
        _load_queries(args.benchmark),
        load_relevance_judgments(args.judgments),
        datasets=load_search_records_from_normalized(args.records),
        limit=args.limit,
    )
    print(
        json.dumps(
            write_calibration_report(report, args.out),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
