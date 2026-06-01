"""Human relevance review queue helpers for search intelligence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

RelevanceLevel = Literal[
    "exact",
    "highly_relevant",
    "relevant",
    "partial",
    "not_relevant",
    "hard_negative",
    "unknown",
]

RELEVANCE_VALUES: set[str] = {
    "exact",
    "highly_relevant",
    "relevant",
    "partial",
    "not_relevant",
    "hard_negative",
    "unknown",
}
POSITIVE_LEVELS = {"exact", "highly_relevant", "relevant"}


@dataclass(frozen=True)
class RelevanceJudgment:
    """Human relevance judgment for a query-result pair."""

    query_id: str
    query_text: str
    dataset_id: str
    relevance: RelevanceLevel
    task_match: int = 0
    modality_match: int = 0
    species_match: int = 0
    analysis_fit: int = 0
    reviewer_id: str = "unassigned"
    confidence: float = 0.0
    notes: str = ""

    def model_dump(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "dataset_id": self.dataset_id,
            "relevance": self.relevance,
            "task_match": self.task_match,
            "modality_match": self.modality_match,
            "species_match": self.species_match,
            "analysis_fit": self.analysis_fit,
            "reviewer_id": self.reviewer_id,
            "confidence": self.confidence,
            "notes": self.notes,
        }


def _bounded_score(value: Any) -> int:
    return max(0, min(int(value or 0), 3))


def judgment_from_dict(payload: dict[str, Any]) -> RelevanceJudgment:
    relevance = str(payload.get("relevance", "unknown"))
    if relevance not in RELEVANCE_VALUES:
        raise ValueError(f"Unknown relevance level: {relevance}")
    return RelevanceJudgment(
        query_id=str(payload.get("query_id", "")),
        query_text=str(payload.get("query_text", payload.get("query", ""))),
        dataset_id=str(payload.get("dataset_id", "")),
        relevance=relevance,  # type: ignore[arg-type]
        task_match=_bounded_score(payload.get("task_match", 0)),
        modality_match=_bounded_score(payload.get("modality_match", 0)),
        species_match=_bounded_score(payload.get("species_match", 0)),
        analysis_fit=_bounded_score(payload.get("analysis_fit", 0)),
        reviewer_id=str(payload.get("reviewer_id", "unassigned")),
        confidence=max(0.0, min(float(payload.get("confidence", 0.0) or 0.0), 1.0)),
        notes=str(payload.get("notes", payload.get("review_notes", ""))),
    )


def load_relevance_judgments(path: str | Path) -> list[RelevanceJudgment]:
    judgments: list[RelevanceJudgment] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                judgments.append(judgment_from_dict(json.loads(line)))
    return judgments


def summarize_relevance_judgments(
    judgments: list[RelevanceJudgment],
) -> dict[str, Any]:
    counts = Counter(judgment.relevance for judgment in judgments)
    positive = sum(counts[level] for level in POSITIVE_LEVELS)
    labeled = len([judgment for judgment in judgments if judgment.relevance != "unknown"])
    hard_negatives = counts["hard_negative"]
    mean_confidence = (
        sum(judgment.confidence for judgment in judgments) / len(judgments)
        if judgments
        else 0.0
    )
    return {
        "judgment_count": len(judgments),
        "labeled_count": labeled,
        "positive_count": positive,
        "hard_negative_count": hard_negatives,
        "precision_like_rate": round(positive / labeled, 4) if labeled else 0.0,
        "mean_confidence": round(mean_confidence, 4),
        "counts_by_relevance": dict(sorted(counts.items())),
    }


def build_review_queue(
    coverage_plan: dict[str, Any],
    benchmark_seeds: dict[str, Any],
    *,
    max_items: int = 25,
) -> list[dict[str, Any]]:
    """Create review items from coverage gaps and generated benchmark seeds."""

    priorities = {
        gap.get("data_form"): gap.get("priority", "medium")
        for gap in coverage_plan.get("gaps", [])
    }
    queue: list[dict[str, Any]] = []
    for query in benchmark_seeds.get("benchmark_queries", []):
        gap = str(query.get("coverage_gap", "general"))
        priority = str(query.get("priority", priorities.get(gap, "medium")))
        queue.append(
            {
                "review_id": f"review:{query.get('id', len(queue) + 1)}",
                "query_id": query.get("id", ""),
                "query_text": query.get("query", ""),
                "coverage_gap": gap,
                "priority": priority,
                "expected_modalities_any": query.get("expected_modalities_any", []),
                "expected_analysis_any": query.get("expected_analysis_any", []),
                "expected_data_standards": query.get("expected_data_standards", []),
                "label_status": "needs_review",
                "review_instruction": (
                    "Run this query, label top results, and add expected dataset IDs "
                    "only after representative corpus records exist."
                ),
            }
        )
    priority_order = {"critical": 0, "high": 1, "medium": 2, "covered": 3}
    queue.sort(key=lambda item: (priority_order.get(item["priority"], 9), item["query_id"]))
    return queue[:max_items]


def _review_markdown(queue: list[dict[str, Any]]) -> str:
    lines = [
        "# Human Relevance Review Queue",
        "",
        f"- Items: {len(queue)}",
        "",
        "| Priority | Query ID | Coverage Gap | Query |",
        "|---|---|---|---|",
    ]
    for item in queue:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["priority"]),
                    str(item["query_id"]),
                    str(item["coverage_gap"]),
                    str(item["query_text"]).replace("|", "/"),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_review_queue(
    queue: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "human_review_queue.json"
    md_path = out / "human_review_queue.md"
    json_path.write_text(json.dumps(queue, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_review_markdown(queue), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a human relevance review queue from coverage reports."
    )
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--benchmark-seeds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-items", type=int, default=25)
    args = parser.parse_args(argv)

    coverage_plan = json.loads(Path(args.coverage).read_text(encoding="utf-8"))
    benchmark_seeds = yaml.safe_load(
        Path(args.benchmark_seeds).read_text(encoding="utf-8")
    ) or {}
    queue = build_review_queue(
        coverage_plan,
        benchmark_seeds,
        max_items=args.max_items,
    )
    print(json.dumps(write_review_queue(queue, args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
