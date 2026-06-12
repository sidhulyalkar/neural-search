"""Agreement metrics for human qrels reviews."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from neural_search.field_state.eval_memory.qrels_schema import QrelsReview
from neural_search.field_state.store import (
    QRELS_AGREEMENT_PATH,
    QRELS_AGREEMENT_REPORT,
    QRELS_REVIEWS_PATH,
    read_jsonl,
    resolve_path,
)


def compute_qrels_agreement(root: Path | None = None) -> dict[str, Any]:
    """Compute simple agreement metrics for qrels reviews."""
    reviews = [
        review
        for review in read_jsonl(QRELS_REVIEWS_PATH, QrelsReview, root)
        if review.relevance_score is not None
    ]
    grouped: dict[str, list[QrelsReview]] = defaultdict(list)
    for review in reviews:
        grouped[review.candidate_id].append(review)
    multiple = {key: value for key, value in grouped.items() if len(value) > 1}
    exact_agreements = 0
    disagreements = 0
    for group in multiple.values():
        values = {
            (
                review.relevance_score,
                review.usefulness_score,
                review.hard_negative_violation,
            )
            for review in group
        }
        if len(values) == 1:
            exact_agreements += 1
        else:
            disagreements += 1
    relevance_distribution = Counter(str(review.relevance_score) for review in reviews)
    hard_negative_distribution = Counter(
        str(review.hard_negative_violation) for review in reviews
    )
    exact_agreement_rate = exact_agreements / len(multiple) if multiple else None
    return {
        "reviewed_candidates": len(grouped),
        "review_records": len(reviews),
        "multiple_review_candidates": len(multiple),
        "exact_agreement_rate": exact_agreement_rate,
        "disagreement_count": disagreements,
        "relevance_distribution": dict(sorted(relevance_distribution.items())),
        "hard_negative_distribution": dict(sorted(hard_negative_distribution.items())),
        "schema_version": "0.3",
    }


def render_qrels_agreement(summary: dict[str, Any]) -> str:
    """Render agreement summary Markdown."""
    lines = [
        "# Qrels Agreement",
        "",
        f"- Reviewed candidates: {summary['reviewed_candidates']}",
        f"- Review records: {summary['review_records']}",
        f"- Candidates with multiple reviews: {summary['multiple_review_candidates']}",
        f"- Exact agreement rate: {summary['exact_agreement_rate']}",
        f"- Disagreement count: {summary['disagreement_count']}",
        "",
        "## Relevance Distribution",
        "",
    ]
    for score, count in summary["relevance_distribution"].items():
        lines.append(f"- {score}: {count}")
    lines.extend(["", "## Hard-Negative Violations", ""])
    for value, count in summary["hard_negative_distribution"].items():
        lines.append(f"- {value}: {count}")
    return "\n".join(lines)


def write_qrels_agreement(root: Path | None = None) -> dict[str, Any]:
    """Write agreement JSON and Markdown reports."""
    summary = compute_qrels_agreement(root)
    json_path = resolve_path(QRELS_AGREEMENT_PATH, root)
    md_path = resolve_path(QRELS_AGREEMENT_REPORT, root)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_qrels_agreement(summary), encoding="utf-8")
    return summary
