"""Adjudicate imported qrels reviews into final qrels artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from neural_search.field_state.eval_memory.qrels_schema import (
    AdjudicatedQrel,
    QrelsReview,
)
from neural_search.field_state.store import (
    ADJUDICATED_QRELS_PATH,
    QRELS_REVIEWS_PATH,
    read_jsonl,
    write_jsonl,
)


def _score_dict(review: QrelsReview) -> dict[str, Any]:
    return {
        "annotator_id": review.annotator_id,
        "relevance_score": review.relevance_score,
        "usefulness_score": review.usefulness_score,
        "hard_negative_violation": review.hard_negative_violation,
        "label_confidence": review.label_confidence,
    }


def _review_id(review: QrelsReview) -> str:
    annotator = review.annotator_id or "unknown"
    return f"{review.candidate_id}:{annotator}:{review.source_note_path or 'no_path'}"


def adjudicate_review_group(reviews: list[QrelsReview]) -> AdjudicatedQrel | None:
    """Adjudicate all reviews for one candidate."""
    valid = [review for review in reviews if review.relevance_score is not None]
    if not valid:
        return None
    explicit = [
        review
        for review in valid
        if review.adjudicated_relevance_score is not None
        or review.review_status == "adjudicated"
    ]
    chosen = explicit[-1] if explicit else valid[0]
    annotator_scores = [_score_dict(review) for review in valid]
    source_review_ids = [_review_id(review) for review in valid]
    values = {
        (
            review.relevance_score,
            review.usefulness_score,
            review.hard_negative_violation,
        )
        for review in valid
    }
    if explicit and chosen.adjudicated_relevance_score is not None:
        return AdjudicatedQrel(
            candidate_id=chosen.candidate_id,
            query_id=chosen.query_id,
            dataset_id=chosen.dataset_id,
            final_relevance_score=chosen.adjudicated_relevance_score,
            final_usefulness_score=chosen.adjudicated_usefulness_score
            if chosen.adjudicated_usefulness_score is not None
            else chosen.usefulness_score,
            final_hard_negative_violation=(
                chosen.adjudicated_hard_negative_violation
                if chosen.adjudicated_hard_negative_violation is not None
                else bool(chosen.hard_negative_violation)
            ),
            adjudication_status="adjudicated",
            annotator_scores=annotator_scores,
            disagreement=len(values) > 1,
            adjudicator_notes=chosen.adjudicator_notes,
            source_review_ids=source_review_ids,
        )
    if len(valid) == 1:
        return AdjudicatedQrel(
            candidate_id=chosen.candidate_id,
            query_id=chosen.query_id,
            dataset_id=chosen.dataset_id,
            final_relevance_score=chosen.relevance_score or 0,
            final_usefulness_score=chosen.usefulness_score,
            final_hard_negative_violation=bool(chosen.hard_negative_violation),
            adjudication_status="single_review",
            annotator_scores=annotator_scores,
            disagreement=False,
            adjudicator_notes=chosen.adjudicator_notes,
            source_review_ids=source_review_ids,
        )
    if len(values) == 1:
        return AdjudicatedQrel(
            candidate_id=chosen.candidate_id,
            query_id=chosen.query_id,
            dataset_id=chosen.dataset_id,
            final_relevance_score=chosen.relevance_score or 0,
            final_usefulness_score=chosen.usefulness_score,
            final_hard_negative_violation=bool(chosen.hard_negative_violation),
            adjudication_status="agreement",
            annotator_scores=annotator_scores,
            disagreement=False,
            adjudicator_notes=None,
            source_review_ids=source_review_ids,
        )
    return AdjudicatedQrel(
        candidate_id=chosen.candidate_id,
        query_id=chosen.query_id,
        dataset_id=chosen.dataset_id,
        final_relevance_score=chosen.relevance_score or 0,
        final_usefulness_score=chosen.usefulness_score,
        final_hard_negative_violation=bool(chosen.hard_negative_violation),
        adjudication_status="needs_adjudication",
        annotator_scores=annotator_scores,
        disagreement=True,
        adjudicator_notes=None,
        source_review_ids=source_review_ids,
    )


def adjudicate_qrels(root: Path | None = None) -> list[AdjudicatedQrel]:
    """Read qrels reviews and write adjudicated qrels."""
    reviews = read_jsonl(QRELS_REVIEWS_PATH, QrelsReview, root)
    grouped: dict[str, list[QrelsReview]] = defaultdict(list)
    for review in reviews:
        grouped[review.candidate_id].append(review)
    adjudicated = [
        qrel
        for qrel in (adjudicate_review_group(group) for group in grouped.values())
        if qrel is not None
    ]
    write_jsonl(ADJUDICATED_QRELS_PATH, adjudicated, root)
    return adjudicated
