"""Bridge: convert raw record/card dicts to DatasetContext."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neural_search.retrieval.usefulness_scorer import DatasetContext


def _extract_labels(items: Any) -> list[str]:
    """Extract label strings from a list of dicts or plain strings."""
    if not items:
        return []
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, Mapping):
            label = item.get("label") or item.get("name") or item.get("id") or ""
            if label:
                result.append(str(label))
    return result


def dataset_context_from_record(
    record: Mapping[str, Any],
    card: Mapping[str, Any] | None = None,
) -> DatasetContext:
    """Convert a raw dataset record (and optional card) to DatasetContext.

    Args:
        record: Raw dataset dict as stored in the corpus.
        card: Optional dataset card dict (DatasetCardV1 serialized).

    Returns:
        DatasetContext ready for score_usefulness().
    """
    dataset_id = str(
        record.get("dataset_id")
        or record.get("id")
        or record.get("source_id")
        or ""
    )

    tasks = _extract_labels(record.get("tasks", []))
    modalities = _extract_labels(record.get("modalities", []))
    species = _extract_labels(record.get("species", []))
    brain_regions = _extract_labels(record.get("brain_regions", []))

    # Card fields (richer metadata)
    affordances: list[str] = []
    data_standards: list[str] = []
    quality_score: float = 0.0
    session_count: int | None = record.get("session_count")
    trial_count: int | None = record.get("trial_count")
    subject_count: int | None = record.get("subject_count")
    has_timestamps: bool = bool(record.get("has_timestamps", False))

    if card:
        aff_raw = card.get("analysis_affordances", [])
        for a in aff_raw:
            if isinstance(a, Mapping):
                aff_id = a.get("affordance_id") or a.get("id") or ""
                if aff_id:
                    affordances.append(str(aff_id))
            elif isinstance(a, str):
                affordances.append(a)

        std_raw = card.get("data_standards", [])
        for s in std_raw:
            if isinstance(s, str):
                data_standards.append(s)
            elif isinstance(s, Mapping):
                std_str = str(s.get("name") or s.get("id") or "")
                if std_str:
                    data_standards.append(std_str)

        raw_qs = card.get("quality_score") or 0.0
        if isinstance(raw_qs, Mapping):
            raw_qs = raw_qs.get("overall_score", 0.0) or 0.0
        quality_score = max(0.0, min(1.0, float(raw_qs)))
        if session_count is None:
            session_count = card.get("session_count") or card.get("n_sessions")
        if trial_count is None:
            trial_count = card.get("trial_count") or card.get("n_trials")
        if subject_count is None:
            subject_count = card.get("subject_count") or card.get("n_subjects")

    return DatasetContext(
        dataset_id=dataset_id,
        modalities=modalities,
        tasks=tasks,
        species=species,
        brain_regions=brain_regions,
        affordances=affordances,
        data_standards=data_standards,
        session_count=session_count,
        trial_count=trial_count,
        subject_count=subject_count,
        has_timestamps=has_timestamps,
        quality_score=quality_score,
    )
