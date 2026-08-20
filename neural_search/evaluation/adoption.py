"""External-user workflow evaluation for Neural Search.

Retrieval metrics answer whether the engine ranks judged datasets correctly.
Adoption metrics answer whether a researcher can actually complete the reuse
workflow: search, recover from metadata gaps, save a useful dataset, generate
or execute a notebook, and make a reuse decision.

The event schema is intentionally simple JSON so browser instrumentation,
notebooks, CLI runs, and usability-study tooling can emit the same records.
These are descriptive usability metrics, never substitutes for qrels or
scientific-validity evaluation.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdoptionEvent:
    session_id: str
    timestamp: str
    event_type: str
    success: bool | None = None
    dataset_id: str | None = None
    usefulness: str | None = None
    would_use_for_analysis: str | None = None
    known_before: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AdoptionEvent:
        event = cls(
            session_id=str(payload["session_id"]),
            timestamp=str(payload["timestamp"]),
            event_type=str(payload["event_type"]),
            success=(
                bool(payload["success"])
                if payload.get("success") is not None
                else None
            ),
            dataset_id=(
                str(payload["dataset_id"])
                if payload.get("dataset_id") is not None
                else None
            ),
            usefulness=(
                str(payload["usefulness"])
                if payload.get("usefulness") is not None
                else None
            ),
            would_use_for_analysis=(
                str(payload["would_use_for_analysis"])
                if payload.get("would_use_for_analysis") is not None
                else None
            ),
            known_before=(
                bool(payload["known_before"])
                if payload.get("known_before") is not None
                else None
            ),
            metadata=dict(payload.get("metadata") or {}),
            schema_version=int(payload.get("schema_version", 1)),
        )
        event.validate()
        return event

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported adoption event schema: {self.schema_version}")
        if not self.session_id.strip():
            raise ValueError("session_id must be non-empty")
        if not self.event_type.strip():
            raise ValueError("event_type must be non-empty")
        _parse_timestamp(self.timestamp)
        if self.usefulness not in {
            None,
            "useful",
            "partially_useful",
            "not_useful",
            "unsure",
        }:
            raise ValueError("invalid usefulness value")
        if self.would_use_for_analysis not in {None, "yes", "maybe", "no"}:
            raise ValueError("invalid would_use_for_analysis value")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("adoption event timestamps must include a timezone")
    return parsed


def load_adoption_events(path: Path) -> list[AdoptionEvent]:
    if not path.is_file():
        return []
    events: list[AdoptionEvent] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("event must be an object")
                events.append(AdoptionEvent.from_dict(payload))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid adoption event at line {line_number}: {exc}"
                ) from exc
    return events


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _mean_per_session(count: int, sessions: int) -> float | None:
    return round(count / sessions, 6) if sessions else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(ordered[lower], 3)
    fraction = rank - lower
    result = ordered[lower] * (1 - fraction) + ordered[upper] * fraction
    return round(result, 3)


def _event_success(event: AdoptionEvent) -> bool:
    return event.success is not False


def _unique_dataset_events(events: Iterable[AdoptionEvent]) -> list[AdoptionEvent]:
    """Avoid repeated ratings of the same dataset/session dominating study metrics."""

    chosen: dict[tuple[str, str], AdoptionEvent] = {}
    anonymous_counter = 0
    for event in events:
        if event.dataset_id:
            key = (event.session_id, event.dataset_id.casefold())
        else:
            anonymous_counter += 1
            key = (event.session_id, f"__event_{anonymous_counter}")
        chosen.setdefault(key, event)
    return list(chosen.values())


def _duration_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "median": round(statistics.median(values), 3) if values else None,
        "p90": _percentile(values, 0.9),
        "observations": len(values),
    }


def evaluate_adoption_events(events: Iterable[AdoptionEvent]) -> dict[str, Any]:
    """Compute workflow-level adoption metrics from researcher sessions."""

    event_list = sorted(
        events,
        key=lambda event: (event.session_id, _parse_timestamp(event.timestamp)),
    )
    by_session: dict[str, list[AdoptionEvent]] = defaultdict(list)
    for event in event_list:
        by_session[event.session_id].append(event)

    search_attempts = [event for event in event_list if event.event_type == "search_attempt"]
    search_successes = [
        event
        for event in event_list
        if event.event_type == "search_success" and _event_success(event)
    ]
    saved = [event for event in event_list if event.event_type == "dataset_saved"]
    notebook_generated = [
        event
        for event in event_list
        if event.event_type == "notebook_generated" and _event_success(event)
    ]
    notebook_executed = [
        event
        for event in event_list
        if event.event_type == "notebook_execution" and _event_success(event)
    ]
    notebook_execution_attempts = [
        event for event in event_list if event.event_type == "notebook_execution"
    ]
    reuse_decisions = _unique_dataset_events(
        event
        for event in event_list
        if event.event_type == "reuse_decision"
        and event.would_use_for_analysis in {"yes", "maybe", "no"}
    )
    positive_reuse_decisions = [
        event
        for event in reuse_decisions
        if event.would_use_for_analysis in {"yes", "maybe"}
    ]
    useful_events = _unique_dataset_events(
        event
        for event in event_list
        if event.usefulness in {"useful", "partially_useful"}
    )
    novel_useful = [event for event in useful_events if event.known_before is False]

    session_first_search_seconds: list[float] = []
    attempt_to_success_seconds: list[float] = []
    sessions_with_useful_dataset = 0
    sessions_with_save = 0
    sessions_with_notebook_generation = 0
    sessions_completed = 0
    metadata_failure_sessions = 0
    metadata_recovered_sessions = 0
    sessions_without_search_attempt = 0

    for session_events in by_session.values():
        session_events = sorted(
            session_events,
            key=lambda event: _parse_timestamp(event.timestamp),
        )
        start = _parse_timestamp(session_events[0].timestamp)
        first_attempt = next(
            (event for event in session_events if event.event_type == "search_attempt"),
            None,
        )
        first_success = next(
            (
                event
                for event in session_events
                if event.event_type == "search_success" and _event_success(event)
            ),
            None,
        )
        if first_attempt is None:
            sessions_without_search_attempt += 1
        if first_success:
            delta = (_parse_timestamp(first_success.timestamp) - start).total_seconds()
            session_first_search_seconds.append(max(0.0, delta))
        if first_attempt is not None:
            first_success_after_attempt = next(
                (
                    event
                    for event in session_events
                    if event.event_type == "search_success"
                    and _event_success(event)
                    and _parse_timestamp(event.timestamp)
                    >= _parse_timestamp(first_attempt.timestamp)
                ),
                None,
            )
            if first_success_after_attempt is not None:
                delta = (
                    _parse_timestamp(first_success_after_attempt.timestamp)
                    - _parse_timestamp(first_attempt.timestamp)
                ).total_seconds()
                attempt_to_success_seconds.append(max(0.0, delta))

        if any(
            event.event_type == "dataset_saved"
            or event.usefulness in {"useful", "partially_useful"}
            for event in session_events
        ):
            sessions_with_useful_dataset += 1
        if any(event.event_type == "dataset_saved" for event in session_events):
            sessions_with_save += 1
        if any(
            event.event_type == "notebook_generated" and _event_success(event)
            for event in session_events
        ):
            sessions_with_notebook_generation += 1
        if any(
            event.event_type == "workflow_complete" and _event_success(event)
            for event in session_events
        ):
            sessions_completed += 1

        failure_indexes = [
            index
            for index, event in enumerate(session_events)
            if event.event_type == "metadata_failure"
        ]
        if failure_indexes:
            metadata_failure_sessions += 1
            if any(
                any(
                    subsequent.event_type == "search_success"
                    and _event_success(subsequent)
                    for subsequent in session_events[index + 1 :]
                )
                for index in failure_indexes
            ):
                metadata_recovered_sessions += 1

    total_sessions = len(by_session)

    return {
        "schema_version": 2,
        "sessions": total_sessions,
        "events": len(event_list),
        "metrics": {
            "search_success_rate": _ratio(len(search_successes), len(search_attempts)),
            "useful_dataset_discovery_rate": _ratio(
                sessions_with_useful_dataset,
                total_sessions,
            ),
            "save_rate_per_session": _ratio(sessions_with_save, total_sessions),
            "notebook_generation_rate_per_session": _ratio(
                sessions_with_notebook_generation,
                total_sessions,
            ),
            "saved_datasets_per_session": _mean_per_session(len(saved), total_sessions),
            "notebooks_generated_per_session": _mean_per_session(
                len(notebook_generated),
                total_sessions,
            ),
            "notebook_execution_success_rate": _ratio(
                len(notebook_executed),
                len(notebook_execution_attempts),
            ),
            "metadata_failure_recovery_rate": _ratio(
                metadata_recovered_sessions,
                metadata_failure_sessions,
            ),
            "positive_reuse_decision_rate": _ratio(
                len(positive_reuse_decisions),
                len(reuse_decisions),
            ),
            "novel_useful_discovery_rate": _ratio(
                len(novel_useful),
                len(useful_events),
            ),
            "workflow_completion_rate": _ratio(sessions_completed, total_sessions),
            "time_to_first_successful_search_seconds": _duration_summary(
                session_first_search_seconds
            ),
            "search_attempt_to_success_seconds": _duration_summary(
                attempt_to_success_seconds
            ),
        },
        "counts": {
            "search_attempts": len(search_attempts),
            "search_successes": len(search_successes),
            "saved_datasets": len(saved),
            "sessions_with_save": sessions_with_save,
            "notebooks_generated": len(notebook_generated),
            "sessions_with_notebook_generation": sessions_with_notebook_generation,
            "notebook_execution_attempts": len(notebook_execution_attempts),
            "notebook_execution_successes": len(notebook_executed),
            "reuse_decisions": len(reuse_decisions),
            "positive_reuse_decisions": len(positive_reuse_decisions),
            "metadata_failure_sessions": metadata_failure_sessions,
            "metadata_recovered_sessions": metadata_recovered_sessions,
            "sessions_completed": sessions_completed,
        },
        "data_quality": {
            "sessions_without_search_attempt": sessions_without_search_attempt,
            "deduplicates_dataset_level_ratings_by_session": True,
            "minimum_sample_size_enforced": False,
        },
        "interpretation": {
            "retrieval_metrics_are_separate": True,
            "gold_relevance_claim": False,
            "descriptive_not_inferential": True,
            "note": (
                "These metrics measure usability and downstream research workflow "
                "completion. They do not replace NDCG/MRR/qrels evaluation, establish "
                "causal product impact, or validate scientific conclusions."
            ),
        },
    }


def evaluate_adoption_file(path: Path) -> dict[str, Any]:
    return evaluate_adoption_events(load_adoption_events(path))
