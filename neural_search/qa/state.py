"""Corpus QA state management for dataset cards and ingestion review."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from neural_search.cards import generate_dataset_card_json
from neural_search.ingestion.demo_seed import build_demo_seed

QA_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "qa"
QA_STATE_PATH = QA_DATA_DIR / "dataset_qa.json"

DatasetQAStatus = Literal[
    "unreviewed",
    "auto_generated",
    "needs_review",
    "reviewed",
    "trusted",
    "rejected",
]

QA_STATUSES: tuple[str, ...] = (
    "unreviewed",
    "auto_generated",
    "needs_review",
    "reviewed",
    "trusted",
    "rejected",
)
UNREVIEWED_STATUSES = {"unreviewed", "auto_generated", "needs_review"}
REVIEWED_STATUSES = {"reviewed", "trusted"}

QA_FIELD_DEFAULTS: dict[str, Any] = {
    "task_labels_verified": False,
    "modality_labels_verified": False,
    "behavior_labels_verified": False,
    "brain_regions_verified": False,
    "linked_papers_verified": False,
    "notebook_tested": False,
    "reviewer_notes": "",
}


def normalize_dataset_id(dataset: Mapping[str, Any] | Any) -> str:
    """Return the stable corpus-facing dataset identifier."""

    if isinstance(dataset, Mapping):
        return str(dataset.get("source_id") or dataset.get("id") or "")
    return str(getattr(dataset, "source_id", None) or getattr(dataset, "id", ""))


def load_qa_state(path: Path = QA_STATE_PATH) -> dict[str, dict[str, Any]]:
    """Load persisted QA state keyed by dataset ID."""

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"QA state root must be a mapping: {path}")
    return {
        str(dataset_id): _normalize_record(record)
        for dataset_id, record in payload.items()
        if isinstance(record, Mapping)
    }


def save_qa_state(state: Mapping[str, Mapping[str, Any]], path: Path = QA_STATE_PATH) -> None:
    """Persist QA state in a deterministic JSON format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        str(dataset_id): _normalize_record(record)
        for dataset_id, record in sorted(state.items())
    }
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_dataset_qa(
    dataset: Mapping[str, Any] | Any,
    state: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return QA fields for a dataset, merging persisted state with defaults."""

    dataset_id = normalize_dataset_id(dataset)
    metadata = _metadata(dataset)
    default_status = str(metadata.get("qa_status") or "auto_generated")
    if default_status not in QA_STATUSES:
        default_status = "needs_review"
    record = {
        "dataset_id": dataset_id,
        "qa_status": default_status,
        **QA_FIELD_DEFAULTS,
    }
    state_record = (state or load_qa_state()).get(dataset_id, {})
    record.update(_normalize_record(state_record, dataset_id=dataset_id, default_status=default_status))
    return record


def attach_qa_to_dataset(
    dataset: Mapping[str, Any],
    state: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a dataset dict with top-level QA fields for API/frontend consumers."""

    qa = get_dataset_qa(dataset, state)
    return {**dataset, **qa}


def attach_qa_to_card(
    card: Any,
    dataset: Mapping[str, Any] | Any,
    state: Mapping[str, Mapping[str, Any]] | None = None,
) -> Any:
    """Mutate a Pydantic dataset card with QA fields and return it."""

    qa = get_dataset_qa(dataset, state)
    for key, value in qa.items():
        if key == "dataset_id":
            continue
        if hasattr(card, key):
            setattr(card, key, value)
    return card


def update_dataset_status(
    dataset_id: str,
    status: DatasetQAStatus,
    *,
    notes: str | None = None,
    path: Path = QA_STATE_PATH,
) -> dict[str, Any]:
    """Set the QA status for a dataset and persist the review timestamp."""

    if status not in QA_STATUSES:
        raise ValueError(f"Unsupported QA status: {status}")
    state = load_qa_state(path)
    now = datetime.now(UTC).isoformat()
    existing = _normalize_record(state.get(dataset_id, {}), dataset_id=dataset_id)
    existing["qa_status"] = status
    existing["reviewed_at"] = now
    if notes is not None:
        existing["reviewer_notes"] = notes
    state[dataset_id] = existing
    save_qa_state(state, path)
    return existing


def update_dataset_qa_fields(
    dataset_id: str,
    updates: Mapping[str, Any],
    *,
    path: Path = QA_STATE_PATH,
) -> dict[str, Any]:
    """Persist reviewer-controlled QA fields for a dataset."""

    state = load_qa_state(path)
    existing = _normalize_record(state.get(dataset_id, {}), dataset_id=dataset_id)
    for key, value in updates.items():
        if key in QA_FIELD_DEFAULTS or key == "qa_status":
            existing[key] = value
    existing["updated_at"] = datetime.now(UTC).isoformat()
    state[dataset_id] = _normalize_record(existing, dataset_id=dataset_id)
    save_qa_state(state, path)
    return state[dataset_id]


def list_unreviewed_records() -> list[dict[str, Any]]:
    """Return seed records that are not accepted or rejected yet."""

    state = load_qa_state()
    records = []
    for record in build_demo_seed():
        dataset = attach_qa_to_dataset(record["dataset"], state)
        if dataset["qa_status"] in UNREVIEWED_STATUSES:
            records.append({**record, "dataset": dataset})
    return records


def reviewed_dataset_cards(statuses: set[str] | None = None) -> list[dict[str, Any]]:
    """Generate reviewed/trusted dataset cards enriched with QA fields."""

    accepted_statuses = statuses or REVIEWED_STATUSES
    state = load_qa_state()
    cards: list[dict[str, Any]] = []
    for record in build_demo_seed():
        dataset = attach_qa_to_dataset(record["dataset"], state)
        if dataset["qa_status"] not in accepted_statuses:
            continue
        card = generate_dataset_card_json(dataset, record["extraction"], record.get("papers", []))
        attach_qa_to_card(card, dataset, state)
        cards.append(
            {
                "dataset": dataset,
                "card": card.model_dump(mode="json"),
                "papers": record.get("papers", []),
            }
        )
    return cards


def qa_counts(records: list[dict[str, Any]] | None = None) -> dict[str, int]:
    """Count QA statuses across a record list or the demo corpus."""

    state = load_qa_state()
    corpus = records or build_demo_seed()
    counts = dict.fromkeys(QA_STATUSES, 0)
    for record in corpus:
        dataset = record.get("dataset", record)
        status = get_dataset_qa(dataset, state)["qa_status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def top_demo_ready(records: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    """Rank trusted/reviewed datasets that are strongest candidates for demos."""

    state = load_qa_state()
    ranked: list[dict[str, Any]] = []
    for record in records:
        dataset = attach_qa_to_dataset(record.get("dataset", record), state)
        if dataset["qa_status"] not in REVIEWED_STATUSES:
            continue
        card = generate_dataset_card_json(dataset, record.get("extraction"), record.get("papers", []))
        status_bonus = 10 if dataset["qa_status"] == "trusted" else 0
        notebook_bonus = 5 if dataset.get("notebook_tested") else 0
        ranked.append(
            {
                "source_id": dataset.get("source_id", dataset.get("id", "unknown")),
                "title": dataset.get("title", "Untitled"),
                "source": dataset.get("source", "unknown"),
                "qa_status": dataset["qa_status"],
                "score": min(100, card.analysis_readiness.score + status_bonus + notebook_bonus),
                "analysis_readiness_score": card.analysis_readiness.score,
                "notebook_tested": dataset.get("notebook_tested", False),
            }
        )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]


def _normalize_record(
    record: Mapping[str, Any],
    *,
    dataset_id: str | None = None,
    default_status: str = "auto_generated",
) -> dict[str, Any]:
    normalized = {
        "dataset_id": str(record.get("dataset_id") or dataset_id or ""),
        "qa_status": str(record.get("qa_status") or default_status),
        **QA_FIELD_DEFAULTS,
    }
    if normalized["qa_status"] not in QA_STATUSES:
        normalized["qa_status"] = "needs_review"
    for key in QA_FIELD_DEFAULTS:
        if key == "reviewer_notes":
            normalized[key] = str(record.get(key, QA_FIELD_DEFAULTS[key]) or "")
        else:
            normalized[key] = bool(record.get(key, QA_FIELD_DEFAULTS[key]))
    for key in ["reviewed_at", "updated_at"]:
        if record.get(key):
            normalized[key] = str(record[key])
    return normalized


def _metadata(dataset: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(dataset, Mapping):
        metadata = dataset.get("metadata_json", {})
    else:
        metadata = getattr(dataset, "metadata_json", {})
    return metadata if isinstance(metadata, Mapping) else {}
