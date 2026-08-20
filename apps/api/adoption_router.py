"""Explicit, opt-in researcher-workflow telemetry for usability studies.

Events are local JSONL research artifacts. They are not analytics beacons. The
write/report endpoints are disabled unless ``NEURAL_SEARCH_ADOPTION_STUDY=1``
is set by the operator, so a public Neural Search deployment cannot be used to
silently poison or grow local benchmark files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from neural_search.evaluation.adoption import AdoptionEvent, evaluate_adoption_file

router = APIRouter(prefix="/api/adoption", tags=["adoption"])
_DEFAULT_EVENTS_PATH = Path("artifacts/frontend/adoption_events.jsonl")
_DEFAULT_MAX_EVENT_BYTES = 64 * 1024
_DEFAULT_MAX_FILE_BYTES = 50 * 1024 * 1024
_WRITE_LOCK = Lock()


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _study_enabled() -> bool:
    return _truthy(os.getenv("NEURAL_SEARCH_ADOPTION_STUDY"))


def _require_study_enabled() -> None:
    if not _study_enabled():
        raise HTTPException(
            status_code=403,
            detail=(
                "Adoption-study telemetry is disabled. The deployment operator must "
                "explicitly set NEURAL_SEARCH_ADOPTION_STUDY=1 to collect local study events."
            ),
        )


def _events_path() -> Path:
    configured = os.getenv("NEURAL_SEARCH_ADOPTION_EVENTS")
    return Path(configured) if configured else _DEFAULT_EVENTS_PATH


def _max_event_bytes() -> int:
    return max(
        1024,
        int(os.getenv("NEURAL_SEARCH_ADOPTION_MAX_EVENT_BYTES") or _DEFAULT_MAX_EVENT_BYTES),
    )


def _max_file_bytes() -> int:
    return max(
        1024 * 1024,
        int(os.getenv("NEURAL_SEARCH_ADOPTION_MAX_FILE_BYTES") or _DEFAULT_MAX_FILE_BYTES),
    )


class AdoptionEventRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=256)
    timestamp: str = Field(min_length=1, max_length=128)
    event_type: str = Field(min_length=1, max_length=128)
    success: bool | None = None
    dataset_id: str | None = Field(default=None, max_length=512)
    usefulness: str | None = Field(default=None, max_length=64)
    would_use_for_analysis: str | None = Field(default=None, max_length=64)
    known_before: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.get("/status")
async def adoption_status() -> dict[str, Any]:
    """Expose whether this deployment opted into local usability-study collection."""

    return {
        "enabled": _study_enabled(),
        "storage": "local_jsonl" if _study_enabled() else "disabled",
        "automatic_analytics": False,
    }


@router.post("/events")
async def record_event(request: AdoptionEventRequest) -> dict[str, Any]:
    """Append one explicit usability-study event to the local event stream."""

    _require_study_enabled()
    try:
        event = AdoptionEvent.from_dict(request.model_dump(mode="json"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    serialized = json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":"))
    encoded = (serialized + "\n").encode("utf-8")
    if len(encoded) > _max_event_bytes():
        raise HTTPException(
            status_code=413,
            detail="Adoption event exceeds the configured per-event size limit.",
        )

    path = _events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        current_size = path.stat().st_size if path.exists() else 0
        if current_size + len(encoded) > _max_file_bytes():
            raise HTTPException(
                status_code=507,
                detail=(
                    "Adoption event store reached its configured size limit. Rotate or "
                    "archive the study file before recording additional events."
                ),
            )
        with path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    return {"recorded": True, "event": event.to_dict()}


@router.get("/report")
async def adoption_report() -> dict[str, Any]:
    """Compute current local external-user workflow metrics."""

    _require_study_enabled()
    path = _events_path()
    try:
        return {"events_path": str(path), **evaluate_adoption_file(path)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
