"""Researcher-workflow telemetry for explicit usability studies.

Events are local JSONL research artifacts. They are not analytics beacons and
are only created when this endpoint is called by a client participating in an
adoption/usability workflow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from neural_search.evaluation.adoption import (
    AdoptionEvent,
    evaluate_adoption_file,
)

router = APIRouter(prefix="/api/adoption", tags=["adoption"])
_DEFAULT_EVENTS_PATH = Path("artifacts/frontend/adoption_events.jsonl")
_WRITE_LOCK = Lock()


def _events_path() -> Path:
    configured = os.getenv("NEURAL_SEARCH_ADOPTION_EVENTS")
    return Path(configured) if configured else _DEFAULT_EVENTS_PATH


class AdoptionEventRequest(BaseModel):
    session_id: str
    timestamp: str
    event_type: str
    success: bool | None = None
    dataset_id: str | None = None
    usefulness: str | None = None
    would_use_for_analysis: str | None = None
    known_before: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.post("/events")
async def record_event(request: AdoptionEventRequest) -> dict[str, Any]:
    """Append one explicit usability-study event to the local event stream."""

    try:
        event = AdoptionEvent.from_dict(request.model_dump(mode="json"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    path = _events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
    return {"recorded": True, "event": event.to_dict()}


@router.get("/report")
async def adoption_report() -> dict[str, Any]:
    """Compute current local external-user workflow metrics."""

    path = _events_path()
    try:
        return {"events_path": str(path), **evaluate_adoption_file(path)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
