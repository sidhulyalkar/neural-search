"""HTTP transport for evidence-aware dataset reanalysis planning."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from neural_search.services import ReanalysisPlanningService

router = APIRouter(prefix="/api/reanalysis", tags=["reanalysis"])
_service = ReanalysisPlanningService()


@router.get("/{dataset_id}")
async def reanalysis_plan(
    dataset_id: str,
    limit: int = Query(default=12, ge=1, le=30),
) -> dict[str, Any]:
    """Return feasible methods, blockers, precedents, and evidence for a dataset."""

    try:
        return _service.plan(dataset_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
