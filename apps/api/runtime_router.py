"""Read-only runtime/profile inspection HTTP transport."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from neural_search.services import RuntimeReadinessService

router = APIRouter(prefix="/api/runtime", tags=["runtime"])
_service = RuntimeReadinessService()


@router.get("/profiles")
async def profiles() -> dict[str, Any]:
    """List supported execution profiles and the active profile name."""

    return _service.profiles()


@router.get("/status")
async def runtime_status(
    profile: str | None = Query(default=None),
) -> dict[str, Any]:
    """Report dependency, artifact, and compatibility readiness."""

    try:
        return _service.status(profile)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/artifacts")
async def artifacts(
    artifact_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Inspect the artifact registry without reading artifact contents."""

    try:
        return _service.artifacts(artifact_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/capabilities")
async def capabilities(
    profile: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return researcher-facing capability availability for one profile."""

    try:
        return _service.capabilities(profile)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/bundles")
async def bundles() -> dict[str, Any]:
    """Return published bundle metadata plus local verified lock state."""

    try:
        return _service.bundles()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
