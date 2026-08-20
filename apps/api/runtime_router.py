"""Runtime/profile inspection endpoints.

These endpoints deliberately expose only environment and artifact readiness.
They never mutate artifacts, rebuild indexes, or disclose secret values.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from neural_search.runtime import (
    PROFILES,
    artifact_status,
    list_artifacts,
    list_profiles,
    profile_status,
)

router = APIRouter(prefix="/api/runtime", tags=["runtime"])


def _default_profile() -> str:
    configured = os.getenv("NEURAL_SEARCH_PROFILE", "demo").strip().lower()
    return configured if configured in PROFILES else "demo"


@router.get("/profiles")
async def profiles() -> dict[str, Any]:
    """List supported execution profiles and the active profile name."""

    return {
        "active_profile": _default_profile(),
        "profiles": list_profiles(),
    }


@router.get("/status")
async def runtime_status(
    profile: str | None = Query(default=None),
) -> dict[str, Any]:
    """Report dependency and artifact readiness for one execution profile."""

    selected = (profile or _default_profile()).strip().lower()
    if selected not in PROFILES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown execution profile: {selected}",
        )
    return profile_status(selected)


@router.get("/artifacts")
async def artifacts(
    artifact_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Inspect the artifact registry without reading artifact contents."""

    if artifact_id is not None:
        try:
            return {"artifacts": [artifact_status(artifact_id)]}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "artifacts": [artifact_status(item["id"]) for item in list_artifacts()]
    }
