"""Application service for execution-profile and artifact readiness.

This module is deliberately independent of FastAPI so CLIs, notebooks, agents,
workers, and HTTP routers can share the same capability semantics.
"""

from __future__ import annotations

import os
from typing import Any

from neural_search.runtime import (
    PROFILES,
    artifact_status,
    list_artifacts,
    list_profiles,
    profile_status,
)


class RuntimeReadinessService:
    """Expose runtime capability state without mutating scientific artifacts."""

    def active_profile(self) -> str:
        configured = os.getenv("NEURAL_SEARCH_PROFILE", "demo").strip().lower()
        return configured if configured in PROFILES else "demo"

    def profiles(self) -> dict[str, Any]:
        return {
            "active_profile": self.active_profile(),
            "profiles": list_profiles(),
        }

    def status(self, profile: str | None = None) -> dict[str, Any]:
        selected = (profile or self.active_profile()).strip().lower()
        if selected not in PROFILES:
            raise ValueError(f"Unknown execution profile: {selected}")
        return profile_status(selected)

    def artifacts(self, artifact_id: str | None = None) -> dict[str, Any]:
        if artifact_id is not None:
            return {"artifacts": [artifact_status(artifact_id)]}
        return {
            "artifacts": [artifact_status(item["id"]) for item in list_artifacts()]
        }
