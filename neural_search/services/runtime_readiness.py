"""Application service for execution-profile, artifact, and capability readiness."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from neural_search.runtime import (
    PROFILES,
    artifact_status,
    capability_status,
    list_artifacts,
    list_profiles,
    load_release_index,
    lock_snapshot,
    profile_status,
    verify_locked_artifacts,
)
from neural_search.runtime.bundles import DEFAULT_INDEX_PATH


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

    def capabilities(self, profile: str | None = None) -> dict[str, Any]:
        selected = (profile or self.active_profile()).strip().lower()
        if selected not in PROFILES:
            raise ValueError(f"Unknown execution profile: {selected}")
        return capability_status(selected)

    def bundles(self) -> dict[str, Any]:
        index_path = Path(os.getenv("NEURAL_SEARCH_ARTIFACT_INDEX") or DEFAULT_INDEX_PATH)
        index = load_release_index(index_path, allow_local_files=True)
        return {
            "index_path": str(index_path),
            "available_bundles": [
                {
                    "name": entry.name,
                    "version": entry.version,
                    "ref": entry.ref,
                    "manifest_url": entry.manifest_url,
                    "compatibility_group": entry.compatibility_group,
                    "deprecated": entry.deprecated,
                }
                for entry in index.bundles
            ],
            "lock": lock_snapshot(),
            "verification": verify_locked_artifacts(),
        }
