"""Runtime health semantics for distributed lineage relationships."""

from __future__ import annotations

from typing import Any

from neural_search.runtime import catalog


def compatibility_status(artifact_ids: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Treat an absent parent as unverifiable when the child declares its lineage ID."""

    raw = catalog.compatibility_status(artifact_ids)
    unknown = list(raw["unknown"])
    incompatible: list[dict[str, Any]] = []
    for item in raw["incompatible"]:
        artifact_id = str(item["artifact_id"])
        lineage = catalog.artifact_status(artifact_id).get("lineage") or {}
        declared = dict(lineage.get("derived_from") or {})
        remaining: list[str] = []
        for issue in list(item.get("issues") or []):
            if issue.startswith("parent_missing:"):
                parent_id = issue.partition(":")[2]
                if declared.get(parent_id):
                    unknown.append(
                        {
                            "artifact_id": artifact_id,
                            "reason": f"parent_not_local:{parent_id}",
                            "expected_parent_lineage": declared[parent_id],
                        }
                    )
                    continue
            remaining.append(issue)
        if remaining:
            incompatible.append({**item, "issues": remaining})
    state = "incompatible" if incompatible else ("unknown" if unknown else "compatible")
    return {
        **raw,
        "state": state,
        "compatible": not incompatible,
        "unknown": unknown,
        "incompatible": incompatible,
    }


def profile_status(name: str) -> dict[str, Any]:
    base = catalog.profile_status(name)
    profile = catalog.PROFILES[name]
    ids = list(dict.fromkeys(profile.required_artifacts + profile.recommended_artifacts + profile.produced_artifacts))
    compatibility = compatibility_status(ids)
    required_incompatible = {
        item["artifact_id"]
        for item in compatibility["incompatible"]
        if item["artifact_id"] in profile.required_artifacts
    }
    ready = bool(base["dependencies_ready"] and base["required_artifacts_ready"] and not required_incompatible)
    health = "unhealthy" if not ready else (
        "degraded"
        if base.get("recommended_missing") or compatibility["state"] == "unknown"
        else "ready"
    )
    return {**base, "ready": ready, "health": health, "compatibility": compatibility}


def build_reproducibility_manifest(
    name: str,
    *,
    checksum_limit_bytes: int = 50 * 1024 * 1024,
) -> dict[str, Any]:
    manifest = catalog.build_reproducibility_manifest(
        name,
        checksum_limit_bytes=checksum_limit_bytes,
    )
    status = profile_status(name)
    return {
        **manifest,
        "profile_ready": status["ready"],
        "profile_health": status["health"],
        "compatibility": status["compatibility"],
    }
