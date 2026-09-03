"""Versioned artifact manifests for inference-derived outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from neural_search.inference.schemas import RunManifest


class ArtifactManifest(BaseModel):
    """Lineage record for an inference-derived scientific artifact."""

    artifact_id: str
    artifact_type: str
    schema_version: str
    content_sha256: str
    path: str
    parent_artifact_ids: list[str] = Field(default_factory=list)
    inference_runs: list[RunManifest] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def write_json_artifact(
    path: Path,
    payload: Any,
    *,
    artifact_type: str,
    schema_version: str,
    inference_runs: list[RunManifest] | None = None,
    parent_artifact_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Write deterministic JSON plus a sidecar manifest carrying lineage and hashes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(rendered, encoding="utf-8")
    content_hash = hashlib.sha256(rendered.encode()).hexdigest()
    manifest = ArtifactManifest(
        artifact_id=f"{artifact_type}:{content_hash[:16]}",
        artifact_type=artifact_type,
        schema_version=schema_version,
        content_sha256=content_hash,
        path=str(path),
        parent_artifact_ids=parent_artifact_ids or [],
        inference_runs=inference_runs or [],
        metadata=metadata or {},
    )
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
