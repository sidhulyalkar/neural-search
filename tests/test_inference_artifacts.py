"""Tests for inference-derived artifact lineage manifests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from neural_search.inference.artifacts import write_json_artifact
from neural_search.inference.schemas import (
    InferenceCapability,
    InferenceMessage,
    InferenceRequest,
    RunManifest,
)


def test_write_json_artifact_creates_checksum_lineage_sidecar(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    run = RunManifest.build(
        provider="nim",
        model="model-a",
        model_profile="scientific_extraction",
        capability=InferenceCapability.STRUCTURED_EXTRACTION,
        request=InferenceRequest(
            messages=[InferenceMessage(role="user", content="extract")],
            capability=InferenceCapability.STRUCTURED_EXTRACTION,
        ),
        started_at=now,
        completed_at=now,
    )
    output = tmp_path / "findings.json"
    manifest = write_json_artifact(
        output,
        [{"finding": "example"}],
        artifact_type="scientific_findings",
        schema_version="1.0",
        inference_runs=[run],
        parent_artifact_ids=["papers:abc123"],
    )

    sidecar = output.with_suffix(".json.manifest.json")
    assert output.is_file()
    assert sidecar.is_file()
    parsed = json.loads(sidecar.read_text(encoding="utf-8"))
    assert parsed["artifact_id"] == manifest.artifact_id
    assert parsed["content_sha256"] == manifest.content_sha256
    assert parsed["parent_artifact_ids"] == ["papers:abc123"]
    assert parsed["inference_runs"][0]["provider"] == "nim"
