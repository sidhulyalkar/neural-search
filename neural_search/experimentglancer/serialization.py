"""Deterministic hashing and JSON serialization for ExperimentGlancer scenes.

Scene generation must be reproducible: the same dataset + query + evidence
should always yield the same ``scene_id`` and the same JSON bytes, so a
shared URL keeps pointing at the same scene.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from neural_search.experimentglancer.schemas import ExperimentGlancerSceneV1


def stable_hash(value: Any, *, length: int = 12) -> str:
    """Return a short, deterministic hash of a JSON-serializable value."""

    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:length]


def make_scene_id(
    *,
    dataset_id: str,
    query: str = "",
    anchor_hint: Any = "",
    requested_layers: Any = (),
    affordance_ids: Any = (),
    deep_introspection: bool = False,
) -> str:
    """Deterministic scene id derived from every input that can change the
    resulting scene contract. Two calls with the same dataset but different
    requested layers, affordances, anchor hint, or introspection depth must
    never collapse onto the same id -- otherwise a shared URL could resolve
    to a scene the requester didn't ask for."""

    safe_dataset_id = str(dataset_id).replace(":", "_").replace("/", "_")
    suffix = stable_hash(
        {
            "dataset_id": dataset_id,
            "query": query,
            "anchor_hint": anchor_hint,
            "requested_layers": sorted(requested_layers),
            "affordance_ids": sorted(affordance_ids),
            "deep_introspection": deep_introspection,
        }
    )
    return f"eg_{safe_dataset_id}_{suffix}"


def to_stable_json(scene: ExperimentGlancerSceneV1) -> str:
    """Serialize a scene to canonical, deterministically ordered JSON."""

    return json.dumps(
        scene.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
