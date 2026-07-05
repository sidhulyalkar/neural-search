"""Disk persistence for generated ExperimentGlancer scenes.

Scenes are cheap to regenerate from a dataset + query, but a shared URL must
keep resolving even after the in-memory cache evicts an entry or the API
process restarts. Every scene is written once (by ``scene_id``, which is
already a deterministic hash — see ``serialization.make_scene_id``) under
``artifacts/experimentglancer/scenes/`` so ``GET /scenes/{scene_id}`` can
rehydrate it later without re-running retrieval.
"""

from __future__ import annotations

import re
from pathlib import Path

from neural_search.experimentglancer.schemas import ExperimentGlancerSceneV1

SCENES_DIR = Path("artifacts/experimentglancer/scenes")

_VALID_SCENE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _scene_path(scene_id: str) -> Path | None:
    """Return the on-disk path for ``scene_id``, or ``None`` if it isn't a
    well-formed id (guards against path traversal via the URL parameter)."""

    if not scene_id or not _VALID_SCENE_ID.match(scene_id):
        return None
    return SCENES_DIR / f"{scene_id}.json"


def save_scene(scene: ExperimentGlancerSceneV1) -> None:
    path = _scene_path(scene.scene_id)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(scene.model_dump_json(), encoding="utf-8")
    tmp_path.replace(path)


def load_scene(scene_id: str) -> ExperimentGlancerSceneV1 | None:
    path = _scene_path(scene_id)
    if path is None or not path.exists():
        return None
    return ExperimentGlancerSceneV1.model_validate_json(path.read_text(encoding="utf-8"))
