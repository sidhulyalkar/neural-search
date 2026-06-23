"""Cognitive Atlas crosswalk for Neural Search behavioral task IDs.

Loads ``data/ontology/task_atlas.yaml``, which maps Neural Search task
ontology IDs to Cognitive Atlas (cognitiveatlas.org) task/term IDs.

Data quality note: of the 87 source mappings, 65 point to a single
Cognitive Atlas concept (``trm_4f244f46ebf58``) whose cached record has
``name=""`` and ``definition_text="None"`` — an empty/placeholder concept
left over from a substring-match fallback in whatever script produced
task_atlas.yaml, not a real semantic match. Propagating those into the
graph would silently attach the *same* meaningless crosswalk ID to a dozen
unrelated tasks. This module filters them out: only mappings with a
non-empty ``cogat_label`` are exposed, leaving genuine coverage at roughly
22 of 87 tasks (~25%), not the 87/87 the file's own ``_meta`` block claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TASK_ATLAS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "ontology" / "task_atlas.yaml"
)


@dataclass(frozen=True)
class CogAtlasMatch:
    cogat_id: str
    cogat_label: str
    match_type: str


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def _task_atlas_index_cached() -> dict[str, CogAtlasMatch]:
    """Build our_task_id -> CogAtlasMatch, excluding empty/placeholder matches."""
    raw = _load_yaml(DEFAULT_TASK_ATLAS_PATH)
    index: dict[str, CogAtlasMatch] = {}
    for entry in raw.get("task_mappings", []) or []:
        label = (entry.get("cogat_label") or "").strip()
        cogat_id = entry.get("cogat_id")
        our_id = entry.get("our_id")
        if not label or not cogat_id or not our_id:
            continue  # placeholder/empty Cognitive Atlas concept — not a real match
        index[our_id] = CogAtlasMatch(
            cogat_id=cogat_id,
            cogat_label=label,
            match_type=entry.get("match_type") or "unknown",
        )
    return index


def get_cogat_match(task_id: str) -> CogAtlasMatch | None:
    """Return the validated Cognitive Atlas crosswalk for a Neural Search task ID, or None."""
    return _task_atlas_index_cached().get(task_id)


def get_cogat_coverage() -> dict[str, int]:
    """Return validated-mapping coverage stats for the task_atlas.yaml crosswalk."""
    raw = _load_yaml(DEFAULT_TASK_ATLAS_PATH)
    return {
        "total_tasks": len(raw.get("task_mappings", []) or []),
        "validated_matches": len(_task_atlas_index_cached()),
    }
