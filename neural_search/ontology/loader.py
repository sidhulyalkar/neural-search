"""Ontology loading and startup validation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from neural_search.ontology.models import (
    REQUIRED_TASK_FIELDS,
    BrainRegion,
    Ontology,
    RecordingScale,
    Task,
)

DEFAULT_ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "ontology"
    / "behavioral_task_ontology.yaml"
)

DEFAULT_BRAIN_REGIONS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "ontology" / "brain_regions.yaml"
)

DEFAULT_RECORDING_SCALES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "ontology" / "recording_scales.yaml"
)


class OntologyValidationError(ValueError):
    """Raised when an ontology file fails schema validation."""


def _load_yaml(path: str | Path) -> dict[str, Any]:
    ontology_path = Path(path)
    if not ontology_path.exists():
        raise OntologyValidationError(f"Ontology file not found: {ontology_path}")
    with ontology_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise OntologyValidationError("Ontology root must be a mapping")
    return raw


def validate_ontology(path: str | Path = DEFAULT_ONTOLOGY_PATH) -> Ontology:
    """Validate an ontology YAML file and return the parsed ontology."""

    raw = _load_yaml(path)
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise OntologyValidationError("Ontology must define a non-empty tasks list")

    errors: list[str] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            errors.append(f"tasks[{index}] must be a mapping")
            continue
        missing = sorted(REQUIRED_TASK_FIELDS - set(task))
        if missing:
            task_id = task.get("id", f"index {index}")
            errors.append(f"task {task_id} missing required fields: {missing}")

    if errors:
        raise OntologyValidationError("; ".join(errors))

    try:
        return Ontology.model_validate(raw)
    except ValidationError as exc:
        raise OntologyValidationError(str(exc)) from exc


@lru_cache(maxsize=8)
def _load_ontology_cached(path_string: str) -> Ontology:
    return validate_ontology(path_string)


def load_ontology(path: str | Path = DEFAULT_ONTOLOGY_PATH) -> Ontology:
    """Load and validate ontology YAML."""

    return _load_ontology_cached(str(Path(path)))


def get_ontology() -> Ontology:
    """Return the default ontology, validated on first startup access."""

    return load_ontology(DEFAULT_ONTOLOGY_PATH)


def reload_ontology(path: str | Path = DEFAULT_ONTOLOGY_PATH) -> Ontology:
    _load_ontology_cached.cache_clear()
    return load_ontology(path)


def get_all_tasks() -> list[Task]:
    return get_ontology().tasks


def get_task_by_id(task_id: str) -> Task | None:
    return get_ontology().task_by_id.get(task_id)


@lru_cache(maxsize=8)
def _load_brain_regions_cached(path_string: str) -> tuple[BrainRegion, ...]:
    raw = _load_yaml(path_string)
    entries = raw.get("brain_regions")
    if not isinstance(entries, list):
        raise OntologyValidationError("Brain-region ontology must define brain_regions list")
    try:
        return tuple(BrainRegion.model_validate(entry) for entry in entries)
    except ValidationError as exc:
        raise OntologyValidationError(str(exc)) from exc


def load_brain_regions(
    path: str | Path = DEFAULT_BRAIN_REGIONS_PATH,
) -> tuple[BrainRegion, ...]:
    """Load search-oriented brain region aliases and parent links."""

    return _load_brain_regions_cached(str(Path(path)))


def get_brain_regions() -> tuple[BrainRegion, ...]:
    """Return the default brain-region ontology."""

    return load_brain_regions(DEFAULT_BRAIN_REGIONS_PATH)


@lru_cache(maxsize=8)
def _load_recording_scales_cached(path_string: str) -> tuple[RecordingScale, ...]:
    raw = _load_yaml(path_string)
    entries = raw.get("recording_scales")
    if not isinstance(entries, list):
        raise OntologyValidationError("Recording-scale ontology must define recording_scales list")
    try:
        return tuple(RecordingScale.model_validate(entry) for entry in entries)
    except ValidationError as exc:
        raise OntologyValidationError(str(exc)) from exc


def load_recording_scales(
    path: str | Path = DEFAULT_RECORDING_SCALES_PATH,
) -> tuple[RecordingScale, ...]:
    """Load search-oriented recording/sampling scale categories."""

    return _load_recording_scales_cached(str(Path(path)))


def get_recording_scales() -> tuple[RecordingScale, ...]:
    """Return the default recording-scale ontology."""

    return load_recording_scales(DEFAULT_RECORDING_SCALES_PATH)


# ── Atlas-ref helpers ──────────────────────────────────────────────────────

def _build_region_index(regions: tuple[BrainRegion, ...]) -> dict[str, BrainRegion]:
    return {r.id: r for r in regions}


@lru_cache(maxsize=1)
def _region_index_cached() -> dict[str, BrainRegion]:
    return _build_region_index(get_brain_regions())


def get_region_atlas_refs(region_id: str) -> dict[str, str]:
    """Return the atlas_refs mapping for a canonical region ID, or {} if unknown."""
    region = _region_index_cached().get(region_id)
    return dict(region.atlas_refs) if region else {}


def get_region_uberon_id(region_id: str) -> str | None:
    """Return the UBERON ID for a canonical region ID, or None."""
    return get_region_atlas_refs(region_id).get("uberon")


def get_region_allen_ccf_id(region_id: str) -> str | None:
    """Return the Allen CCF v3 mouse structure ID for a canonical region ID, or None."""
    return get_region_atlas_refs(region_id).get("allen_ccf_mouse")


def get_regions_by_uberon(uberon_id: str) -> list[BrainRegion]:
    """Return all canonical regions that map to a given UBERON ID."""
    return [
        r for r in get_brain_regions()
        if r.atlas_refs.get("uberon") == uberon_id
    ]


def get_regions_by_allen_ccf(structure_id: str) -> list[BrainRegion]:
    """Return all canonical regions that map to a given Allen CCF structure ID."""
    return [
        r for r in get_brain_regions()
        if r.atlas_refs.get("allen_ccf_mouse") == structure_id
    ]
