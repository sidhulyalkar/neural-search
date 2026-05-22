"""Ontology loading and startup validation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from neural_search.ontology.models import Ontology, REQUIRED_TASK_FIELDS


DEFAULT_ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "ontology"
    / "behavioral_task_ontology.yaml"
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


def get_all_tasks():
    return get_ontology().tasks


def get_task_by_id(task_id: str):
    return get_ontology().task_by_id.get(task_id)

