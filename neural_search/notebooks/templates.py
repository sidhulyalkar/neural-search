"""Notebook template loading and dataset-feature matching."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "notebooks" / "templates.yaml"
)


def _get_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _values(obj: Any, name: str) -> set[str]:
    value = _get_value(obj, name, []) or []
    if isinstance(value, str):
        return {value.casefold(), value.replace("_", " ").casefold()}
    values = set()
    for item in value:
        text = str(item)
        values.add(text.casefold())
        values.add(text.replace("_", " ").casefold())
    return values


def _normalize_list(values: list[Any]) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        text = str(value)
        normalized.add(text.casefold())
        normalized.add(text.replace("_", " ").casefold())
    return normalized


@lru_cache(maxsize=4)
def load_notebook_templates(path: str | Path = DEFAULT_TEMPLATE_PATH) -> list[dict[str, Any]]:
    """Load notebook templates from YAML."""

    template_path = Path(path)
    if not template_path.exists():
        return []
    with template_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    templates = payload.get("templates", [])
    if not isinstance(templates, list):
        raise ValueError(f"Template file must contain a templates list: {template_path}")
    return [template for template in templates if isinstance(template, dict) and template.get("id")]


def get_notebook_template(template_id: str) -> dict[str, Any] | None:
    """Return a notebook template by ID."""

    for template in load_notebook_templates():
        if template.get("id") == template_id:
            return template
    return None


def evaluate_template_for_dataset(
    template: dict[str, Any],
    dataset: Any,
) -> dict[str, Any]:
    """Evaluate whether a dataset satisfies a template's required features."""

    required = template.get("required_features", {}) or {}
    missing: list[str] = []

    modalities = _values(dataset, "modalities")
    tasks = _values(dataset, "tasks")
    standards = _values(dataset, "data_standards")

    if required.get("modalities_any"):
        expected = _normalize_list(required["modalities_any"])
        if not (modalities & expected):
            missing.append(
                "requires one modality: " + ", ".join(required["modalities_any"])
            )

    if required.get("tasks_any"):
        expected = _normalize_list(required["tasks_any"])
        if not (tasks & expected):
            missing.append("requires one task: " + ", ".join(required["tasks_any"]))

    if required.get("data_standards_any"):
        expected = _normalize_list(required["data_standards_any"])
        if not (standards & expected):
            missing.append(
                "requires one data standard: " + ", ".join(required["data_standards_any"])
            )

    for flag in ["has_trials", "has_behavior", "has_raw_data", "has_processed_data"]:
        if required.get(flag) is True and not bool(_get_value(dataset, flag, False)):
            missing.append(f"requires {flag}")

    return {
        "id": template["id"],
        "title": template.get("title", template["id"]),
        "description": template.get("description", ""),
        "required_features": required,
        "available": not missing,
        "missing_requirements": missing,
        "recipes": template.get("recipes", []),
    }


def available_templates_for_dataset(dataset: Any) -> list[dict[str, Any]]:
    """Return all templates annotated with availability for one dataset."""

    return [
        evaluate_template_for_dataset(template, dataset)
        for template in load_notebook_templates()
    ]
