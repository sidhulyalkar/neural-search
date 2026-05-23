"""Data-driven analysis recipe loading and matching."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from neural_search.ontology import normalize_text


DEFAULT_RECIPE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "recipes" / "analysis_recipes.yaml"
)


def _normalize_id(value: str) -> str:
    return normalize_text(value).replace(" ", "_")


@lru_cache(maxsize=4)
def load_analysis_recipes(path: str | Path = DEFAULT_RECIPE_PATH) -> list[dict[str, Any]]:
    """Load analysis recipes from YAML."""

    recipe_path = Path(path)
    if not recipe_path.exists():
        return []
    with recipe_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    recipes = payload.get("recipes", [])
    if not isinstance(recipes, list):
        raise ValueError(f"Recipe file must contain a recipes list: {recipe_path}")
    normalized: list[dict[str, Any]] = []
    for recipe in recipes:
        if not isinstance(recipe, dict):
            continue
        recipe_id = str(recipe.get("id", "")).strip()
        task_ids = [str(task) for task in recipe.get("task_ids", [])]
        if not recipe_id or not task_ids:
            continue
        normalized.append(
            {
                **recipe,
                "id": recipe_id,
                "task_ids": task_ids,
                "match_task_ids": [_normalize_id(task) for task in task_ids],
            }
        )
    return normalized


def get_recipe(recipe_id: str, path: str | Path = DEFAULT_RECIPE_PATH) -> dict[str, Any] | None:
    """Return one recipe by ID."""

    for recipe in load_analysis_recipes(path):
        if recipe["id"] == recipe_id:
            return recipe
    return None


def match_recipes_for_tasks(
    task_ids: list[str] | set[str] | tuple[str, ...],
    path: str | Path = DEFAULT_RECIPE_PATH,
) -> list[dict[str, Any]]:
    """Match recipes to ontology task IDs."""

    normalized_tasks = {_normalize_id(task) for task in task_ids}
    matches = []
    for recipe in load_analysis_recipes(path):
        overlap = normalized_tasks & set(recipe.get("match_task_ids", []))
        if not overlap:
            continue
        matches.append(
            {
                **recipe,
                "match_score": len(overlap) / max(len(recipe.get("match_task_ids", [])), 1),
                "matched_tasks": sorted(overlap),
            }
        )
    return sorted(matches, key=lambda item: (item["match_score"], item["id"]), reverse=True)
