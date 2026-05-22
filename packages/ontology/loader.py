"""Load ontology from YAML files."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from .models import Ontology, Task


DEFAULT_ONTOLOGY_PATH = Path(__file__).parent.parent.parent / "data" / "ontology"


def load_ontology(
    ontology_dir: Optional[Path] = None,
    task_file: str = "behavioral_task_ontology.yaml",
) -> Ontology:
    """
    Load the ontology from YAML files.

    Args:
        ontology_dir: Directory containing ontology YAML files.
        task_file: Name of the task ontology file.

    Returns:
        Loaded Ontology object.
    """
    if ontology_dir is None:
        ontology_dir = DEFAULT_ONTOLOGY_PATH

    ontology_dir = Path(ontology_dir)

    # Load tasks
    tasks: list[Task] = []
    task_path = ontology_dir / task_file
    if task_path.exists():
        with open(task_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data and "tasks" in data:
                for task_data in data["tasks"]:
                    tasks.append(Task(**task_data))

    return Ontology(tasks=tasks)


@lru_cache(maxsize=1)
def get_ontology() -> Ontology:
    """
    Get the cached ontology instance.

    Uses LRU cache to avoid reloading on every call.
    """
    return load_ontology()


def reload_ontology() -> Ontology:
    """Force reload the ontology, clearing cache."""
    get_ontology.cache_clear()
    return get_ontology()
