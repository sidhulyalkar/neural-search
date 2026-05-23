"""Starter notebook generation and validation."""

from importlib import import_module
from typing import Any

from neural_search.notebooks.generator import generate_nwb_starter_notebook
from neural_search.notebooks.templates import (
    available_templates_for_dataset,
    evaluate_template_for_dataset,
    get_notebook_template,
    load_notebook_templates,
)


def __getattr__(name: str) -> Any:
    if name == "validate_notebook_structure":
        module = import_module("neural_search.notebooks.validate")
        return module.validate_notebook_structure
    raise AttributeError(f"module 'neural_search.notebooks' has no attribute {name!r}")

__all__ = [
    "available_templates_for_dataset",
    "evaluate_template_for_dataset",
    "generate_nwb_starter_notebook",
    "get_notebook_template",
    "load_notebook_templates",
    "validate_notebook_structure",
]
