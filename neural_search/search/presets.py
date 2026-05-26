"""Retrieval configuration presets for different environments."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PRESETS_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "retrieval_presets.yaml"

_PRESETS_CACHE: dict[str, Any] | None = None


def _load_presets() -> dict[str, Any]:
    global _PRESETS_CACHE
    if _PRESETS_CACHE is None:
        if PRESETS_PATH.exists():
            with open(PRESETS_PATH, encoding="utf-8") as f:
                _PRESETS_CACHE = yaml.safe_load(f) or {}
        else:
            _PRESETS_CACHE = {}
    return _PRESETS_CACHE


def list_presets() -> list[str]:
    """Return available preset names."""
    presets = _load_presets()
    return list(presets.get("presets", {}).keys())


def load_preset(name: str) -> dict[str, Any]:
    """Load a named retrieval configuration preset.

    Args:
        name: Preset name (ci, local, exploratory, benchmark)

    Returns:
        Retrieval configuration dictionary

    Raises:
        ValueError: If preset name not found
    """
    presets = _load_presets()
    preset_configs = presets.get("presets", {})

    if name not in preset_configs:
        available = ", ".join(preset_configs.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    return dict(preset_configs[name])


def get_preset_description(name: str) -> str:
    """Get the description for a preset."""
    preset = load_preset(name)
    return preset.get("description", "")


def merge_with_preset(
    base_config: Mapping[str, Any] | None,
    preset_name: str,
) -> dict[str, Any]:
    """Merge a base config with a preset, with base taking precedence.

    Args:
        base_config: User-provided config overrides
        preset_name: Name of preset to use as base

    Returns:
        Merged configuration dictionary
    """
    preset = load_preset(preset_name)

    if not base_config:
        return preset

    result = dict(preset)
    for key, value in base_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value

    return result
