"""Shared artifact helpers for Concept Memory reproducibility."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DETERMINISTIC_ENV_VAR = "NEURAL_SEARCH_DETERMINISTIC_ARTIFACTS"
DETERMINISTIC_TIMESTAMP = "1970-01-01T00:00:00+00:00"

_VOLATILE_KEYS = frozenset({
    "created_at",
    "generated_at",
    "validated_at",
})


def deterministic_enabled(enabled: bool | None = None) -> bool:
    """Return whether deterministic artifact mode is active."""
    if enabled is not None:
        return enabled
    return os.environ.get(DETERMINISTIC_ENV_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def artifact_timestamp(deterministic: bool | None = None) -> str:
    """Return a fixed or wall-clock UTC timestamp for generated artifacts."""
    if deterministic_enabled(deterministic):
        return DETERMINISTIC_TIMESTAMP
    return datetime.now(UTC).isoformat()


@contextmanager
def deterministic_artifacts(enabled: bool) -> Iterator[None]:
    """Temporarily set deterministic artifact mode for build orchestration."""
    previous = os.environ.get(DETERMINISTIC_ENV_VAR)
    if enabled:
        os.environ[DETERMINISTIC_ENV_VAR] = "1"
    else:
        os.environ.pop(DETERMINISTIC_ENV_VAR, None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(DETERMINISTIC_ENV_VAR, None)
        else:
            os.environ[DETERMINISTIC_ENV_VAR] = previous


def strip_volatile_fields(value: Any) -> Any:
    """Return a JSON-compatible value with approved volatile fields removed."""
    if isinstance(value, dict):
        return {
            key: strip_volatile_fields(item)
            for key, item in sorted(value.items())
            if key not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [strip_volatile_fields(item) for item in value]
    return value


def semantic_bytes_for_path(path: Path) -> bytes:
    """Return deterministic bytes for hashing an artifact semantically."""
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".jsonl"):
        records: list[Any] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(strip_volatile_fields(json.loads(line)))
            except json.JSONDecodeError:
                records.append(line)
        payload: Any = records
    elif path.suffix == ".json":
        try:
            payload = strip_volatile_fields(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            payload = path.read_text(encoding="utf-8")
    else:
        payload = path.read_text(encoding="utf-8")
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def count_records(path: Path) -> int | None:
    """Return record count for JSONL/JSON artifacts when cheaply available."""
    if path.suffix == ".jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            for key in ("nodes", "by_id", "artifacts"):
                value = data.get(key)
                if isinstance(value, list | dict):
                    return len(value)
    return None
