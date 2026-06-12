"""Artifact manifest generation for Concept Memory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from neural_search.field_state.concept_memory.artifact_utils import (
    count_records,
    deterministic_enabled,
    semantic_bytes_for_path,
)

MANIFEST_PATH = Path("artifacts/field_state/concept_memory/manifest.json")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _input_hashes(input_dir: Path) -> dict[str, str]:
    if not input_dir.exists():
        return {}
    hashes: dict[str, str] = {}
    for path in sorted(input_dir.glob("*.jsonl")):
        hashes[str(path.name)] = _sha256_path(path)
    return hashes


def write_manifest(
    *,
    root: Path,
    artifact_paths: dict[str, Path],
    deterministic: bool | None = None,
    corpus_path: Path | None = None,
    graphml_status: dict[str, Any] | None = None,
) -> Path:
    """Write a reproducibility manifest for generated concept-memory artifacts."""
    base = root
    manifest_path = base / MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_entries: list[dict[str, Any]] = []
    for label, path in sorted(artifact_paths.items()):
        if not path.exists() or path == manifest_path:
            continue
        rel_path = path.relative_to(base) if path.is_relative_to(base) else path
        artifact_entries.append({
            "label": label,
            "path": str(rel_path),
            "record_count": count_records(path),
            "byte_sha256": _sha256_path(path),
            "semantic_sha256": _sha256_bytes(semantic_bytes_for_path(path)),
        })

    corpus_info: dict[str, Any] | None = None
    if corpus_path is not None:
        resolved_corpus = corpus_path if corpus_path.is_absolute() else base / corpus_path
        corpus_info = {
            "path": str(corpus_path),
            "exists": resolved_corpus.exists(),
            "byte_sha256": _sha256_path(resolved_corpus) if resolved_corpus.exists() else None,
        }

    payload: dict[str, Any] = {
        "schema_version": "0.4.1",
        "build_mode": "deterministic" if deterministic_enabled(deterministic) else "normal",
        "deterministic": deterministic_enabled(deterministic),
        "artifacts": artifact_entries,
        "source_corpus": corpus_info,
        "field_state_input_hashes": _input_hashes(base / "artifacts" / "field_state"),
        "graphml_export": graphml_status or {
            "optional": True,
            "status": "not_attempted",
            "path": None,
            "error": None,
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def read_manifest(root: Path) -> dict[str, Any] | None:
    """Read manifest if present."""
    path = root / MANIFEST_PATH
    if not path.exists():
        return None
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
