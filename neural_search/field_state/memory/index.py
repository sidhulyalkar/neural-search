"""Memory index generation for the Obsidian mirror."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_search.field_state.obsidian.paths import vault_relative, vault_root
from neural_search.field_state.obsidian.reader import ObsidianNote, read_obsidian_notes


def content_hash(text: str) -> str:
    """Hash full note content for deterministic change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def memory_index_path(vault_path: Path) -> Path:
    """Return the memory index path."""
    return vault_root(vault_path) / "90_System/memory_index.json"


def note_index_entry(vault_path: Path, note: ObsidianNote) -> dict[str, Any]:
    """Build one memory-index note entry.

    The content hash covers the full note, including human blocks, so the index
    can detect human edits as well as regenerated machine content.
    """
    return {
        "field_state_id": note.frontmatter.get("field_state_id"),
        "type": note.frontmatter.get("type"),
        "title": note.frontmatter.get("title"),
        "path": vault_relative(vault_path, note.path),
        "status": note.frontmatter.get("status"),
        "review_status": note.frontmatter.get("review_status"),
        "content_hash": content_hash(note.raw_text),
        "source_artifacts": note.frontmatter.get("source_artifacts", []),
        "schema_version": note.frontmatter.get("schema_version"),
    }


def build_memory_index(vault_path: Path, field: str) -> dict[str, Any]:
    """Build the current memory index from vault notes."""
    notes = read_obsidian_notes(vault_path, field)
    return {
        "schema_version": "0.2",
        "field": field,
        "generated_at": datetime.now(UTC).isoformat(),
        "hash_strategy": "sha256 of full Markdown note content",
        "notes": [note_index_entry(vault_path, note) for note in notes],
    }


def write_memory_index(vault_path: Path, field: str) -> Path:
    """Write the memory index JSON file."""
    path = memory_index_path(vault_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_memory_index(vault_path, field), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def read_memory_index(vault_path: Path) -> dict[str, Any] | None:
    """Read the memory index if present."""
    path = memory_index_path(vault_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return data
