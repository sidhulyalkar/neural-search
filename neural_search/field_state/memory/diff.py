"""Memory diff report for the Obsidian mirror."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from neural_search.field_state.memory.index import content_hash, read_memory_index
from neural_search.field_state.obsidian.paths import vault_root
from neural_search.field_state.obsidian.reader import read_obsidian_notes
from neural_search.field_state.obsidian.sync import append_sync_log
from neural_search.field_state.obsidian.templates import (
    GENERATED_BEGIN,
    GENERATED_END,
    HUMAN_BEGIN,
    HUMAN_END,
)


@dataclass
class MemoryDiff:
    """Diff categories for a field-state vault."""

    field: str
    unreviewed: list[str] = dataclass_field(default_factory=list)
    human_edited: list[str] = dataclass_field(default_factory=list)
    missing_from_index: list[str] = dataclass_field(default_factory=list)
    index_missing_from_vault: list[str] = dataclass_field(default_factory=list)
    duplicate_ids: list[str] = dataclass_field(default_factory=list)
    schema_mismatches: list[str] = dataclass_field(default_factory=list)
    missing_markers: list[str] = dataclass_field(default_factory=list)


def compute_memory_diff(vault_path: Path, field: str) -> MemoryDiff:
    """Compute a concise memory diff."""
    notes = read_obsidian_notes(vault_path, field)
    index = read_memory_index(vault_path) or {"notes": []}
    indexed_notes: list[dict[str, Any]] = list(index.get("notes", []))
    indexed_by_id = {
        str(entry.get("field_state_id", "")): entry
        for entry in indexed_notes
        if entry.get("field_state_id")
    }
    note_ids: dict[str, int] = {}
    diff = MemoryDiff(field=field)

    for note in notes:
        note_id = note.field_state_id
        if not note_id:
            continue
        note_ids[note_id] = note_ids.get(note_id, 0) + 1
        if note.frontmatter.get("review_status") == "unreviewed":
            diff.unreviewed.append(note_id)
        if "<!-- Add notes here. -->" not in note.human_block:
            diff.human_edited.append(note_id)
        if note_id not in indexed_by_id:
            diff.missing_from_index.append(note_id)
        elif content_hash(note.raw_text) != indexed_by_id[note_id].get("content_hash"):
            diff.human_edited.append(note_id)
        if note.frontmatter.get("schema_version") != "0.2":
            diff.schema_mismatches.append(note_id)
        for marker in (GENERATED_BEGIN, GENERATED_END, HUMAN_BEGIN, HUMAN_END):
            if marker not in note.body:
                diff.missing_markers.append(note_id)
                break

    for note_id, count in note_ids.items():
        if count > 1:
            diff.duplicate_ids.append(note_id)
    for note_id, entry in indexed_by_id.items():
        note_path = vault_path / str(entry.get("path", ""))
        if note_id not in note_ids or not note_path.exists():
            diff.index_missing_from_vault.append(note_id)

    diff.human_edited = sorted(set(diff.human_edited))
    return diff


def render_memory_diff(diff: MemoryDiff) -> str:
    """Render memory diff Markdown."""
    sections = {
        "Notes generated but not reviewed": diff.unreviewed,
        "Notes edited by human or changed since index": diff.human_edited,
        "Notes missing from index": diff.missing_from_index,
        "Notes in index missing from vault": diff.index_missing_from_vault,
        "Duplicate field_state_id values": diff.duplicate_ids,
        "Schema version mismatches": diff.schema_mismatches,
        "Notes missing generated/human markers": diff.missing_markers,
    }
    lines = ["# Field-State Memory Diff", "", f"Field: {diff.field}", ""]
    for title, items in sections.items():
        lines.extend([f"## {title}", ""])
        if items:
            lines.extend(f"- `{item}`" for item in sorted(set(items)))
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines)


def write_memory_diff(vault_path: Path, field: str) -> Path:
    """Write memory diff report."""
    diff = compute_memory_diff(vault_path, field)
    path = vault_root(vault_path) / "90_System/memory_diff.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_memory_diff(diff), encoding="utf-8")
    append_sync_log(
        vault_path,
        operation="memory-diff",
        field=field,
        warnings=[
            f"human_edited={len(diff.human_edited)}",
            f"duplicates={len(diff.duplicate_ids)}",
        ],
    )
    return path
