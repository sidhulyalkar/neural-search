"""Import human review metadata from Obsidian notes into overlay JSONL files."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from neural_search.field_state.obsidian.paths import vault_relative
from neural_search.field_state.obsidian.reader import ObsidianNote, read_obsidian_notes
from neural_search.field_state.obsidian.sync import append_sync_log
from neural_search.field_state.store import ARTIFACT_DIR, resolve_path, write_jsonl

OVERLAY_PATHS: dict[str, Path] = {
    "claim": ARTIFACT_DIR / "reviewed_claims.jsonl",
    "benchmark_gap": ARTIFACT_DIR / "reviewed_benchmark_gaps.jsonl",
    "opportunity": ARTIFACT_DIR / "reviewed_opportunities.jsonl",
    "field_snapshot": ARTIFACT_DIR / "reviewed_snapshots.jsonl",
    "codex_task": ARTIFACT_DIR / "reviewed_tasks.jsonl",
}


class ReviewOverlay(BaseModel):
    """Human review overlay imported from an Obsidian note."""

    field_state_id: str
    type: str
    field: str
    title: str
    status: str
    review_status: str
    human_block: str
    reviewer_notes: str
    human_reviewer: str | None = None
    reviewed_at: str | None = None
    human_priority: str | None = None
    human_tags: list[str] = Field(default_factory=list)
    source_record_id: str | None = None
    source_note_path: str
    imported_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = "0.2"


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def extract_reviewer_notes(human_block: str) -> str:
    """Extract reviewer notes from a human block."""
    marker = "## Reviewer notes"
    index = human_block.find(marker)
    if index == -1:
        return human_block.strip()
    notes = human_block[index + len(marker) :].strip()
    end_marker = "<!-- FIELDSTATE:END human -->"
    if end_marker in notes:
        notes = notes[: notes.index(end_marker)]
    return notes.strip()


def overlay_from_note(vault_path: Path, note: ObsidianNote) -> ReviewOverlay:
    """Build one overlay record from an Obsidian note."""
    frontmatter = note.frontmatter
    source_record_id = (
        frontmatter.get("claim_id")
        or frontmatter.get("gap_id")
        or frontmatter.get("opportunity_id")
        or frontmatter.get("snapshot_id")
        or frontmatter.get("related_opportunity")
    )
    return ReviewOverlay(
        field_state_id=str(frontmatter.get("field_state_id", "")),
        type=str(frontmatter.get("type", "")),
        field=str(frontmatter.get("field", "")),
        title=str(frontmatter.get("title", "")),
        status=str(frontmatter.get("status", "")),
        review_status=str(frontmatter.get("review_status", "unreviewed")),
        human_block=note.human_block,
        reviewer_notes=extract_reviewer_notes(note.human_block),
        human_reviewer=frontmatter.get("human_reviewer"),
        reviewed_at=frontmatter.get("reviewed_at"),
        human_priority=frontmatter.get("human_priority"),
        human_tags=_as_string_list(frontmatter.get("human_tags")),
        source_record_id=str(source_record_id) if source_record_id else None,
        source_note_path=vault_relative(vault_path, note.path),
    )


def import_review_overlays(
    vault_path: Path,
    *,
    field: str,
    root: Path | None = None,
) -> dict[str, Path]:
    """Import human review metadata from Obsidian into overlay JSONL files."""
    notes = read_obsidian_notes(vault_path, field)
    grouped: dict[str, list[ReviewOverlay]] = {note_type: [] for note_type in OVERLAY_PATHS}
    warnings: list[str] = []
    for note in notes:
        note_type = note.note_type
        if note_type not in grouped:
            continue
        if not note.field_state_id:
            warnings.append(f"Skipping note without field_state_id: {note.path}")
            continue
        grouped[note_type].append(overlay_from_note(vault_path, note))

    paths: dict[str, Path] = {}
    for note_type, records in grouped.items():
        paths[note_type] = write_jsonl(OVERLAY_PATHS[note_type], records, root)

    append_sync_log(
        vault_path,
        operation="import-obsidian",
        field=field,
        notes_created=sum(len(records) for records in grouped.values()),
        warnings=warnings,
    )
    # Ensure parent artifact directory exists even when write_jsonl was rooted.
    resolve_path(ARTIFACT_DIR, root).mkdir(parents=True, exist_ok=True)
    return paths
