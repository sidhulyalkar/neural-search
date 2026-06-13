"""Sync-log helpers for the Obsidian memory mirror."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from neural_search.field_state.obsidian.paths import vault_root


def sync_log_path(vault_path: Path) -> Path:
    """Return the sync log path."""
    return vault_root(vault_path) / "90_System/sync_log.md"


def append_sync_log(
    vault_path: Path,
    *,
    operation: str,
    field: str,
    notes_created: int = 0,
    notes_updated: int = 0,
    notes_skipped: int = 0,
    warnings: list[str] | None = None,
) -> Path:
    """Append a concise sync-log entry."""
    path = sync_log_path(vault_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    warning_items = warnings or []
    entry = [
        f"## {datetime.now(UTC).isoformat()} - {operation}",
        "",
        f"Field: {field}",
        "",
        f"- Notes created: {notes_created}",
        f"- Notes updated: {notes_updated}",
        f"- Notes skipped: {notes_skipped}",
        f"- Warnings: {len(warning_items)}",
    ]
    for warning in warning_items:
        entry.append(f"  - {warning}")
    entry.append("")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(entry))
        handle.write("\n")
    return path
