"""Read and parse field-state Obsidian notes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_search.field_state.obsidian.frontmatter import parse_frontmatter
from neural_search.field_state.obsidian.paths import (
    FIELD_STATE_ROOT,
    assert_inside_vault,
)
from neural_search.field_state.obsidian.templates import (
    GENERATED_BEGIN,
    GENERATED_END,
    HUMAN_BEGIN,
    HUMAN_END,
)

SUPPORTED_NOTE_TYPES = {
    "claim",
    "benchmark_gap",
    "opportunity",
    "field_snapshot",
    "qrels_review",
    "hard_negative_review",
    "affordance_review",
    "eval_snapshot",
    "codex_task",
    "decision_log",
}


@dataclass(frozen=True)
class ObsidianNote:
    """Parsed Obsidian note."""

    path: Path
    frontmatter: dict[str, Any]
    body: str
    generated_block: str
    human_block: str
    raw_text: str

    @property
    def field_state_id(self) -> str:
        return str(self.frontmatter.get("field_state_id", ""))

    @property
    def note_type(self) -> str:
        return str(self.frontmatter.get("type", ""))


def _extract_block(text: str, begin: str, end: str, *, include_markers: bool = False) -> str:
    start = text.find(begin)
    if start == -1:
        return ""
    content_start = start + len(begin)
    stop = text.find(end, content_start)
    if stop == -1:
        return ""
    if include_markers:
        return text[start : stop + len(end)].strip("\n")
    return text[content_start:stop].strip("\n")


def extract_human_block(text: str) -> str:
    """Extract the human-preserved block from a note."""
    return _extract_block(text, HUMAN_BEGIN, HUMAN_END, include_markers=True)


def extract_generated_block(text: str) -> str:
    """Extract the machine-generated block from a note."""
    return _extract_block(text, GENERATED_BEGIN, GENERATED_END)


def read_note(path: Path) -> ObsidianNote:
    """Read one Obsidian note."""
    text = path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(text)
    return ObsidianNote(
        path=path,
        frontmatter=frontmatter,
        body=body,
        generated_block=extract_generated_block(body),
        human_block=extract_human_block(body),
        raw_text=text,
    )


def read_obsidian_notes(vault_path: Path, field: str) -> list[ObsidianNote]:
    """Scan a vault for supported field-state notes."""
    root = vault_path / FIELD_STATE_ROOT
    if not root.exists():
        return []
    notes: list[ObsidianNote] = []
    for path in sorted(root.rglob("*.md")):
        assert_inside_vault(vault_path, path)
        try:
            note = read_note(path)
        except ValueError:
            continue
        if note.frontmatter.get("field") != field:
            continue
        if note.frontmatter.get("type") in SUPPORTED_NOTE_TYPES:
            notes.append(note)
    return notes
