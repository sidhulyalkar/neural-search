"""Conservative YAML frontmatter helpers for Obsidian notes."""

from __future__ import annotations

from typing import Any

import yaml  # type: ignore[import-untyped]


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from Markdown text.

    Returns an empty dict when the note has no frontmatter. Invalid YAML raises
    ValueError so callers do not silently corrupt review notes.
    """
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("frontmatter start marker found without closing marker")
    raw = text[4:end]
    body = text[end + 5 :]
    try:
        parsed = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return dict(parsed), body


def render_frontmatter(data: dict[str, Any]) -> str:
    """Render frontmatter with stable key ordering preserved by insertion order."""
    rendered = yaml.safe_dump(
        data,
        allow_unicode=False,
        default_flow_style=False,
        sort_keys=False,
    ).strip()
    return f"---\n{rendered}\n---\n"


def split_note(text: str) -> tuple[dict[str, Any], str]:
    """Split a note into frontmatter and Markdown body."""
    return parse_frontmatter(text)


def compose_note(frontmatter: dict[str, Any], body: str) -> str:
    """Compose frontmatter and body into a Markdown note."""
    return f"{render_frontmatter(frontmatter)}\n{body.lstrip()}"
