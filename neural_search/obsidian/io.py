"""Safe Obsidian vault note reading and writing.

Write-safety rule: fields listed in HUMAN_OWNED_FIELDS are NEVER overwritten
by export scripts if a human has already set them to a non-None value.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from neural_search.obsidian.templates import render_frontmatter

HUMAN_OWNED_FIELDS = {"label", "confidence", "audit_status"}

_FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def parse_note(content: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body_str) from a Markdown note string."""
    match = _FM_RE.match(content)
    if not match:
        return {}, content
    try:
        fm = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    body = content[match.end():]
    return fm, body


def read_note(path: Path) -> tuple[dict, str]:
    """Read and parse an existing vault note."""
    return parse_note(path.read_text(encoding="utf-8"))


def safe_write_note(path: Path, frontmatter: dict, body: str) -> None:
    """Write a note, preserving any human-owned frontmatter fields."""
    if path.exists():
        existing_fm, existing_body = read_note(path)
        for field in HUMAN_OWNED_FIELDS:
            existing_val = existing_fm.get(field)
            if existing_val is not None:
                frontmatter[field] = existing_val
        if existing_body.strip():
            body = existing_body

    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_frontmatter(frontmatter) + "\n" + body
    path.write_text(content, encoding="utf-8")
