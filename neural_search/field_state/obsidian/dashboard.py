"""Dashboard generation for the field-state Obsidian mirror."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neural_search.field_state.memory.index import read_memory_index
from neural_search.field_state.obsidian.paths import vault_root


def _link_for(entry: dict[str, Any]) -> str:
    title = str(entry.get("title") or entry.get("field_state_id") or "Untitled")
    path = Path(str(entry.get("path", "")))
    stem = path.stem or title
    return f"- [[{stem}|{title}]]"


def render_dashboard(vault_path: Path, field: str) -> str:
    """Render a plugin-free Obsidian dashboard."""
    index = read_memory_index(vault_path) or {"notes": []}
    entries: list[dict[str, Any]] = list(index.get("notes", []))
    opportunities = [item for item in entries if item.get("type") == "opportunity"][:10]
    claims = [item for item in entries if item.get("type") == "claim"][:10]
    gaps = [item for item in entries if item.get("type") == "benchmark_gap"][:10]
    qrels = [item for item in entries if item.get("type") == "qrels_review"][:10]
    tasks = [item for item in entries if item.get("type") == "codex_task"][:10]
    decisions = [item for item in entries if item.get("type") == "decision_log"][:10]

    def render_links(items: list[dict[str, Any]]) -> list[str]:
        return [_link_for(item) for item in items] or ["- none"]

    lines = [
        "# Field-State Dashboard",
        "",
        f"Field: `{field}`",
        "",
        "## Latest snapshot",
        "",
        "- [[latest_snapshot]]",
        "",
        "## Top opportunities",
        "",
        *render_links(opportunities),
        "",
        "## Weak claims needing review",
        "",
        *render_links(claims),
        "",
        "## Active benchmark gaps",
        "",
        *render_links(gaps),
        "",
        "## Qrels review",
        "",
        *render_links(qrels),
        "",
        "## Codex tasks",
        "",
        *render_links(tasks),
        "",
        "## Decision log",
        "",
        *render_links(decisions),
        "",
        "## System",
        "",
        "- [[memory_index]]",
        "- [[sync_log]]",
        "- [[validation_report]]",
        "- [[memory_diff]]",
        "",
    ]
    return "\n".join(lines)


def write_dashboard(vault_path: Path, field: str) -> Path:
    """Write the dashboard note."""
    path = vault_root(vault_path) / "00_Dashboard/Field-State Dashboard.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard(vault_path, field), encoding="utf-8")
    return path
