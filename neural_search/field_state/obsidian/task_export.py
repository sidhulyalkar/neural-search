"""Codex task and decision-log note generation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from neural_search.field_state.obsidian.frontmatter import (
    compose_note,
    parse_frontmatter,
)
from neural_search.field_state.obsidian.paths import (
    ensure_vault_structure,
    field_state_id,
    note_path,
)
from neural_search.field_state.obsidian.reader import extract_human_block
from neural_search.field_state.obsidian.sync import append_sync_log
from neural_search.field_state.obsidian.templates import (
    default_human_block,
    render_codex_task_note,
    render_decision_note,
)
from neural_search.field_state.obsidian.writer import atomic_write_text
from neural_search.field_state.store import read_opportunities


def _compose_preserving_human(
    path: Path,
    frontmatter: dict[str, object],
    body: str,
    note_type: str,
) -> str:
    if not path.exists():
        return compose_note(frontmatter, body)
    old_text = path.read_text(encoding="utf-8")
    old_frontmatter, old_body = parse_frontmatter(old_text)
    human_block = extract_human_block(old_body) or default_human_block(note_type)
    body = body.replace(default_human_block(note_type), human_block)
    if old_frontmatter.get("created_at"):
        frontmatter["created_at"] = old_frontmatter["created_at"]
    return compose_note(frontmatter, body)


def export_codex_task(
    *,
    opportunity_id: str,
    vault_path: Path,
    field: str = "neuroscience_dataset_reuse",
    root: Path | None = None,
) -> Path:
    """Generate a Codex task note from one opportunity."""
    ensure_vault_structure(vault_path)
    opportunities = read_opportunities(root)
    match = None
    for opportunity in opportunities:
        generated_id = field_state_id("opportunity", opportunity.opportunity_id, opportunity.title)
        if opportunity_id in {opportunity.opportunity_id, generated_id}:
            match = opportunity
            break
    if match is None:
        raise ValueError(f"unknown opportunity id: {opportunity_id}")

    title = f"Build {match.title}"
    task_id = field_state_id("codex_task", match.opportunity_id, title)
    now = datetime.now(UTC).isoformat()
    frontmatter: dict[str, object] = {
        "type": "codex_task",
        "field_state_id": task_id,
        "field": field,
        "title": title,
        "status": "todo",
        "review_status": "unreviewed",
        "created_at": now,
        "updated_at": now,
        "generated_at": now,
        "source_artifacts": ["artifacts/field_state/opportunities.jsonl"],
        "schema_version": "0.2",
        "tags": ["field-state", "codex-task"],
        "related_opportunity": match.opportunity_id,
    }
    body = render_codex_task_note(
        title=title,
        goal=f"Advance this field-state opportunity: {match.title}.",
        context=match.description,
        related_opportunity=match.opportunity_id,
        files_to_inspect=[
            "neural_search/field_state/",
            "reports/eval/",
            "artifacts/field_state/",
            "data/corpus/normalized/combined_corpus.jsonl",
        ],
        implementation_requirements=[
            "Preserve generated artifacts and human review overlays separately.",
            "Keep outputs deterministic and inspectable in plain text.",
            "Avoid external network calls and heavyweight storage.",
        ],
        acceptance_criteria=[
            "Relevant field-state artifacts are updated.",
            "Reports or snapshots can reference the resulting evidence.",
            "Focused tests cover the new behavior.",
        ],
        tests=[
            "Add or update targeted pytest coverage.",
            "Run ruff on touched field-state files.",
            "Run mypy on neural_search/field_state.",
        ],
        risks=[
            "Human review may be mistaken for verified truth.",
            "Scope may expand beyond validation support.",
        ],
    )
    path = note_path(vault_path, "codex_task", title, "todo")
    atomic_write_text(path, _compose_preserving_human(path, frontmatter, body, "codex_task"))
    append_sync_log(vault_path, operation="export-task", field=field, notes_created=1)
    return path


def add_decision_note(
    *,
    vault_path: Path,
    field: str,
    title: str,
    decision: str | None = None,
    rationale: str | None = None,
    evidence: str | None = None,
    consequences: str | None = None,
    revisit_criteria: str | None = None,
) -> Path:
    """Create or update a decision-log note."""
    ensure_vault_structure(vault_path)
    now = datetime.now(UTC).isoformat()
    frontmatter: dict[str, object] = {
        "type": "decision_log",
        "field_state_id": field_state_id("decision_log", title, title),
        "field": field,
        "title": title,
        "status": "active",
        "review_status": "reviewed",
        "created_at": now,
        "updated_at": now,
        "generated_at": now,
        "source_artifacts": [],
        "schema_version": "0.2",
        "tags": ["field-state", "decision"],
    }
    body = render_decision_note(
        title=title,
        decision=decision or "<!-- Add decision here. -->",
        rationale=rationale or "<!-- Add rationale here. -->",
        evidence=evidence or "<!-- Add evidence here. -->",
        consequences=consequences or "<!-- Add consequences here. -->",
        revisit_criteria=revisit_criteria or "<!-- Add revisit criteria here. -->",
    )
    path = note_path(vault_path, "decision_log", title, "active")
    atomic_write_text(path, _compose_preserving_human(path, frontmatter, body, "decision_log"))
    append_sync_log(vault_path, operation="decision-add", field=field, notes_created=1)
    return path
