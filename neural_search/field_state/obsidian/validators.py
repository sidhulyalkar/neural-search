"""Validation checks for the field-state Obsidian mirror."""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from neural_search.field_state.memory.index import memory_index_path
from neural_search.field_state.obsidian.paths import (
    FIELD_STATE_ROOT,
    assert_inside_vault,
    vault_root,
)
from neural_search.field_state.obsidian.reader import ObsidianNote, read_obsidian_notes
from neural_search.field_state.obsidian.sync import append_sync_log
from neural_search.field_state.obsidian.templates import (
    GENERATED_BEGIN,
    GENERATED_END,
    HUMAN_BEGIN,
    HUMAN_END,
)

REQUIRED_FRONTMATTER = {
    "type",
    "field_state_id",
    "field",
    "title",
    "status",
    "review_status",
    "schema_version",
}

VALID_STATUS_BY_TYPE: dict[str, set[str]] = {
    "claim": {
        "active",
        "needs_validation",
        "partially_tested",
        "retired",
        "reviewed",
        "deprecated",
    },
    "benchmark_gap": {"open", "in_progress", "addressed", "active", "resolved"},
    "opportunity": {
        "candidate",
        "next_up",
        "in_progress",
        "done",
        "deferred",
        "active",
        "validated",
        "rejected",
    },
    "field_snapshot": {"current", "archived"},
    "qrels_review": {
        "unreviewed",
        "reviewed",
        "needs_adjudication",
        "adjudicated",
        "rejected",
    },
    "hard_negative_review": {"unreviewed", "reviewed", "resolved"},
    "affordance_review": {"unreviewed", "reviewed", "validated", "rejected"},
    "eval_snapshot": {"current", "archived"},
    "codex_task": {"todo", "active", "done"},
    "decision_log": {"active", "superseded"},
}


@dataclass(frozen=True)
class ValidationIssue:
    """One validation issue."""

    severity: str
    path: str
    message: str


@dataclass
class ValidationResult:
    """Validation result for an Obsidian vault."""

    field: str
    issues: list[ValidationIssue] = dataclass_field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")


def _add_issue(
    issues: list[ValidationIssue],
    severity: str,
    path: Path,
    message: str,
) -> None:
    issues.append(ValidationIssue(severity=severity, path=str(path), message=message))


def _validate_note(vault_path: Path, note: ObsidianNote, issues: list[ValidationIssue]) -> None:
    frontmatter = note.frontmatter
    for required in REQUIRED_FRONTMATTER:
        if required not in frontmatter or frontmatter.get(required) in {None, ""}:
            _add_issue(issues, "error", note.path, f"missing required frontmatter: {required}")

    body = note.body
    for marker in (GENERATED_BEGIN, GENERATED_END, HUMAN_BEGIN, HUMAN_END):
        if marker not in body:
            _add_issue(issues, "error", note.path, f"missing block marker: {marker}")

    note_type = str(frontmatter.get("type", ""))
    status = str(frontmatter.get("status", ""))
    valid_status = VALID_STATUS_BY_TYPE.get(note_type)
    if valid_status is not None and status not in valid_status:
        _add_issue(issues, "error", note.path, f"invalid status for {note_type}: {status}")

    if note_type == "opportunity":
        for key in (
            "novelty_score",
            "feasibility_score",
            "impact_score",
            "uncertainty_reduction_score",
            "personal_fit_score",
            "risk_score",
            "total_score",
        ):
            if key not in frontmatter:
                _add_issue(issues, "error", note.path, f"opportunity missing score field: {key}")
    if note_type == "claim" and "evidence_level" not in frontmatter:
        _add_issue(issues, "error", note.path, "claim missing evidence_level")
    if note_type == "benchmark_gap" and not (
        frontmatter.get("why_it_matters") or "## Required validation" in body
    ):
        _add_issue(
            issues,
            "error",
            note.path,
            "benchmark gap missing rationale or validation need",
        )

    try:
        assert_inside_vault(vault_path, note.path)
    except ValueError as exc:
        _add_issue(issues, "error", note.path, str(exc))


def validate_obsidian_memory(vault_path: Path, field: str) -> ValidationResult:
    """Validate an Obsidian field-state vault."""
    issues: list[ValidationIssue] = []
    root = vault_path / FIELD_STATE_ROOT
    if root.exists():
        for path in sorted(root.rglob("*.md")):
            relative = path.relative_to(root).as_posix()
            if relative.startswith(("00_Dashboard/", "90_System/")):
                continue
            text = path.read_text(encoding="utf-8")
            if not text.startswith("---\n"):
                _add_issue(issues, "error", path, "missing frontmatter")
            for marker in (GENERATED_BEGIN, GENERATED_END, HUMAN_BEGIN, HUMAN_END):
                if marker not in text:
                    _add_issue(issues, "error", path, f"missing block marker: {marker}")

    notes = read_obsidian_notes(vault_path, field)
    seen: dict[str, Path] = {}
    for note in notes:
        _validate_note(vault_path, note, issues)
        note_id = note.field_state_id
        if note_id:
            if note_id in seen:
                _add_issue(
                    issues,
                    "error",
                    note.path,
                    f"duplicate field_state_id also used by {seen[note_id]}: {note_id}",
                )
            seen[note_id] = note.path

    index_path = memory_index_path(vault_path)
    if not index_path.exists():
        _add_issue(issues, "error", index_path, "memory index does not exist")
    else:
        try:
            index_data: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _add_issue(issues, "error", index_path, f"memory index is not parseable: {exc}")
        else:
            if index_data.get("schema_version") != "0.2":
                _add_issue(issues, "error", index_path, "memory index schema_version is not 0.2")

    return ValidationResult(field=field, issues=issues)


def render_validation_report(result: ValidationResult) -> str:
    """Render validation result Markdown."""
    lines = [
        "# Field-State Memory Validation",
        "",
        f"- Field: {result.field}",
        f"- Valid: {result.is_valid}",
        f"- Errors: {result.error_count}",
        f"- Warnings: {result.warning_count}",
        "",
        "## Issues",
        "",
    ]
    if not result.issues:
        lines.append("- none")
    for issue in result.issues:
        lines.append(f"- {issue.severity.upper()}: `{issue.path}` - {issue.message}")
    return "\n".join(lines)


def write_validation_report(vault_path: Path, result: ValidationResult) -> Path:
    """Write validation report to the vault."""
    path = vault_root(vault_path) / "90_System/validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_validation_report(result), encoding="utf-8")
    append_sync_log(
        vault_path,
        operation="memory-validate",
        field=result.field,
        warnings=[issue.message for issue in result.issues],
    )
    return path
