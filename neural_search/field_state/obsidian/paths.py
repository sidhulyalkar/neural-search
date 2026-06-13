"""Vault path helpers for the field-state Obsidian mirror."""

from __future__ import annotations

import re
from pathlib import Path

FIELD_STATE_ROOT = Path("Field-State")

FOLDERS = [
    "00_Dashboard",
    "10_Snapshots/snapshots",
    "20_Claims/weak",
    "20_Claims/reviewed",
    "20_Claims/deprecated",
    "30_Benchmark_Gaps/active",
    "30_Benchmark_Gaps/resolved",
    "40_Opportunities/candidate",
    "40_Opportunities/active",
    "40_Opportunities/validated",
    "40_Opportunities/rejected",
    "60_Evaluation/qrels_review/unreviewed",
    "60_Evaluation/qrels_review/reviewed",
    "60_Evaluation/qrels_review/needs_adjudication",
    "60_Evaluation/qrels_review/adjudicated",
    "60_Evaluation/hard_negatives",
    "60_Evaluation/affordance_validation",
    "60_Evaluation/eval_snapshots",
    "70_Codex_Tasks/todo",
    "70_Codex_Tasks/active",
    "70_Codex_Tasks/done",
    "80_Decision_Log/decisions",
    "90_System",
]


def vault_root(vault_path: Path) -> Path:
    """Return the field-state root inside an Obsidian vault."""
    return vault_path / FIELD_STATE_ROOT


def ensure_vault_structure(vault_path: Path) -> list[Path]:
    """Create the required Obsidian folder structure."""
    created: list[Path] = []
    for folder in FOLDERS:
        path = vault_root(vault_path) / folder
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created


def sanitize_filename(value: str) -> str:
    """Return a filesystem-safe Markdown filename stem."""
    clean = re.sub(r"[^A-Za-z0-9._ -]+", "", value).strip()
    clean = re.sub(r"\s+", " ", clean)
    clean = clean.replace("/", "-")
    return clean[:120] or "untitled"


def slugify(value: str) -> str:
    """Return a stable lowercase slug."""
    clean = re.sub(r"[^A-Za-z0-9]+", "-", value.lower()).strip("-")
    return clean or "untitled"


def field_state_id(note_type: str, source_id: str, title: str = "") -> str:
    """Build a stable canonical field-state ID."""
    prefix = {
        "claim": "claim",
        "benchmark_gap": "benchmark_gap",
        "opportunity": "opportunity",
        "field_snapshot": "snapshot",
        "qrels_review": "qrels_candidate",
        "hard_negative_review": "hard_negative",
        "affordance_review": "affordance_review",
        "codex_task": "codex_task",
        "decision_log": "decision",
    }.get(note_type, note_type)
    raw = source_id or title
    for removable in (f"{prefix}_", f"{prefix}:"):
        if raw.startswith(removable):
            raw = raw[len(removable) :]
    return f"{prefix}:{slugify(raw)}"


def assert_inside_vault(vault_path: Path, path: Path) -> None:
    """Reject paths outside the vault."""
    vault_resolved = vault_path.resolve()
    path_resolved = path.resolve()
    if vault_resolved != path_resolved and vault_resolved not in path_resolved.parents:
        raise ValueError(f"path escapes vault: {path}")


def vault_relative(vault_path: Path, path: Path) -> str:
    """Return a POSIX vault-relative path after traversal validation."""
    assert_inside_vault(vault_path, path)
    return path.resolve().relative_to(vault_path.resolve()).as_posix()


def note_dir(vault_path: Path, note_type: str, status: str) -> Path:
    """Return the directory for a note type/status pair."""
    root = vault_root(vault_path)
    if note_type == "claim":
        if status in {"reviewed", "trusted"}:
            return root / "20_Claims/reviewed"
        if status in {"deprecated", "retired"}:
            return root / "20_Claims/deprecated"
        return root / "20_Claims/weak"
    if note_type == "benchmark_gap":
        if status in {"resolved", "addressed"}:
            return root / "30_Benchmark_Gaps/resolved"
        return root / "30_Benchmark_Gaps/active"
    if note_type == "opportunity":
        normalized = status if status in {"active", "validated", "rejected"} else "candidate"
        return root / "40_Opportunities" / normalized
    if note_type == "field_snapshot":
        return root / "10_Snapshots/snapshots"
    if note_type == "qrels_review":
        normalized = status if status in {"reviewed", "needs_adjudication", "adjudicated"} else "unreviewed"
        return root / "60_Evaluation/qrels_review" / normalized
    if note_type == "hard_negative_review":
        return root / "60_Evaluation/hard_negatives"
    if note_type == "affordance_review":
        return root / "60_Evaluation/affordance_validation"
    if note_type == "eval_snapshot":
        return root / "60_Evaluation/eval_snapshots"
    if note_type == "codex_task":
        normalized = status if status in {"active", "done"} else "todo"
        return root / "70_Codex_Tasks" / normalized
    if note_type == "decision_log":
        return root / "80_Decision_Log/decisions"
    return root / "90_System"


def note_path(
    vault_path: Path,
    note_type: str,
    title: str,
    status: str = "candidate",
    *,
    latest_snapshot: bool = False,
) -> Path:
    """Return a stable note path for a generated note."""
    if latest_snapshot:
        return vault_root(vault_path) / "10_Snapshots/latest_snapshot.md"
    directory = note_dir(vault_path, note_type, status)
    path = directory / f"{sanitize_filename(title)}.md"
    assert_inside_vault(vault_path, path)
    return path
