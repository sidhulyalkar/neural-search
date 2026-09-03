"""Deterministic local repository inventory for scientific software audits."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field


DEFAULT_POLICY_FILES = (
    "README.md",
    "README.rst",
    "CONTRIBUTING.md",
    "CONTRIBUTING.rst",
    "CODE_OF_CONDUCT.md",
    "CHANGELOG.md",
    "NEWS.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE.md",
)

DEFAULT_SOURCE_SUFFIXES = {
    ".py",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".m",
    ".jl",
    ".r",
    ".rs",
}


class RepositoryFile(BaseModel):
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    suffix: str


class RepositorySnapshot(BaseModel):
    root: str
    revision: str | None
    files: list[RepositoryFile]
    policy_files: dict[str, str] = Field(default_factory=dict)
    source_file_count: int = 0
    total_source_bytes: int = 0


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def inventory_repository(
    root: Path,
    *,
    source_suffixes: set[str] | None = None,
    max_file_bytes: int = 2_000_000,
) -> RepositorySnapshot:
    """Inventory text-like source files without interpreting model output.

    The inventory follows the local checkout supplied by the researcher and excludes
    `.git`, virtual environments, generated build directories, and oversized files.
    """

    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {root}")
    suffixes = {suffix.lower() for suffix in (source_suffixes or DEFAULT_SOURCE_SUFFIXES)}
    ignored_parts = {".git", ".venv", "venv", "node_modules", "dist", "build", "__pycache__"}
    files: list[RepositoryFile] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ignored_parts.intersection(path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in suffixes:
            continue
        size = path.stat().st_size
        if size > max_file_bytes:
            continue
        data = path.read_bytes()
        files.append(
            RepositoryFile(
                path=path.relative_to(root).as_posix(),
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=size,
                suffix=path.suffix.lower(),
            )
        )
        total_bytes += size

    policies: dict[str, str] = {}
    for relative in DEFAULT_POLICY_FILES:
        candidate = root / relative
        if candidate.is_file():
            policies[relative] = hashlib.sha256(candidate.read_bytes()).hexdigest()

    return RepositorySnapshot(
        root=str(root),
        revision=_git_revision(root),
        files=files,
        policy_files=policies,
        source_file_count=len(files),
        total_source_bytes=total_bytes,
    )


def read_source_component(
    snapshot: RepositorySnapshot,
    path: str,
    *,
    start_line: int = 1,
    end_line: int | None = None,
    max_chars: int = 100_000,
) -> str:
    """Read a source component that is present in the frozen repository inventory."""

    indexed = {item.path for item in snapshot.files}
    if path not in indexed:
        raise ValueError(f"path is not part of repository snapshot: {path}")
    if start_line < 1:
        raise ValueError("start_line must be >= 1")
    target = Path(snapshot.root) / path
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = lines[start_line - 1 : end_line]
    rendered = "\n".join(selected)
    if len(rendered) > max_chars:
        raise ValueError(f"source component exceeds max_chars={max_chars}")
    return rendered
