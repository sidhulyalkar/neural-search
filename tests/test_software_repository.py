"""Tests for deterministic scientific repository inventory."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_search.software.repository import inventory_repository, read_source_component


def test_repository_inventory_hashes_source_and_policy_files(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Test package\n", encoding="utf-8")
    source = tmp_path / "analysis.py"
    source.write_text("def statistic(x):\n    return x + 1\n", encoding="utf-8")
    generated = tmp_path / "build"
    generated.mkdir()
    (generated / "ignored.py").write_text("ignored = True\n", encoding="utf-8")

    snapshot = inventory_repository(tmp_path)

    assert snapshot.source_file_count == 1
    assert snapshot.files[0].path == "analysis.py"
    assert len(snapshot.files[0].sha256) == 64
    assert "README.md" in snapshot.policy_files
    assert read_source_component(snapshot, "analysis.py", start_line=1, end_line=1) == "def statistic(x):"


def test_repository_reader_rejects_paths_outside_inventory(tmp_path: Path) -> None:
    (tmp_path / "analysis.py").write_text("value = 1\n", encoding="utf-8")
    snapshot = inventory_repository(tmp_path)

    with pytest.raises(ValueError, match="not part of repository snapshot"):
        read_source_component(snapshot, "README.md")
