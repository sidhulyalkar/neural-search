"""Tests for the incremental field-state update pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _make_record(dataset_id: str, title: str = "Test", source: str = "dandi") -> dict:
    return {
        "dataset_id": dataset_id,
        "source": source,
        "source_id": dataset_id.split(":")[-1],
        "title": title,
        "description": "test description",
        "modalities": [],
        "species": [],
        "brain_regions": [],
        "tasks": [],
        "data_standards": [],
        "linked_papers": [],
        "usability_flags": {},
        "analysis_affordances": [],
        "behavioral_events": [],
        "file_formats": [],
    }


class TestChangeDetection:
    """Test the update pipeline's change detection logic via the module directly."""

    def test_new_record_detected(self) -> None:
        from scripts.field_state.update_field_state import detect_changes
        records = [_make_record("dataset:a:1", "Title A")]
        prev = {}
        new_r, changed_r, removed = detect_changes(records, prev)
        assert len(new_r) == 1
        assert len(changed_r) == 0
        assert len(removed) == 0

    def test_changed_record_detected(self) -> None:
        from scripts.field_state.update_field_state import _record_hash, detect_changes
        rec = _make_record("dataset:a:1", "Old Title")
        prev = {"dataset:a:1": _record_hash(rec)}
        updated_rec = _make_record("dataset:a:1", "New Title")
        new_r, changed_r, removed = detect_changes([updated_rec], prev)
        assert len(changed_r) == 1
        assert len(new_r) == 0

    def test_removed_record_detected(self) -> None:
        from scripts.field_state.update_field_state import detect_changes
        records = []
        prev = {"dataset:a:1": "somehash"}
        new_r, changed_r, removed = detect_changes(records, prev)
        assert "dataset:a:1" in removed

    def test_unchanged_record_not_in_changed(self) -> None:
        from scripts.field_state.update_field_state import _record_hash, detect_changes
        rec = _make_record("dataset:a:1", "Stable Title")
        prev = {"dataset:a:1": _record_hash(rec)}
        new_r, changed_r, removed = detect_changes([rec], prev)
        assert len(new_r) == 0
        assert len(changed_r) == 0
        assert len(removed) == 0


class TestDryRun:
    """Test that --dry-run does not write artifacts."""

    def test_dry_run_no_output(self, tmp_path: Path) -> None:
        # Write minimal corpus
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(json.dumps(_make_record("dataset:dandi:000001")) + "\n", encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "scripts/field_state/update_field_state.py",
             "--corpus", str(corpus),
             "--out-dir", str(tmp_path / "out"),
             "--snapshots-dir", str(tmp_path / "snaps"),
             "--dry-run"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        # No artifacts should be written in dry-run mode
        assert not (tmp_path / "out" / "memory_graph_nodes.jsonl").exists()

    def test_full_run_writes_artifacts(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(json.dumps(_make_record("dataset:dandi:000001")) + "\n", encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "scripts/field_state/update_field_state.py",
             "--corpus", str(corpus),
             "--out-dir", str(tmp_path / "out"),
             "--snapshots-dir", str(tmp_path / "snaps"),
             "--force"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "out" / "memory_graph_nodes.jsonl").exists()
        assert (tmp_path / "out" / "memory_graph_edges.jsonl").exists()
        assert (tmp_path / "out" / "memory_graph_manifest.json").exists()
