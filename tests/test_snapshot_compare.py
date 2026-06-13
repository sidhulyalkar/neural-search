"""Tests for snapshot comparison and drift monitoring."""

from __future__ import annotations

import json
from pathlib import Path


def _write_corpus_manifest(path: Path, record_hashes: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"record_hashes": record_hashes, "record_count": len(record_hashes)}),
        encoding="utf-8",
    )


def _write_graph_manifest(path: Path, nodes: int, edges: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "total_nodes": nodes,
            "total_edges": edges,
            "node_counts_by_type": {"dataset": nodes // 2, "source_archive": nodes // 2},
            "edge_counts_by_type": {"dataset_from_source": edges},
        }),
        encoding="utf-8",
    )


class TestSnapshotDiff:
    def test_added_datasets_detected(self, tmp_path: Path) -> None:
        old_dir = tmp_path / "snap_old"
        new_dir = tmp_path / "snap_new"
        _write_corpus_manifest(old_dir / "corpus_manifest.json", {"ds:a:1": "hash1"})
        _write_corpus_manifest(new_dir / "corpus_manifest.json", {"ds:a:1": "hash1", "ds:b:2": "hash2"})
        _write_graph_manifest(old_dir / "memory_graph_manifest.json", 2, 1)
        _write_graph_manifest(new_dir / "memory_graph_manifest.json", 4, 2)

        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "scripts/field_state/compare_snapshots.py",
             "--old", str(old_dir), "--new", str(new_dir),
             "--out-dir", str(tmp_path / "reports")],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0, result.stderr
        diff_json = json.loads((tmp_path / "reports" / "snapshot_diff.json").read_text())
        assert "ds:b:2" in diff_json["corpus"]["added"]

    def test_removed_datasets_detected(self, tmp_path: Path) -> None:
        old_dir = tmp_path / "snap_old"
        new_dir = tmp_path / "snap_new"
        _write_corpus_manifest(old_dir / "corpus_manifest.json", {"ds:a:1": "hash1", "ds:b:2": "hash2"})
        _write_corpus_manifest(new_dir / "corpus_manifest.json", {"ds:a:1": "hash1"})
        _write_graph_manifest(old_dir / "memory_graph_manifest.json", 4, 2)
        _write_graph_manifest(new_dir / "memory_graph_manifest.json", 2, 1)

        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "scripts/field_state/compare_snapshots.py",
             "--old", str(old_dir), "--new", str(new_dir),
             "--out-dir", str(tmp_path / "reports")],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0, result.stderr
        diff_json = json.loads((tmp_path / "reports" / "snapshot_diff.json").read_text())
        assert "ds:b:2" in diff_json["corpus"]["removed"]

    def test_changed_datasets_detected(self, tmp_path: Path) -> None:
        old_dir = tmp_path / "snap_old"
        new_dir = tmp_path / "snap_new"
        _write_corpus_manifest(old_dir / "corpus_manifest.json", {"ds:a:1": "oldhash"})
        _write_corpus_manifest(new_dir / "corpus_manifest.json", {"ds:a:1": "newhash"})
        _write_graph_manifest(old_dir / "memory_graph_manifest.json", 2, 1)
        _write_graph_manifest(new_dir / "memory_graph_manifest.json", 2, 1)

        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "scripts/field_state/compare_snapshots.py",
             "--old", str(old_dir), "--new", str(new_dir),
             "--out-dir", str(tmp_path / "reports")],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0, result.stderr
        diff_json = json.loads((tmp_path / "reports" / "snapshot_diff.json").read_text())
        assert "ds:a:1" in diff_json["corpus"]["changed"]

    def test_no_changes_all_unchanged(self, tmp_path: Path) -> None:
        old_dir = tmp_path / "snap_old"
        new_dir = tmp_path / "snap_new"
        hashes = {"ds:a:1": "hash1", "ds:b:2": "hash2"}
        _write_corpus_manifest(old_dir / "corpus_manifest.json", hashes)
        _write_corpus_manifest(new_dir / "corpus_manifest.json", hashes)
        _write_graph_manifest(old_dir / "memory_graph_manifest.json", 4, 2)
        _write_graph_manifest(new_dir / "memory_graph_manifest.json", 4, 2)

        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "scripts/field_state/compare_snapshots.py",
             "--old", str(old_dir), "--new", str(new_dir),
             "--out-dir", str(tmp_path / "reports")],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0, result.stderr
        diff_json = json.loads((tmp_path / "reports" / "snapshot_diff.json").read_text())
        assert diff_json["corpus"]["added"] == []
        assert diff_json["corpus"]["removed"] == []
        assert diff_json["corpus"]["changed"] == []
        assert diff_json["corpus"]["unchanged_count"] == 2

    def test_large_removal_triggers_warning(self, tmp_path: Path) -> None:
        old_dir = tmp_path / "snap_old"
        new_dir = tmp_path / "snap_new"
        old_hashes = {f"ds:a:{i}": f"hash{i}" for i in range(10)}
        new_hashes = {"ds:a:0": "hash0"}  # Removed 90%
        _write_corpus_manifest(old_dir / "corpus_manifest.json", old_hashes)
        _write_corpus_manifest(new_dir / "corpus_manifest.json", new_hashes)
        _write_graph_manifest(old_dir / "memory_graph_manifest.json", 20, 10)
        _write_graph_manifest(new_dir / "memory_graph_manifest.json", 2, 1)

        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "scripts/field_state/compare_snapshots.py",
             "--old", str(old_dir), "--new", str(new_dir),
             "--out-dir", str(tmp_path / "reports")],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        diff_json = json.loads((tmp_path / "reports" / "snapshot_diff.json").read_text())
        assert len(diff_json["warnings"]) >= 1
