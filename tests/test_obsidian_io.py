"""Tests for Obsidian vault I/O — frontmatter roundtrip and write-safety."""
from __future__ import annotations

from neural_search.obsidian.io import parse_note, read_note, safe_write_note


class TestParseNote:
    def test_parses_frontmatter_and_body(self):
        content = "---\ntype: dataset\ndataset_id: dandi:1\n---\n\n# Title\n\nBody text."
        fm, body = parse_note(content)
        assert fm["type"] == "dataset"
        assert fm["dataset_id"] == "dandi:1"
        assert "Body text." in body

    def test_no_frontmatter_returns_empty_dict(self):
        content = "Just a body with no frontmatter."
        fm, body = parse_note(content)
        assert fm == {}
        assert "Just a body" in body

    def test_malformed_frontmatter_returns_empty_dict(self):
        content = "---\n: broken: yaml: \n---\nBody."
        fm, body = parse_note(content)
        assert isinstance(fm, dict)


class TestSafeWriteNote:
    def test_creates_new_file(self, tmp_path):
        note_path = tmp_path / "test.md"
        safe_write_note(note_path, {"type": "dataset", "label": None}, "# Body")
        assert note_path.exists()
        content = note_path.read_text()
        assert "type: dataset" in content

    def test_preserves_human_label_on_overwrite(self, tmp_path):
        note_path = tmp_path / "annot.md"
        safe_write_note(note_path, {"type": "annotation", "label": None, "audit_status": "pending"}, "")
        content = note_path.read_text().replace("label: null", "label: 2")
        note_path.write_text(content)
        safe_write_note(note_path, {"type": "annotation", "label": None, "audit_status": "pending"}, "")
        fm, _ = read_note(note_path)
        assert fm.get("label") == 2

    def test_preserves_audit_status_done(self, tmp_path):
        note_path = tmp_path / "annot2.md"
        safe_write_note(note_path, {"audit_status": "pending"}, "")
        content = note_path.read_text().replace("audit_status: pending", "audit_status: done")
        note_path.write_text(content)
        safe_write_note(note_path, {"audit_status": "pending"}, "")
        fm, _ = read_note(note_path)
        assert fm.get("audit_status") == "done"

    def test_creates_parent_dirs(self, tmp_path):
        note_path = tmp_path / "deep" / "nested" / "note.md"
        safe_write_note(note_path, {"x": 1}, "")
        assert note_path.exists()
