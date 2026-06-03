from __future__ import annotations
import json
import tempfile
from pathlib import Path
import pytest
from scripts.harvest_corpus import (
    load_seen_ids,
    append_new_records,
    deduplicate_combined,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_seen_ids_empty_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "corpus.jsonl"
        assert load_seen_ids(p) == set()


def test_load_seen_ids_existing_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "corpus.jsonl"
        _write_jsonl(p, [{"source_id": "a"}, {"source_id": "b"}])
        ids = load_seen_ids(p)
        assert ids == {"a", "b"}


def test_append_new_records_skips_seen() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "corpus.jsonl"
        _write_jsonl(p, [{"source_id": "a", "title": "old"}])
        records = [
            {"source_id": "a", "title": "duplicate"},
            {"source_id": "b", "title": "new"},
        ]
        added = append_new_records(p, records, seen_ids={"a"})
        assert added == 1
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[-1])["source_id"] == "b"


def test_deduplicate_combined_removes_duplicates() -> None:
    with tempfile.TemporaryDirectory() as td:
        f1 = Path(td) / "a.jsonl"
        f2 = Path(td) / "b.jsonl"
        out = Path(td) / "combined.jsonl"
        _write_jsonl(f1, [{"source": "dandi", "source_id": "000001", "title": "A"}])
        _write_jsonl(f2, [
            {"source": "dandi", "source_id": "000001", "title": "A-dup"},
            {"source": "openneuro", "source_id": "ds000001", "title": "B"},
        ])
        count = deduplicate_combined([f1, f2], out)
        assert count == 2
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
