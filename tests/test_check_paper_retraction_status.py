"""Tests for scripts/check_paper_retraction_status.py's DOI-collection logic."""

from __future__ import annotations

import json

import scripts.check_paper_retraction_status as script


def _write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_collects_unique_real_dois_across_files(tmp_path, monkeypatch):
    openalex = tmp_path / "openalex.jsonl"
    _write_jsonl(
        openalex,
        [
            {"dataset_record_id": "a", "match_method": "doi_exact", "paper_doi": "10.1/x"},
            {"dataset_record_id": "b", "match_method": "not_found", "paper_doi": None},
        ],
    )
    crossref = tmp_path / "crossref.jsonl"
    _write_jsonl(
        crossref,
        [
            # Same DOI as openalex row -- should dedup.
            {"dataset_record_id": "c", "match_method": "crossref_doi_exact", "paper_doi": "10.1/x"},
            {"dataset_record_id": "d", "match_method": "crossref_doi_exact", "paper_doi": "10.2/y"},
        ],
    )
    monkeypatch.setattr(script, "LINK_FILES", [openalex, crossref])

    dois = script.collect_unique_dois()

    assert dois == ["10.1/x", "10.2/y"]


def test_missing_files_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "LINK_FILES", [tmp_path / "missing.jsonl"])
    assert script.collect_unique_dois() == []
