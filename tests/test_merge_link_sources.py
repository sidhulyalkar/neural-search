"""Tests for neural_search.literature.merge_links.merge_link_sources."""

from __future__ import annotations

import json

from neural_search.literature.merge_links import merge_link_sources


def _write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _read_jsonl(path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_union_across_sources_keeps_both_rows_for_same_dataset(tmp_path) -> None:
    openalex_path = tmp_path / "openalex.jsonl"
    _write_jsonl(
        openalex_path,
        [{"dataset_record_id": "zenodo:1", "paper_source": "openalex", "match_method": "doi_exact", "confidence": 1.0}],
    )
    datacite_path = tmp_path / "datacite.jsonl"
    _write_jsonl(
        datacite_path,
        [
            {
                "dataset_record_id": "zenodo:1",
                "paper_source": "datacite",
                "match_method": "datacite_related_identifier",
                "confidence": 1.0,
            }
        ],
    )
    out_path = tmp_path / "merged.jsonl"

    counts = merge_link_sources(
        source_paths={"openalex": openalex_path, "datacite": datacite_path}, out_path=out_path
    )

    rows = _read_jsonl(out_path)
    assert len(rows) == 2
    assert counts == {"openalex": 1, "datacite": 1}
    sources = {r["paper_source"] for r in rows}
    assert sources == {"openalex", "datacite"}


def test_duplicate_within_source_keeps_highest_confidence(tmp_path) -> None:
    path = tmp_path / "crossref.jsonl"
    _write_jsonl(
        path,
        [
            {"dataset_record_id": "zenodo:1", "paper_source": "crossref", "match_method": "crossref_title_fuzzy", "confidence": 0.7},
            {"dataset_record_id": "zenodo:1", "paper_source": "crossref", "match_method": "crossref_doi_exact", "confidence": 1.0},
        ],
    )
    out_path = tmp_path / "merged.jsonl"

    merge_link_sources(source_paths={"crossref": path}, out_path=out_path)

    rows = _read_jsonl(out_path)
    assert len(rows) == 1
    assert rows[0]["match_method"] == "crossref_doi_exact"
    assert rows[0]["confidence"] == 1.0


def test_missing_source_file_is_skipped_gracefully(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.jsonl"
    out_path = tmp_path / "merged.jsonl"

    counts = merge_link_sources(source_paths={"crossref": missing}, out_path=out_path)

    assert counts == {}
    assert _read_jsonl(out_path) == []


def test_not_found_rows_are_preserved_for_accounting(tmp_path) -> None:
    path = tmp_path / "pubmed.jsonl"
    _write_jsonl(
        path,
        [{"dataset_record_id": "zenodo:1", "paper_source": "pubmed", "match_method": "not_found", "confidence": 0.0}],
    )
    out_path = tmp_path / "merged.jsonl"

    merge_link_sources(source_paths={"pubmed": path}, out_path=out_path)

    rows = _read_jsonl(out_path)
    assert len(rows) == 1
    assert rows[0]["match_method"] == "not_found"
