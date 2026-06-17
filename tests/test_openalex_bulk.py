"""Tests for OpenAlex bulk ingestion module (TDD: written before implementation)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from neural_search.ingestion.openalex_bulk import (
    BulkIngester,
    normalize_bulk_work,
    reconstruct_abstract,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_WORK = {
    "id": "https://openalex.org/W2741809807",
    "doi": "https://doi.org/10.1038/nature12373",
    "title": "A map of human genome variation",
    "abstract_inverted_index": {"A": [0], "map": [1], "of": [2]},
    "publication_year": 2022,
    "cited_by_count": 512,
    "concepts": [
        {"id": "https://openalex.org/C169760540", "display_name": "Neuroscience"},
        {"id": "https://openalex.org/C86803240", "display_name": "Biology"},
    ],
    "authorships": [
        {"author": {"display_name": "Jane Doe", "orcid": None}},
        {"author": {"display_name": "John Smith", "orcid": None}},
    ],
    "primary_location": {
        "source": {"display_name": "Nature"}
    },
    "open_access": {"is_oa": True, "oa_url": "https://pmc.example.com/articles/1234"},
    "topics": [
        {"display_name": "Genome-wide association study"},
        {"display_name": "Single-nucleotide polymorphism"},
    ],
}


# ---------------------------------------------------------------------------
# TestReconstructAbstract
# ---------------------------------------------------------------------------

class TestReconstructAbstract:
    def test_empty_returns_none(self):
        assert reconstruct_abstract(None) is None

    def test_empty_dict_returns_none(self):
        assert reconstruct_abstract({}) is None

    def test_single_word(self):
        result = reconstruct_abstract({"hello": [0]})
        assert result == "hello"

    def test_multi_word_correct_order(self):
        result = reconstruct_abstract({"world": [1], "hello": [0]})
        assert result == "hello world"

    def test_word_with_multiple_positions(self):
        result = reconstruct_abstract({"the": [0, 2], "cat": [1]})
        assert result == "the cat the"

    def test_returns_string_not_none_for_valid_input(self):
        result = reconstruct_abstract({"foo": [0]})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestNormalizeBulkWork
# ---------------------------------------------------------------------------

class TestNormalizeBulkWork:
    def test_full_record(self):
        result = normalize_bulk_work(MINIMAL_WORK)
        expected_keys = {
            "paper_id", "source", "source_id", "title", "abstract",
            "doi", "url", "year", "authors", "linked_datasets",
            "extracted_labels", "raw_payload_path", "created_at",
            "extractor_version", "citation_count", "venue",
            "concept_ids", "open_access_url", "topics",
        }
        assert expected_keys.issubset(result.keys())

    def test_no_abstract(self):
        work = dict(MINIMAL_WORK)
        work = {**work, "abstract_inverted_index": None}
        result = normalize_bulk_work(work)
        assert result["abstract"] is None

    def test_empty_abstract_index_gives_none(self):
        work = {**MINIMAL_WORK, "abstract_inverted_index": {}}
        result = normalize_bulk_work(work)
        assert result["abstract"] is None

    def test_no_doi(self):
        work = {**MINIMAL_WORK, "doi": None}
        result = normalize_bulk_work(work)
        assert result["doi"] is None

    def test_concept_ids_extracted(self):
        result = normalize_bulk_work(MINIMAL_WORK)
        assert "C169760540" in result["concept_ids"]
        assert "C86803240" in result["concept_ids"]
        for cid in result["concept_ids"]:
            assert not cid.startswith("https://")

    def test_citation_count(self):
        result = normalize_bulk_work(MINIMAL_WORK)
        assert result["citation_count"] == 512

    def test_venue_from_primary_location(self):
        result = normalize_bulk_work(MINIMAL_WORK)
        assert result["venue"] == "Nature"

    def test_no_venue_when_missing(self):
        work = {**MINIMAL_WORK, "primary_location": None}
        result = normalize_bulk_work(work)
        assert result["venue"] is None

    def test_no_venue_when_no_source(self):
        work = {**MINIMAL_WORK, "primary_location": {"source": None}}
        result = normalize_bulk_work(work)
        assert result["venue"] is None

    def test_topics_extracted(self):
        result = normalize_bulk_work(MINIMAL_WORK)
        assert result["topics"] == [
            "Genome-wide association study",
            "Single-nucleotide polymorphism",
        ]

    def test_topics_empty_when_missing(self):
        work = {**MINIMAL_WORK, "topics": None}
        result = normalize_bulk_work(work)
        assert result["topics"] == []

    def test_authors_are_strings(self):
        result = normalize_bulk_work(MINIMAL_WORK)
        assert isinstance(result["authors"], list)
        for author in result["authors"]:
            assert isinstance(author, str)
        assert "Jane Doe" in result["authors"]

    def test_source_is_openalex(self):
        result = normalize_bulk_work(MINIMAL_WORK)
        assert result["source"] == "openalex"

    def test_source_id_strips_url_prefix(self):
        result = normalize_bulk_work(MINIMAL_WORK)
        assert result["source_id"] == "W2741809807"
        assert not result["source_id"].startswith("https://")

    def test_extracted_labels_always_empty(self):
        result = normalize_bulk_work(MINIMAL_WORK)
        assert result["extracted_labels"] == []

    def test_linked_datasets_always_empty(self):
        result = normalize_bulk_work(MINIMAL_WORK)
        assert result["linked_datasets"] == []

    def test_open_access_url_extracted(self):
        result = normalize_bulk_work(MINIMAL_WORK)
        assert result["open_access_url"] == "https://pmc.example.com/articles/1234"

    def test_open_access_url_none_when_missing(self):
        work = {**MINIMAL_WORK, "open_access": None}
        result = normalize_bulk_work(work)
        assert result["open_access_url"] is None

    def test_year_mapped(self):
        result = normalize_bulk_work(MINIMAL_WORK)
        assert result["year"] == 2022


# ---------------------------------------------------------------------------
# TestBulkIngesterCheckpoint
# ---------------------------------------------------------------------------

class TestBulkIngesterCheckpoint:
    def test_load_missing_checkpoint_returns_star_cursor(self, tmp_path: Path):
        ingester = BulkIngester(out_dir=tmp_path)
        cursor, count = ingester.load_checkpoint()
        assert cursor == "*"
        assert count == 0

    def test_save_and_reload_checkpoint(self, tmp_path: Path):
        ingester = BulkIngester(out_dir=tmp_path)
        ingester.save_checkpoint("IjxkYXRhIjp7InBhZ2UiOjF9fQ==", 500)
        cursor, count = ingester.load_checkpoint()
        assert cursor == "IjxkYXRhIjp7InBhZ2UiOjF9fQ=="
        assert count == 500

    def test_checkpoint_has_tier_field(self, tmp_path: Path):
        ingester = BulkIngester(out_dir=tmp_path, tier="tier2")
        ingester.save_checkpoint("abc", 100)
        checkpoint_data = json.loads((tmp_path / ".checkpoint.json").read_text())
        assert checkpoint_data["tier"] == "tier2"

    def test_checkpoint_has_last_updated_field(self, tmp_path: Path):
        ingester = BulkIngester(out_dir=tmp_path)
        ingester.save_checkpoint("*", 0)
        checkpoint_data = json.loads((tmp_path / ".checkpoint.json").read_text())
        assert "last_updated" in checkpoint_data

    def test_checkpoint_path_inside_out_dir(self, tmp_path: Path):
        ingester = BulkIngester(out_dir=tmp_path)
        assert ingester._checkpoint_path == tmp_path / ".checkpoint.json"


# ---------------------------------------------------------------------------
# TestBulkIngesterRun
# ---------------------------------------------------------------------------

def _make_fetch_page_side_effect(*page_batches: list[dict]):
    """
    Returns a side_effect callable that yields page_batches in order.
    After all pages exhausted, returns ([], None) to signal end of results.
    """
    calls = list(page_batches)
    calls_iter = iter(calls)

    def side_effect(cursor: str):
        try:
            records = next(calls_iter)
            next_cursor = "next_cursor_token" if records else None
            return records, next_cursor
        except StopIteration:
            return [], None

    return side_effect


def _make_sample_records(n: int) -> list[dict]:
    """Create n minimal normalized record dicts."""
    return [
        {
            "paper_id": f"paper:openalex:W{i}",
            "source": "openalex",
            "source_id": f"W{i}",
            "title": f"Paper {i}",
            "abstract": None,
            "doi": None,
            "url": None,
            "year": 2022,
            "authors": [],
            "linked_datasets": [],
            "extracted_labels": [],
            "raw_payload_path": None,
            "created_at": "2026-06-17T00:00:00+00:00",
            "extractor_version": "v0.3.0",
            "citation_count": 0,
            "venue": None,
            "concept_ids": [],
            "open_access_url": None,
            "topics": [],
        }
        for i in range(n)
    ]


class TestBulkIngesterRun:
    def test_writes_shard_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        ingester = BulkIngester(out_dir=tmp_path, shard_size=5)
        records = _make_sample_records(5)

        side_effect = _make_fetch_page_side_effect(records, [])
        monkeypatch.setattr(ingester, "_fetch_page", side_effect)

        ingester.run()

        shard_files = list(tmp_path.glob("tier1_batch_*.jsonl"))
        assert len(shard_files) == 1

    def test_max_records_respected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        ingester = BulkIngester(out_dir=tmp_path, shard_size=100)
        # Each page returns 10 records, but max_records=15
        page_a = _make_sample_records(10)
        page_b = _make_sample_records(10)

        call_count = 0

        def side_effect(cursor: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return page_a, "cursor_2"
            elif call_count == 2:
                return page_b, "cursor_3"
            else:
                return [], None

        monkeypatch.setattr(ingester, "_fetch_page", side_effect)

        total = ingester.run(max_records=15)
        assert total <= 15

    def test_checkpoint_saved_after_flush(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        ingester = BulkIngester(out_dir=tmp_path, shard_size=3)
        records = _make_sample_records(3)

        side_effect = _make_fetch_page_side_effect(records, [])
        monkeypatch.setattr(ingester, "_fetch_page", side_effect)

        ingester.run()

        assert (tmp_path / ".checkpoint.json").exists()

    def test_returns_total_count(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        ingester = BulkIngester(out_dir=tmp_path, shard_size=100)
        records = _make_sample_records(7)

        side_effect = _make_fetch_page_side_effect(records, [])
        monkeypatch.setattr(ingester, "_fetch_page", side_effect)

        total = ingester.run()
        assert total == 7

    def test_shard_contains_valid_jsonl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        ingester = BulkIngester(out_dir=tmp_path, shard_size=2)
        records = _make_sample_records(2)

        side_effect = _make_fetch_page_side_effect(records, [])
        monkeypatch.setattr(ingester, "_fetch_page", side_effect)

        ingester.run()

        shard = next(tmp_path.glob("tier1_batch_*.jsonl"))
        lines = shard.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "paper_id" in obj

    def test_multiple_shards_written(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        ingester = BulkIngester(out_dir=tmp_path, shard_size=3)
        page_a = _make_sample_records(3)
        page_b = _make_sample_records(3)

        side_effect = _make_fetch_page_side_effect(page_a, page_b, [])
        monkeypatch.setattr(ingester, "_fetch_page", side_effect)

        ingester.run()

        shard_files = sorted(tmp_path.glob("tier1_batch_*.jsonl"))
        assert len(shard_files) == 2

    def test_empty_run_returns_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        ingester = BulkIngester(out_dir=tmp_path, shard_size=10)

        monkeypatch.setattr(ingester, "_fetch_page", lambda cursor: ([], None))

        total = ingester.run()
        assert total == 0
