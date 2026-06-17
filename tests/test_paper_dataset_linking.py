"""Tests for neural_search.literature.linking (TDD: written before implementation)."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neural_search.literature.linking import (
    DatasetPaperLink,
    link_corpus_to_literature,
    lookup_by_doi,
    lookup_by_title,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

OPENALEX_WORK_RESPONSE = {
    "id": "https://openalex.org/W2741809807",
    "doi": "https://doi.org/10.1038/s41593-020-0636-4",
    "title": "A NWB-based dataset and processing pipeline",
    "publication_year": 2020,
}

OPENALEX_SEARCH_RESPONSE = {
    "results": [
        {
            "id": "https://openalex.org/W9999999999",
            "doi": "https://doi.org/10.1234/fake.2021",
            "title": "A NWB-based dataset and processing pipeline of human single-neuron activity",
            "publication_year": 2021,
        }
    ]
}


def _make_mock_response(status_code: int, json_body: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_body
    mock_resp.raise_for_status.side_effect = (
        None
        if status_code < 400
        else Exception(f"HTTP {status_code}")
    )
    return mock_resp


# ---------------------------------------------------------------------------
# TestLookupByDoi
# ---------------------------------------------------------------------------


class TestLookupByDoi:
    def test_found_returns_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resp = _make_mock_response(200, OPENALEX_WORK_RESPONSE)
        mock_get = MagicMock(return_value=mock_resp)
        monkeypatch.setattr("neural_search.literature.linking._http_get", mock_get)

        result = lookup_by_doi("10.1038/s41593-020-0636-4")

        assert result is not None
        assert result["openalex_id"] == "W2741809807"
        assert result["doi"] == "10.1038/s41593-020-0636-4"
        assert result["title"] == "A NWB-based dataset and processing pipeline"
        assert result["year"] == 2020

    def test_not_found_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resp = _make_mock_response(404, {})
        mock_resp.raise_for_status.side_effect = Exception("HTTP 404")
        mock_get = MagicMock(return_value=mock_resp)
        monkeypatch.setattr("neural_search.literature.linking._http_get", mock_get)

        result = lookup_by_doi("10.9999/does-not-exist")

        assert result is None

    def test_http_error_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_get = MagicMock(side_effect=Exception("Connection refused"))
        monkeypatch.setattr("neural_search.literature.linking._http_get", mock_get)

        result = lookup_by_doi("10.1038/anything")

        assert result is None


# ---------------------------------------------------------------------------
# TestLookupByTitle
# ---------------------------------------------------------------------------


class TestLookupByTitle:
    def test_high_similarity_returns_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_resp = _make_mock_response(200, OPENALEX_SEARCH_RESPONSE)
        mock_get = MagicMock(return_value=mock_resp)
        monkeypatch.setattr("neural_search.literature.linking._http_get", mock_get)

        result = lookup_by_title(
            "A NWB-based dataset and processing pipeline of human single-neuron activity"
        )

        assert result is not None
        assert result["openalex_id"] == "W9999999999"

    def test_low_similarity_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        low_sim_response = {
            "results": [
                {
                    "id": "https://openalex.org/W1111111111",
                    "doi": None,
                    "title": "Completely unrelated paper about chemistry",
                    "publication_year": 2019,
                }
            ]
        }
        mock_resp = _make_mock_response(200, low_sim_response)
        mock_get = MagicMock(return_value=mock_resp)
        monkeypatch.setattr("neural_search.literature.linking._http_get", mock_get)

        result = lookup_by_title(
            "A NWB-based dataset and processing pipeline of human single-neuron activity"
        )

        assert result is None

    def test_no_results_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        empty_response = {"results": []}
        mock_resp = _make_mock_response(200, empty_response)
        mock_get = MagicMock(return_value=mock_resp)
        monkeypatch.setattr("neural_search.literature.linking._http_get", mock_get)

        result = lookup_by_title("Some dataset title")

        assert result is None

    def test_http_error_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_get = MagicMock(side_effect=Exception("Timeout"))
        monkeypatch.setattr("neural_search.literature.linking._http_get", mock_get)

        result = lookup_by_title("Some title")

        assert result is None


# ---------------------------------------------------------------------------
# TestDatasetPaperLink
# ---------------------------------------------------------------------------


class TestDatasetPaperLink:
    def test_doi_exact_confidence_is_one(self) -> None:
        link = DatasetPaperLink(
            dataset_record_id="dandi:000004",
            paper_openalex_id="W2741809807",
            paper_doi="10.1038/s41593-020-0636-4",
            paper_title="A NWB-based dataset",
            paper_year=2020,
            match_method="doi_exact",
            confidence=1.0,
        )
        assert link.confidence == 1.0
        assert link.match_method == "doi_exact"

    def test_not_found_confidence_is_zero(self) -> None:
        link = DatasetPaperLink(
            dataset_record_id="dandi:000999",
            paper_openalex_id="",
            paper_doi=None,
            paper_title=None,
            paper_year=None,
            match_method="not_found",
            confidence=0.0,
        )
        assert link.confidence == 0.0
        assert link.match_method == "not_found"

    def test_dataclass_fields_present(self) -> None:
        field_names = {f.name for f in fields(DatasetPaperLink)}
        expected = {
            "dataset_record_id",
            "paper_openalex_id",
            "paper_doi",
            "paper_title",
            "paper_year",
            "match_method",
            "confidence",
        }
        assert expected == field_names


# ---------------------------------------------------------------------------
# TestLinkCorpusToLiterature
# ---------------------------------------------------------------------------

CORPUS_WITH_DOI = json.dumps(
    {
        "source": "dandi",
        "source_id": "000004",
        "title": "A NWB-based dataset and processing pipeline of human single-neuron activity",
        "doi": "10.1038/s41593-020-0636-4",
    }
)

CORPUS_WITHOUT_DOI = json.dumps(
    {
        "source": "dandi",
        "source_id": "000785",
        "title": "Projection-specific routing of odor information",
        "doi": None,
    }
)

CORPUS_WITHOUT_DOI_FIELD = json.dumps(
    {
        "source": "openneuro",
        "source_id": "ds001234",
        "title": "A fMRI study of working memory",
    }
)


def _write_corpus(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


class TestLinkCorpusToLiterature:
    def test_creates_output_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus_file = tmp_path / "corpus.jsonl"
        _write_corpus(corpus_file, [CORPUS_WITH_DOI])
        out_file = tmp_path / "links.jsonl"

        doi_result = {
            "openalex_id": "W2741809807",
            "doi": "10.1038/s41593-020-0636-4",
            "title": "A NWB-based dataset",
            "year": 2020,
        }
        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_doi",
            lambda doi: doi_result,
        )
        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_title",
            lambda title, year=None: None,
        )

        link_corpus_to_literature(corpus_file, out_file)

        assert out_file.exists()

    def test_doi_lookup_attempted_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus_file = tmp_path / "corpus.jsonl"
        _write_corpus(corpus_file, [CORPUS_WITH_DOI])
        out_file = tmp_path / "links.jsonl"

        doi_calls: list[str] = []
        title_calls: list[str] = []

        def fake_doi(doi: str) -> dict:
            doi_calls.append(doi)
            return {
                "openalex_id": "W111",
                "doi": doi,
                "title": "Test",
                "year": 2020,
            }

        def fake_title(title: str, year: int | None = None) -> None:
            title_calls.append(title)
            return None

        monkeypatch.setattr("neural_search.literature.linking.lookup_by_doi", fake_doi)
        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_title", fake_title
        )

        link_corpus_to_literature(corpus_file, out_file)

        assert len(doi_calls) == 1
        assert doi_calls[0] == "10.1038/s41593-020-0636-4"
        assert len(title_calls) == 0

    def test_skip_without_doi(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus_file = tmp_path / "corpus.jsonl"
        _write_corpus(
            corpus_file, [CORPUS_WITHOUT_DOI, CORPUS_WITHOUT_DOI_FIELD]
        )
        out_file = tmp_path / "links.jsonl"

        title_calls: list[str] = []

        def fake_title(title: str, year: int | None = None) -> None:
            title_calls.append(title)
            return None

        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_doi",
            lambda doi: None,
        )
        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_title", fake_title
        )

        link_corpus_to_literature(corpus_file, out_file, skip_without_doi=True)

        assert len(title_calls) == 0

    def test_writes_valid_jsonl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus_file = tmp_path / "corpus.jsonl"
        _write_corpus(corpus_file, [CORPUS_WITH_DOI])
        out_file = tmp_path / "links.jsonl"

        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_doi",
            lambda doi: {
                "openalex_id": "W2741809807",
                "doi": doi,
                "title": "A NWB-based dataset",
                "year": 2020,
            },
        )
        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_title",
            lambda title, year=None: None,
        )

        link_corpus_to_literature(corpus_file, out_file)

        lines = [l for l in out_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["dataset_record_id"] == "dandi:000004"
        assert record["paper_openalex_id"] == "W2741809807"
        assert record["match_method"] == "doi_exact"
        assert record["confidence"] == 1.0

    def test_returns_list_of_links(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus_file = tmp_path / "corpus.jsonl"
        _write_corpus(corpus_file, [CORPUS_WITH_DOI, CORPUS_WITHOUT_DOI])
        out_file = tmp_path / "links.jsonl"

        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_doi",
            lambda doi: {
                "openalex_id": "W111",
                "doi": doi,
                "title": "Found",
                "year": 2020,
            },
        )
        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_title",
            lambda title, year=None: None,
        )

        result = link_corpus_to_literature(corpus_file, out_file)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, DatasetPaperLink) for r in result)

    def test_handles_corpus_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus_dir = tmp_path / "corpus_shards"
        corpus_dir.mkdir()
        (corpus_dir / "shard_0.jsonl").write_text(CORPUS_WITH_DOI + "\n")
        (corpus_dir / "shard_1.jsonl").write_text(CORPUS_WITHOUT_DOI_FIELD + "\n")
        out_file = tmp_path / "links.jsonl"

        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_doi",
            lambda doi: {
                "openalex_id": "W222",
                "doi": doi,
                "title": "Some paper",
                "year": 2021,
            },
        )
        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_title",
            lambda title, year=None: None,
        )

        result = link_corpus_to_literature(corpus_dir, out_file)

        assert len(result) == 2

    def test_title_fallback_when_doi_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus_file = tmp_path / "corpus.jsonl"
        _write_corpus(corpus_file, [CORPUS_WITH_DOI])
        out_file = tmp_path / "links.jsonl"

        title_calls: list[str] = []

        def fake_title(title: str, year: int | None = None) -> dict:
            title_calls.append(title)
            return {
                "openalex_id": "W_TITLE",
                "doi": None,
                "title": title,
                "year": 2020,
            }

        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_doi",
            lambda doi: None,
        )
        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_title", fake_title
        )

        result = link_corpus_to_literature(corpus_file, out_file)

        assert len(title_calls) == 1
        assert result[0].match_method == "title_fuzzy"
        assert 0.7 <= result[0].confidence <= 0.9

    def test_no_doi_falls_back_to_title(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus_file = tmp_path / "corpus.jsonl"
        _write_corpus(corpus_file, [CORPUS_WITHOUT_DOI])
        out_file = tmp_path / "links.jsonl"

        title_calls: list[str] = []

        def fake_title(title: str, year: int | None = None) -> dict:
            title_calls.append(title)
            return {
                "openalex_id": "W_NODOI",
                "doi": None,
                "title": title,
                "year": 2022,
                "similarity": 0.85,
            }

        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_doi",
            lambda doi: None,
        )
        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_title", fake_title
        )

        result = link_corpus_to_literature(corpus_file, out_file)

        assert len(title_calls) == 1
        assert result[0].match_method == "title_fuzzy"
        assert result[0].confidence == 0.85

    def test_no_doi_not_found_returns_not_found_link(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus_file = tmp_path / "corpus.jsonl"
        _write_corpus(corpus_file, [CORPUS_WITHOUT_DOI_FIELD])
        out_file = tmp_path / "links.jsonl"

        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_doi",
            lambda doi: None,
        )
        monkeypatch.setattr(
            "neural_search.literature.linking.lookup_by_title",
            lambda title, year=None: None,
        )

        result = link_corpus_to_literature(corpus_file, out_file)

        assert result[0].match_method == "not_found"
        assert result[0].confidence == 0.0
