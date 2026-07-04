"""Tests for neural_search.literature.linking (TDD: written before implementation)."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neural_search.literature.linking import (
    DatasetPaperLink,
    link_corpus_to_literature,
    link_corpus_to_local_literature,
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
            "paper_source",
            "paper_source_id",
        }
        assert expected == field_names

    def test_paper_source_id_defaults_from_openalex_id(self) -> None:
        """New paper_source/paper_source_id fields (added for Crossref/DataCite/
        Semantic Scholar/PubMed support) must not break existing OpenAlex-only
        call sites that never set them explicitly."""

        link = DatasetPaperLink(
            dataset_record_id="dandi:000004",
            paper_openalex_id="W123",
            paper_doi="10.1/x",
            paper_title="t",
            paper_year=2020,
            match_method="doi_exact",
            confidence=1.0,
        )
        assert link.paper_source == "openalex"
        assert link.paper_source_id == "W123"


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

        lines = [line for line in out_file.read_text().splitlines() if line.strip()]
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

    def test_local_literature_links_by_doi(self, tmp_path: Path) -> None:
        corpus_file = tmp_path / "corpus.jsonl"
        paper_file = tmp_path / "papers.jsonl"
        out_file = tmp_path / "links.jsonl"
        _write_corpus(corpus_file, [CORPUS_WITH_DOI])
        _write_corpus(
            paper_file,
            [
                json.dumps(
                    {
                        "paper_id": "paper:openalex:W2741809807",
                        "source_id": "W2741809807",
                        "doi": "https://doi.org/10.1038/s41593-020-0636-4",
                        "title": "A NWB-based dataset and processing pipeline",
                        "year": 2020,
                    }
                )
            ],
        )

        links = link_corpus_to_local_literature(corpus_file, paper_file, out_file)

        assert links[0].match_method == "doi_exact"
        assert links[0].paper_openalex_id == "W2741809807"
        assert links[0].confidence == 1.0

    def test_local_literature_links_by_title(self, tmp_path: Path) -> None:
        corpus_file = tmp_path / "corpus.jsonl"
        paper_file = tmp_path / "papers.jsonl"
        out_file = tmp_path / "links.jsonl"
        _write_corpus(corpus_file, [CORPUS_WITHOUT_DOI_FIELD])
        _write_corpus(
            paper_file,
            [
                json.dumps(
                    {
                        "paper_id": "paper:openalex:W_TITLE",
                        "source_id": "W_TITLE",
                        "doi": None,
                        "title": "A fMRI study of working memory",
                        "year": 2021,
                    }
                )
            ],
        )

        links = link_corpus_to_local_literature(corpus_file, paper_file, out_file)

        assert links[0].match_method == "title_fuzzy_local"
        assert links[0].paper_openalex_id == "W_TITLE"
        assert links[0].confidence >= 0.82

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


# ---------------------------------------------------------------------------
# TestTransientLookupError
# ---------------------------------------------------------------------------


class TestTransientLookupError:
    """Regression tests for a 2026-07-02 incident: a full-corpus live-linking
    run exhausted this environment's OpenAlex request budget after ~70
    records (HTTP 429), and the bare `except Exception: return None` that
    used to wrap every lookup silently converted every subsequent
    budget-exhausted request into a false "not_found" for the remaining
    ~7,100 records, corrupting the run with no visible error."""

    def test_lookup_by_doi_raises_on_429(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from neural_search.literature.linking import TransientLookupError

        mock_resp = MagicMock(status_code=429, text="Insufficient budget")
        monkeypatch.setattr(
            "neural_search.literature.linking._http_get", lambda *a, **k: mock_resp
        )

        with pytest.raises(TransientLookupError):
            lookup_by_doi("10.1038/anything")

    def test_lookup_by_doi_raises_on_5xx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from neural_search.literature.linking import TransientLookupError

        mock_resp = MagicMock(status_code=503, text="Service unavailable")
        monkeypatch.setattr(
            "neural_search.literature.linking._http_get", lambda *a, **k: mock_resp
        )

        with pytest.raises(TransientLookupError):
            lookup_by_doi("10.1038/anything")

    def test_lookup_by_title_raises_on_429(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from neural_search.literature.linking import TransientLookupError

        mock_resp = MagicMock(status_code=429, text="Insufficient budget")
        monkeypatch.setattr(
            "neural_search.literature.linking._http_get", lambda *a, **k: mock_resp
        )

        with pytest.raises(TransientLookupError):
            lookup_by_title("Some title")

    def test_genuine_404_still_returns_none_not_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_resp = MagicMock(status_code=404)
        mock_resp.raise_for_status.side_effect = Exception("404 Not Found")
        monkeypatch.setattr(
            "neural_search.literature.linking._http_get", lambda *a, **k: mock_resp
        )

        assert lookup_by_doi("10.9999/does-not-exist") is None

    def test_link_corpus_to_literature_stops_early_on_transient_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from neural_search.literature.linking import TransientLookupError

        corpus_file = tmp_path / "corpus.jsonl"
        _write_corpus(corpus_file, [CORPUS_WITH_DOI, CORPUS_WITHOUT_DOI_FIELD])
        out_file = tmp_path / "links.jsonl"

        calls = []

        def fake_doi(doi):
            calls.append(doi)
            raise TransientLookupError("budget exhausted")

        monkeypatch.setattr("neural_search.literature.linking.lookup_by_doi", fake_doi)

        result = link_corpus_to_literature(corpus_file, out_file)

        # must stop immediately, not process the second record as "not_found"
        assert result == []
        assert len(calls) == 1
        assert out_file.read_text() == ""
