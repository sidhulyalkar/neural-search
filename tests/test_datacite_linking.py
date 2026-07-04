"""Tests for neural_search.literature.datacite.

Fixture response shapes are copied from a real, live DataCite API lookup
made during development (2026-07-02) against `10.5281/zenodo.11236154`, to
keep these mocks honest rather than guessed.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from neural_search.literature.api_client import TransientLookupError
from neural_search.literature.datacite import (
    extract_related_paper_dois,
    fetch_datacite_record,
    link_corpus_to_datacite,
)

DATACITE_RECORD_WITH_PAPER = {
    "data": {
        "id": "10.5281/zenodo.11236154",
        "type": "dois",
        "attributes": {
            "doi": "10.5281/zenodo.11236154",
            "relatedIdentifiers": [
                {
                    "relationType": "IsSupplementTo",
                    "relatedIdentifier": "10.1038/s41467-024-49226-9",
                    "resourceTypeGeneral": "JournalArticle",
                    "relatedIdentifierType": "DOI",
                },
                {
                    "relationType": "IsVersionOf",
                    "relatedIdentifier": "10.5281/zenodo.11236153",
                    "relatedIdentifierType": "DOI",
                },
            ],
        },
    }
}

DATACITE_RECORD_NO_PAPER = {
    "data": {
        "id": "10.5281/zenodo.15466504",
        "type": "dois",
        "attributes": {
            "doi": "10.5281/zenodo.15466504",
            "relatedIdentifiers": [
                {
                    "relationType": "IsVersionOf",
                    "relatedIdentifier": "10.5281/zenodo.15466503",
                    "relatedIdentifierType": "DOI",
                },
            ],
        },
    }
}


class TestExtractRelatedPaperDois:
    def test_extracts_is_supplement_to(self) -> None:
        hits = extract_related_paper_dois(DATACITE_RECORD_WITH_PAPER)
        assert hits == [
            {
                "doi": "10.1038/s41467-024-49226-9",
                "relation_type": "IsSupplementTo",
                "resource_type_general": "JournalArticle",
            }
        ]

    def test_excludes_is_version_of(self) -> None:
        hits = extract_related_paper_dois(DATACITE_RECORD_NO_PAPER)
        assert hits == []

    def test_excludes_is_part_of(self) -> None:
        record = {
            "data": {
                "attributes": {
                    "relatedIdentifiers": [
                        {
                            "relationType": "IsPartOf",
                            "relatedIdentifier": "10.5281/zenodo.999",
                            "relatedIdentifierType": "DOI",
                        }
                    ]
                }
            }
        }
        assert extract_related_paper_dois(record) == []

    def test_excludes_non_doi_related_identifier_type(self) -> None:
        record = {
            "data": {
                "attributes": {
                    "relatedIdentifiers": [
                        {
                            "relationType": "IsCitedBy",
                            "relatedIdentifier": "some-url",
                            "relatedIdentifierType": "URL",
                        }
                    ]
                }
            }
        }
        assert extract_related_paper_dois(record) == []


class TestFetchDataciteRecord:
    @respx.mock
    def test_genuine_404_returns_none(self) -> None:
        respx.get("https://api.datacite.org/dois/10.9999/missing").mock(
            return_value=httpx.Response(404, json={"errors": [{"status": "404"}]})
        )
        assert fetch_datacite_record("10.9999/missing") is None

    @respx.mock
    def test_429_raises_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
        respx.get("https://api.datacite.org/dois/10.5281/zenodo.11236154").mock(
            return_value=httpx.Response(429, text="rate limited")
        )
        with pytest.raises(TransientLookupError):
            fetch_datacite_record("10.5281/zenodo.11236154")

    @respx.mock
    def test_5xx_raises_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
        respx.get("https://api.datacite.org/dois/10.5281/zenodo.11236154").mock(
            return_value=httpx.Response(503, text="unavailable")
        )
        with pytest.raises(TransientLookupError):
            fetch_datacite_record("10.5281/zenodo.11236154")

    @respx.mock
    def test_200_returns_json(self) -> None:
        respx.get("https://api.datacite.org/dois/10.5281/zenodo.11236154").mock(
            return_value=httpx.Response(200, json=DATACITE_RECORD_WITH_PAPER)
        )
        assert fetch_datacite_record("10.5281/zenodo.11236154") == DATACITE_RECORD_WITH_PAPER


class TestLinkCorpusToDatacite:
    @respx.mock
    def test_resolves_is_supplement_to_relation(self, tmp_path) -> None:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            json.dumps(
                {
                    "source": "zenodo",
                    "source_id": "11236154",
                    "doi": "10.5281/zenodo.11236154",
                    "title": "t",
                }
            )
            + "\n"
        )
        out_path = tmp_path / "out.jsonl"
        respx.get("https://api.datacite.org/dois/10.5281/zenodo.11236154").mock(
            return_value=httpx.Response(200, json=DATACITE_RECORD_WITH_PAPER)
        )

        links = link_corpus_to_datacite(corpus, out_path)

        assert len(links) == 1
        link = links[0]
        assert link.dataset_record_id == "zenodo:11236154"
        assert link.match_method == "datacite_related_identifier"
        assert link.confidence == 1.0
        assert link.paper_source == "datacite"
        assert link.paper_source_id == "10.1038/s41467-024-49226-9"
        assert link.paper_doi == "10.1038/s41467-024-49226-9"
        assert out_path.exists()

    @respx.mock
    def test_dataset_without_doi_is_not_applicable(self, tmp_path) -> None:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            json.dumps({"source": "dandi", "source_id": "000785", "title": "t"}) + "\n"
        )
        out_path = tmp_path / "out.jsonl"

        links = link_corpus_to_datacite(corpus, out_path)

        assert len(links) == 1
        assert links[0].match_method == "not_applicable_no_dataset_doi"
        assert links[0].confidence == 0.0

    @respx.mock
    def test_dataset_with_doi_but_no_related_paper_is_not_found(self, tmp_path) -> None:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            json.dumps(
                {
                    "source": "zenodo",
                    "source_id": "15466504",
                    "doi": "10.5281/zenodo.15466504",
                    "title": "t",
                }
            )
            + "\n"
        )
        out_path = tmp_path / "out.jsonl"
        respx.get("https://api.datacite.org/dois/10.5281/zenodo.15466504").mock(
            return_value=httpx.Response(200, json=DATACITE_RECORD_NO_PAPER)
        )

        links = link_corpus_to_datacite(corpus, out_path)

        assert links[0].match_method == "not_found"

    @respx.mock
    def test_transient_error_aborts_but_keeps_prior_results(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            "\n".join(
                json.dumps(
                    {
                        "source": "zenodo",
                        "source_id": str(i),
                        "doi": f"10.5281/zenodo.{i}",
                        "title": "t",
                    }
                )
                for i in range(3)
            )
            + "\n"
        )
        out_path = tmp_path / "out.jsonl"
        call_count = 0

        def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(200, json=DATACITE_RECORD_WITH_PAPER)
            return httpx.Response(429, text="rate limited")

        respx.get(url__regex=r"https://api\.datacite\.org/dois/.*").mock(side_effect=_handler)

        links = link_corpus_to_datacite(corpus, out_path)

        assert len(links) == 1
        assert links[0].match_method == "datacite_related_identifier"
