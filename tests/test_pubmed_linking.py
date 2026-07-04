"""Tests for neural_search.literature.pubmed.

Fixture response shapes are copied from real, live NCBI E-utilities and
bioRxiv API lookups made during development (2026-07-02): PMID 32132545
(`10.1038/s41597-020-0415-9`) and bioRxiv DOI `10.1101/2024.07.09.602729`.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from neural_search.literature.api_client import TransientLookupError
from neural_search.literature.api_config import LiteratureAPIConfig
from neural_search.literature.pubmed import (
    link_corpus_to_pubmed,
    lookup_biorxiv_by_doi,
    lookup_by_doi,
    lookup_by_title,
)

_CONFIG = LiteratureAPIConfig(
    contact_email="test@example.org",
    crossref_mailto="test@example.org",
    semantic_scholar_api_key=None,
    ncbi_api_key=None,
    ncbi_tool_email="test@example.org",
    bluesky_handle=None,
    bluesky_app_password=None,
)

ESEARCH_HIT = {
    "esearchresult": {"count": "1", "retmax": "1", "retstart": "0", "idlist": ["32132545"]}
}
ESEARCH_MISS = {
    "esearchresult": {"count": "0", "retmax": "0", "retstart": "0", "idlist": []}
}
ESUMMARY_HIT = {
    "result": {
        "uids": ["32132545"],
        "32132545": {
            "uid": "32132545",
            "pubdate": "2020 Mar 4",
            "title": "A NWB-based dataset and processing pipeline of human single-neuron activity during a declarative memory task.",
            "articleids": [
                {"idtype": "pubmed", "value": "32132545"},
                {"idtype": "doi", "value": "10.1038/s41597-020-0415-9"},
            ],
        },
    }
}
BIORXIV_HIT = {
    "messages": [{"status": "ok"}],
    "collection": [
        {
            "title": "Influence of asymmetric microchannels in the structure and function of engineered neuronal circuits",
            "doi": "10.1101/2024.07.09.602729",
            "date": "2024-07-13",
        }
    ],
}
BIORXIV_MISS = {"messages": [{"status": "no posts found"}], "collection": []}


class TestLookupByDoi:
    @respx.mock
    def test_returns_normalized_dict(self) -> None:
        respx.get(url__regex=r".*esearch\.fcgi.*").mock(return_value=httpx.Response(200, json=ESEARCH_HIT))
        respx.get(url__regex=r".*esummary\.fcgi.*").mock(return_value=httpx.Response(200, json=ESUMMARY_HIT))

        hit = lookup_by_doi("10.1038/s41597-020-0415-9", config=_CONFIG)

        assert hit == {
            "pmid": "32132545",
            "doi": "10.1038/s41597-020-0415-9",
            "title": "A NWB-based dataset and processing pipeline of human single-neuron activity during a declarative memory task",
            "year": 2020,
        }

    @respx.mock
    def test_empty_idlist_is_genuine_not_found(self) -> None:
        respx.get(url__regex=r".*esearch\.fcgi.*").mock(return_value=httpx.Response(200, json=ESEARCH_MISS))
        assert lookup_by_doi("10.9999/missing", config=_CONFIG) is None

    @respx.mock
    def test_429_raises_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
        respx.get(url__regex=r".*esearch\.fcgi.*").mock(return_value=httpx.Response(429))
        with pytest.raises(TransientLookupError):
            lookup_by_doi("10.1038/s41597-020-0415-9", config=_CONFIG)

    @respx.mock
    def test_sends_api_key_when_configured(self) -> None:
        config = LiteratureAPIConfig(
            contact_email="t@example.org",
            crossref_mailto="t@example.org",
            semantic_scholar_api_key=None,
            ncbi_api_key="ncbi-key",
            ncbi_tool_email="t@example.org",
            bluesky_handle=None,
            bluesky_app_password=None,
        )
        route = respx.get(url__regex=r".*esearch\.fcgi.*").mock(
            return_value=httpx.Response(200, json=ESEARCH_MISS)
        )
        lookup_by_doi("10.1/x", config=config)
        assert route.calls.last.request.url.params["api_key"] == "ncbi-key"


class TestLookupByTitle:
    @respx.mock
    def test_returns_hit_above_threshold(self) -> None:
        respx.get(url__regex=r".*esearch\.fcgi.*").mock(return_value=httpx.Response(200, json=ESEARCH_HIT))
        respx.get(url__regex=r".*esummary\.fcgi.*").mock(return_value=httpx.Response(200, json=ESUMMARY_HIT))

        hit = lookup_by_title(
            "A NWB-based dataset and processing pipeline of human single-neuron activity", config=_CONFIG
        )
        assert hit is not None
        assert hit["doi"] == "10.1038/s41597-020-0415-9"
        assert hit["year"] == 2020


class TestLookupBiorxivByDoi:
    @respx.mock
    def test_returns_normalized_dict(self) -> None:
        respx.get("https://api.biorxiv.org/details/biorxiv/10.1101/2024.07.09.602729").mock(
            return_value=httpx.Response(200, json=BIORXIV_HIT)
        )
        hit = lookup_biorxiv_by_doi("10.1101/2024.07.09.602729")
        assert hit == {
            "doi": "10.1101/2024.07.09.602729",
            "title": "Influence of asymmetric microchannels in the structure and function of engineered neuronal circuits",
            "year": 2024,
        }

    @respx.mock
    def test_http_200_with_empty_collection_is_not_found(self) -> None:
        """Regression test: bioRxiv returns HTTP 200 even for a nonexistent
        DOI -- must check `collection` emptiness, not rely on status code."""

        respx.get("https://api.biorxiv.org/details/biorxiv/10.1101/9999.99.99.999999").mock(
            return_value=httpx.Response(200, json=BIORXIV_MISS)
        )
        assert lookup_biorxiv_by_doi("10.1101/9999.99.99.999999") is None

    @respx.mock
    def test_429_raises_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
        respx.get("https://api.biorxiv.org/details/biorxiv/10.1101/2024.07.09.602729").mock(
            return_value=httpx.Response(429)
        )
        with pytest.raises(TransientLookupError):
            lookup_biorxiv_by_doi("10.1101/2024.07.09.602729")


class TestLinkCorpusToPubmed:
    @respx.mock
    def test_biorxiv_doi_routes_to_biorxiv_not_pubmed(self, tmp_path) -> None:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            json.dumps(
                {"source": "zenodo", "source_id": "1", "doi": "10.1101/2024.07.09.602729", "title": "t"}
            )
            + "\n"
        )
        out_path = tmp_path / "out.jsonl"
        respx.get("https://api.biorxiv.org/details/biorxiv/10.1101/2024.07.09.602729").mock(
            return_value=httpx.Response(200, json=BIORXIV_HIT)
        )

        links = link_corpus_to_pubmed(corpus, out_path)

        assert len(links) == 1
        assert links[0].match_method == "biorxiv_doi_exact"
        assert links[0].paper_source == "biorxiv"

    @respx.mock
    def test_regular_doi_routes_to_pubmed(self, tmp_path) -> None:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            json.dumps(
                {"source": "neuromorpho", "source_id": "X", "doi": "10.1038/s41597-020-0415-9", "title": "t"}
            )
            + "\n"
        )
        out_path = tmp_path / "out.jsonl"
        respx.get(url__regex=r".*esearch\.fcgi.*").mock(return_value=httpx.Response(200, json=ESEARCH_HIT))
        respx.get(url__regex=r".*esummary\.fcgi.*").mock(return_value=httpx.Response(200, json=ESUMMARY_HIT))

        links = link_corpus_to_pubmed(corpus, out_path)

        assert len(links) == 1
        assert links[0].match_method == "pubmed_doi_exact"
        assert links[0].paper_source == "pubmed"
        assert links[0].paper_source_id == "32132545"

    @respx.mock
    def test_no_match_is_not_found(self, tmp_path) -> None:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            json.dumps({"source": "zenodo", "source_id": "1", "doi": "10.9999/missing", "title": "t"}) + "\n"
        )
        out_path = tmp_path / "out.jsonl"
        respx.get(url__regex=r".*esearch\.fcgi.*").mock(return_value=httpx.Response(200, json=ESEARCH_MISS))

        links = link_corpus_to_pubmed(corpus, out_path)

        assert links[0].match_method == "not_found"

    @respx.mock
    def test_transient_error_aborts_but_keeps_prior_results(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            "\n".join(
                json.dumps({"source": "zenodo", "source_id": str(i), "doi": f"10.1/{i}", "title": "t"})
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
                return httpx.Response(200, json=ESEARCH_HIT)
            if call_count == 2:
                return httpx.Response(200, json=ESUMMARY_HIT)
            return httpx.Response(429)

        respx.get(url__regex=r".*eutils\.ncbi\.nlm\.nih\.gov.*").mock(side_effect=_handler)

        links = link_corpus_to_pubmed(corpus, out_path)

        assert len(links) == 1
