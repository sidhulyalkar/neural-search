"""Tests for neural_search.literature.semantic_scholar.

Fixture response shapes are copied from a real, live Semantic Scholar Graph
API lookup made during development (2026-07-02) against
`10.1038/s41597-020-0415-9`.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from neural_search.literature.api_client import TransientLookupError
from neural_search.literature.api_config import LiteratureAPIConfig
from neural_search.literature.semantic_scholar import (
    link_corpus_to_semantic_scholar,
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

_CONFIG_WITH_KEY = LiteratureAPIConfig(
    contact_email="test@example.org",
    crossref_mailto="test@example.org",
    semantic_scholar_api_key="s2-key",
    ncbi_api_key=None,
    ncbi_tool_email="test@example.org",
    bluesky_handle=None,
    bluesky_app_password=None,
)

DOI_RESPONSE = {
    "paperId": "effc945255806851b6994d76854eb827e9c2336a",
    "externalIds": {
        "PubMedCentral": "7055261",
        "MAG": "3007951719",
        "DOI": "10.1038/s41597-020-0415-9",
        "CorpusId": 211836584,
        "PubMed": "32132545",
    },
    "title": "A NWB-based dataset and processing pipeline of human single-neuron activity during a declarative memory task",
    "year": 2020,
}

SEARCH_RESPONSE = {"data": [DOI_RESPONSE]}


class TestLookupByDoi:
    @respx.mock
    def test_returns_normalized_dict(self) -> None:
        respx.get("https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41597-020-0415-9").mock(
            return_value=httpx.Response(200, json=DOI_RESPONSE)
        )
        hit = lookup_by_doi("10.1038/s41597-020-0415-9", config=_CONFIG)
        assert hit == {
            "paper_id": "effc945255806851b6994d76854eb827e9c2336a",
            "doi": "10.1038/s41597-020-0415-9",
            "title": "A NWB-based dataset and processing pipeline of human single-neuron activity during a declarative memory task",
            "year": 2020,
        }

    @respx.mock
    def test_sends_api_key_header_when_configured(self) -> None:
        route = respx.get(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41597-020-0415-9"
        ).mock(return_value=httpx.Response(200, json=DOI_RESPONSE))
        lookup_by_doi("10.1038/s41597-020-0415-9", config=_CONFIG_WITH_KEY)
        assert route.calls.last.request.headers["x-api-key"] == "s2-key"

    @respx.mock
    def test_no_key_sends_no_api_key_header(self) -> None:
        route = respx.get(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41597-020-0415-9"
        ).mock(return_value=httpx.Response(200, json=DOI_RESPONSE))
        lookup_by_doi("10.1038/s41597-020-0415-9", config=_CONFIG)
        assert "x-api-key" not in route.calls.last.request.headers

    @respx.mock
    def test_genuine_404_returns_none(self) -> None:
        respx.get("https://api.semanticscholar.org/graph/v1/paper/DOI:10.9999/missing").mock(
            return_value=httpx.Response(404)
        )
        assert lookup_by_doi("10.9999/missing", config=_CONFIG) is None

    @respx.mock
    def test_429_raises_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
        respx.get("https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41597-020-0415-9").mock(
            return_value=httpx.Response(429)
        )
        with pytest.raises(TransientLookupError):
            lookup_by_doi("10.1038/s41597-020-0415-9", config=_CONFIG)


class TestLookupByTitle:
    @respx.mock
    def test_returns_hit_above_threshold(self) -> None:
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE)
        )
        hit = lookup_by_title(
            "A NWB-based dataset and processing pipeline of human single-neuron activity", config=_CONFIG
        )
        assert hit is not None
        assert hit["doi"] == "10.1038/s41597-020-0415-9"

    @respx.mock
    def test_no_results_returns_none(self) -> None:
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        assert lookup_by_title("anything", config=_CONFIG) is None


class TestLinkCorpusToSemanticScholar:
    @respx.mock
    def test_doi_exact_match(self, tmp_path) -> None:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            json.dumps(
                {"source": "neuromorpho", "source_id": "X", "doi": "10.1038/s41597-020-0415-9", "title": "t"}
            )
            + "\n"
        )
        out_path = tmp_path / "out.jsonl"
        respx.get("https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41597-020-0415-9").mock(
            return_value=httpx.Response(200, json=DOI_RESPONSE)
        )

        links = link_corpus_to_semantic_scholar(corpus, out_path)

        assert len(links) == 1
        link = links[0]
        assert link.match_method == "semantic_scholar_doi_exact"
        assert link.paper_source == "semantic_scholar"
        assert link.paper_source_id == "effc945255806851b6994d76854eb827e9c2336a"

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
                return httpx.Response(200, json=DOI_RESPONSE)
            return httpx.Response(429)

        respx.get(url__regex=r"https://api\.semanticscholar\.org/graph/v1/paper/DOI:.*").mock(
            side_effect=_handler
        )

        links = link_corpus_to_semantic_scholar(corpus, out_path)

        assert len(links) == 1
