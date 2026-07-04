"""Tests for neural_search.literature.crossref.

Fixture response shapes are copied from real, live Crossref API lookups made
during development (2026-07-02): `10.1038/s41597-020-0415-9` (doi/title
lookup) and `10.1016/j.micpro.2020.103768` (a real Elsevier retraction
notice, to keep the update-to mock honest rather than guessed).
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from neural_search.literature.api_client import TransientLookupError
from neural_search.literature.api_config import LiteratureAPIConfig
from neural_search.literature.crossref import (
    fetch_retraction_status,
    link_corpus_to_crossref,
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

WORKS_RESPONSE = {
    "message": {
        "DOI": "10.1038/s41597-020-0415-9",
        "title": ["A NWB-based dataset and processing pipeline of human single-neuron activity"],
        "published": {"date-parts": [[2020, 3, 4]]},
    }
}

SEARCH_RESPONSE = {
    "message": {
        "items": [
            {
                "DOI": "10.1038/s41597-020-0415-9",
                "title": ["A NWB-based dataset and processing pipeline of human single-neuron activity"],
                "published-print": {"date-parts": [[2020, 3, 4]]},
            }
        ]
    }
}

RETRACTION_RESPONSE = {
    "message": {
        "DOI": "10.1016/j.micpro.2020.103768",
        "title": ["RETRACTED: Cross-Cultural communication of language learning social software"],
        "update-to": [
            {
                "DOI": "10.1016/j.micpro.2020.103768",
                "type": "retraction",
                "label": "Retraction",
                "source": "publisher",
                "updated": {"date-parts": [[2021, 3, 1]]},
            }
        ],
    }
}

NO_UPDATES_RESPONSE = {
    "message": {"DOI": "10.7717/peerj.8178", "title": ["A paper"], "update-to": None}
}


class TestLookupByDoi:
    @respx.mock
    def test_returns_normalized_dict(self) -> None:
        respx.get("https://api.crossref.org/works/10.1038/s41597-020-0415-9").mock(
            return_value=httpx.Response(200, json=WORKS_RESPONSE)
        )
        hit = lookup_by_doi("10.1038/s41597-020-0415-9", config=_CONFIG)
        assert hit == {
            "doi": "10.1038/s41597-020-0415-9",
            "title": "A NWB-based dataset and processing pipeline of human single-neuron activity",
            "year": 2020,
        }

    @respx.mock
    def test_genuine_404_returns_none(self) -> None:
        respx.get("https://api.crossref.org/works/10.9999/missing").mock(
            return_value=httpx.Response(404)
        )
        assert lookup_by_doi("10.9999/missing", config=_CONFIG) is None

    @respx.mock
    def test_429_raises_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
        respx.get("https://api.crossref.org/works/10.1038/s41597-020-0415-9").mock(
            return_value=httpx.Response(429)
        )
        with pytest.raises(TransientLookupError):
            lookup_by_doi("10.1038/s41597-020-0415-9", config=_CONFIG)


class TestLookupByTitle:
    @respx.mock
    def test_returns_hit_above_threshold(self) -> None:
        respx.get("https://api.crossref.org/works").mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE)
        )
        hit = lookup_by_title(
            "A NWB-based dataset and processing pipeline of human single-neuron activity", config=_CONFIG
        )
        assert hit is not None
        assert hit["doi"] == "10.1038/s41597-020-0415-9"
        assert hit["year"] == 2020

    @respx.mock
    def test_returns_none_below_threshold(self) -> None:
        respx.get("https://api.crossref.org/works").mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE)
        )
        hit = lookup_by_title("Completely unrelated title about something else entirely", config=_CONFIG)
        assert hit is None

    @respx.mock
    def test_no_results_returns_none(self) -> None:
        respx.get("https://api.crossref.org/works").mock(
            return_value=httpx.Response(200, json={"message": {"items": []}})
        )
        assert lookup_by_title("anything", config=_CONFIG) is None


class TestFetchRetractionStatus:
    @respx.mock
    def test_detects_real_retraction_shape(self) -> None:
        respx.get("https://api.crossref.org/works/10.1016/j.micpro.2020.103768").mock(
            return_value=httpx.Response(200, json=RETRACTION_RESPONSE)
        )
        status = fetch_retraction_status("10.1016/j.micpro.2020.103768", config=_CONFIG)
        assert status["status"] == "retracted"
        assert status["related_dois"] == ["10.1016/j.micpro.2020.103768"]
        assert status["source"] == "crossref"

    @respx.mock
    def test_no_update_to_is_none_status(self) -> None:
        respx.get("https://api.crossref.org/works/10.7717/peerj.8178").mock(
            return_value=httpx.Response(200, json=NO_UPDATES_RESPONSE)
        )
        status = fetch_retraction_status("10.7717/peerj.8178", config=_CONFIG)
        assert status["status"] == "none"
        assert status["related_dois"] == []

    @respx.mock
    def test_correction_type_detected(self) -> None:
        response = {
            "message": {
                "DOI": "10.1/x",
                "update-to": [{"DOI": "10.1/y", "type": "correction"}],
            }
        }
        respx.get("https://api.crossref.org/works/10.1/x").mock(
            return_value=httpx.Response(200, json=response)
        )
        status = fetch_retraction_status("10.1/x", config=_CONFIG)
        assert status["status"] == "corrected"

    @respx.mock
    def test_404_is_conservative_none_not_raised(self) -> None:
        respx.get("https://api.crossref.org/works/10.9999/missing").mock(
            return_value=httpx.Response(404)
        )
        status = fetch_retraction_status("10.9999/missing", config=_CONFIG)
        assert status["status"] == "none"

    @respx.mock
    def test_429_still_raises_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
        respx.get("https://api.crossref.org/works/10.1/x").mock(return_value=httpx.Response(429))
        with pytest.raises(TransientLookupError):
            fetch_retraction_status("10.1/x", config=_CONFIG)


class TestLinkCorpusToCrossref:
    @respx.mock
    def test_doi_exact_match(self, tmp_path) -> None:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            json.dumps(
                {
                    "source": "neuromorpho",
                    "source_id": "X",
                    "doi": "10.1038/s41597-020-0415-9",
                    "title": "t",
                }
            )
            + "\n"
        )
        out_path = tmp_path / "out.jsonl"
        respx.get("https://api.crossref.org/works/10.1038/s41597-020-0415-9").mock(
            return_value=httpx.Response(200, json=WORKS_RESPONSE)
        )

        links = link_corpus_to_crossref(corpus, out_path)

        assert len(links) == 1
        link = links[0]
        assert link.match_method == "crossref_doi_exact"
        assert link.paper_source == "crossref"
        assert link.confidence == 1.0
        assert out_path.exists()

    @respx.mock
    def test_no_match_is_not_found(self, tmp_path) -> None:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            json.dumps({"source": "zenodo", "source_id": "1", "doi": "10.9999/missing", "title": "t"})
            + "\n"
        )
        out_path = tmp_path / "out.jsonl"
        respx.get("https://api.crossref.org/works/10.9999/missing").mock(
            return_value=httpx.Response(404)
        )
        respx.get("https://api.crossref.org/works").mock(
            return_value=httpx.Response(200, json={"message": {"items": []}})
        )

        links = link_corpus_to_crossref(corpus, out_path)

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
                return httpx.Response(200, json=WORKS_RESPONSE)
            return httpx.Response(429)

        respx.get(url__regex=r"https://api\.crossref\.org/works/.*").mock(side_effect=_handler)

        links = link_corpus_to_crossref(corpus, out_path)

        assert len(links) == 1
