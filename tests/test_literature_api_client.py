"""Tests for neural_search.literature.api_client -- the shared
TransientLookupError type used across OpenAlex, Crossref, DataCite,
Semantic Scholar, and PubMed clients."""

from __future__ import annotations

import httpx
import pytest
import respx

from neural_search.literature.api_client import (
    TransientLookupError,
    get_or_raise_transient,
    raise_if_transient,
)


def test_raises_on_429() -> None:
    resp = httpx.Response(429, text="quota exceeded")
    with pytest.raises(TransientLookupError, match="Crossref"):
        raise_if_transient(resp, source="Crossref")


def test_raises_on_5xx() -> None:
    resp = httpx.Response(503, text="unavailable")
    with pytest.raises(TransientLookupError, match="DataCite"):
        raise_if_transient(resp, source="DataCite")


def test_does_not_raise_on_404() -> None:
    resp = httpx.Response(404, text="not found")
    raise_if_transient(resp, source="Crossref")  # should not raise


def test_does_not_raise_on_200() -> None:
    resp = httpx.Response(200, json={})
    raise_if_transient(resp, source="Crossref")  # should not raise


def test_linking_module_reexports_same_exception_type() -> None:
    from neural_search.literature.linking import TransientLookupError as LinkingError

    assert LinkingError is TransientLookupError


class TestGetOrRaiseTransient:
    """Regression tests for a real incident (2026-07-02): a corpus-scale
    Crossref run crashed the whole process with an uncaught
    httpx.ReadTimeout after http_get_with_retry exhausted its retries,
    because callers only caught TransientLookupError, not raw httpx
    exceptions. get_or_raise_transient converts transport-level failures
    into TransientLookupError too, so a run aborts gracefully (saving
    partial progress) instead of crashing uncaught."""

    @respx.mock
    def test_transport_error_after_retries_becomes_transient_lookup_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
        respx.get("https://example.test/x").mock(side_effect=httpx.ReadTimeout("timed out"))

        with pytest.raises(TransientLookupError):
            get_or_raise_transient("https://example.test/x", source="TestSource", max_retries=1)

    @respx.mock
    def test_status_level_transience_still_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
        respx.get("https://example.test/x").mock(return_value=httpx.Response(429))

        with pytest.raises(TransientLookupError):
            get_or_raise_transient("https://example.test/x", source="TestSource")

    @respx.mock
    def test_success_returns_response(self) -> None:
        respx.get("https://example.test/x").mock(return_value=httpx.Response(200, json={"ok": True}))

        resp = get_or_raise_transient("https://example.test/x", source="TestSource")

        assert resp.status_code == 200
