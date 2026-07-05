"""Tests for neural_search.http_utils.http_get_with_retry -- the shared
retry/backoff primitive used by every new literature-linking and discourse
API client (Crossref, DataCite, Semantic Scholar, PubMed, Bluesky)."""

from __future__ import annotations

import httpx
import pytest
import respx

from neural_search.http_utils import http_get_with_retry

URL = "https://example.test/api/thing"


@respx.mock
def test_returns_first_response_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
    route = respx.get(URL).mock(return_value=httpx.Response(200, json={"ok": True}))

    resp = http_get_with_retry(URL)

    assert resp.status_code == 200
    assert route.call_count == 1


@respx.mock
def test_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json={"ok": True})]
    )

    resp = http_get_with_retry(URL, max_retries=3)

    assert resp.status_code == 200
    assert route.call_count == 2


@respx.mock
def test_retries_on_5xx_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(200)]
    )

    resp = http_get_with_retry(URL, max_retries=3)

    assert resp.status_code == 200
    assert route.call_count == 2


@respx.mock
def test_returns_final_bad_response_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
    route = respx.get(URL).mock(return_value=httpx.Response(429))

    resp = http_get_with_retry(URL, max_retries=2)

    assert resp.status_code == 429
    assert route.call_count == 3  # initial attempt + 2 retries


@respx.mock
def test_does_not_retry_genuine_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("neural_search.http_utils.time.sleep", lambda *_: None)
    route = respx.get(URL).mock(return_value=httpx.Response(404))

    resp = http_get_with_retry(URL, max_retries=3)

    assert resp.status_code == 404
    assert route.call_count == 1


@respx.mock
def test_follows_redirects() -> None:
    """Regression test for a real incident (2026-07-03): Crossref returns
    HTTP 301 for case-normalized/superseded DOIs. httpx.get()'s default
    (follow_redirects=False) left the raw 301 response, and a caller's
    resp.raise_for_status() then raised an uncaught httpx.HTTPStatusError,
    crashing a corpus-scale Crossref run partway through."""

    redirect_url = "https://example.test/api/thing"
    final_url = "https://example.test/api/final-thing"
    respx.get(redirect_url).mock(
        return_value=httpx.Response(301, headers={"Location": final_url})
    )
    respx.get(final_url).mock(return_value=httpx.Response(200, json={"ok": True}))

    resp = http_get_with_retry(redirect_url)

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
