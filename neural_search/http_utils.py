"""Generic HTTP GET with retry/backoff, shared across literature-linking and
discourse ingestion modules that hit distinct, independently rate-limited
external APIs (Crossref, DataCite, Semantic Scholar, PubMed, Bluesky)."""

from __future__ import annotations

import random
import time

import httpx

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def http_get_with_retry(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    max_retries: int = 3,
    backoff_base: float = 0.5,
    timeout: float = 10.0,
) -> httpx.Response:
    """GET with exponential backoff + jitter on 429/5xx and transport errors.

    Returns the final response regardless of status, including after
    exhausting retries -- this helper only controls retry timing. Callers
    decide what a bad status means (e.g. via a TransientLookupError-raising
    check), consistent with how neural_search.literature.linking already
    separates "genuinely not found" from "retryable/quota failure".
    """

    last_response: httpx.Response | None = None
    for attempt in range(max_retries + 1):
        try:
            # follow_redirects=True: found necessary by a real incident
            # (2026-07-03) -- Crossref returns HTTP 301 for DOIs it has
            # case-normalized or superseded (e.g. a publisher-submitted
            # uppercase DOI redirecting to Crossref's canonical lowercase
            # form). httpx.get()'s default (follow_redirects=False) leaves
            # the raw 301 response, and a caller's `resp.raise_for_status()`
            # then raises an uncaught httpx.HTTPStatusError -- crashing a
            # corpus-scale run instead of transparently resolving to the
            # paper the redirect points to.
            last_response = httpx.get(
                url, params=params, headers=headers, timeout=timeout, follow_redirects=True
            )
        except httpx.TransportError:
            if attempt == max_retries:
                raise
            _sleep_backoff(attempt, backoff_base)
            continue

        if last_response.status_code not in _TRANSIENT_STATUS_CODES or attempt == max_retries:
            return last_response
        _sleep_backoff(attempt, backoff_base)

    assert last_response is not None  # pragma: no cover - loop always returns/raises above
    return last_response


def _sleep_backoff(attempt: int, backoff_base: float) -> None:
    delay = backoff_base * (2**attempt) + random.uniform(0, backoff_base)
    time.sleep(delay)
