"""Shared transient-error detection for literature-linking API clients
(OpenAlex, Crossref, DataCite, Semantic Scholar, PubMed).

Generalizes the pattern first built in neural_search.literature.linking for
OpenAlex alone: distinguish a genuine "not found" from a retryable/quota
failure (HTTP 429/5xx). See TransientLookupError's docstring for the
incident that made this distinction necessary -- every new literature API
client added after that incident must use it from the start, not retrofit
it after a silent-corruption incident of its own.
"""

from __future__ import annotations

from typing import Any

import httpx

from neural_search.http_utils import http_get_with_retry

__all__ = ["TransientLookupError", "raise_if_transient", "http_get_with_retry", "get_or_raise_transient"]


class TransientLookupError(Exception):
    """Raised for retryable/quota failures (HTTP 429, 5xx) from any literature
    API client -- must never be silently treated as "paper not found". See
    neural_search.literature.linking.TransientLookupError's original
    docstring for the incident (OpenAlex budget exhaustion, 2026-07-02) that
    made this distinction necessary; this class generalizes it so Crossref,
    DataCite, Semantic Scholar, and PubMed clients share one exception type
    rather than each defining their own."""


def raise_if_transient(resp: httpx.Response, source: str) -> None:
    """Raise TransientLookupError if `resp` looks like a retryable failure.

    `source` names the API in the error message (e.g. "Crossref", "DataCite")
    since multiple clients now share this check.
    """

    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientLookupError(
            f"{source} request failed with status {resp.status_code}: {resp.text[:300]}"
        )


def get_or_raise_transient(url: str, *, source: str, **kwargs: Any) -> httpx.Response:
    """`http_get_with_retry` + `raise_if_transient`, plus converting
    transport-level failures (timeouts, connection errors -- anything left
    after `http_get_with_retry`'s own retries are exhausted) into
    `TransientLookupError` too.

    Found necessary by a real incident (2026-07-02): a corpus-scale Crossref
    run crashed the whole process with an uncaught `httpx.ReadTimeout` after
    retries were exhausted, because callers only caught `TransientLookupError`
    around their `link_corpus_to_*` loops, not raw `httpx` exceptions. A
    persistent network failure after retries has the same "can't tell if
    this is a genuine not-found, abort and preserve partial progress"
    semantics as an HTTP 429/5xx -- it must not crash the whole script
    (like this incident did) nor be silently absorbed as "not found" (like
    the pre-existing bare `except Exception: return None` in
    `neural_search.literature.linking` would have done before its own
    2026-07-02 fix).
    """

    try:
        resp = http_get_with_retry(url, **kwargs)
    except httpx.TransportError as exc:
        raise TransientLookupError(f"{source} request failed (transport error): {exc}") from exc
    raise_if_transient(resp, source=source)
    return resp
