"""Link corpus datasets to papers via Crossref, and check paper retraction/
correction status -- a second, independently-metered literature source
alongside OpenAlex and DataCite.

Response shapes below are confirmed against the real, live Crossref API
during development (2026-07-02), not guessed:
- `works/{doi}` and `works?query.bibliographic=...` both return
  `{"message": {"DOI": ..., "title": [...], "published"/"published-print"/
  "published-online": {"date-parts": [[year, ...]]}, "update-to": [...]}}`.
  `title` is a list (Crossref supports multiple titles per work); this
  module uses the first entry.
- Retraction/correction status lives in the top-level `update-to` field on
  the *original* work (not a `relation.is-corrected-by` entry -- confirmed
  by querying Crossref's own relation-type validation error, which does not
  list any retraction/correction relation type at all). Each `update-to`
  entry has `type` ("retraction" | "correction" | ...), `DOI`, and `updated`
  fields -- verified live on `10.1016/j.micpro.2020.103768`, a real
  Elsevier retraction notice.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from neural_search.literature.api_client import TransientLookupError, get_or_raise_transient
from neural_search.literature.api_config import LiteratureAPIConfig, load_literature_api_config
from neural_search.literature.corpus_io import (
    DatasetPaperLink,
    iter_corpus_records,
    make_not_found_link,
)
from neural_search.literature.title_match import title_similarity

_CROSSREF_BASE = "https://api.crossref.org/works"
_TITLE_MATCH_THRESHOLD = 0.75  # matches linking.py's live OpenAlex title-fuzzy threshold
_RETRACTION_TYPES = {"retraction"}
_CORRECTION_TYPES = {"correction", "erratum", "corrigendum"}


def _extract_year(message: dict) -> int | None:
    for key in ("published", "published-print", "published-online"):
        date_parts = message.get(key, {}).get("date-parts")
        if date_parts and date_parts[0]:
            return date_parts[0][0]
    return None


def _extract_title(message: dict) -> str | None:
    titles = message.get("title") or []
    return titles[0] if titles else None


def lookup_by_doi(doi: str, *, config: LiteratureAPIConfig | None = None) -> dict | None:
    """Query Crossref by DOI. Returns normalized dict, or None only on a genuine 404.

    Raises TransientLookupError on 429/5xx -- see neural_search.literature.linking's
    OpenAlex client for the incident that made this discipline mandatory.
    """

    config = config or load_literature_api_config()
    resp = get_or_raise_transient(
        f"{_CROSSREF_BASE}/{doi}", params={"mailto": config.crossref_mailto}, source="Crossref"
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    message = resp.json()["message"]
    return {
        "doi": message.get("DOI"),
        "title": _extract_title(message),
        "year": _extract_year(message),
    }


def lookup_by_title(title: str, *, config: LiteratureAPIConfig | None = None) -> dict | None:
    """Search Crossref by title. Returns dict if similarity >= threshold, else
    None on a genuine no-match. Raises TransientLookupError on 429/5xx."""

    config = config or load_literature_api_config()
    resp = get_or_raise_transient(
        _CROSSREF_BASE,
        params={"query.bibliographic": title, "rows": 1, "mailto": config.crossref_mailto},
        source="Crossref",
    )
    resp.raise_for_status()
    items = resp.json()["message"].get("items", [])
    if not items:
        return None
    hit = items[0]
    hit_title = _extract_title(hit) or ""
    similarity = title_similarity(title, hit_title)
    if similarity < _TITLE_MATCH_THRESHOLD:
        return None
    return {
        "doi": hit.get("DOI"),
        "title": hit_title,
        "year": _extract_year(hit),
        "similarity": similarity,
    }


def fetch_retraction_status(doi: str, *, config: LiteratureAPIConfig | None = None) -> dict[str, Any]:
    """Check a paper's retraction/correction status via Crossref's `update-to`.

    Returns `{"status": "retracted"|"corrected"|"none", "related_dois": [...],
    "source": "crossref", "checked_at": <iso8601>}`. Conservative: any lookup
    failure (including a genuine 404, e.g. a non-Crossref DOI) reports
    "none" rather than raising, since this is a best-effort enrichment pass,
    not a required linking step -- see paper_node_builder.py's call site
    (called once per already-resolved paper node, not per corpus record).
    Still raises TransientLookupError on 429/5xx so a budget wall isn't
    silently absorbed as "not retracted".
    """

    from datetime import UTC, datetime

    config = config or load_literature_api_config()
    resp = get_or_raise_transient(
        f"{_CROSSREF_BASE}/{doi}", params={"mailto": config.crossref_mailto}, source="Crossref"
    )
    checked_at = datetime.now(UTC).isoformat()
    if resp.status_code != 200:
        return {"status": "none", "related_dois": [], "source": "crossref", "checked_at": checked_at}

    message = resp.json()["message"]
    updates = message.get("update-to") or []
    related_dois = sorted({u["DOI"] for u in updates if u.get("DOI")})
    types = {u.get("type") for u in updates}
    if types & _RETRACTION_TYPES:
        status = "retracted"
    elif types & _CORRECTION_TYPES:
        status = "corrected"
    else:
        status = "none"
    return {
        "status": status,
        "related_dois": related_dois,
        "source": "crossref",
        "checked_at": checked_at,
    }


def link_corpus_to_crossref(corpus_path: Path, out_path: Path) -> list[DatasetPaperLink]:
    """Link corpus datasets to papers via Crossref DOI/title matching.

    Stops immediately on TransientLookupError, matching
    link_corpus_to_literature's behavior -- records resolved before the
    failure are still written to `out_path`.
    """

    config = load_literature_api_config()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    links: list[DatasetPaperLink] = []

    with out_path.open("w") as out_fh:
        for i, rec in enumerate(iter_corpus_records(corpus_path)):
            record_id = f"{rec['source']}:{rec['source_id']}"
            doi = rec.get("doi") or None
            title = rec.get("title") or ""

            try:
                link = _resolve_crossref_link(record_id, doi, title, config)
            except TransientLookupError as exc:
                print(
                    f"Aborting after {i} records: transient Crossref lookup "
                    f"failure ({exc}). Records processed so far are saved; "
                    f"re-run once resolved rather than trusting the "
                    f"remaining corpus as genuinely unmatched."
                )
                break

            out_fh.write(json.dumps(asdict(link)) + "\n")
            links.append(link)

    return links


def _resolve_crossref_link(
    record_id: str, doi: str | None, title: str, config: LiteratureAPIConfig
) -> DatasetPaperLink:
    if doi:
        hit = lookup_by_doi(doi, config=config)
        if hit:
            return DatasetPaperLink(
                dataset_record_id=record_id,
                paper_openalex_id="",
                paper_doi=hit["doi"],
                paper_title=hit["title"],
                paper_year=hit["year"],
                match_method="crossref_doi_exact",
                confidence=1.0,
                paper_source="crossref",
                paper_source_id=hit["doi"] or "",
            )

    hit = lookup_by_title(title, config=config) if title else None
    if hit:
        return DatasetPaperLink(
            dataset_record_id=record_id,
            paper_openalex_id="",
            paper_doi=hit["doi"],
            paper_title=hit["title"],
            paper_year=hit["year"],
            match_method="crossref_title_fuzzy",
            confidence=min(0.9, max(0.7, hit.get("similarity", 0.8))),
            paper_source="crossref",
            paper_source_id=hit["doi"] or "",
        )

    return make_not_found_link(record_id, paper_source="crossref")
