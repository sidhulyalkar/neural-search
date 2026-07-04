"""Link corpus datasets to papers via Semantic Scholar -- a third literature
source alongside OpenAlex, DataCite, and Crossref.

Response shape confirmed against the real, live Semantic Scholar Graph API
during development (2026-07-02): both `paper/DOI:{doi}` and `paper/search`
return a flat dict (or `{"data": [...]}` for search) with `paperId`, `title`,
`year`, and `externalIds` (a dict including `DOI`, `PubMed`, etc.).

**Real, demonstrated rate-limit fragility** (confirmed live, not assumed):
the unauthenticated tier returned HTTP 429 on the `paper/search` endpoint
even after exponential backoff up to ~40s total wait, while the `paper/DOI:`
lookup endpoint succeeded moments later -- the unauthenticated quota appears
to be tight and possibly shared across users/endpoints in this environment,
not just a per-process rate limit `http_get_with_retry`'s backoff can fully
absorb. A corpus-scale run may need to abort early via TransientLookupError
and be resumed later, or run with an API key (`config.semantic_scholar_api_key`)
for realistic throughput -- this is a genuine external constraint, not a
code defect, same category as the OpenAlex budget wall documented in
neural_search.literature.linking.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from neural_search.literature.api_client import TransientLookupError, get_or_raise_transient
from neural_search.literature.api_config import LiteratureAPIConfig, load_literature_api_config
from neural_search.literature.corpus_io import (
    DatasetPaperLink,
    iter_corpus_records,
    make_not_found_link,
)
from neural_search.literature.title_match import title_similarity

_S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
_FIELDS = "title,year,externalIds"
_TITLE_MATCH_THRESHOLD = 0.75


def _headers(config: LiteratureAPIConfig) -> dict[str, str] | None:
    if config.semantic_scholar_api_key:
        return {"x-api-key": config.semantic_scholar_api_key}
    return None


def _normalize_hit(paper: dict) -> dict:
    external_ids = paper.get("externalIds") or {}
    return {
        "paper_id": paper.get("paperId") or "",
        "doi": external_ids.get("DOI"),
        "title": paper.get("title"),
        "year": paper.get("year"),
    }


def lookup_by_doi(doi: str, *, config: LiteratureAPIConfig | None = None) -> dict | None:
    """Query Semantic Scholar by DOI. Returns normalized dict, or None only
    on a genuine 404. Raises TransientLookupError on 429/5xx."""

    config = config or load_literature_api_config()
    resp = get_or_raise_transient(
        f"{_S2_BASE}/DOI:{doi}",
        params={"fields": _FIELDS},
        headers=_headers(config),
        source="Semantic Scholar",
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return _normalize_hit(resp.json())


def lookup_by_title(title: str, *, config: LiteratureAPIConfig | None = None) -> dict | None:
    """Search Semantic Scholar by title. Returns dict if similarity >=
    threshold, else None on a genuine no-match. Raises TransientLookupError
    on 429/5xx."""

    config = config or load_literature_api_config()
    resp = get_or_raise_transient(
        f"{_S2_BASE}/search",
        params={"query": title, "fields": _FIELDS, "limit": 1},
        headers=_headers(config),
        source="Semantic Scholar",
    )
    resp.raise_for_status()
    results = resp.json().get("data", [])
    if not results:
        return None
    hit = _normalize_hit(results[0])
    similarity = title_similarity(title, hit["title"] or "")
    if similarity < _TITLE_MATCH_THRESHOLD:
        return None
    hit["similarity"] = similarity
    return hit


def link_corpus_to_semantic_scholar(corpus_path: Path, out_path: Path) -> list[DatasetPaperLink]:
    """Link corpus datasets to papers via Semantic Scholar DOI/title matching.

    Stops immediately on TransientLookupError -- see this module's docstring
    for the real, demonstrated fragility of the unauthenticated tier.
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
                link = _resolve_s2_link(record_id, doi, title, config)
            except TransientLookupError as exc:
                print(
                    f"Aborting after {i} records: transient Semantic Scholar "
                    f"lookup failure ({exc}). Records processed so far are "
                    f"saved; re-run later (or with an API key) rather than "
                    f"trusting the remaining corpus as genuinely unmatched."
                )
                break

            out_fh.write(json.dumps(asdict(link)) + "\n")
            links.append(link)

    return links


def _resolve_s2_link(
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
                match_method="semantic_scholar_doi_exact",
                confidence=1.0,
                paper_source="semantic_scholar",
                paper_source_id=hit["paper_id"],
            )

    hit = lookup_by_title(title, config=config) if title else None
    if hit:
        return DatasetPaperLink(
            dataset_record_id=record_id,
            paper_openalex_id="",
            paper_doi=hit["doi"],
            paper_title=hit["title"],
            paper_year=hit["year"],
            match_method="semantic_scholar_title_fuzzy",
            confidence=min(0.9, max(0.7, hit.get("similarity", 0.8))),
            paper_source="semantic_scholar",
            paper_source_id=hit["paper_id"],
        )

    return make_not_found_link(record_id, paper_source="semantic_scholar")
