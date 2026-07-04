"""Link corpus datasets to papers via PubMed (NCBI E-utilities) and bioRxiv
-- a fourth/fifth literature source alongside OpenAlex, DataCite, and
Crossref. bioRxiv is folded into this module rather than a separate one:
its DOIs are structurally distinct (`10.1101/...`), easy to route by
prefix, and it needs only one extra lookup function plus its own small
response normalizer.

Response shapes confirmed against the real, live NCBI/bioRxiv APIs during
development (2026-07-02):
- `esearch.fcgi` returns `{"esearchresult": {"count": "N", "idlist": [...]}}`
  -- an empty `idlist` (count "0") is the genuine not-found signal, no
  special HTTP status involved.
- `esummary.fcgi` returns `{"result": {"<pmid>": {"title": ..., "pubdate":
  "2020 Mar 4", "articleids": [{"idtype": "doi", "value": "10.xxxx/yyyy"},
  ...]}}}` -- the DOI is inside `articleids`, not a top-level field; `pubdate`
  is a free-text string, year extracted via leading 4-digit token.
- bioRxiv's `details/biorxiv/{doi}` returns **HTTP 200 even when the DOI
  doesn't exist** -- `{"messages": [{"status": "no posts found"}], "collection":
  []}` -- so "not found" must be detected via an empty `collection`, not an
  HTTP status code. This is exactly the "different APIs signal transient
  failure slightly differently" case anticipated when `raise_if_transient`
  was designed to take a `source` name rather than being hardcoded to one
  API's conventions.
"""

from __future__ import annotations

import json
import re
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

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_BIORXIV_BASE = "https://api.biorxiv.org/details/biorxiv"
_TITLE_MATCH_THRESHOLD = 0.75
_BIORXIV_DOI_PREFIX = "10.1101/"


def _ncbi_params(config: LiteratureAPIConfig, **extra: str) -> dict[str, str]:
    params = {"retmode": "json", "tool": "neural-search", "email": config.ncbi_tool_email, **extra}
    if config.ncbi_api_key:
        params["api_key"] = config.ncbi_api_key
    return params


def _extract_year(pubdate: str) -> int | None:
    match = re.match(r"(\d{4})", pubdate or "")
    return int(match.group(1)) if match else None


def _extract_doi(article_ids: list[dict]) -> str | None:
    for entry in article_ids or []:
        if entry.get("idtype") == "doi":
            return entry.get("value")
    return None


def _esearch(term: str, config: LiteratureAPIConfig) -> list[str]:
    resp = get_or_raise_transient(
        f"{_EUTILS_BASE}/esearch.fcgi",
        params=_ncbi_params(config, db="pubmed", term=term, retmax="1"),
        source="PubMed",
    )
    resp.raise_for_status()
    return resp.json()["esearchresult"].get("idlist", [])


def _esummary(pmid: str, config: LiteratureAPIConfig) -> dict:
    resp = get_or_raise_transient(
        f"{_EUTILS_BASE}/esummary.fcgi",
        params=_ncbi_params(config, db="pubmed", id=pmid),
        source="PubMed",
    )
    resp.raise_for_status()
    return resp.json()["result"][pmid]


def lookup_by_doi(doi: str, *, config: LiteratureAPIConfig | None = None) -> dict | None:
    """Query PubMed by DOI. Returns normalized dict, or None on a genuine
    no-match (empty esearch result). Raises TransientLookupError on 429/5xx."""

    config = config or load_literature_api_config()
    ids = _esearch(f"{doi}[doi]", config)
    if not ids:
        return None
    summary = _esummary(ids[0], config)
    return {
        "pmid": ids[0],
        "doi": _extract_doi(summary.get("articleids", [])),
        "title": summary.get("title", "").rstrip("."),
        "year": _extract_year(summary.get("pubdate", "")),
    }


def lookup_by_title(title: str, *, config: LiteratureAPIConfig | None = None) -> dict | None:
    """Search PubMed by title. Returns dict if similarity >= threshold, else
    None on a genuine no-match. Raises TransientLookupError on 429/5xx."""

    config = config or load_literature_api_config()
    ids = _esearch(f"{title}[title]", config)
    if not ids:
        return None
    summary = _esummary(ids[0], config)
    hit_title = summary.get("title", "").rstrip(".")
    similarity = title_similarity(title, hit_title)
    if similarity < _TITLE_MATCH_THRESHOLD:
        return None
    return {
        "pmid": ids[0],
        "doi": _extract_doi(summary.get("articleids", [])),
        "title": hit_title,
        "year": _extract_year(summary.get("pubdate", "")),
        "similarity": similarity,
    }


def lookup_biorxiv_by_doi(doi: str) -> dict | None:
    """Query bioRxiv by DOI. Returns None on a genuine no-match (empty
    `collection`, even though bioRxiv returns HTTP 200 either way).
    Raises TransientLookupError on 429/5xx."""

    resp = get_or_raise_transient(f"{_BIORXIV_BASE}/{doi}", source="bioRxiv")
    resp.raise_for_status()
    collection = resp.json().get("collection", [])
    if not collection:
        return None
    hit = collection[0]
    year_match = re.match(r"(\d{4})", hit.get("date", ""))
    return {
        "doi": hit.get("doi"),
        "title": hit.get("title"),
        "year": int(year_match.group(1)) if year_match else None,
    }


def link_corpus_to_pubmed(corpus_path: Path, out_path: Path) -> list[DatasetPaperLink]:
    """Link corpus datasets to papers via PubMed and bioRxiv.

    For each record: if the DOI has the bioRxiv prefix (`10.1101/`), route to
    `lookup_biorxiv_by_doi` (bioRxiv DOIs are never indexed as PubMed DOI
    matches in the same way). Otherwise try PubMed DOI, then PubMed title.
    Stops immediately on TransientLookupError from either source.
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
                link = _resolve_pubmed_link(record_id, doi, title, config)
            except TransientLookupError as exc:
                print(
                    f"Aborting after {i} records: transient PubMed/bioRxiv "
                    f"lookup failure ({exc}). Records processed so far are "
                    f"saved; re-run once resolved rather than trusting the "
                    f"remaining corpus as genuinely unmatched."
                )
                break

            out_fh.write(json.dumps(asdict(link)) + "\n")
            links.append(link)

    return links


def _resolve_pubmed_link(
    record_id: str, doi: str | None, title: str, config: LiteratureAPIConfig
) -> DatasetPaperLink:
    if doi and doi.startswith(_BIORXIV_DOI_PREFIX):
        hit = lookup_biorxiv_by_doi(doi)
        if hit:
            return DatasetPaperLink(
                dataset_record_id=record_id,
                paper_openalex_id="",
                paper_doi=hit["doi"],
                paper_title=hit["title"],
                paper_year=hit["year"],
                match_method="biorxiv_doi_exact",
                confidence=1.0,
                paper_source="biorxiv",
                paper_source_id=hit["doi"] or "",
            )
        return make_not_found_link(record_id, paper_source="biorxiv")

    if doi:
        hit = lookup_by_doi(doi, config=config)
        if hit:
            return DatasetPaperLink(
                dataset_record_id=record_id,
                paper_openalex_id="",
                paper_doi=hit["doi"],
                paper_title=hit["title"],
                paper_year=hit["year"],
                match_method="pubmed_doi_exact",
                confidence=1.0,
                paper_source="pubmed",
                paper_source_id=hit["pmid"],
            )

    hit = lookup_by_title(title, config=config) if title else None
    if hit:
        return DatasetPaperLink(
            dataset_record_id=record_id,
            paper_openalex_id="",
            paper_doi=hit["doi"],
            paper_title=hit["title"],
            paper_year=hit["year"],
            match_method="pubmed_title_fuzzy",
            confidence=min(0.9, max(0.7, hit.get("similarity", 0.8))),
            paper_source="pubmed",
            paper_source_id=hit["pmid"],
        )

    return make_not_found_link(record_id, paper_source="pubmed")
