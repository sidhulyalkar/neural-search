"""Link corpus datasets to OpenAlex papers via DOI lookup and title fuzzy matching."""

from __future__ import annotations

import difflib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class DatasetPaperLink:
    dataset_record_id: str
    paper_openalex_id: str
    paper_doi: str | None
    paper_title: str | None
    paper_year: int | None
    match_method: str  # "doi_exact" | "title_fuzzy" | "not_found"
    confidence: float  # 1.0 doi_exact, 0.7-0.9 title_fuzzy, 0.0 not_found


_OPENALEX_BASE = "https://api.openalex.org"
_MAILTO = "sid.soccer.21@gmail.com"
_RATE_LIMIT_SLEEP = 0.15


def _http_get(url: str, params: dict | None = None):
    try:
        import httpx
        return httpx.get(url, params=params, timeout=10)
    except ImportError:
        import urllib.request
        import urllib.parse

        if params:
            url = url + "?" + urllib.parse.urlencode(params)

        class _FakeResp:
            def __init__(self, body: bytes, status: int) -> None:
                self._body = body
                self.status_code = status

            def json(self) -> dict:
                return json.loads(self._body)

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise Exception(f"HTTP {self.status_code}")

        with urllib.request.urlopen(url, timeout=10) as r:
            return _FakeResp(r.read(), r.status)


def _parse_openalex_id(openalex_url: str) -> str:
    return openalex_url.rstrip("/").split("/")[-1]


def _parse_doi(doi_url: str | None) -> str | None:
    if not doi_url:
        return None
    return doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "")


def lookup_by_doi(doi: str) -> dict | None:
    """Query OpenAlex by DOI. Returns normalized dict or None on 404/error."""
    url = f"{_OPENALEX_BASE}/works/https://doi.org/{doi}"
    try:
        resp = _http_get(url)
        resp.raise_for_status()
        data = resp.json()
        return {
            "openalex_id": _parse_openalex_id(data["id"]),
            "doi": _parse_doi(data.get("doi")),
            "title": data.get("title"),
            "year": data.get("publication_year"),
        }
    except Exception:
        return None
    finally:
        time.sleep(_RATE_LIMIT_SLEEP)


def lookup_by_title(title: str, year: int | None = None) -> dict | None:
    """Search OpenAlex by title. Returns dict if similarity >= 0.75, else None."""
    params: dict = {
        "search": title,
        "per_page": 1,
        "mailto": _MAILTO,
    }
    try:
        resp = _http_get(f"{_OPENALEX_BASE}/works", params=params)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None
        hit = results[0]
        hit_title = hit.get("title") or ""
        ratio = difflib.SequenceMatcher(None, title.lower(), hit_title.lower()).ratio()
        if ratio < 0.75:
            return None
        return {
            "openalex_id": _parse_openalex_id(hit["id"]),
            "doi": _parse_doi(hit.get("doi")),
            "title": hit_title,
            "year": hit.get("publication_year"),
            "similarity": ratio,
        }
    except Exception:
        return None
    finally:
        time.sleep(_RATE_LIMIT_SLEEP)


def _iter_corpus_records(corpus_path: Path) -> Iterator[dict]:
    paths = (
        sorted(corpus_path.glob("*.jsonl"))
        if corpus_path.is_dir()
        else [corpus_path]
    )
    for p in paths:
        with p.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _make_not_found_link(record_id: str) -> DatasetPaperLink:
    return DatasetPaperLink(
        dataset_record_id=record_id,
        paper_openalex_id="",
        paper_doi=None,
        paper_title=None,
        paper_year=None,
        match_method="not_found",
        confidence=0.0,
    )


def link_corpus_to_literature(
    corpus_path: Path,
    out_path: Path,
    *,
    max_workers: int = 1,
    skip_without_doi: bool = False,
) -> list[DatasetPaperLink]:
    """Process all corpus records and link to OpenAlex papers."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    links: list[DatasetPaperLink] = []

    with out_path.open("w") as out_fh:
        for rec in _iter_corpus_records(corpus_path):
            record_id = f"{rec['source']}:{rec['source_id']}"
            doi = rec.get("doi") or None
            title = rec.get("title") or ""

            link = _resolve_link(record_id, doi, title, skip_without_doi)
            out_fh.write(json.dumps(asdict(link)) + "\n")
            links.append(link)

    return links


def _resolve_link(
    record_id: str,
    doi: str | None,
    title: str,
    skip_without_doi: bool,
) -> DatasetPaperLink:
    if doi:
        hit = lookup_by_doi(doi)
        if hit:
            return DatasetPaperLink(
                dataset_record_id=record_id,
                paper_openalex_id=hit["openalex_id"],
                paper_doi=hit["doi"],
                paper_title=hit["title"],
                paper_year=hit["year"],
                match_method="doi_exact",
                confidence=1.0,
            )
        hit = lookup_by_title(title)
        if hit:
            return DatasetPaperLink(
                dataset_record_id=record_id,
                paper_openalex_id=hit["openalex_id"],
                paper_doi=hit["doi"],
                paper_title=hit["title"],
                paper_year=hit["year"],
                match_method="title_fuzzy",
                confidence=min(0.9, max(0.7, hit.get("similarity", 0.8))),
            )
        return _make_not_found_link(record_id)

    if skip_without_doi:
        return _make_not_found_link(record_id)

    hit = lookup_by_title(title)
    if hit:
        return DatasetPaperLink(
            dataset_record_id=record_id,
            paper_openalex_id=hit["openalex_id"],
            paper_doi=hit["doi"],
            paper_title=hit["title"],
            paper_year=hit["year"],
            match_method="title_fuzzy",
            confidence=min(0.9, max(0.7, hit.get("similarity", 0.8))),
        )
    return _make_not_found_link(record_id)
