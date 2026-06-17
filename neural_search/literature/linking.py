"""Link corpus datasets to OpenAlex papers via DOI lookup and title fuzzy matching."""

from __future__ import annotations

import difflib
import json
import re
import time
from collections import Counter, defaultdict
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
_TITLE_MATCH_THRESHOLD = 0.82
_MAX_TITLE_CANDIDATES = 250
_MAX_TOKEN_POSTINGS = 8_000
_TITLE_STOPWORDS = {
    "and",
    "are",
    "based",
    "data",
    "dataset",
    "datasets",
    "during",
    "for",
    "from",
    "human",
    "mouse",
    "neural",
    "neuronal",
    "of",
    "the",
    "using",
    "with",
}


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


def _doi_key(doi: str | None) -> str | None:
    parsed = _parse_doi(doi)
    return parsed.lower().strip() if parsed else None


def _normalize_title(title: str | None) -> str:
    if not title:
        return ""
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def _title_tokens(title: str | None) -> list[str]:
    normalized = _normalize_title(title)
    return [
        token
        for token in normalized.split()
        if len(token) >= 4 and token not in _TITLE_STOPWORDS
    ]


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


def _iter_paper_records(paper_shards: Path) -> Iterator[dict]:
    paths = (
        sorted(paper_shards.glob("*.jsonl"))
        if paper_shards.is_dir()
        else [paper_shards]
    )
    for path in paths:
        with path.open() as fh:
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


class LocalPaperIndex:
    """DOI and token-candidate index over local normalized OpenAlex shards."""

    def __init__(self, paper_shards: Path) -> None:
        self.records: list[dict] = []
        self.doi_index: dict[str, int] = {}
        self.token_index: dict[str, list[int]] = defaultdict(list)
        self._build(paper_shards)

    def _build(self, paper_shards: Path) -> None:
        for record in _iter_paper_records(paper_shards):
            idx = len(self.records)
            self.records.append(record)
            doi = _doi_key(record.get("doi"))
            if doi and doi not in self.doi_index:
                self.doi_index[doi] = idx
            for token in set(_title_tokens(record.get("title"))):
                self.token_index[token].append(idx)

    def lookup_by_doi(self, doi: str) -> dict | None:
        key = _doi_key(doi)
        if not key or key not in self.doi_index:
            return None
        return self._hit(self.records[self.doi_index[key]], similarity=1.0)

    def lookup_by_title(self, title: str) -> dict | None:
        normalized = _normalize_title(title)
        tokens = _title_tokens(title)
        if not normalized or not tokens:
            return None

        candidate_counts: Counter[int] = Counter()
        for token in tokens:
            postings = self.token_index.get(token, [])
            if len(postings) > _MAX_TOKEN_POSTINGS:
                continue
            candidate_counts.update(postings)

        if not candidate_counts:
            return None

        best_record: dict | None = None
        best_ratio = 0.0
        for idx, _ in candidate_counts.most_common(_MAX_TITLE_CANDIDATES):
            record = self.records[idx]
            paper_title = _normalize_title(record.get("title"))
            if not paper_title:
                continue
            ratio = difflib.SequenceMatcher(None, normalized, paper_title).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_record = record

        if best_record is None or best_ratio < _TITLE_MATCH_THRESHOLD:
            return None
        return self._hit(best_record, similarity=best_ratio)

    def _hit(self, record: dict, *, similarity: float) -> dict:
        source_id = record.get("source_id")
        if not source_id and record.get("paper_id"):
            source_id = str(record["paper_id"]).split(":")[-1]
        return {
            "openalex_id": source_id or "",
            "doi": _parse_doi(record.get("doi")),
            "title": record.get("title"),
            "year": record.get("year"),
            "similarity": similarity,
        }


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


def link_corpus_to_local_literature(
    corpus_path: Path,
    paper_shards: Path,
    out_path: Path,
    *,
    max_records: int | None = None,
    skip_without_doi: bool = False,
    progress_every: int | None = None,
) -> list[DatasetPaperLink]:
    """Link corpus datasets against locally ingested OpenAlex paper shards."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building local paper index from {paper_shards}...", flush=True)
    index = LocalPaperIndex(paper_shards)
    print(f"Local paper index ready: {len(index.records)} papers", flush=True)
    records = _iter_corpus_records(corpus_path)
    links: list[DatasetPaperLink] = []

    with out_path.open("w") as out_fh:
        for i, rec in enumerate(records, start=1):
            if max_records is not None and i > max_records:
                break
            record_id = f"{rec['source']}:{rec['source_id']}"
            doi = rec.get("doi") or None
            title = rec.get("title") or ""
            link = _resolve_local_link(record_id, doi, title, skip_without_doi, index)
            out_fh.write(json.dumps(asdict(link)) + "\n")
            links.append(link)
            if progress_every and i % progress_every == 0:
                print(f"  Processed {i} records...", flush=True)

    return links


def _resolve_local_link(
    record_id: str,
    doi: str | None,
    title: str,
    skip_without_doi: bool,
    index: LocalPaperIndex,
) -> DatasetPaperLink:
    if doi:
        hit = index.lookup_by_doi(doi)
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
        hit = index.lookup_by_title(title)
        if hit:
            return DatasetPaperLink(
                dataset_record_id=record_id,
                paper_openalex_id=hit["openalex_id"],
                paper_doi=hit["doi"],
                paper_title=hit["title"],
                paper_year=hit["year"],
                match_method="title_fuzzy_local",
                confidence=min(0.95, max(0.7, hit.get("similarity", 0.82))),
            )
        return _make_not_found_link(record_id)

    if skip_without_doi:
        return _make_not_found_link(record_id)

    hit = index.lookup_by_title(title)
    if hit:
        return DatasetPaperLink(
            dataset_record_id=record_id,
            paper_openalex_id=hit["openalex_id"],
            paper_doi=hit["doi"],
            paper_title=hit["title"],
            paper_year=hit["year"],
            match_method="title_fuzzy_local",
            confidence=min(0.95, max(0.7, hit.get("similarity", 0.82))),
        )
    return _make_not_found_link(record_id)


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
