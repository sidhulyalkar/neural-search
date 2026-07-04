"""Link corpus datasets to OpenAlex papers via DOI lookup and title fuzzy matching."""

from __future__ import annotations

import difflib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from neural_search.literature.api_client import TransientLookupError
from neural_search.literature.corpus_io import DatasetPaperLink
from neural_search.literature.corpus_io import iter_corpus_records as _iter_corpus_records
from neural_search.literature.corpus_io import iter_paper_records as _iter_paper_records
from neural_search.literature.corpus_io import make_not_found_link as _make_not_found_link
from neural_search.literature.title_match import normalize_title as _normalize_title
from neural_search.literature.title_match import title_tokens as _title_tokens

# DatasetPaperLink and TransientLookupError now live in corpus_io.py /
# api_client.py respectively (shared by the other literature-linking
# sources added afterward); re-imported above for backward compatibility --
# existing callers importing them from this module are unaffected.

_OPENALEX_BASE = "https://api.openalex.org"
_MAILTO = "sid.soccer.21@gmail.com"
_RATE_LIMIT_SLEEP = 0.15
_TITLE_MATCH_THRESHOLD = 0.82
_MAX_TITLE_CANDIDATES = 250
_MAX_TOKEN_POSTINGS = 8_000


def _http_get(url: str, params: dict | None = None):
    try:
        import httpx
        return httpx.get(url, params=params, timeout=10)
    except ImportError:
        import urllib.parse
        import urllib.request

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


def _raise_if_transient(resp) -> None:
    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientLookupError(
            f"OpenAlex request failed with status {resp.status_code}: {resp.text[:300]}"
        )


def lookup_by_doi(doi: str) -> dict | None:
    """Query OpenAlex by DOI. Returns normalized dict, or None only on a genuine 404.

    Raises TransientLookupError on 429/5xx — callers must not treat that the
    same as "not found" (see TransientLookupError's docstring)."""

    url = f"{_OPENALEX_BASE}/works/https://doi.org/{doi}"
    try:
        resp = _http_get(url)
        _raise_if_transient(resp)
        resp.raise_for_status()
        data = resp.json()
        return {
            "openalex_id": _parse_openalex_id(data["id"]),
            "doi": _parse_doi(data.get("doi")),
            "title": data.get("title"),
            "year": data.get("publication_year"),
        }
    except TransientLookupError:
        raise
    except Exception:
        return None
    finally:
        time.sleep(_RATE_LIMIT_SLEEP)


def lookup_by_title(title: str, year: int | None = None) -> dict | None:
    """Search OpenAlex by title. Returns dict if similarity >= 0.75, else None
    on a genuine no-match. Raises TransientLookupError on 429/5xx — see
    lookup_by_doi."""

    params: dict = {
        "search": title,
        "per_page": 1,
        "mailto": _MAILTO,
    }
    try:
        resp = _http_get(f"{_OPENALEX_BASE}/works", params=params)
        _raise_if_transient(resp)
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
    except TransientLookupError:
        raise
    except Exception:
        return None
    finally:
        time.sleep(_RATE_LIMIT_SLEEP)


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
    """Process all corpus records and link to OpenAlex papers.

    Stops immediately on TransientLookupError (HTTP 429/5xx, e.g. a quota or
    budget exhaustion) rather than continuing and writing false "not_found"
    results for every remaining record — see TransientLookupError's
    docstring for the incident that made this necessary. Records already
    resolved before the failure are still written to `out_path`.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    links: list[DatasetPaperLink] = []

    with out_path.open("w") as out_fh:
        for i, rec in enumerate(_iter_corpus_records(corpus_path)):
            record_id = f"{rec['source']}:{rec['source_id']}"
            doi = rec.get("doi") or None
            title = rec.get("title") or ""

            try:
                link = _resolve_link(record_id, doi, title, skip_without_doi)
            except TransientLookupError as exc:
                print(
                    f"Aborting after {i}/{'?'} records: transient lookup failure "
                    f"({exc}). Records processed so far are saved; re-run once "
                    f"resolved (e.g. quota reset) rather than trusting the "
                    f"remaining corpus as genuinely unmatched."
                )
                break
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
