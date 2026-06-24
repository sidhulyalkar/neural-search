"""Cursor-based bulk downloader from OpenAlex for neuroscience literature."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from neural_search.normalized import stable_normalized_id

logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"
POLITE_EMAIL = "neuralsearch@example.com"
NEURO_CONCEPT_ID = "C169760540"
RATE_LIMIT_DELAY = 0.12
PAGE_SIZE = 200
MAX_RETRIES = 6
RETRY_BACKOFF_BASE_S = 2.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
SELECT_FIELDS = (
    "id,doi,title,abstract_inverted_index,publication_year,"
    "concepts,cited_by_count,authorships,primary_location,"
    "open_access,topics"
)

TIER_FILTERS: dict[str, str] = {
    "tier1": "concepts.id:C169760540,type:article,cited_by_count:>99",
    "tier2": "concepts.id:C169760540,type:article,has_abstract:true,open_access.is_oa:true",
    "tier3": "concepts.id:C169760540,type:article",
}


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Reconstruct plain text from an OpenAlex inverted abstract index."""
    if not inverted_index:
        return None

    pairs: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pairs.append((pos, word))

    if not pairs:
        return None

    pairs.sort(key=lambda x: x[0])
    return " ".join(word for _, word in pairs)


def _strip_openalex_url(value: str) -> str:
    return value.replace("https://openalex.org/", "")


def normalize_bulk_work(work: dict) -> dict:
    """Flatten an OpenAlex work dict into all NormalizedPaperRecord fields."""
    raw_id = work.get("id", "")
    source_id = _strip_openalex_url(raw_id)

    try:
        paper_id = stable_normalized_id("paper", "openalex", source_id)
    except Exception:
        paper_id = f"paper:openalex:{source_id}"

    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

    doi = work.get("doi")

    authors: list[str] = []
    for authorship in work.get("authorships") or []:
        name = (authorship.get("author") or {}).get("display_name")
        if name:
            authors.append(name)

    concept_ids: list[str] = []
    for concept in work.get("concepts") or []:
        cid = concept.get("id", "")
        concept_ids.append(_strip_openalex_url(cid))

    primary_location = work.get("primary_location") or {}
    source_info = primary_location.get("source") or {}
    venue: str | None = source_info.get("display_name") or None

    open_access = work.get("open_access") or {}
    open_access_url: str | None = open_access.get("oa_url") or None

    topics: list[str] = [
        t["display_name"]
        for t in (work.get("topics") or [])
        if t.get("display_name")
    ]

    return {
        "paper_id": paper_id,
        "source": "openalex",
        "source_id": source_id,
        "title": work.get("title") or "",
        "abstract": abstract,
        "doi": doi,
        "url": doi or raw_id or None,
        "year": work.get("publication_year"),
        "authors": authors,
        "linked_datasets": [],
        "extracted_labels": [],
        "raw_payload_path": None,
        "created_at": datetime.now(UTC).isoformat(),
        "extractor_version": "v0.3.0",
        "citation_count": work.get("cited_by_count") or 0,
        "venue": venue,
        "concept_ids": concept_ids,
        "open_access_url": open_access_url,
        "topics": topics,
    }


class BulkIngester:
    """Cursor-paginated bulk downloader with checkpoint/resume and JSONL shards."""

    def __init__(
        self,
        out_dir: Path,
        tier: str = "tier1",
        shard_size: int = 10_000,
    ) -> None:
        self.out_dir = out_dir
        self.tier = tier
        self.shard_size = shard_size
        self._checkpoint_path = out_dir / ".checkpoint.json"

    def load_checkpoint(self) -> tuple[str, int]:
        if not self._checkpoint_path.exists():
            return "*", 0
        data = json.loads(self._checkpoint_path.read_text())
        return data.get("cursor", "*"), data.get("records_fetched", 0)

    def save_checkpoint(self, cursor: str, count: int) -> None:
        payload = {
            "cursor": cursor,
            "records_fetched": count,
            "tier": self.tier,
            "last_updated": datetime.now(UTC).isoformat(),
        }
        self._checkpoint_path.write_text(json.dumps(payload, indent=2))

    def _fetch_page(self, cursor: str) -> tuple[list[dict], str | None]:
        time.sleep(RATE_LIMIT_DELAY)
        params = {
            "filter": TIER_FILTERS[self.tier],
            "per_page": PAGE_SIZE,
            "cursor": cursor,
            "select": SELECT_FIELDS,
            "mailto": POLITE_EMAIL,
        }
        data = self._get_with_retry(params)

        results = data.get("results", [])
        records = [normalize_bulk_work(w) for w in results]
        next_cursor: str | None = (data.get("meta") or {}).get("next_cursor")
        return records, next_cursor

    def _get_with_retry(self, params: dict) -> dict:
        """GET /works with exponential-backoff retry on 429/5xx.

        OpenAlex's polite pool documents a 10 req/s ceiling; sustained bulk
        crawling can still occasionally exceed it and get a 429. Honor
        Retry-After when present, otherwise back off exponentially
        (2s, 4s, 8s, ...) capped at MAX_RETRIES attempts.
        """
        for attempt in range(MAX_RETRIES + 1):
            with httpx.Client(timeout=60.0) as client:
                response = client.get(f"{OPENALEX_BASE}/works", params=params)

            if response.status_code not in RETRYABLE_STATUS_CODES:
                response.raise_for_status()
                return response.json()

            if attempt == MAX_RETRIES:
                response.raise_for_status()

            retry_after = response.headers.get("Retry-After")
            wait_s = float(retry_after) if retry_after else RETRY_BACKOFF_BASE_S * (2**attempt)
            # Without this, a long Retry-After wait (OpenAlex's rate-limit window
            # reset can be several minutes) looks indistinguishable from a hung
            # process -- no output at all until the next successful page fetch.
            remaining = response.headers.get("X-RateLimit-Remaining")
            msg = (
                f"Rate limited (HTTP {response.status_code}, attempt {attempt + 1}/"
                f"{MAX_RETRIES + 1}) -- waiting {wait_s:.0f}s before retrying"
            )
            if remaining is not None:
                msg += f" (X-RateLimit-Remaining={remaining})"
            logger.warning(msg)
            print(msg, flush=True)
            time.sleep(wait_s)

        raise AssertionError("unreachable")  # loop always returns or raises above

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.tier}_batch_{shard_idx:04d}.jsonl"
        lines = "\n".join(json.dumps(r) for r in records)
        path.write_text(lines + "\n" if lines else "")
        return path

    def run(self, max_records: int | None = None) -> int:
        cursor, count = self.load_checkpoint()
        shard_idx = count // self.shard_size
        buffer: list[dict] = []

        try:
            while True:
                if max_records is not None and count >= max_records:
                    break

                records, next_cursor = self._fetch_page(cursor)

                if not records:
                    break

                if max_records is not None:
                    remaining = max_records - count
                    records = records[:remaining]

                buffer.extend(records)
                count += len(records)

                while len(buffer) >= self.shard_size:
                    shard = buffer[: self.shard_size]
                    buffer = buffer[self.shard_size :]
                    self._write_shard(shard, shard_idx)
                    shard_idx += 1
                    self.save_checkpoint(cursor, count)

                if next_cursor is None:
                    break

                cursor = next_cursor

        except KeyboardInterrupt:
            pass

        if buffer:
            self._write_shard(buffer, shard_idx)
            shard_idx += 1

        self.save_checkpoint(cursor, count)
        return count
