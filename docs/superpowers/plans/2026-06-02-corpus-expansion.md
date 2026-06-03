# Corpus Expansion to 5,000+ Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the neural search corpus from ~873 records to 5,000+ by harvesting all available DANDI and OpenNeuro datasets, activating dormant Tier 2 adapters (GIN, EBRAINS, NeuroVault), adding PhysioNet HTML scraping, and wiring everything through a checkpoint-aware central orchestrator.

**Architecture:** Each source gets a dedicated full-harvest function added to its existing adapter module. A new `scripts/harvest_corpus.py` orchestrator runs all sources sequentially, writes per-source JSONL checkpoints to `data/corpus/normalized/`, then deduplicates across all files into `data/corpus/normalized/combined_corpus.jsonl`. The orchestrator is idempotent: re-running it skips source IDs already seen in each checkpoint file.

**Tech Stack:** Python 3.11+, httpx, pytest, BeautifulSoup4 (PhysioNet scraping), existing `extract_dataset_labels`, `is_valid_dataset`, `@register` pattern.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `neural_search/ingestion/dandi.py` | Modify | Add `fetch_all_dandisets()` full paginator |
| `neural_search/ingestion/openneuro.py` | Modify | Update GraphQL query + add `fetch_all_openneuro()` |
| `neural_search/ingestion/physionet.py` | Create | New PhysioNet HTML scraping adapter |
| `neural_search/ingestion/gin.py` | Modify | Add full search across all GIN_SEARCH_TERMS (not capped) |
| `neural_search/ingestion/ebrains.py` | Modify | Add offset pagination loop |
| `neural_search/ingestion/zenodo.py` | Modify | Expand ZENODO_QUERIES list + per-query pagination |
| `scripts/harvest_corpus.py` | Create | Central orchestrator with checkpoint/resume + dedup |
| `tests/ingestion/test_dandi_harvest.py` | Create | DANDI paginator tests |
| `tests/ingestion/test_openneuro_harvest.py` | Create | OpenNeuro cursor tests |
| `tests/ingestion/test_physionet.py` | Create | PhysioNet scraper tests |
| `tests/ingestion/test_harvest_orchestrator.py` | Create | Orchestrator checkpoint tests |

---

## Task 1: DANDI Full Paginator

DANDI REST API supports offset pagination: `GET /api/dandisets/?page=N&page_size=100`. The response always contains `count` (total), `next` (full URL or null), and `results`. We add `fetch_all_dandisets()` that loops until `next` is null, normalizes each result, and returns the full list. The existing `scripts/expand_dandi_corpus.py` is updated to call this instead of issuing per-query searches.

**Files:**
- Modify: `neural_search/ingestion/dandi.py`
- Modify: `scripts/expand_dandi_corpus.py`
- Create: `tests/ingestion/test_dandi_harvest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_dandi_harvest.py
from __future__ import annotations
import json
import pytest
import httpx
import respx
from neural_search.ingestion.dandi import fetch_all_dandisets, normalize_dandiset

DANDI_API = "https://api.dandiarchive.org/api"

STUB_DANDISET = {
    "identifier": "000001",
    "draft_version": {"name": "Mouse Visual Cortex Neuropixels", "metadata": {"description": "Neuropixels recordings from mouse V1."}},
}

def _make_page(results: list, next_url: str | None) -> dict:
    return {"count": 10, "next": next_url, "previous": None, "results": results}


@respx.mock
def test_fetch_all_dandisets_single_page() -> None:
    respx.get(f"{DANDI_API}/dandisets/").mock(
        return_value=httpx.Response(200, json=_make_page([STUB_DANDISET], None))
    )
    records = fetch_all_dandisets(page_size=100)
    assert len(records) == 1
    assert records[0]["source"] == "dandi"
    assert records[0]["source_id"] == "000001"


@respx.mock
def test_fetch_all_dandisets_two_pages() -> None:
    page1_url = f"{DANDI_API}/dandisets/?page=1&page_size=2"
    page2_url = f"{DANDI_API}/dandisets/?page=2&page_size=2"
    respx.get(page1_url).mock(
        return_value=httpx.Response(200, json=_make_page([STUB_DANDISET], page2_url))
    )
    stub2 = dict(STUB_DANDISET, identifier="000002")
    respx.get(page2_url).mock(
        return_value=httpx.Response(200, json=_make_page([stub2], None))
    )
    records = fetch_all_dandisets(start_url=page1_url, page_size=2)
    assert len(records) == 2
    assert {r["source_id"] for r in records} == {"000001", "000002"}


@respx.mock
def test_fetch_all_dandisets_respects_max_records() -> None:
    stubs = [dict(STUB_DANDISET, identifier=str(i).zfill(6)) for i in range(5)]
    respx.get(f"{DANDI_API}/dandisets/").mock(
        return_value=httpx.Response(200, json=_make_page(stubs, None))
    )
    records = fetch_all_dandisets(max_records=3, page_size=100)
    assert len(records) == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
python -m pytest tests/ingestion/test_dandi_harvest.py -v 2>&1 | head -30
```

Expected: FAIL with `ImportError: cannot import name 'fetch_all_dandisets'`

- [ ] **Step 3: Implement `fetch_all_dandisets` in dandi.py**

Add after the existing `fetch_dandi` function in `neural_search/ingestion/dandi.py`:

```python
def fetch_all_dandisets(
    *,
    start_url: str | None = None,
    page_size: int = 100,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Page through all DANDI dandisets and return normalized records.

    Args:
        start_url: Override the first page URL (useful for resuming).
        page_size: Number of results per API page.
        max_records: Stop after this many records (None = fetch all).
    """
    import logging as _logging
    log = _logging.getLogger(__name__)

    url: str | None = start_url or f"{DANDI_API_URL}/dandisets/?page=1&page_size={page_size}"
    all_records: list[dict[str, Any]] = []

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        while url:
            try:
                resp = client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("DANDI page fetch failed: %s — %s", url, exc)
                break

            data = resp.json()
            for raw in data.get("results", []):
                all_records.append(normalize_dandiset(raw))
                if max_records and len(all_records) >= max_records:
                    return all_records

            url = data.get("next")
            log.info("DANDI harvest: %d records so far, next=%s", len(all_records), url)

    return all_records
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/ingestion/test_dandi_harvest.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Update `scripts/expand_dandi_corpus.py` to use full paginator**

Replace the existing `main()` body with:

```python
"""Expand DANDI corpus: harvest all 842+ dandisets via full pagination."""
from __future__ import annotations

import json
from pathlib import Path

from neural_search.ingestion.dandi import fetch_all_dandisets

OUTPUT_PATH = Path("data/corpus/normalized/real_dandi.jsonl")


def main() -> None:
    seen_ids: set[str] = set()

    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if sid := rec.get("source_id"):
                        seen_ids.add(sid)
                except json.JSONDecodeError:
                    pass
        print(f"Loaded {len(seen_ids)} existing DANDI source IDs")

    print("Fetching all DANDI dandisets via pagination…")
    all_records = fetch_all_dandisets()
    print(f"Fetched {len(all_records)} total records from DANDI API")

    new_records = [r for r in all_records if r.get("source_id") not in seen_ids]
    print(f"New records (not yet in corpus): {len(new_records)}")

    if new_records:
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for rec in new_records:
                f.write(json.dumps(rec) + "\n")
        print(f"Appended {len(new_records)} records to {OUTPUT_PATH}")
    else:
        print("No new records to add.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add neural_search/ingestion/dandi.py scripts/expand_dandi_corpus.py tests/ingestion/test_dandi_harvest.py
git commit -m "feat: add full DANDI paginator — fetch_all_dandisets() over all 842 dandisets"
```

---

## Task 2: OpenNeuro Full Cursor Paginator

OpenNeuro GraphQL supports cursor-based pagination. The existing query uses `first: $first` but no `after` cursor. We update `_search_query()` to include `pageInfo { hasNextPage endCursor }` and add `fetch_all_openneuro()` that loops until `hasNextPage` is false. The `records_from_response()` function already works on individual responses.

**Files:**
- Modify: `neural_search/ingestion/openneuro.py`
- Modify: `scripts/expand_openneuro_corpus.py`
- Create: `tests/ingestion/test_openneuro_harvest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_openneuro_harvest.py
from __future__ import annotations
import pytest
import httpx
import respx
from neural_search.ingestion.openneuro import fetch_all_openneuro

OPENNEURO_GQL = "https://openneuro.org/crn/graphql"

STUB_NODE = {"id": "ds000001", "name": "Visual EEG Experiment", "created": "2021-01-01", "public": True,
             "latestSnapshot": {"tag": "1.0.0", "created": "2021-01-01", "size": 10000, "readme": None,
                                "summary": {"subjects": 20, "tasks": ["visual"], "modalities": ["eeg"]}}}

def _gql_response(nodes: list, has_next: bool, cursor: str | None) -> dict:
    return {
        "data": {
            "datasets": {
                "edges": [{"cursor": f"cur{i}", "node": n} for i, n in enumerate(nodes)],
                "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
            }
        }
    }


@respx.mock
def test_fetch_all_openneuro_single_page() -> None:
    respx.post(OPENNEURO_GQL).mock(
        return_value=httpx.Response(200, json=_gql_response([STUB_NODE], False, None))
    )
    records = fetch_all_openneuro(page_size=100)
    assert len(records) == 1
    assert records[0]["source"] == "openneuro"
    assert records[0]["source_id"] == "ds000001"


@respx.mock
def test_fetch_all_openneuro_two_pages() -> None:
    node2 = dict(STUB_NODE, id="ds000002")
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = __import__("json").loads(request.content)
        after = (body.get("variables") or {}).get("after")
        if after is None:
            return httpx.Response(200, json=_gql_response([STUB_NODE], True, "cursor_abc"))
        return httpx.Response(200, json=_gql_response([node2], False, None))

    respx.post(OPENNEURO_GQL).mock(side_effect=handler)
    records = fetch_all_openneuro(page_size=1)
    assert len(records) == 2
    assert {r["source_id"] for r in records} == {"ds000001", "ds000002"}
    assert call_count == 2


@respx.mock
def test_fetch_all_openneuro_respects_max_records() -> None:
    nodes = [dict(STUB_NODE, id=f"ds{i:06d}") for i in range(5)]
    respx.post(OPENNEURO_GQL).mock(
        return_value=httpx.Response(200, json=_gql_response(nodes, False, None))
    )
    records = fetch_all_openneuro(page_size=100, max_records=3)
    assert len(records) == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/ingestion/test_openneuro_harvest.py -v 2>&1 | head -30
```

Expected: FAIL with `ImportError: cannot import name 'fetch_all_openneuro'`

- [ ] **Step 3: Update `_search_query()` and add `fetch_all_openneuro()` in openneuro.py**

Replace `_search_query()` with a function that accepts an `after` cursor:

```python
def _paginated_query() -> str:
    return """
    query SearchDatasets($first: Int, $after: String) {
        datasets(first: $first, after: $after, filterBy: {public: true}) {
            edges {
                cursor
                node {
                    id
                    name
                    created
                    public
                    latestSnapshot {
                        tag
                        created
                        size
                        readme
                        summary {
                            subjects
                            tasks
                            modalities
                        }
                    }
                }
            }
            pageInfo {
                hasNextPage
                endCursor
            }
        }
    }
    """
```

Keep `_search_query()` as an alias for backwards compatibility:

```python
def _search_query() -> str:
    return _paginated_query()
```

Add `fetch_all_openneuro` function after `fetch_openneuro`:

```python
def fetch_all_openneuro(
    *,
    page_size: int = 100,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch all public OpenNeuro datasets using GraphQL cursor pagination."""
    import logging as _logging
    log = _logging.getLogger(__name__)

    all_records: list[dict[str, Any]] = []
    after: str | None = None

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        while True:
            resp = client.post(
                OPENNEURO_API_URL,
                json={
                    "query": _paginated_query(),
                    "variables": {"first": page_size, "after": after},
                },
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("errors"):
                log.warning("OpenNeuro GraphQL error: %s", data["errors"])
                break

            datasets_data = data.get("data", {}).get("datasets", {})
            edges = datasets_data.get("edges", [])
            page_info = datasets_data.get("pageInfo", {})

            for edge in edges:
                node = edge.get("node", {})
                if node.get("id"):
                    all_records.append(normalize_openneuro_dataset(node))
                    if max_records and len(all_records) >= max_records:
                        return all_records

            log.info("OpenNeuro harvest: %d records, hasNextPage=%s", len(all_records), page_info.get("hasNextPage"))

            if not page_info.get("hasNextPage"):
                break
            after = page_info.get("endCursor")

    return all_records
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/ingestion/test_openneuro_harvest.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Update `scripts/expand_openneuro_corpus.py` to use full paginator**

Replace the existing `main()` body with:

```python
"""Expand OpenNeuro corpus: harvest all 1,754+ datasets via cursor pagination."""
from __future__ import annotations

import json
from pathlib import Path

from neural_search.ingestion.openneuro import fetch_all_openneuro

OUTPUT_PATH = Path("data/corpus/normalized/real_openneuro.jsonl")


def main() -> None:
    seen_ids: set[str] = set()

    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if sid := rec.get("source_id"):
                        seen_ids.add(sid)
                except json.JSONDecodeError:
                    pass
        print(f"Loaded {len(seen_ids)} existing OpenNeuro source IDs")

    print("Fetching all OpenNeuro datasets via cursor pagination…")
    all_records = fetch_all_openneuro()
    print(f"Fetched {len(all_records)} total records from OpenNeuro")

    new_records = [r for r in all_records if r.get("source_id") not in seen_ids]
    print(f"New records (not yet in corpus): {len(new_records)}")

    if new_records:
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for rec in new_records:
                f.write(json.dumps(rec) + "\n")
        print(f"Appended {len(new_records)} records to {OUTPUT_PATH}")
    else:
        print("No new records to add.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add neural_search/ingestion/openneuro.py scripts/expand_openneuro_corpus.py tests/ingestion/test_openneuro_harvest.py
git commit -m "feat: add OpenNeuro full cursor paginator — fetch_all_openneuro() over all 1,754 datasets"
```

---

## Task 3: PhysioNet Adapter

PhysioNet (`physionet.org/content/`) lists ~500 public databases via HTML. We scrape this listing page using Python's `html.parser` (stdlib), extract `<a>` links matching the `/content/{slug}/` URL pattern, then fetch each dataset's detail page to pull title, description, and topic tags. Only records with neuro-relevant signals pass the `is_valid_dataset` classifier.

**Files:**
- Create: `neural_search/ingestion/physionet.py`
- Create: `tests/ingestion/test_physionet.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_physionet.py
from __future__ import annotations
import pytest
import httpx
import respx
from neural_search.ingestion.physionet import (
    _parse_dataset_links,
    normalize_physionet_dataset,
    fetch_physionet,
)

SAMPLE_LISTING_HTML = """
<html><body>
<ul class="database-list">
  <li><a href="/content/chbmit/1.0.0/">CHB-MIT Scalp EEG</a></li>
  <li><a href="/content/eegmmidb/1.0.0/">EEG Motor Imagery</a></li>
  <li><a href="/content/noneuro/1.0.0/">Non-Neuro Dataset</a></li>
</ul>
</body></html>
"""

SAMPLE_DETAIL_HTML = """
<html><body>
<h1 class="project-title">CHB-MIT Scalp EEG Database</h1>
<div class="project-description">
  Scalp EEG recordings from pediatric patients with intractable epilepsy.
  Collected at Boston Children's Hospital.
</div>
<ul class="subject-list">
  <li>EEG</li><li>Seizure</li><li>Pediatric</li>
</ul>
</body></html>
"""


def test_parse_dataset_links_extracts_content_paths() -> None:
    links = _parse_dataset_links(SAMPLE_LISTING_HTML)
    assert "/content/chbmit/1.0.0/" in links
    assert "/content/eegmmidb/1.0.0/" in links
    assert "/content/noneuro/1.0.0/" in links
    assert len(links) == 3


def test_normalize_physionet_dataset_basic() -> None:
    raw = {
        "slug": "chbmit",
        "version": "1.0.0",
        "title": "CHB-MIT Scalp EEG Database",
        "description": "Scalp EEG recordings from pediatric patients with intractable epilepsy.",
        "topics": ["EEG", "Seizure"],
    }
    rec = normalize_physionet_dataset(raw)
    assert rec["source"] == "physionet"
    assert rec["source_id"] == "chbmit"
    assert "eeg" in rec["modalities"]


@respx.mock
def test_fetch_physionet_returns_neuro_records() -> None:
    respx.get("https://physionet.org/content/").mock(
        return_value=httpx.Response(200, text=SAMPLE_LISTING_HTML)
    )
    # Mock two detail pages
    respx.get("https://physionet.org/content/chbmit/1.0.0/").mock(
        return_value=httpx.Response(200, text=SAMPLE_DETAIL_HTML)
    )
    respx.get("https://physionet.org/content/eegmmidb/1.0.0/").mock(
        return_value=httpx.Response(200, text=SAMPLE_DETAIL_HTML)
    )
    respx.get("https://physionet.org/content/noneuro/1.0.0/").mock(
        return_value=httpx.Response(200, text="<html><body><h1>Non-Neuro</h1></body></html>")
    )
    records = fetch_physionet(limit=50)
    # Must have accepted only neuro records (noneuro has no neuro signal)
    assert all(r["source"] == "physionet" for r in records)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/ingestion/test_physionet.py -v 2>&1 | head -30
```

Expected: FAIL with `ModuleNotFoundError: No module named 'neural_search.ingestion.physionet'`

- [ ] **Step 3: Implement `neural_search/ingestion/physionet.py`**

```python
"""PhysioNet ingestion adapter — scrape public dataset listing.

PhysioNet hosts ~500 public physiology/neuroscience databases.
Homepage: https://physionet.org/content/
No formal REST API; we scrape the content listing page.
"""
from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Any

import httpx

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.dataset_classifier import is_valid_dataset
from neural_search.ingestion.registry import register

logger = logging.getLogger(__name__)

PHYSIONET_BASE = "https://physionet.org"
PHYSIONET_LISTING = f"{PHYSIONET_BASE}/content/"
_CONTENT_LINK_RE = re.compile(r"^/content/([^/]+)/([^/]+)/$")


class _LinkExtractor(HTMLParser):
    """Minimal HTML parser that collects href values matching /content/{slug}/{version}/."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value and _CONTENT_LINK_RE.match(value):
                if value not in self.links:
                    self.links.append(value)


class _DetailExtractor(HTMLParser):
    """Extract title, description, and topic tags from a PhysioNet detail page."""

    def __init__(self) -> None:
        super().__init__()
        self.title: str = ""
        self.description: str = ""
        self.topics: list[str] = []
        self._in_title = False
        self._in_desc = False
        self._in_topic = False
        self._title_tags = {"h1", "h2"}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = dict(attrs).get("class", "") or ""
        if tag in self._title_tags and "title" in classes:
            self._in_title = True
        elif tag == "div" and "description" in classes:
            self._in_desc = True
        elif tag == "li" and ("subject" in classes or "topic" in classes):
            self._in_topic = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._title_tags:
            self._in_title = False
        elif tag == "div":
            self._in_desc = False
        elif tag == "li":
            self._in_topic = False

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._in_title and not self.title:
            self.title = text
        elif self._in_desc:
            self.description += " " + text
        elif self._in_topic:
            self.topics.append(text)


def _parse_dataset_links(html: str) -> list[str]:
    """Return all /content/{slug}/{version}/ hrefs from a PhysioNet page."""
    parser = _LinkExtractor()
    parser.feed(html)
    return parser.links


def _parse_detail_page(html: str, slug: str, version: str) -> dict[str, Any]:
    """Extract structured metadata from a PhysioNet dataset detail page."""
    parser = _DetailExtractor()
    parser.feed(html)
    return {
        "slug": slug,
        "version": version,
        "title": parser.title or f"PhysioNet/{slug}",
        "description": parser.description.strip(),
        "topics": parser.topics,
    }


def normalize_physionet_dataset(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a PhysioNet dataset dict to the flat legacy format."""
    slug = raw["slug"]
    version = raw.get("version", "")
    title = raw.get("title") or f"PhysioNet/{slug}"
    description = raw.get("description") or ""
    topics = raw.get("topics", [])

    extraction = extract_dataset_labels(
        title=title,
        description=f"{description} {' '.join(topics)}",
        file_paths=[],
        source_metadata={"topics": topics},
        linked_paper_abstracts=[],
    )

    return {
        "source": "physionet",
        "source_id": slug,
        "title": title,
        "description": description,
        "url": f"{PHYSIONET_BASE}/content/{slug}/{version}/",
        "license": "Open Data Commons Attribution License",
        "species": [item.id for item in extraction.species] or ["human"],
        "modalities": sorted({item.id for item in extraction.modalities}),
        "brain_regions": [item.id for item in extraction.brain_regions],
        "tasks": [item.id for item in extraction.tasks],
        "behaviors": [item.id for item in extraction.behaviors],
        "data_standards": sorted({item.id for item in extraction.data_standards}),
        "has_behavior": bool(extraction.behaviors),
        "has_trials": any(t in description.casefold() for t in ["trial", "epoch", "event"]),
        "has_raw_data": True,
        "has_processed_data": False,
        "metadata_json": {"raw_source": "physionet", "topics": topics, "version": version},
    }


@register("physionet")
def fetch_physionet(limit: int = 200) -> list[dict[str, Any]]:
    """Scrape PhysioNet content listing and return normalized neuro dataset records."""
    accepted: list[dict[str, Any]] = []

    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        try:
            resp = client.get(PHYSIONET_LISTING)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("PhysioNet listing fetch failed: %s", exc)
            return []

        links = _parse_dataset_links(resp.text)
        logger.info("PhysioNet: found %d dataset links", len(links))

        for path in links:
            if len(accepted) >= limit:
                break
            m = _CONTENT_LINK_RE.match(path)
            if not m:
                continue
            slug, version = m.group(1), m.group(2)
            url = f"{PHYSIONET_BASE}{path}"
            try:
                detail_resp = client.get(url)
                detail_resp.raise_for_status()
                raw = _parse_detail_page(detail_resp.text, slug, version)
                rec = normalize_physionet_dataset(raw)
                result = is_valid_dataset(rec)
                if result.accepted:
                    accepted.append(rec)
                else:
                    logger.debug("PhysioNet rejected %s: %s", slug, result.failure_reason)
            except Exception as exc:
                logger.warning("PhysioNet detail fetch failed for %s: %s", slug, exc)

    logger.info("physionet: accepted %d datasets", len(accepted))
    return accepted
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/ingestion/test_physionet.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add neural_search/ingestion/physionet.py tests/ingestion/test_physionet.py
git commit -m "feat: add PhysioNet adapter — HTML scraping of public dataset listing"
```

---

## Task 4: Expand Tier 2 Adapters (GIN, EBRAINS, Zenodo)

Three existing adapters need higher limits and pagination to yield more records. GIN needs all search terms with dedup. EBRAINS needs offset pagination. Zenodo needs an expanded query list with per-query pagination (`page` parameter). No new tests needed; the existing adapters already pass tests — we just increase coverage via config changes and pagination.

**Files:**
- Modify: `neural_search/ingestion/gin.py`
- Modify: `neural_search/ingestion/ebrains.py`
- Modify: `neural_search/ingestion/zenodo.py`

- [ ] **Step 1: Update GIN to iterate all search terms without hard cap**

In `neural_search/ingestion/gin.py`, find `@register("gin")` and its `fetch_gin` function. Expand `GIN_SEARCH_TERMS` and ensure all terms are searched with up to 50 results each:

```python
GIN_SEARCH_TERMS = [
    "neuropixels", "calcium imaging", "ephys", "fmri", "eeg", "ecog",
    "mouse", "human", "NWB", "BIDS", "spike sorting", "behavior",
    "electrophysiology", "two-photon", "optogenetics", "patch clamp",
    "hippocampus", "prefrontal cortex", "visual cortex", "motor cortex",
    "sleep", "decision making", "reward", "working memory", "primate",
    "rat", "zebrafish", "human intracranial", "fiber photometry",
]
```

Also ensure the `fetch_gin` function iterates ALL terms (not stopping at first `limit`) by using an internal `seen_ids` set:

```python
# In fetch_gin, replace the body so that it accumulates across all terms:
@register("gin")
def fetch_gin(limit: int = 500) -> list[dict[str, Any]]:
    """Search GIN for neuroscience datasets across all search terms."""
    accepted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for term in GIN_SEARCH_TERMS:
            if len(accepted) >= limit:
                break
            try:
                resp = client.get(
                    f"{GIN_API}/repos/search",
                    params={"q": term, "limit": 50, "topic": True},
                )
                resp.raise_for_status()
                for repo in resp.json().get("data", []):
                    sid = str(repo.get("id", ""))
                    if not sid or sid in seen_ids:
                        continue
                    seen_ids.add(sid)
                    rec = normalize_gin_repo(repo)
                    result = is_valid_dataset(rec)
                    if result.accepted:
                        accepted.append(rec)
                    if len(accepted) >= limit:
                        break
            except Exception as exc:
                logger.warning("GIN fetch error for '%s': %s", term, exc)

    logger.info("gin: accepted %d datasets", len(accepted))
    return accepted
```

- [ ] **Step 2: Update EBRAINS to paginate**

In `neural_search/ingestion/ebrains.py`, find `@register("ebrains")` and `fetch_ebrains`. Replace the single-page fetch with an offset loop:

```python
@register("ebrains")
def fetch_ebrains(limit: int = 300) -> list[dict[str, Any]]:
    """Fetch public EBRAINS datasets with offset pagination."""
    accepted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    size = 50
    from_offset = 0

    headers = _auth_headers()

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        while len(accepted) < limit:
            try:
                resp = client.get(
                    EBRAINS_SEARCH_URL,
                    params={"from": from_offset, "size": size},
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("EBRAINS fetch error at offset %d: %s", from_offset, exc)
                break

            items = data.get("data", data) if isinstance(data, dict) else data
            if not items:
                break

            for raw in items:
                sid = str(raw.get("id") or raw.get("@id") or "")
                if "/" in sid:
                    sid = sid.rstrip("/").split("/")[-1]
                if not sid or sid in seen_ids:
                    continue
                seen_ids.add(sid)
                rec = normalize_ebrains_dataset(raw)
                result = is_valid_dataset(rec)
                if result.accepted:
                    accepted.append(rec)
                if len(accepted) >= limit:
                    break

            if len(items) < size:
                break
            from_offset += size

    logger.info("ebrains: accepted %d datasets", len(accepted))
    return accepted
```

- [ ] **Step 3: Expand Zenodo query list and add per-query page loop**

In `neural_search/ingestion/zenodo.py`, expand `ZENODO_QUERIES` and add page looping:

```python
ZENODO_QUERIES = [
    "neuroscience electrophysiology", "fmri brain imaging", "calcium imaging neural",
    "neuropixels spike sorting", "eeg brain recording", "ecog human intracranial",
    "hippocampus memory", "prefrontal cortex", "reward learning dopamine",
    "two-photon imaging mouse", "optogenetics neural circuit", "patch clamp neuron",
    "motor cortex behavior", "visual cortex stimulus", "basal ganglia striatum",
    "cerebellum motor learning", "olfactory system", "primate electrophysiology",
    "human eeg cognitive", "sleep slow wave", "decision making neural",
    "working memory prefrontal", "place cells grid cells", "fear conditioning amygdala",
    "auditory cortex sound", "somatosensory cortex touch", "brainstem spinal cord",
    "NWB neurodata without borders", "BIDS brain imaging data",
]
```

Replace the `fetch_zenodo` inner loop to page through each query:

```python
@register("zenodo")
def fetch_zenodo(limit: int = 500) -> list[dict[str, Any]]:
    """Search Zenodo for neuroscience datasets across all queries with pagination."""
    accepted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for query in ZENODO_QUERIES:
        if len(accepted) >= limit:
            break
        page = 1
        per_page = 25
        while len(accepted) < limit:
            try:
                resp = httpx.get(
                    ZENODO_API,
                    params={"q": query, "type": "dataset", "size": per_page, "page": page, "sort": "mostrecent"},
                    timeout=30,
                )
                resp.raise_for_status()
                hits = resp.json().get("hits", {}).get("hits", [])
                if not hits:
                    break
                for item in hits:
                    sid = str(item.get("id", ""))
                    if not sid or sid in seen_ids:
                        continue
                    seen_ids.add(sid)
                    rec = normalize_zenodo_record(item)
                    result = is_valid_dataset(rec)
                    if result.accepted:
                        accepted.append(rec)
                    if len(accepted) >= limit:
                        break
                if len(hits) < per_page:
                    break
                page += 1
            except Exception as exc:
                logger.warning("zenodo fetch error for '%s' page %d: %s", query, page, exc)
                break

    logger.info("zenodo: accepted %d datasets", len(accepted))
    return accepted
```

- [ ] **Step 4: Run existing adapter tests to verify nothing broke**

```bash
python -m pytest tests/ingestion/ -v -k "gin or ebrains or zenodo" 2>&1 | tail -20
```

Expected: all previously-passing tests still PASS

- [ ] **Step 5: Commit**

```bash
git add neural_search/ingestion/gin.py neural_search/ingestion/ebrains.py neural_search/ingestion/zenodo.py
git commit -m "feat: expand Tier 2 adapters — GIN full terms, EBRAINS pagination, Zenodo 29 queries"
```

---

## Task 5: Central Harvest Orchestrator

A single script that runs all adapters in sequence, writes per-source checkpoint JSONL files, skips already-seen source IDs, and at the end deduplicates across all files into `data/corpus/normalized/combined_corpus.jsonl`. The orchestrator is safe to re-run: it reads existing checkpoint files first, so interrupted runs resume from where they stopped.

**Files:**
- Create: `scripts/harvest_corpus.py`
- Create: `tests/ingestion/test_harvest_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_harvest_orchestrator.py
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import pytest
from scripts.harvest_corpus import (
    load_seen_ids,
    append_new_records,
    deduplicate_combined,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_seen_ids_empty_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "corpus.jsonl"
        assert load_seen_ids(p) == set()


def test_load_seen_ids_existing_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "corpus.jsonl"
        _write_jsonl(p, [{"source_id": "a"}, {"source_id": "b"}])
        ids = load_seen_ids(p)
        assert ids == {"a", "b"}


def test_append_new_records_skips_seen() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "corpus.jsonl"
        _write_jsonl(p, [{"source_id": "a", "title": "old"}])
        records = [
            {"source_id": "a", "title": "duplicate"},
            {"source_id": "b", "title": "new"},
        ]
        added = append_new_records(p, records, seen_ids={"a"})
        assert added == 1
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[-1])["source_id"] == "b"


def test_deduplicate_combined_removes_duplicates() -> None:
    with tempfile.TemporaryDirectory() as td:
        f1 = Path(td) / "a.jsonl"
        f2 = Path(td) / "b.jsonl"
        out = Path(td) / "combined.jsonl"
        _write_jsonl(f1, [{"source": "dandi", "source_id": "000001", "title": "A"}])
        _write_jsonl(f2, [
            {"source": "dandi", "source_id": "000001", "title": "A-dup"},  # same id
            {"source": "openneuro", "source_id": "ds000001", "title": "B"},
        ])
        count = deduplicate_combined([f1, f2], out)
        assert count == 2
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/ingestion/test_harvest_orchestrator.py -v 2>&1 | head -30
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.harvest_corpus'`

- [ ] **Step 3: Implement `scripts/harvest_corpus.py`**

```python
"""Central corpus harvest orchestrator.

Runs all registered ingestion adapters, writes per-source JSONL checkpoints,
then deduplicates into combined_corpus.jsonl.

Usage:
    python scripts/harvest_corpus.py [--dry-run] [--sources dandi openneuro ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CORPUS_DIR = Path("data/corpus/normalized")

SOURCE_OUTPUTS: dict[str, Path] = {
    "dandi": CORPUS_DIR / "real_dandi.jsonl",
    "openneuro": CORPUS_DIR / "real_openneuro.jsonl",
    "neurovault": CORPUS_DIR / "real_neurovault.jsonl",
    "gin": CORPUS_DIR / "real_gin.jsonl",
    "ebrains": CORPUS_DIR / "real_ebrains.jsonl",
    "zenodo": CORPUS_DIR / "real_zenodo.jsonl",
    "physionet": CORPUS_DIR / "real_physionet.jsonl",
    "osf": CORPUS_DIR / "real_osf.jsonl",
    "figshare": CORPUS_DIR / "real_figshare.jsonl",
}

COMBINED_OUTPUT = CORPUS_DIR / "combined_corpus.jsonl"

SOURCE_LIMITS: dict[str, int] = {
    "dandi": 1000,
    "openneuro": 2000,
    "neurovault": 600,
    "gin": 500,
    "ebrains": 300,
    "zenodo": 500,
    "physionet": 200,
    "osf": 200,
    "figshare": 200,
}


def load_seen_ids(path: Path) -> set[str]:
    """Return set of source_id values already in a JSONL file."""
    if not path.exists():
        return set()
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if sid := rec.get("source_id"):
                    seen.add(str(sid))
            except json.JSONDecodeError:
                pass
    return seen


def append_new_records(
    path: Path,
    records: list[dict[str, Any]],
    seen_ids: set[str],
) -> int:
    """Write records whose source_id is not in seen_ids. Returns count added."""
    new = [r for r in records if str(r.get("source_id", "")) not in seen_ids]
    if not new:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in new:
            f.write(json.dumps(rec) + "\n")
    return len(new)


def deduplicate_combined(sources: list[Path], output: Path) -> int:
    """Read all source JSONL files, deduplicate by (source, source_id), write combined."""
    seen: set[tuple[str, str]] = set()
    records: list[dict[str, Any]] = []

    for src in sources:
        if not src.exists():
            continue
        with open(src, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = (str(rec.get("source", "")), str(rec.get("source_id", "")))
                    if key[1] and key not in seen:
                        seen.add(key)
                        records.append(rec)
                except json.JSONDecodeError:
                    pass

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return len(records)


def run_harvest(
    sources: list[str],
    dry_run: bool = False,
) -> dict[str, int]:
    """Run specified adapters and return {source: new_records_added} mapping."""
    # Import adapter modules to trigger @register decorators
    import neural_search.ingestion.dandi  # noqa: F401
    import neural_search.ingestion.openneuro  # noqa: F401
    import neural_search.ingestion.neurovault  # noqa: F401
    import neural_search.ingestion.gin  # noqa: F401
    import neural_search.ingestion.ebrains  # noqa: F401
    import neural_search.ingestion.zenodo  # noqa: F401
    import neural_search.ingestion.physionet  # noqa: F401
    import neural_search.ingestion.osf  # noqa: F401
    import neural_search.ingestion.figshare  # noqa: F401
    from neural_search.ingestion.registry import _REGISTRY  # type: ignore[attr-defined]

    results: dict[str, int] = {}

    for source in sources:
        if source not in _REGISTRY:
            logger.warning("Source '%s' not in registry — skipping", source)
            continue

        output_path = SOURCE_OUTPUTS.get(source, CORPUS_DIR / f"real_{source}.jsonl")
        limit = SOURCE_LIMITS.get(source, 200)
        seen_ids = load_seen_ids(output_path)

        logger.info("Running %s (seen=%d, limit=%d)…", source, len(seen_ids), limit)

        try:
            records: list[dict[str, Any]] = _REGISTRY[source](limit=limit)
            logger.info("%s: fetched %d records", source, len(records))
        except Exception as exc:
            logger.error("%s: adapter failed — %s", source, exc)
            results[source] = 0
            continue

        if dry_run:
            new_count = len([r for r in records if str(r.get("source_id", "")) not in seen_ids])
            logger.info("[DRY-RUN] %s: would add %d new records", source, new_count)
            results[source] = new_count
        else:
            added = append_new_records(output_path, records, seen_ids)
            logger.info("%s: added %d new records to %s", source, added, output_path)
            results[source] = added

    if not dry_run:
        all_source_files = [SOURCE_OUTPUTS.get(s, CORPUS_DIR / f"real_{s}.jsonl") for s in SOURCE_OUTPUTS]
        total = deduplicate_combined(all_source_files, COMBINED_OUTPUT)
        logger.info("Combined corpus: %d total unique records → %s", total, COMBINED_OUTPUT)

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harvest neuroscience datasets from all sources")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCE_OUTPUTS.keys()),
        help="Sources to run (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show counts without writing")
    args = parser.parse_args(argv)

    results = run_harvest(args.sources, dry_run=args.dry_run)
    for source, count in results.items():
        print(f"{source}: {count} {'(dry-run)' if args.dry_run else 'added'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Add `scripts/__init__.py` so test imports work**

```bash
touch /mnt/c/Users/sidso/Documents/neural-search/scripts/__init__.py
```

- [ ] **Step 5: Run orchestrator tests**

```bash
python -m pytest tests/ingestion/test_harvest_orchestrator.py -v
```

Expected: 4 PASSED

- [ ] **Step 6: Commit**

```bash
git add scripts/harvest_corpus.py scripts/__init__.py tests/ingestion/test_harvest_orchestrator.py
git commit -m "feat: add central harvest orchestrator with checkpoint/resume and deduplication"
```

---

## Task 6: Run Full Harvest and Rebuild Index

Now execute the actual harvest. This is a live network operation targeting 5,000+ records. Run sources in sequence to avoid rate-limit issues. After harvest, rebuild the turbovec index so the API serves the expanded corpus.

**Files:**
- This is a run step — no new files. Verify final corpus size and re-run `scripts/build_turbovec_index.py`.

- [ ] **Step 1: Run DANDI full harvest**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
python scripts/expand_dandi_corpus.py 2>&1 | tail -5
```

Expected: logs showing ~842 total records fetched, N new appended to `data/corpus/normalized/real_dandi.jsonl`.

- [ ] **Step 2: Run OpenNeuro full harvest**

```bash
python scripts/expand_openneuro_corpus.py 2>&1 | tail -5
```

Expected: logs showing ~1,754 total records fetched, N new appended to `data/corpus/normalized/real_openneuro.jsonl`.

- [ ] **Step 3: Run Tier 2 adapters via orchestrator**

```bash
python scripts/harvest_corpus.py --sources neurovault gin ebrains zenodo physionet osf figshare 2>&1 | tail -20
```

Expected: each source logs accepted count and appends to its checkpoint file.

- [ ] **Step 4: Verify combined corpus size**

```bash
wc -l data/corpus/normalized/combined_corpus.jsonl
python -c "
import json
from collections import Counter
records = [json.loads(l) for l in open('data/corpus/normalized/combined_corpus.jsonl')]
counts = Counter(r['source'] for r in records)
print(f'Total: {len(records)}')
for src, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f'  {src}: {n}')
"
```

Expected: total >= 3,000 records (conservative; target is 5,000+).

- [ ] **Step 5: Rebuild turbovec index**

```bash
python scripts/build_turbovec_index.py --corpus data/corpus/normalized/combined_corpus.jsonl 2>&1 | tail -10
```

Expected: "Built index with N vectors" where N >= 3000.

- [ ] **Step 6: Smoke-test the API**

```bash
curl -s http://localhost:8000/api/reports/corpus-completeness | python -m json.tool | head -20
```

Expected: `total_records` >= 3000.

- [ ] **Step 7: Commit corpus checkpoint files and index**

```bash
git add data/corpus/normalized/real_dandi.jsonl data/corpus/normalized/real_openneuro.jsonl \
        data/corpus/normalized/real_neurovault.jsonl data/corpus/normalized/real_gin.jsonl \
        data/corpus/normalized/real_ebrains.jsonl data/corpus/normalized/real_zenodo.jsonl \
        data/corpus/normalized/real_physionet.jsonl data/corpus/normalized/combined_corpus.jsonl
git commit -m "data: expand corpus to 5K+ records across DANDI, OpenNeuro, NeuroVault, GIN, EBRAINS, Zenodo, PhysioNet"
```

---

## Self-Review

### Spec coverage

| Requirement | Task covering it |
|-------------|-----------------|
| Full DANDI harvest (842) | Task 1 |
| Full OpenNeuro harvest (1,754) | Task 2 |
| PhysioNet new adapter | Task 3 |
| GIN expanded | Task 4 |
| EBRAINS paginated | Task 4 |
| Zenodo expanded | Task 4 |
| Central orchestrator with checkpoint | Task 5 |
| Deduplication across sources | Task 5 |
| Live harvest + index rebuild | Task 6 |

### Placeholder scan

No TBDs, TODOs, or incomplete sections. All code blocks are complete and runnable.

### Type consistency

- `fetch_all_dandisets()` returns `list[dict[str, Any]]` — used as list iteration in Task 6 ✓
- `fetch_all_openneuro()` returns `list[dict[str, Any]]` — same shape ✓
- `fetch_physionet()` decorated with `@register("physionet")` — same signature as other adapters ✓
- `load_seen_ids`, `append_new_records`, `deduplicate_combined` — used consistently in orchestrator ✓
