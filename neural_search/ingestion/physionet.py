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
    """Collect href values matching /content/{slug}/{version}/."""

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
        "accession": slug,
        "title": title,
        "description": description,
        "url": f"{PHYSIONET_BASE}/content/{slug}/{version}/",
        "license": "ODC-By",
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


def _collect_all_listing_links(client: httpx.Client, max_pages: int = 60) -> list[str]:
    """Paginate through the PhysioNet content listing and collect all dataset links."""
    seen: set[str] = set()
    links: list[str] = []
    for page in range(1, max_pages + 1):
        try:
            resp = client.get(PHYSIONET_LISTING, params={"page": page})
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("PhysioNet listing page %d failed: %s", page, exc)
            break
        batch = _parse_dataset_links(resp.text)
        new_links = [lnk for lnk in batch if lnk not in seen]
        if not new_links:
            break
        seen.update(new_links)
        links.extend(new_links)
        logger.debug("PhysioNet listing page %d: %d new links (%d total)", page, len(new_links), len(links))
    return links


@register("physionet")
def fetch_physionet(limit: int = 200) -> list[dict[str, Any]]:
    """Scrape PhysioNet content listing (all pages) and return normalized neuro dataset records."""
    accepted: list[dict[str, Any]] = []

    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        links = _collect_all_listing_links(client)
        logger.info("PhysioNet: found %d dataset links across all pages", len(links))

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
