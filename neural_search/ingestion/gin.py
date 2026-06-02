"""G-Node GIN ingestion adapter.

GIN (Gin Is Not GitHub) hosts neuroscience datasets with BIDS/NWB metadata.
API: https://gin.g-node.org/api/v1/ (Gitea-compatible)

Searches for repos tagged with neuroscience terms.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.registry import register

logger = logging.getLogger(__name__)

GIN_API = "https://gin.g-node.org/api/v1"
GIN_SEARCH_TERMS = [
    "neuropixels", "calcium imaging", "ephys", "fmri", "eeg", "ecog",
    "mouse", "human", "NWB", "BIDS", "spike sorting", "behavior",
]


def normalize_gin_repo(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a GIN repository to the flat legacy dict format."""
    source_id = str(raw.get("id", ""))
    full_name = raw.get("full_name") or raw.get("name") or source_id
    title = raw.get("name") or full_name
    description = raw.get("description", "")
    url = raw.get("html_url") or f"https://gin.g-node.org/{full_name}"

    extraction = extract_dataset_labels(
        title=title,
        description=description,
        file_paths=[],
        source_metadata=raw,
        linked_paper_abstracts=[],
    )

    license_raw = raw.get("license")
    license_name = (
        license_raw.get("name") if isinstance(license_raw, dict) else None
    )

    return {
        "source": "gin",
        "source_id": source_id,
        "title": title,
        "description": description,
        "url": url,
        "license": license_name,
        "species": [item.id for item in extraction.species],
        "modalities": sorted({item.id for item in extraction.modalities}),
        "brain_regions": [item.id for item in extraction.brain_regions],
        "tasks": [item.id for item in extraction.tasks],
        "behaviors": [item.id for item in extraction.behaviors],
        "data_standards": ["BIDS"] if (raw.get("name") or "").upper().startswith("BIDS") else [],
        "has_behavior": bool(extraction.behaviors),
        "has_trials": any(t in str(description).lower() for t in ["trial", "stimulus"]),
        "has_raw_data": True,
        "has_processed_data": False,
        "metadata_json": {
            "raw_source": "gin",
            "full_name": full_name,
            "updated": raw.get("updated"),
            "stars": raw.get("stars_count", 0),
        },
    }


@register("gin")
def fetch_gin(limit: int = 100) -> list[dict[str, Any]]:
    """Search GIN for neuroscience dataset repositories."""
    seen_ids: set[str] = set()
    records: list[dict[str, Any]] = []

    for term in GIN_SEARCH_TERMS:
        if len(records) >= limit:
            break
        try:
            resp = httpx.get(
                f"{GIN_API}/repos/search",
                params={"q": term, "limit": 50, "topic": True},
                timeout=30,
            )
            resp.raise_for_status()
            for repo in resp.json().get("data", []):
                rid = str(repo.get("id", ""))
                if not rid or rid in seen_ids:
                    continue
                seen_ids.add(rid)
                records.append(normalize_gin_repo(repo))
                if len(records) >= limit:
                    break
        except Exception as exc:
            logger.warning("GIN fetch error for term '%s': %s", term, exc)

    logger.info("GIN: fetched %d repositories", len(records))
    return records
