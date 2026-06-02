"""figshare ingestion adapter — Tier 2.

figshare hosts datasets, figures, posters, and papers.
ALL records must pass DatasetInclusionClassifier.

API: https://api.figshare.com/v2/articles?item_type=3
item_type=3 is 'dataset'.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.dataset_classifier import is_valid_dataset
from neural_search.ingestion.registry import register
from neural_search.ingestion.osf import _log_rejection

logger = logging.getLogger(__name__)

FIGSHARE_API = "https://api.figshare.com/v2"
FIGSHARE_SEARCH_TERMS = [
    "neuroscience", "electrophysiology", "fmri", "calcium imaging",
    "neuropixels", "eeg", "hippocampus", "cortex", "spike sorting",
]


def normalize_figshare_item(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a figshare article."""
    source_id = str(raw.get("id", ""))
    title = raw.get("title") or f"figshare {source_id}"
    description = raw.get("description") or raw.get("abstract") or ""
    doi = raw.get("doi") or raw.get("DOI")
    license_info = raw.get("license", {})
    license_name = (
        license_info.get("name", "") if isinstance(license_info, dict)
        else str(license_info or "")
    )
    tags = [
        t.get("value") or t if isinstance(t, dict) else str(t)
        for t in raw.get("tags", [])
    ]
    categories = [c.get("title", "") for c in raw.get("categories", []) if isinstance(c, dict)]
    defined_type = str(raw.get("defined_type_name") or raw.get("defined_type") or "")

    extraction = extract_dataset_labels(
        title=title,
        description=f"{description} {' '.join(tags)} {' '.join(categories)}",
        file_paths=[],
        source_metadata=raw,
        linked_paper_abstracts=[],
    )

    return {
        "source": "figshare",
        "source_id": source_id,
        "title": title,
        "description": description,
        "url": raw.get("url_public_html") or f"https://figshare.com/articles/{source_id}",
        "license": license_name or None,
        "doi": doi,
        "keywords": tags,
        "resource_type": defined_type or "dataset",
        "species": [item.id for item in extraction.species],
        "modalities": sorted({item.id for item in extraction.modalities}),
        "brain_regions": [item.id for item in extraction.brain_regions],
        "tasks": [item.id for item in extraction.tasks],
        "behaviors": [item.id for item in extraction.behaviors],
        "data_standards": sorted({item.id for item in extraction.data_standards}),
        "has_behavior": bool(extraction.behaviors),
        "has_trials": False,
        "has_raw_data": True,
        "has_processed_data": False,
        "metadata_json": {"raw_source": "figshare", "doi": doi, "categories": categories},
    }


@register("figshare")
def fetch_figshare(limit: int = 100) -> list[dict[str, Any]]:
    """Search figshare for neuroscience datasets (item_type=3)."""
    accepted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for term in FIGSHARE_SEARCH_TERMS:
        if len(accepted) >= limit:
            break
        try:
            resp = httpx.post(
                f"{FIGSHARE_API}/articles/search",
                json={"search_for": term, "item_type": 3, "page_size": 30},
                timeout=30,
            )
            resp.raise_for_status()
            for item in resp.json():
                sid = str(item.get("id", ""))
                if not sid or sid in seen_ids:
                    continue
                seen_ids.add(sid)
                rec = normalize_figshare_item(item)
                result = is_valid_dataset(rec)
                if result.accepted:
                    accepted.append(rec)
                else:
                    _log_rejection(rec, result.failure_reason, "figshare")
                if len(accepted) >= limit:
                    break
        except Exception as exc:
            logger.warning("figshare fetch error for '%s': %s", term, exc)

    logger.info("figshare: accepted %d datasets", len(accepted))
    return accepted
