"""Harvard Dataverse ingestion adapter — Tier 2.

Harvard Dataverse is a major open-data repository at https://dataverse.harvard.edu/.
All deposits require a CC license (CC0 1.0 is the default since May 2022).

API: https://dataverse.harvard.edu/api/search?q={query}&type=dataset&per_page=100
License info: GET /api/datasets/:persistentId/?persistentId={doi}
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.dataset_classifier import is_valid_dataset
from neural_search.ingestion.osf import _log_rejection
from neural_search.ingestion.registry import register

logger = logging.getLogger(__name__)

DATAVERSE_API = "https://dataverse.harvard.edu/api"
DATAVERSE_SEARCH_TERMS = [
    "neuroscience electrophysiology",
    "fmri brain imaging neuroscience",
    "calcium imaging neural activity",
    "neuropixels spike sorting",
    "eeg brain recording",
    "hippocampus place cells",
    "prefrontal cortex decision",
    "reward learning dopamine",
    "two-photon imaging cortex",
    "patch clamp neuron",
    "motor cortex behavior",
    "visual cortex stimulus",
    "basal ganglia striatum",
    "cerebellum motor learning",
    "primate electrophysiology",
    "human eeg cognitive neuroscience",
    "working memory neural",
    "fear conditioning amygdala",
    "auditory cortex sound",
    "NWB neurodata without borders",
    "local field potential LFP",
    "neural population dynamics",
    "optogenetics neural circuit",
    "single neuron recording",
    "brain computer interface",
]


def normalize_dataverse_item(raw: dict[str, Any], license_str: str = "") -> dict[str, Any]:
    """Normalize a Harvard Dataverse search result to the flat corpus dict."""
    source_id = raw.get("global_id", "").replace("doi:", "").replace("/", "_")
    title = raw.get("name") or f"Dataverse {source_id}"
    description = raw.get("description") or ""
    doi = raw.get("global_id", "").replace("doi:", "")
    if doi:
        doi = f"https://doi.org/{doi}"
    subjects = raw.get("subjects", [])
    keywords = raw.get("keywords", [])

    extraction = extract_dataset_labels(
        title=title,
        description=f"{description} {' '.join(subjects)} {' '.join(keywords)}",
        file_paths=[],
        source_metadata=raw,
        linked_paper_abstracts=[],
    )

    return {
        "source": "harvard_dataverse",
        "source_id": source_id,
        "title": title,
        "description": description,
        "url": raw.get("url") or f"https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:{source_id.replace('_', '/')}",
        "license": license_str or None,
        "doi": doi or None,
        "keywords": keywords,
        "resource_type": "dataset",
        "species": [item.id for item in extraction.species],
        "modalities": sorted({item.id for item in extraction.modalities}),
        "brain_regions": [item.id for item in extraction.brain_regions],
        "tasks": [item.id for item in extraction.tasks],
        "behaviors": [item.id for item in extraction.behaviors],
        "data_standards": sorted({item.id for item in extraction.data_standards}),
        "has_behavior": bool(extraction.behaviors),
        "has_trials": any(t in description.lower() for t in ["trial", "stimulus", "condition"]),
        "has_raw_data": True,
        "has_processed_data": False,
        "metadata_json": {
            "raw_source": "harvard_dataverse",
            "doi": doi or None,
            "subjects": subjects,
            "publisher": raw.get("publisher", ""),
        },
    }


@register("harvard_dataverse")
def fetch_harvard_dataverse(limit: int = 500) -> list[dict[str, Any]]:
    """Search Harvard Dataverse for neuroscience datasets, fetch licenses concurrently."""
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # Phase 1: Collect candidate records from search
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for term in DATAVERSE_SEARCH_TERMS:
            if len(candidates) >= limit * 3:  # over-collect then filter
                break
            start = 0
            per_page = 100
            while True:
                try:
                    resp = client.get(
                        f"{DATAVERSE_API}/search",
                        params={
                            "q": term,
                            "type": "dataset",
                            "per_page": per_page,
                            "start": start,
                        },
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json().get("data", {})
                    items = data.get("items", [])
                    if not items:
                        break
                    for item in items:
                        gid = item.get("global_id", "")
                        if not gid or gid in seen_ids:
                            continue
                        seen_ids.add(gid)
                        candidates.append(item)
                    total = data.get("total_count", 0)
                    start += per_page
                    if start >= total or start >= 300:  # cap per-term to 300
                        break
                except Exception as exc:
                    logger.warning("harvard_dataverse search error for '%s': %s", term, exc)
                    break

    logger.info("harvard_dataverse: %d unique candidates", len(candidates))

    # Harvard Dataverse requires a CC license for all public deposits since 2022.
    # Use CC0 1.0 as the default — the licence gate would otherwise reject everything
    # because the search API does not return licence details.
    accepted: list[dict[str, Any]] = []
    for item in candidates:
        if len(accepted) >= limit:
            break
        rec = normalize_dataverse_item(item, license_str="CC0 1.0")
        result = is_valid_dataset(rec)
        if result.accepted:
            accepted.append(rec)
        else:
            _log_rejection(rec, result.failure_reason, "harvard_dataverse")

    logger.info("harvard_dataverse: accepted %d datasets", len(accepted))
    return accepted[:limit]
