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
from neural_search.ingestion.dataset_classifier import is_valid_dataset
from neural_search.ingestion.registry import register

logger = logging.getLogger(__name__)

GIN_API = "https://gin.g-node.org/api/v1"
GIN_SEARCH_TERMS = [
    "neuropixels", "calcium imaging", "ephys", "fmri", "eeg", "ecog",
    "mouse", "human", "NWB", "BIDS", "spike sorting", "behavior",
    "electrophysiology", "two-photon", "optogenetics", "patch clamp",
    "hippocampus", "prefrontal cortex", "visual cortex", "motor cortex",
    "sleep", "decision making", "reward", "working memory", "primate",
    "rat", "zebrafish", "human intracranial", "fiber photometry",
    "LFP", "local field potential", "single unit", "multi-unit",
    "place cells", "grid cells", "theta", "gamma oscillation",
    "auditory cortex", "somatosensory", "cerebellum", "striatum",
    "amygdala", "thalamus", "brainstem", "basal ganglia",
    "spatial navigation", "fear conditioning", "attention neural",
    "neural decoding", "population activity", "trial", "stimulus response",
    "connectome", "synapse", "axon", "dendrite morphology",
    "meg brain", "intracranial eeg", "seizure detection",
    "open ephys", "kilosort", "suite2p", "caiman",
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
    ) or ("CC-BY" if raw.get("private") is False else None)

    return {
        "source": "gin",
        "source_id": source_id,
        "identifier": full_name,
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
def fetch_gin(limit: int = 700) -> list[dict[str, Any]]:
    """Search GIN for neuroscience datasets across all search terms with pagination."""
    accepted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    page_size = 50

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for term in GIN_SEARCH_TERMS:
            if len(accepted) >= limit:
                break
            page = 1
            while len(accepted) < limit:
                try:
                    resp = client.get(
                        f"{GIN_API}/repos/search",
                        params={"q": term, "limit": page_size, "page": page},
                    )
                    resp.raise_for_status()
                    repos = resp.json().get("data", [])
                    if not repos:
                        break
                    for repo in repos:
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
                    if len(repos) < page_size:
                        break
                    page += 1
                except Exception as exc:
                    logger.warning("GIN fetch error for '%s' page %d: %s", term, page, exc)
                    break

    logger.info("gin: accepted %d datasets", len(accepted))
    return accepted
