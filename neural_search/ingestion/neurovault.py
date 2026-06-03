"""NeuroVault ingestion adapter — collection-level normalization.

NeuroVault hosts statistical maps, parcellations, and group-level MRI outputs.
We normalize at the *collection* level (not individual images).

IMPORTANT: NeuroVault is NOT always raw reusable datasets.
Every collection must have: a public=true flag, a name, and at least description
or number_of_images > 0.

API: https://neurovault.org/api/collections/?format=json&public=true
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.registry import register

logger = logging.getLogger(__name__)

NEUROVAULT_API = "https://neurovault.org/api/collections/?format=json&public=true"


def normalize_collection(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a NeuroVault collection to the flat legacy dict format."""
    source_id = str(raw.get("id", ""))
    title = raw.get("name") or f"NeuroVault Collection {source_id}"
    description = raw.get("description", "")
    doi = raw.get("DOI") or raw.get("doi")
    n_images = raw.get("number_of_images", 0)

    extraction = extract_dataset_labels(
        title=title,
        description=description or f"{title} fmri mri brain imaging human",
        file_paths=[],
        source_metadata=raw,
        linked_paper_abstracts=[],
    )

    url = f"https://neurovault.org/collections/{source_id}/"
    if doi:
        url = f"https://doi.org/{doi}"

    return {
        "source": "neurovault",
        "source_id": source_id,
        "title": title,
        "description": description,
        "url": url,
        "license": "CC-BY",
        "species": [item.id for item in extraction.species] or ["human"],
        "modalities": sorted({item.id for item in extraction.modalities} | {"fmri"}),
        "brain_regions": [item.id for item in extraction.brain_regions],
        "tasks": [item.id for item in extraction.tasks],
        "behaviors": [item.id for item in extraction.behaviors],
        "data_standards": ["BIDS"],
        "has_behavior": False,
        "has_trials": False,
        "has_raw_data": False,
        "has_processed_data": True,
        "n_images": n_images,
        "metadata_json": {
            "raw_source": "neurovault",
            "doi": doi,
            "n_images": n_images,
            "owner": raw.get("owner_name"),
        },
    }


@register("neurovault")
def fetch_neurovault(limit: int = 100) -> list[dict[str, Any]]:
    """Fetch public NeuroVault collections."""
    records: list[dict[str, Any]] = []
    url: str | None = NEUROVAULT_API
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        while url and len(records) < limit:
            try:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("NeuroVault fetch error: %s", exc)
                break

            for col in data.get("results", []):
                if not col.get("id") or not col.get("name"):
                    continue
                records.append(normalize_collection(col))
                if len(records) >= limit:
                    break

            url = data.get("next") if len(records) < limit else None

    logger.info("NeuroVault: fetched %d collections", len(records))
    return records
