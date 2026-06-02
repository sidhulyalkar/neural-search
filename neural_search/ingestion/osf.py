"""OSF (Open Science Framework) ingestion adapter — Tier 2.

OSF hosts mixed content. ALL records must pass DatasetInclusionClassifier
before being included in the corpus. Rejected records go to
data/corpus/rejected/tier2_rejected.jsonl.

API: https://api.osf.io/v2/nodes/?filter[public]=true
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.dataset_classifier import is_valid_dataset
from neural_search.ingestion.registry import register

logger = logging.getLogger(__name__)

OSF_API = "https://api.osf.io/v2"
OSF_NEURO_TAGS = [
    "neuroscience", "neuroscience data", "fmri", "eeg", "electrophysiology",
    "calcium imaging", "neuropixels", "spike sorting", "hippocampus", "cortex",
]
REJECTION_LOG = Path("data/corpus/rejected/tier2_rejected.jsonl")


def _log_rejection(record: dict, reason: str, source: str) -> None:
    REJECTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = json.dumps({"source": source, "id": record.get("source_id"), "reason": reason})
    with REJECTION_LOG.open("a") as f:
        f.write(entry + "\n")


def normalize_osf_project(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OSF project node to the flat legacy dict format."""
    source_id = str(raw.get("id", ""))
    attrs = raw.get("attributes", raw)
    title = attrs.get("title") or f"OSF {source_id}"
    description = attrs.get("description", "")
    tags = attrs.get("tags", [])
    license_raw = attrs.get("license")
    license_name = (
        license_raw.get("name", "") if isinstance(license_raw, dict)
        else str(license_raw or "")
    )

    extraction = extract_dataset_labels(
        title=title,
        description=description,
        file_paths=[],
        source_metadata={**attrs, "tags": tags},
        linked_paper_abstracts=[],
    )

    return {
        "source": "osf",
        "source_id": source_id,
        "title": title,
        "description": description,
        "url": f"https://osf.io/{source_id}/",
        "license": license_name or None,
        "keywords": tags,
        "resource_type": "dataset" if attrs.get("category") == "data" else attrs.get("category", ""),
        "species": [item.id for item in extraction.species],
        "modalities": sorted({item.id for item in extraction.modalities}),
        "brain_regions": [item.id for item in extraction.brain_regions],
        "tasks": [item.id for item in extraction.tasks],
        "behaviors": [item.id for item in extraction.behaviors],
        "data_standards": sorted({item.id for item in extraction.data_standards}),
        "has_behavior": bool(extraction.behaviors),
        "has_trials": any(t in str(description).lower() for t in ["trial", "stimulus"]),
        "has_raw_data": True,
        "has_processed_data": False,
        "metadata_json": {"raw_source": "osf", "tags": tags},
    }


@register("osf")
def fetch_osf(limit: int = 100) -> list[dict[str, Any]]:
    """Fetch public OSF project nodes that mention neuroscience keywords."""
    accepted: list[dict[str, Any]] = []
    for tag in OSF_NEURO_TAGS:
        if len(accepted) >= limit:
            break
        try:
            resp = httpx.get(
                f"{OSF_API}/nodes/",
                params={"filter[public]": "true", "filter[tags]": tag, "page[size]": 50},
                timeout=30,
            )
            resp.raise_for_status()
            for node in resp.json().get("data", []):
                rec = normalize_osf_project(node)
                result = is_valid_dataset(rec)
                if result.accepted:
                    accepted.append(rec)
                else:
                    _log_rejection(rec, result.failure_reason, "osf")
                if len(accepted) >= limit:
                    break
        except Exception as exc:
            logger.warning("OSF fetch error for tag '%s': %s", tag, exc)

    logger.info("OSF: accepted %d datasets (rejections logged)", len(accepted))
    return accepted
