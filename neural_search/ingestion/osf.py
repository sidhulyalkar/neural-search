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
    "neuroscience", "electrophysiology", "fmri", "eeg", "calcium imaging",
    "neuropixels", "spike sorting", "hippocampus", "prefrontal cortex",
    "neural data", "brain imaging", "two-photon imaging", "patch clamp",
    "optogenetics", "NWB", "BIDS", "single unit recording", "LFP",
    "place cells", "decision making", "working memory", "fear conditioning",
    "motor cortex", "visual cortex", "auditory cortex", "cerebellum",
    "spatial navigation", "reward", "dopamine", "ecog", "meg brain",
    "neural oscillation", "brain connectivity", "deep brain stimulation",
    "retina electrophysiology", "spinal cord recording", "attention task",
    "perceptual decision", "sensorimotor", "local field potential",
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
    # License lives in embeds (requires embed[]=license in request), not attributes
    embed_lic = (
        raw.get("embeds", {})
        .get("license", {})
        .get("data", {})
        .get("attributes", {})
    )
    license_name = embed_lic.get("name", "") if embed_lic else ""

    extraction = extract_dataset_labels(
        title=title,
        description=description,
        file_paths=[],
        source_metadata={**attrs, "tags": tags},
        linked_paper_abstracts=[],
    )

    # OSF URLs are persistent identifiers even without a DOI
    osf_url = f"https://osf.io/{source_id}/"

    return {
        "source": "osf",
        "source_id": source_id,
        "title": title,
        "description": description,
        "url": osf_url,
        "doi": osf_url,  # OSF URL serves as persistent identifier for classifier
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
def fetch_osf(limit: int = 200) -> list[dict[str, Any]]:
    """Fetch public OSF project nodes that mention neuroscience keywords, with pagination."""
    accepted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for tag in OSF_NEURO_TAGS:
            if len(accepted) >= limit:
                break
            next_url: str | None = f"{OSF_API}/nodes/"
            params: dict = {
                "filter[public]": "true",
                "filter[tags]": tag,
                "page[size]": 50,
                "embed[]": "license",
            }
            pages_this_tag = 0
            while next_url and len(accepted) < limit and pages_this_tag < 4:
                try:
                    resp = client.get(next_url, params=params, timeout=30)
                    resp.raise_for_status()
                    payload = resp.json()
                    params = {}  # subsequent pages use full URL from links.next
                    pages_this_tag += 1
                    for node in payload.get("data", []):
                        sid = str(node.get("id", ""))
                        if not sid or sid in seen_ids:
                            continue
                        seen_ids.add(sid)
                        rec = normalize_osf_project(node)
                        result = is_valid_dataset(rec)
                        if result.accepted:
                            accepted.append(rec)
                        else:
                            _log_rejection(rec, result.failure_reason, "osf")
                        if len(accepted) >= limit:
                            break
                    next_url = (payload.get("links") or {}).get("next")
                except Exception as exc:
                    logger.warning("OSF fetch error for tag '%s': %s", tag, exc)
                    break

    logger.info("OSF: accepted %d datasets (rejections logged)", len(accepted))
    return accepted
