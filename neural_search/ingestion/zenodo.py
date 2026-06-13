"""zenodo ingestion adapter — Tier 2.

zenodo hosts open research outputs. ALL records must pass DatasetInclusionClassifier.
API: https://zenodo.org/api/records?type=dataset&q=neuroscience
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.dataset_classifier import is_valid_dataset
from neural_search.ingestion.registry import register

logger = logging.getLogger(__name__)

ZENODO_API = "https://zenodo.org/api/records"
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
    "single neuron firing rate", "local field potential LFP", "multiunit activity brain",
    "neural population dynamics", "brain atlas connectome", "synaptic plasticity",
    "seizure epilepsy EEG", "deep brain stimulation", "brain computer interface BCI",
    "spinal cord neural", "peripheral nervous system", "autonomic nervous system",
    "retina photoreceptor", "lateral geniculate nucleus", "superior colliculus",
    "dopamine serotonin striatum", "acetylcholine cholinergic", "GABA inhibitory",
    "glutamate excitatory", "action potential spike train", "membrane potential patch",
    "calcium transient fluorescence", "GCaMP indicator neural", "voltage imaging",
    "widefield mesoscale imaging", "fiber photometry in vivo", "miniscope calcium",
    "spatial navigation maze", "head direction cells", "grid cells entorhinal",
    "theta oscillation hippocampus", "sharp wave ripple", "gamma oscillation cortex",
    "sensory perception threshold", "attention neural", "task performance neural",
]


def normalize_zenodo_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a zenodo record."""
    source_id = str(raw.get("id", ""))
    meta = raw.get("metadata", raw)
    title = meta.get("title") or f"zenodo {source_id}"
    description = meta.get("description") or ""
    doi = raw.get("doi") or meta.get("doi")
    license_raw = meta.get("license")
    license_id = (
        license_raw.get("id", "") if isinstance(license_raw, dict)
        else str(license_raw or "")
    )
    keywords = meta.get("keywords", [])
    rtype_raw = meta.get("resource_type")
    resource_type = (
        rtype_raw.get("type", "dataset") if isinstance(rtype_raw, dict) else "dataset"
    )

    extraction = extract_dataset_labels(
        title=title,
        description=f"{description} {' '.join(str(k) for k in keywords)}",
        file_paths=[],
        source_metadata=meta,
        linked_paper_abstracts=[],
    )

    return {
        "source": "zenodo",
        "source_id": source_id,
        "title": title,
        "description": description,
        "url": (raw.get("links") or {}).get("html") or f"https://zenodo.org/records/{source_id}",
        "license": license_id or None,
        "doi": doi,
        "keywords": [str(k) for k in keywords],
        "resource_type": resource_type,
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
        "metadata_json": {"raw_source": "zenodo", "doi": doi},
    }


@register("zenodo")
def fetch_zenodo(limit: int = 500) -> list[dict[str, Any]]:
    """Search Zenodo for neuroscience datasets across all queries with pagination."""
    accepted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for query in ZENODO_QUERIES:
            if len(accepted) >= limit:
                break
            page = 1
            per_page = 100
            while len(accepted) < limit:
                try:
                    resp = client.get(
                        ZENODO_API,
                        params={"q": query, "type": "dataset", "size": per_page, "page": page, "sort": "mostrecent"},
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
