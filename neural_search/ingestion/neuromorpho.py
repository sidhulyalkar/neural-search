"""NeuroMorpho.Org ingestion adapter.

NeuroMorpho is the largest online inventory of digitally reconstructed neurons,
organized by archive (publication set). Each archive corresponds to a paper or
dataset deposited by a lab; one corpus record is created per archive.

API: https://neuromorpho.org/api/
Total archives: ~1,074 covering ~297K individual neuron reconstructions.
License: CC BY 4.0 (https://neuromorpho.org/about.jsp)
"""
from __future__ import annotations

import logging
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx

from neural_search.ingestion.registry import register

logger = logging.getLogger(__name__)

NEUROMORPHO_API = "https://neuromorpho.org/api"

_BRAIN_REGION_MAP: dict[str, str] = {
    "neocortex": "cortex",
    "prefrontal": "prefrontal_cortex",
    "motor": "motor_cortex",
    "somatosensory": "somatosensory_cortex",
    "visual": "visual_cortex",
    "auditory": "auditory_cortex",
    "hippocampus": "hippocampus",
    "ca1": "hippocampus",
    "ca3": "hippocampus",
    "dentate gyrus": "hippocampus",
    "entorhinal cortex": "entorhinal_cortex",
    "cerebellum": "cerebellum",
    "olfactory bulb": "olfactory_bulb",
    "main olfactory bulb": "olfactory_bulb",
    "basal ganglia": "basal_ganglia",
    "striatum": "striatum",
    "substantia nigra": "basal_ganglia",
    "thalamus": "thalamus",
    "amygdala": "amygdala",
    "spinal cord": "spinal_cord",
    "retina": "retina",
    "brainstem": "brainstem",
    "hypothalamus": "hypothalamus",
    "frontal lobe": "frontal_cortex",
    "temporal lobe": "temporal_cortex",
    "parietal lobe": "parietal_cortex",
}

_SPECIES_MAP: dict[str, str] = {
    "mouse": "mouse",
    "rat": "rat",
    "human": "human",
    "monkey": "monkey",
    "cat": "cat",
    "drosophila": "drosophila",
    "zebrafish": "zebrafish",
    "rabbit": "rabbit",
    "guinea pig": "guinea_pig",
    "turtle": "turtle",
    "ferret": "ferret",
    "macaque monkey": "monkey",
    "rhesus monkey": "monkey",
}


def _map_region(raw: str) -> str | None:
    key = raw.lower().strip()
    for pattern, mapped in _BRAIN_REGION_MAP.items():
        if pattern in key:
            return mapped
    return None


def _sample_archive_neurons(archive: str, n: int = 5) -> list[dict]:
    """Fetch a small sample of neurons from one archive to extract metadata."""
    for attempt in range(3):
        try:
            resp = httpx.get(
                f"{NEUROMORPHO_API}/neuron/select",
                params={"q": f"archive:{archive}", "page": 0, "size": n},
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json().get("_embedded", {}).get("neuronResources", [])
        except Exception as exc:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                logger.debug("neuromorpho: failed to sample archive %s: %s", archive, exc)
    return []


def _build_archive_record(archive: str, neuron_count: int, neurons: list[dict]) -> dict[str, Any]:
    """Create a flat corpus record for one NeuroMorpho archive."""
    species_set: set[str] = set()
    regions_set: set[str] = set()
    cell_types: list[str] = []
    dois: list[str] = []

    for n in neurons:
        sp = _SPECIES_MAP.get((n.get("species") or "").lower().strip())
        if sp:
            species_set.add(sp)
        for r in n.get("brain_region") or []:
            mapped = _map_region(r)
            if mapped:
                regions_set.add(mapped)
        for ct in (n.get("cell_type") or [])[:3]:
            if ct and ct not in cell_types:
                cell_types.append(ct)
        for doi in n.get("reference_doi") or []:
            if doi and doi not in dois:
                dois.append(doi)

    doi = dois[0] if dois else None
    regions = sorted(regions_set)
    species = sorted(species_set)
    ct_str = ", ".join(cell_types[:5]) if cell_types else "neurons"
    region_str = ", ".join(r.replace("_", " ") for r in regions[:4]) if regions else "various brain regions"
    sp_str = ", ".join(species) if species else "multiple species"

    description = (
        f"NeuroMorpho archive '{archive}' containing {neuron_count:,} digitally reconstructed "
        f"{ct_str} from {sp_str} in {region_str}. "
        f"SWC-format 3D morphological reconstructions with metadata on soma, dendrites, and axons."
    )

    return {
        "source": "neuromorpho",
        "source_id": archive,
        "title": f"NeuroMorpho: {archive} ({neuron_count} neurons)",
        "description": description,
        "url": f"https://neuromorpho.org/byarchive.jsp?archiveName={urllib.parse.quote(archive)}",
        "license": "CC BY 4.0",
        "doi": doi,
        "species": species,
        "modalities": ["neuron_morphology"],
        "brain_regions": regions,
        "tasks": [],
        "behaviors": [],
        "data_standards": ["SWC"],
        "has_behavior": False,
        "has_trials": False,
        "has_raw_data": True,
        "has_processed_data": True,
        "metadata_json": {
            "raw_source": "neuromorpho",
            "archive": archive,
            "neuron_count": neuron_count,
            "cell_types": cell_types[:10],
            "doi": doi,
        },
    }


def _fetch_all_archives() -> list[tuple[str, int]]:
    """Return (archive_name, neuron_count) pairs for all NeuroMorpho archives."""
    archives: list[tuple[str, int]] = []
    page = 0
    page_size = 100

    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        while True:
            try:
                resp = client.get(
                    f"{NEUROMORPHO_API}/neuron/partition/archive",
                    params={"page": page, "size": page_size},
                    timeout=20,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("neuromorpho: archive listing failed at page %d: %s", page, exc)
                break

            fields = data.get("fields", {})
            archives.extend(fields.items())

            page_info = data.get("page", {})
            total_pages = page_info.get("totalPages", 1)
            if page >= total_pages - 1:
                break
            page += 1

    logger.info("neuromorpho: found %d archives", len(archives))
    return archives


@register("neuromorpho")
def fetch_neuromorpho(limit: int = 500) -> list[dict[str, Any]]:
    """Fetch NeuroMorpho archives as corpus records (one per archive, concurrent sampling)."""
    archives = _fetch_all_archives()

    target_archives = [(a, c) for a, c in archives if a][:limit]
    records: list[dict[str, Any]] = []

    def _process(item: tuple[str, int]) -> dict[str, Any]:
        archive, neuron_count = item
        neurons = _sample_archive_neurons(archive)
        return _build_archive_record(archive, neuron_count, neurons)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_process, item): item for item in target_archives}
        done = 0
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception as exc:
                archive = futures[future][0]
                logger.debug("neuromorpho: archive %s failed: %s", archive, exc)
            done += 1
            if done % 50 == 0:
                logger.info("neuromorpho: %d/%d records", done, len(target_archives))

    logger.info("neuromorpho: fetched %d archive records", len(records))
    return records
