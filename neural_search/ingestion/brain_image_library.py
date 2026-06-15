"""Brain Image Library ingestion adapter.

BIL is the BRAIN Initiative archive for light/optical microscopy data. The
metadata API exposes query endpoints that return BIL IDs and retrieve endpoints
that return structured metadata for each dataset.

Documentation: https://www.brainimagelibrary.org/metadataapi.html
API: https://api.brainimagelibrary.org
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.registry import register

logger = logging.getLogger(__name__)

BIL_API = "https://api.brainimagelibrary.org"
BIL_SOURCE_URL = "https://www.brainimagelibrary.org"
BIL_RETRIEVE_WORKERS = 12

BIL_QUERIES: list[tuple[str, dict[str, str]]] = [
    ("query/fulltext", {"text": "brain"}),
    ("query/dataset", {"title": "mouse"}),
    ("query/dataset", {"title": "human"}),
    ("query/dataset", {"title": "cortex"}),
    ("query/dataset", {"title": "microscopy"}),
    ("query/instrument", {"microscopetype": "Two Photon"}),
    ("query/instrument", {"microscopetype": "light sheet"}),
    ("query/specimen", {"species": "Mouse"}),
    ("query/specimen", {"species": "Human"}),
    ("query/specimen", {"preparationtype": "CLARITY"}),
    ("query/dataset", {"title": "retina"}),
    ("query/dataset", {"title": "barrel cortex"}),
    ("query/dataset", {"title": "septum"}),
    ("query/dataset", {"title": "cerebellum"}),
    ("query/dataset", {"title": "spinal cord"}),
    ("query/instrument", {"microscopetype": "confocal"}),
]

_SPECIES_MAP = {
    "mouse": "mouse",
    "mus musculus": "mouse",
    "human": "human",
    "homo sapiens": "human",
    "rat": "rat",
    "rattus norvegicus": "rat",
    "marmoset": "marmoset",
    "zebrafish": "zebrafish",
}

_BRAIN_REGION_TOKENS = {
    "cortex": "cortex",
    "visual cortex": "visual_cortex",
    "motor cortex": "motor_cortex",
    "hippocampus": "hippocampus",
    "thalamus": "thalamus",
    "hypothalamus": "hypothalamus",
    "cerebellum": "cerebellum",
    "striatum": "striatum",
    "amygdala": "amygdala",
    "brainstem": "brainstem",
    "whole brain": "whole_brain",
    "brain": "whole_brain",
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _first_mapping(value: Any) -> dict[str, Any]:
    for item in _as_list(value):
        if isinstance(item, dict):
            return item
    return {}


def _string_values(*values: Any) -> list[str]:
    strings: list[str] = []
    for value in values:
        for item in _as_list(value):
            if item is None:
                continue
            text = str(item).strip()
            if text and text.lower() not in {"none", "null", "nan"}:
                strings.append(text)
    return strings


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _query_ids(client: httpx.Client, endpoint: str, params: dict[str, str]) -> list[str]:
    response = client.get(f"{BIL_API}/{endpoint}", params=params)
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("success", "")).lower() != "true":
        logger.warning("BIL query failed: %s %s -> %s", endpoint, params, payload.get("message"))
        return []
    return [str(item) for item in payload.get("bildids", []) if item]


def _retrieve_metadata(client: httpx.Client, bildid: str) -> list[dict[str, Any]]:
    response = client.get(f"{BIL_API}/retrieve", params={"bildid": bildid})
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("success", "")).lower() != "true":
        logger.warning("BIL retrieve failed: %s -> %s", bildid, payload.get("message"))
        return []
    return [item for item in payload.get("retjson", []) if isinstance(item, dict)]


def _extract_species(specimens: list[Any]) -> list[str]:
    species: list[str] = []
    for specimen in specimens:
        if not isinstance(specimen, dict):
            continue
        raw = str(specimen.get("species") or "").strip()
        mapped = _SPECIES_MAP.get(raw.lower())
        if mapped:
            species.append(mapped)
        elif raw:
            species.append(raw.lower().replace(" ", "_"))
    return _dedupe(species)


def _extract_brain_regions(record: dict[str, Any]) -> list[str]:
    text_parts: list[str] = []
    for section in ("Specimen", "Dataset", "Image", "Instrument", "Submission", "Contributors"):
        for item in _as_list(record.get(section)):
            if isinstance(item, dict):
                text_parts.extend(str(value) for value in item.values() if value)
    haystack = " ".join(text_parts)
    haystack_lower = haystack.lower().replace("_", " ")
    token_regions = [region for token, region in _BRAIN_REGION_TOKENS.items() if token in haystack_lower]
    try:
        from neural_search.ontology import (
            match_brain_regions,  # lazy import to avoid circular imports
        )

        matches = match_brain_regions(haystack)
        return _dedupe([m.id for m in matches] + token_regions)
    except Exception:
        return _dedupe(token_regions)


def _extract_modalities(dataset: dict[str, Any], instruments: list[Any], images: list[Any]) -> list[str]:
    modalities = ["microscopy"]
    general = str(dataset.get("generalmodality") or "").lower()
    technique = str(dataset.get("technique") or "").lower()
    if "population imaging" in general:
        modalities.append("population_imaging")
    if "two" in technique and "photon" in technique:
        modalities.append("two_photon")
    if "light" in technique and "sheet" in technique:
        modalities.append("light_sheet")
    if "morpholog" in technique or "neurite" in str(images).lower():
        modalities.append("morphology")
    for instrument in instruments:
        if not isinstance(instrument, dict):
            continue
        microscope = str(instrument.get("microscopetype") or "").lower()
        if "two" in microscope and "photon" in microscope:
            modalities.append("two_photon")
        if "light" in microscope and "sheet" in microscope:
            modalities.append("light_sheet")
        if "confocal" in microscope:
            modalities.append("confocal_microscopy")
    return _dedupe(modalities)


def _extract_formats(images: list[Any], assets: list[Any]) -> list[str]:
    values: list[str] = []
    for image in images:
        if isinstance(image, dict):
            values.extend(_string_values(image.get("files"), image.get("dimensionorder")))
    for asset in assets:
        if isinstance(asset, dict) and asset.get("manifestfile"):
            values.append("manifest_json")
    return _dedupe(values)


def _child_assets(assets: list[Any], dataset: dict[str, Any]) -> list[dict[str, Any]]:
    child_assets: list[dict[str, Any]] = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        manifest = asset.get("manifestfile")
        if manifest:
            child_assets.append(
                {
                    "record_type": "child_asset",
                    "asset_type": "manifest",
                    "path": str(manifest),
                    "url": str(manifest),
                    "file_format": "json",
                    "metadata_json": {
                        "bildid": asset.get("bildid"),
                        "bildoi": asset.get("bildoi"),
                    },
                }
            )
    directory = dataset.get("bildirectory")
    if directory:
        child_assets.append(
            {
                "record_type": "child_asset",
                "asset_type": "bil_directory",
                "path": str(directory),
                "url": str(directory),
                "metadata_json": {"access_method": "BIL archive directory"},
            }
        )
    return child_assets


def normalize_bil_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize one BIL retrieve metadata object to a flat corpus record."""
    submission = _first_mapping(raw.get("Submission"))
    dataset = _first_mapping(raw.get("Dataset"))
    specimens = _as_list(raw.get("Specimen"))
    instruments = _as_list(raw.get("Instrument"))
    images = _as_list(raw.get("Image"))
    assets = _as_list(raw.get("Assets"))
    first_asset = _first_mapping(assets)

    source_id = str(
        first_asset.get("bildid")
        or submission.get("doi")
        or submission.get("submission_uuid")
        or dataset.get("doi")
    ).strip()
    title = str(dataset.get("title") or source_id or "Brain Image Library dataset").strip()
    description = " ".join(
        _string_values(dataset.get("abstract"), dataset.get("methods"), dataset.get("technicalinfo"))
    )
    doi = dataset.get("doi") or submission.get("doi") or first_asset.get("bildoi")
    url = str(doi or f"{BIL_API}/retrieve?bildid={source_id}")

    species = _extract_species(specimens)
    modalities = _extract_modalities(dataset, instruments, images)
    brain_regions = _extract_brain_regions(raw)
    formats = _extract_formats(images, assets)
    child_assets = _child_assets(assets, dataset)

    extraction = extract_dataset_labels(
        title=title,
        description=description,
        file_paths=[asset.get("path", "") for asset in child_assets],
        source_metadata=raw,
        linked_paper_abstracts=[],
    )

    standards = sorted({item.id for item in extraction.data_standards} | set(formats))
    if "manifest_json" in formats:
        standards.append("BIL manifest")

    return {
        "source": "brain_image_library",
        "source_id": source_id,
        "source_type": "canonical_dataset",
        "record_type": "dataset_collection",
        "title": title,
        "description": description,
        "url": url,
        "access_url": url,
        "license": dataset.get("rightsidentifier") or dataset.get("rights"),
        "doi": doi,
        "species": species or [item.id for item in extraction.species],
        "modalities": sorted(set(modalities) | {item.id for item in extraction.modalities}),
        "brain_regions": brain_regions or [item.id for item in extraction.brain_regions],
        "tasks": [item.id for item in extraction.tasks],
        "behaviors": [item.id for item in extraction.behaviors],
        "data_standards": _dedupe(standards),
        "formats": formats,
        "analysis_affordances": [
            "brain_image_analysis",
            "microscopy_image_registration",
            "morphology_analysis",
            "atlas_mapping",
        ],
        "assets": child_assets,
        "has_behavior": False,
        "has_trials": False,
        "has_raw_data": True,
        "has_processed_data": True,
        "metadata_json": {
            "raw_source": "brain_image_library",
            "submission_uuid": submission.get("submission_uuid"),
            "bildirectory": dataset.get("bildirectory"),
            "dataset_size": dataset.get("dataset_size"),
            "rightsuri": dataset.get("rightsuri"),
            "contributors": raw.get("Contributors", []),
            "instrument": instruments,
            "image": images,
            "child_assets": child_assets,
        },
    }


def _passes_quality_gate(record: dict[str, Any]) -> bool:
    required = [
        record.get("source_id"),
        record.get("title"),
        record.get("url") or record.get("access_url"),
        record.get("modalities"),
    ]
    return all(bool(value) for value in required)


@register("brain_image_library")
def fetch_brain_image_library(limit: int = 500) -> list[dict[str, Any]]:
    """Harvest BIL metadata records from API query/retrieve endpoints."""
    accepted: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    candidate_ids: list[str] = []

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for endpoint, params in BIL_QUERIES:
            if len(candidate_ids) >= limit:
                break
            try:
                ids = _query_ids(client, endpoint, params)
            except Exception as exc:
                logger.warning("BIL query error for %s %s: %s", endpoint, params, exc)
                continue
            for bildid in ids:
                if len(candidate_ids) >= limit:
                    break
                if bildid in seen_query_ids:
                    continue
                seen_query_ids.add(bildid)
                candidate_ids.append(bildid)

    def retrieve_one(bildid: str) -> list[dict[str, Any]]:
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                return _retrieve_metadata(client, bildid)
        except Exception as exc:
            logger.warning("BIL retrieve error for %s: %s", bildid, exc)
            return []

    seen_record_ids: set[str] = set()
    workers = min(BIL_RETRIEVE_WORKERS, max(1, len(candidate_ids)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for records in executor.map(retrieve_one, candidate_ids):
            for raw in records:
                rec = normalize_bil_record(raw)
                source_id = str(rec.get("source_id") or "")
                if not source_id or source_id in seen_record_ids:
                    continue
                seen_record_ids.add(source_id)
                if _passes_quality_gate(rec):
                    accepted.append(rec)
                if len(accepted) >= limit:
                    break
            if len(accepted) >= limit:
                break

    logger.info("brain_image_library: accepted %d datasets", len(accepted))
    return accepted
