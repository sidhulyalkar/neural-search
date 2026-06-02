"""EBRAINS Knowledge Graph ingestion adapter.

EBRAINS uses openMINDS/JSON-LD metadata via the KG Core API.
API: https://search.kg.ebrains.eu/api/

Token is optional for public datasets. Set EBRAINS_TOKEN env var if needed.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.registry import register

logger = logging.getLogger(__name__)

EBRAINS_SEARCH_URL = "https://search.kg.ebrains.eu/api/groups/public/types/Dataset/instances"


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("EBRAINS_TOKEN", "")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _extract_first(fields: dict, *keys: str) -> str | None:
    for key in keys:
        v = fields.get(key)
        if isinstance(v, list) and v:
            v = v[0]
        if isinstance(v, dict):
            v = v.get("value") or v.get("name") or v.get("label")
        if v and isinstance(v, str):
            return v
    return None


def normalize_ebrains_dataset(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize an EBRAINS dataset instance."""
    source_id = str(raw.get("id") or raw.get("@id") or "")
    if "/" in source_id:
        source_id = source_id.rstrip("/").split("/")[-1]

    fields = raw.get("fields", raw)
    title = _extract_first(fields, "name", "title") or f"EBRAINS {source_id}"
    description = _extract_first(fields, "description", "abstract") or ""

    extraction = extract_dataset_labels(
        title=title,
        description=description,
        file_paths=[],
        source_metadata=raw,
        linked_paper_abstracts=[],
    )

    doi = _extract_first(fields, "doi", "identifier")
    url = f"https://search.kg.ebrains.eu/instances/{source_id}"

    return {
        "source": "ebrains",
        "source_id": source_id,
        "title": title,
        "description": description,
        "url": url,
        "license": _extract_first(fields, "license", "rights"),
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
        "metadata_json": {
            "raw_source": "ebrains",
            "doi": doi,
        },
    }


@register("ebrains")
def fetch_ebrains(limit: int = 100) -> list[dict[str, Any]]:
    """Fetch public EBRAINS datasets."""
    records: list[dict[str, Any]] = []
    start = 0
    page_size = 20

    while len(records) < limit:
        try:
            resp = httpx.get(
                EBRAINS_SEARCH_URL,
                params={"from": start, "size": page_size},
                headers=_auth_headers(),
                timeout=30,
            )
            if resp.status_code == 401:
                logger.warning(
                    "EBRAINS API returned 401 — set EBRAINS_TOKEN env var. "
                    "Register at https://ebrains.eu/register"
                )
                break
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("EBRAINS fetch error: %s", exc)
            break

        hits = data.get("data", data.get("hits", []))
        if not hits:
            break

        for item in hits:
            if len(records) >= limit:
                break
            try:
                records.append(normalize_ebrains_dataset(item))
            except Exception as exc:
                logger.debug("EBRAINS normalize error: %s", exc)

        start += page_size
        if start >= data.get("total", start + 1):
            break

    logger.info("EBRAINS: fetched %d datasets", len(records))
    return records
