"""DANDI live ingestion connector with safe dry-run mode."""

from __future__ import annotations

import argparse
import json
from typing import Any

import httpx

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.demo_seed import DEFAULT_DATABASE_URL
from neural_search.ingestion.live import (
    print_cli_error,
    print_normalized_records,
    save_dataset_records,
    save_raw_response,
)

DANDI_API_URL = "https://api.dandiarchive.org/api"


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _asset_summary(dandiset: dict[str, Any]) -> dict[str, Any]:
    assets = dandiset.get("assetsSummary") or dandiset.get("assets_summary") or {}
    if isinstance(assets, dict):
        return assets
    return {}


def normalize_dandiset(raw: dict[str, Any]) -> dict[str, Any]:
    version = raw.get("most_recent_published_version") or raw.get("draft_version") or raw
    metadata = version.get("metadata") or raw.get("metadata") or {}
    assets = _asset_summary(raw) or _asset_summary(version)
    source_id = str(raw.get("identifier") or raw.get("id") or metadata.get("identifier"))
    title = (
        metadata.get("name")
        or version.get("name")
        or raw.get("name")
        or f"DANDI {source_id}"
    )
    description = metadata.get("description") or version.get("description") or raw.get("description")
    text = " ".join(str(part) for part in [title, description, metadata, assets])
    extraction = extract_dataset_labels(
        title=title,
        description=description,
        file_paths=[],
        source_metadata={**metadata, "assets": assets},
        linked_paper_abstracts=[],
    )
    return {
        "source": "dandi",
        "source_id": source_id,
        "title": title,
        "description": description,
        "url": raw.get("url") or f"https://dandiarchive.org/dandiset/{source_id}",
        "license": metadata.get("license") or raw.get("license"),
        "species": [item.id for item in extraction.species],
        "modalities": sorted({item.id for item in extraction.modalities}),
        "brain_regions": [item.id for item in extraction.brain_regions],
        "tasks": [item.id for item in extraction.tasks],
        "behaviors": [item.id for item in extraction.behaviors],
        "data_standards": sorted({item.id for item in extraction.data_standards} | {"NWB"}),
        "has_behavior": bool(extraction.behaviors) or "behavior" in text.casefold(),
        "has_trials": any(term in text.casefold() for term in ["trial", "trials", "event"]),
        "has_raw_data": True,
        "has_processed_data": any(term in text.casefold() for term in ["processed", "derived"]),
        "metadata_json": {
            "raw_source": "dandi",
            "version": version.get("version"),
            "asset_summary": assets,
            "measurement_technique": _as_list(metadata.get("measurementTechnique")),
        },
    }


def fetch_dandi(query: str, limit: int) -> dict[str, Any]:
    params = {"search": query, "page_size": limit}
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(f"{DANDI_API_URL}/dandisets/", params=params)
        response.raise_for_status()
        return response.json()


def records_from_response(payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    results = payload.get("results", payload if isinstance(payload, list) else [])
    return [normalize_dandiset(item) for item in results[:limit]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.ingestion.dandi")
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    args = parser.parse_args(argv)

    try:
        payload = fetch_dandi(args.query, args.limit)
        if args.save or args.save_raw:
            raw_path = save_raw_response("dandi", args.query, payload)
            print(json.dumps({"raw_saved": str(raw_path)}, indent=2))
        records = records_from_response(payload, args.limit)
        print_normalized_records(records)
        if args.dry_run or not args.save:
            return 0
        summary = save_dataset_records(records, args.database_url, args.force)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print_cli_error("dandi", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
