"""OpenNeuro connector for ingesting BIDS datasets."""

from __future__ import annotations

import argparse
import json
import logging
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
from neural_search.ingestion.registry import register
from neural_search.normalized import (
    evidence_label_from_extraction,
    stable_normalized_id,
)
from neural_search.schemas import NormalizedDatasetRecord, UsabilityFlags

OPENNEURO_API_URL = "https://openneuro.org/crn/graphql"

logger = logging.getLogger(__name__)


def _paginated_query() -> str:
    return """
    query SearchDatasets($first: Int, $after: String) {
        datasets(first: $first, after: $after, filterBy: {public: true}) {
            edges {
                cursor
                node {
                    id
                    name
                    created
                    public
                    latestSnapshot {
                        tag
                        created
                        size
                        readme
                        summary {
                            subjects
                            tasks
                            modalities
                        }
                    }
                }
            }
            pageInfo {
                hasNextPage
                endCursor
            }
        }
    }
    """


def _search_query() -> str:
    return """
    query SearchDatasets($modality: String, $first: Int) {
        datasets(first: $first, modality: $modality, filterBy: {public: true}) {
            edges {
                node {
                    id
                    name
                    created
                    public
                    latestSnapshot {
                        tag
                        created
                        size
                        readme
                        summary {
                            subjects
                            tasks
                            modalities
                        }
                    }
                }
            }
        }
    }
    """


def fetch_openneuro(modality: str | None, limit: int) -> dict[str, Any]:
    """Fetch datasets from OpenNeuro, optionally filtered by modality.

    Args:
        modality: BIDS modality to filter by (e.g., 'eeg', 'func', 'anat', 'meg', 'ieeg')
                  Pass None to get all public datasets.
        limit: Maximum number of datasets to fetch.
    """
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.post(
            OPENNEURO_API_URL,
            json={
                "query": _search_query(),
                "variables": {"modality": modality, "first": limit},
            },
        )
        response.raise_for_status()
        data = response.json()
    if data.get("errors"):
        raise RuntimeError(json.dumps(data["errors"], indent=2))
    return data


def fetch_all_openneuro(
    *,
    page_size: int = 100,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch all public OpenNeuro datasets using GraphQL cursor pagination."""
    all_records: list[dict[str, Any]] = []
    after: str | None = None

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        while True:
            resp = client.post(
                OPENNEURO_API_URL,
                json={
                    "query": _paginated_query(),
                    "variables": {"first": page_size, "after": after},
                },
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("errors"):
                # OpenNeuro returns partial results alongside errors (e.g. a single
                # broken latestSnapshot). Log and continue; don't abort the crawl.
                logger.warning("OpenNeuro GraphQL partial error: %s", data["errors"][:2])

            datasets_data = data.get("data", {}).get("datasets", {})
            edges = datasets_data.get("edges", [])
            page_info = datasets_data.get("pageInfo", {})

            for edge in edges:
                edge = edge or {}
                node = edge.get("node") or {}
                if node.get("id"):
                    all_records.append(normalize_openneuro_dataset(node))
                    if max_records is not None and len(all_records) >= max_records:
                        return all_records

            logger.info("OpenNeuro harvest: %d records, hasNextPage=%s", len(all_records), page_info.get("hasNextPage"))

            if not page_info.get("hasNextPage"):
                break
            after = page_info.get("endCursor")

    return all_records


async def search_datasets(
    query: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Search OpenNeuro for datasets.

    Args:
        query: Search query string.
        limit: Maximum number of results.

    Returns:
        List of dataset records.
    """
    data = fetch_openneuro(query, limit)
    return records_from_response(data, limit)


def normalize_openneuro_dataset(node: dict[str, Any]) -> dict[str, Any]:
    snapshot = node.get("latestSnapshot", {}) or {}
    summary = snapshot.get("summary", {}) or {}
    title = node.get("name") or node.get("id")
    description = node.get("description") or snapshot.get("readme")
    modalities = [str(value).casefold() for value in summary.get("modalities", []) or []]
    text = " ".join(str(part) for part in [title, description, summary, modalities])
    extraction = extract_dataset_labels(
        title=title,
        description=description,
        file_paths=[],
        source_metadata={"summary": summary, "modalities": modalities, "standard": "BIDS"},
        linked_paper_abstracts=[],
    )
    tasks = sorted({*summary.get("tasks", []), *(item.id for item in extraction.tasks)})
    return {
        "source": "openneuro",
        "source_id": node["id"],
        "title": title,
        "description": description,
        "url": f"https://openneuro.org/datasets/{node['id']}",
        "license": None,
        "species": [item.id for item in extraction.species],
        "modalities": sorted({*modalities, *(item.id for item in extraction.modalities)}),
        "brain_regions": [item.id for item in extraction.brain_regions],
        "tasks": tasks,
        "behaviors": [item.id for item in extraction.behaviors],
        "data_standards": ["BIDS"],
        "has_behavior": bool(extraction.behaviors) or "events.tsv" in text.casefold(),
        "has_trials": any(term in text.casefold() for term in ["trial", "events.tsv", "task"]),
        "has_raw_data": True,
        "has_processed_data": "derivative" in text.casefold(),
        "metadata_json": {
            "raw_source": "openneuro",
            "subjects": summary.get("subjects"),
            "snapshot_tag": snapshot.get("tag"),
            "size_bytes": snapshot.get("size"),
            "created": node.get("created"),
        },
    }


def normalize_openneuro_record(
    node: dict[str, Any],
    raw_payload_path: str | None = None,
) -> NormalizedDatasetRecord:
    """Normalize a raw OpenNeuro node into the v0.3 provenance-aware schema."""

    legacy = normalize_openneuro_dataset(node)
    metadata = legacy.get("metadata_json", {})
    extraction = extract_dataset_labels(
        title=legacy.get("title"),
        description=legacy.get("description"),
        file_paths=[],
        source_metadata={**metadata, "standard": "BIDS"},
        linked_paper_abstracts=[],
    )
    source_value = " ".join(
        str(part) for part in [legacy.get("title"), legacy.get("description"), metadata] if part
    )
    return NormalizedDatasetRecord(
        dataset_id=stable_normalized_id("dataset", "openneuro", legacy["source_id"]),
        source="openneuro",
        source_id=legacy["source_id"],
        title=legacy["title"],
        description=legacy.get("description"),
        url=legacy.get("url"),
        raw_payload_path=raw_payload_path,
        species=[
            evidence_label_from_extraction(
                label, "species", source_field="summary", source_value=source_value
            )
            for label in extraction.species
        ],
        modalities=[
            evidence_label_from_extraction(
                label, "modality", source_field="summary", source_value=source_value
            )
            for label in extraction.modalities
        ],
        brain_regions=[
            evidence_label_from_extraction(
                label, "brain_region", source_field="summary", source_value=source_value
            )
            for label in extraction.brain_regions
        ],
        tasks=[
            evidence_label_from_extraction(
                label, "task", source_field="summary", source_value=source_value
            )
            for label in extraction.tasks
        ],
        behavioral_events=[
            evidence_label_from_extraction(
                label, "behavioral_event", source_field="summary", source_value=source_value
            )
            for label in extraction.behaviors
        ],
        data_standards=[
            evidence_label_from_extraction(
                label, "data_standard", source_field="summary", source_value=source_value
            )
            for label in extraction.data_standards
        ],
        usability_flags=UsabilityFlags(
            has_trials=legacy.get("has_trials"),
            has_behavior=legacy.get("has_behavior"),
            has_neural_data=bool(legacy.get("modalities")),
            has_raw_data=legacy.get("has_raw_data"),
            has_processed_data=legacy.get("has_processed_data"),
            has_standard_format=True,
        ),
        missing_fields=extraction.missing_fields,
    )


def records_from_response(data: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    edges = data.get("data", {}).get("datasets", {}).get("edges", [])

    for edge in edges:
        node = edge.get("node", {})
        if node.get("id"):
            results.append(normalize_openneuro_dataset(node))

    return results[:limit]


@register("openneuro")
def fetch_openneuro_records(limit: int = 2000) -> list[dict[str, Any]]:
    """Registry adapter for full OpenNeuro cursor pagination."""
    return fetch_all_openneuro(max_records=limit)


async def get_dataset(dataset_id: str) -> dict[str, Any] | None:
    """
    Fetch a specific dataset by ID.

    Args:
        dataset_id: OpenNeuro dataset ID (e.g., 'ds000001').

    Returns:
        Dataset record or None.
    """
    graphql_query = """
    query GetDataset($id: ID!) {
        dataset(id: $id) {
            id
            name
            description
            created
            public
            latestSnapshot {
                tag
                created
                size
                readme
                summary {
                    subjects
                    tasks
                    modalities
                }
            }
        }
    }
    """

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            OPENNEURO_API_URL,
            json={
                "query": graphql_query,
                "variables": {"id": dataset_id},
            },
        )
        response.raise_for_status()
        data = response.json()

    node = data.get("data", {}).get("dataset")
    if not node:
        return None

    snapshot = node.get("latestSnapshot", {}) or {}
    summary = snapshot.get("summary", {}) or {}

    return {
        "source": "openneuro",
        "source_id": node["id"],
        "title": node.get("name", node["id"]),
        "description": node.get("description"),
        "url": f"https://openneuro.org/datasets/{node['id']}",
        "data_standards": ["BIDS"],
        "modalities": summary.get("modalities", []),
        "tasks": summary.get("tasks", []),
        "metadata_json": {
            "subjects": summary.get("subjects"),
            "snapshot_tag": snapshot.get("tag"),
            "size_bytes": snapshot.get("size"),
            "readme": snapshot.get("readme"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.ingestion.openneuro")
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    args = parser.parse_args(argv)

    try:
        payload = fetch_openneuro(args.query, args.limit)
        if args.save or args.save_raw:
            raw_path = save_raw_response("openneuro", args.query, payload)
            print(json.dumps({"raw_saved": str(raw_path)}, indent=2))
        records = records_from_response(payload, args.limit)
        print_normalized_records(records)
        if args.dry_run or not args.save:
            return 0
        summary = save_dataset_records(records, args.database_url, args.force)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print_cli_error("openneuro", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
