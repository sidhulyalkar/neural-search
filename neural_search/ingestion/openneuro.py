"""OpenNeuro connector for ingesting BIDS datasets."""

from __future__ import annotations

from typing import Any

import httpx

OPENNEURO_API_URL = "https://openneuro.org/crn/graphql"


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
    graphql_query = """
    query SearchDatasets($query: String, $first: Int) {
        datasets(first: $first, query: $query) {
            edges {
                node {
                    id
                    name
                    description
                    created
                    public
                    permissions {
                        userPermissions {
                            access
                        }
                    }
                    latestSnapshot {
                        tag
                        created
                        size
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

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            OPENNEURO_API_URL,
            json={
                "query": graphql_query,
                "variables": {"query": query, "first": limit},
            },
        )
        response.raise_for_status()
        data = response.json()

    results = []
    edges = data.get("data", {}).get("datasets", {}).get("edges", [])

    for edge in edges:
        node = edge.get("node", {})
        snapshot = node.get("latestSnapshot", {}) or {}
        summary = snapshot.get("summary", {}) or {}

        results.append({
            "id": f"openneuro:{node['id']}",
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
            },
        })

    return results


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
        "id": f"openneuro:{node['id']}",
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
