"""OpenAlex connector for linking papers to datasets."""

from __future__ import annotations

import argparse
import json
from typing import Any

import httpx

from neural_search.ingestion.demo_seed import DEFAULT_DATABASE_URL
from neural_search.ingestion.live import (
    print_cli_error,
    print_normalized_records,
    save_paper_records,
    save_raw_response,
)

OPENALEX_API_URL = "https://api.openalex.org"


def fetch_openalex(query: str, limit: int) -> dict[str, Any]:
    params = {
        "search": query,
        "per_page": limit,
        "mailto": "neuralsearch@example.com",
    }
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(f"{OPENALEX_API_URL}/works", params=params)
        response.raise_for_status()
        return response.json()


async def search_works(
    query: str,
    limit: int = 25,
    filter_concepts: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Search OpenAlex for academic works (papers).

    Args:
        query: Search query string.
        limit: Maximum number of results.
        filter_concepts: Optional concept IDs to filter by.

    Returns:
        List of paper records.
    """
    params = {
        "search": query,
        "per_page": limit,
        "mailto": "neuralsearch@example.com",  # Polite pool
    }

    if filter_concepts:
        params["filter"] = ",".join(f"concepts.id:{c}" for c in filter_concepts)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{OPENALEX_API_URL}/works",
            params=params,
        )
        response.raise_for_status()
        data = response.json()

    return records_from_response(data, limit)


def _get_abstract(work: dict[str, Any]) -> str | None:
    """Reconstruct abstract from inverted index."""
    abstract_index = work.get("abstract_inverted_index")
    if not abstract_index:
        return None

    # OpenAlex stores abstracts as inverted index: {"word": [positions]}
    words: list[tuple[int, str]] = []
    for word, positions in abstract_index.items():
        for pos in positions:
            words.append((pos, word))

    words.sort(key=lambda x: x[0])
    return " ".join(word for _, word in words)


async def get_work(work_id: str) -> dict[str, Any] | None:
    """
    Fetch a specific work by ID or DOI.

    Args:
        work_id: OpenAlex work ID or DOI.

    Returns:
        Paper record or None.
    """
    # Handle DOI format
    if work_id.startswith("10."):
        work_id = f"https://doi.org/{work_id}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{OPENALEX_API_URL}/works/{work_id}",
            params={"mailto": "neuralsearch@example.com"},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        work = response.json()

    return normalize_work(work)


def normalize_work(work: dict[str, Any]) -> dict[str, Any]:
    authors = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author", {})
        name = author.get("display_name")
        if name:
            authors.append({"name": name, "orcid": author.get("orcid")})

    concepts = [c.get("display_name") for c in work.get("concepts", [])]

    openalex_id = work.get("id")
    source_id = str(openalex_id or "").replace("https://openalex.org/", "")
    return {
        "id": source_id,
        "source": "openalex",
        "source_id": source_id,
        "openalex_id": openalex_id,
        "doi": work.get("doi"),
        "title": work.get("title", ""),
        "abstract": _get_abstract(work),
        "publication_year": work.get("publication_year"),
        "authors_json": authors,
        "url": work.get("doi") or work.get("id"),
        "concepts": concepts[:10],
        "citation_count": work.get("cited_by_count", 0),
        "metadata_json": {
            "type": work.get("type"),
            "is_oa": work.get("open_access", {}).get("is_oa"),
            "venue": work.get("primary_location", {}).get("source", {}).get("display_name"),
        },
    }


def records_from_response(data: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    return [normalize_work(work) for work in data.get("results", [])[:limit]]


async def search_papers_for_dataset(
    dataset_title: str,
    dataset_doi: str | None = None,
    author_names: list[str] | None = None,
    task_terms: list[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Search for papers that might be related to a dataset.

    Combines multiple search strategies:
    1. Search by dataset DOI
    2. Search by dataset title
    3. Search by author names + task terms

    Args:
        dataset_title: Title of the dataset.
        dataset_doi: DOI of the dataset if available.
        author_names: Names of dataset contributors.
        task_terms: Task-related search terms.
        limit: Maximum number of results.

    Returns:
        List of potentially related papers.
    """
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # Strategy 1: Search by DOI
    if dataset_doi:
        doi_results = await search_works(dataset_doi, limit=5)
        for paper in doi_results:
            if paper["id"] not in seen_ids:
                seen_ids.add(paper["id"])
                results.append(paper)

    # Strategy 2: Search by title
    title_results = await search_works(dataset_title, limit=limit // 2)
    for paper in title_results:
        if paper["id"] not in seen_ids:
            seen_ids.add(paper["id"])
            results.append(paper)

    # Strategy 3: Search by author + task terms
    if author_names and task_terms and len(results) < limit:
        for author in author_names[:2]:
            for term in task_terms[:2]:
                query = f"{author} {term}"
                author_results = await search_works(query, limit=3)
                for paper in author_results:
                    if paper["id"] not in seen_ids:
                        seen_ids.add(paper["id"])
                        results.append(paper)
                        if len(results) >= limit:
                            break

    return results[:limit]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.ingestion.openalex")
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    args = parser.parse_args(argv)

    try:
        payload = fetch_openalex(args.query, args.limit)
        if args.save_raw:
            raw_path = save_raw_response("openalex", args.query, payload)
            print(json.dumps({"raw_saved": str(raw_path)}, indent=2))
        records = records_from_response(payload, args.limit)
        print_normalized_records(records)
        if args.dry_run:
            return 0
        summary = save_paper_records(records, args.database_url, args.force)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print_cli_error("openalex", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
