"""Service layer for deterministic source ingestion workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neural_search.ingestion import dandi, openalex, openneuro
from neural_search.ingestion.demo_seed import DEFAULT_DATABASE_URL
from neural_search.ingestion.live import (
    save_dataset_records,
    save_paper_records,
    save_raw_response,
)


@dataclass(frozen=True)
class IngestionRunResult:
    """Summary of a single source ingestion run."""

    source: str
    query: str
    fetched: int
    normalized: int
    saved: int = 0
    skipped: int = 0
    raw_response_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dataset_ids: list[str] = field(default_factory=list)
    paper_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


Fetcher = Callable[[str, int], dict[str, Any]]


def ingest_dandi(
    query: str,
    limit: int = 10,
    *,
    save: bool = False,
    force: bool = False,
    database_url: str = DEFAULT_DATABASE_URL,
    payload: dict[str, Any] | None = None,
    fetcher: Fetcher = dandi.fetch_dandi,
) -> IngestionRunResult:
    """Fetch, normalize, optionally persist DANDI records."""

    payload = _payload_or_fetch(payload, fetcher, query, limit)
    raw_paths = _save_raw_if_requested("dandi", query, payload, save)
    records = dandi.records_from_response(payload, limit)
    saved, skipped = _save_datasets_if_requested(records, save, database_url, force)
    return IngestionRunResult(
        source="dandi",
        query=query,
        fetched=_count_items(payload, "results"),
        normalized=len(records),
        saved=saved,
        skipped=skipped,
        raw_response_paths=raw_paths,
        warnings=_normalization_warnings(records),
        dataset_ids=[str(record["source_id"]) for record in records],
    )


def ingest_openneuro(
    query: str,
    limit: int = 10,
    *,
    save: bool = False,
    force: bool = False,
    database_url: str = DEFAULT_DATABASE_URL,
    payload: dict[str, Any] | None = None,
    fetcher: Fetcher = openneuro.fetch_openneuro,
) -> IngestionRunResult:
    """Fetch, normalize, optionally persist OpenNeuro dataset records."""

    payload = _payload_or_fetch(payload, fetcher, query, limit)
    raw_paths = _save_raw_if_requested("openneuro", query, payload, save)
    records = openneuro.records_from_response(payload, limit)
    saved, skipped = _save_datasets_if_requested(records, save, database_url, force)
    return IngestionRunResult(
        source="openneuro",
        query=query,
        fetched=_count_openneuro_edges(payload),
        normalized=len(records),
        saved=saved,
        skipped=skipped,
        raw_response_paths=raw_paths,
        warnings=_normalization_warnings(records),
        dataset_ids=[str(record["source_id"]) for record in records],
    )


def ingest_openalex(
    query: str,
    limit: int = 10,
    *,
    save: bool = False,
    force: bool = False,
    database_url: str = DEFAULT_DATABASE_URL,
    payload: dict[str, Any] | None = None,
    fetcher: Fetcher = openalex.fetch_openalex,
) -> IngestionRunResult:
    """Fetch, normalize, optionally persist OpenAlex paper records."""

    payload = _payload_or_fetch(payload, fetcher, query, limit)
    raw_paths = _save_raw_if_requested("openalex", query, payload, save)
    records = openalex.records_from_response(payload, limit)
    saved, skipped = _save_papers_if_requested(records, save, database_url, force)
    return IngestionRunResult(
        source="openalex",
        query=query,
        fetched=_count_items(payload, "results"),
        normalized=len(records),
        saved=saved,
        skipped=skipped,
        raw_response_paths=raw_paths,
        warnings=[],
        paper_ids=[str(record.get("source_id") or record.get("id")) for record in records],
    )


def ingest_source(
    source: str,
    query: str,
    limit: int = 10,
    *,
    save: bool = False,
    force: bool = False,
    database_url: str = DEFAULT_DATABASE_URL,
) -> IngestionRunResult:
    """Dispatch ingestion for a named source."""

    normalized_source = source.casefold()
    if normalized_source == "dandi":
        return ingest_dandi(query, limit, save=save, force=force, database_url=database_url)
    if normalized_source == "openneuro":
        return ingest_openneuro(
            query,
            limit,
            save=save,
            force=force,
            database_url=database_url,
        )
    if normalized_source == "openalex":
        return ingest_openalex(
            query,
            limit,
            save=save,
            force=force,
            database_url=database_url,
        )
    raise ValueError(f"Unsupported ingestion source: {source}")


def _payload_or_fetch(
    payload: dict[str, Any] | None,
    fetcher: Fetcher,
    query: str,
    limit: int,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not query.strip():
        raise ValueError("query is required")
    return payload if payload is not None else fetcher(query, limit)


def _save_raw_if_requested(
    source: str,
    query: str,
    payload: dict[str, Any],
    save: bool,
) -> list[str]:
    if not save:
        return []
    path: Path = save_raw_response(source, query, payload)
    return [str(path)]


def _save_datasets_if_requested(
    records: list[dict[str, Any]],
    save: bool,
    database_url: str,
    force: bool,
) -> tuple[int, int]:
    if not save:
        return 0, 0
    summary = save_dataset_records(records, database_url=database_url, force=force)
    return summary["saved"], summary["skipped"]


def _save_papers_if_requested(
    records: list[dict[str, Any]],
    save: bool,
    database_url: str,
    force: bool,
) -> tuple[int, int]:
    if not save:
        return 0, 0
    summary = save_paper_records(records, database_url=database_url, force=force)
    return summary["saved"], summary["skipped"]


def _count_items(payload: dict[str, Any], key: str) -> int:
    items = payload.get(key)
    if isinstance(items, list):
        return len(items)
    if isinstance(payload, list):
        return len(payload)
    return 0


def _count_openneuro_edges(payload: dict[str, Any]) -> int:
    edges = payload.get("data", {}).get("datasets", {}).get("edges", [])
    return len(edges) if isinstance(edges, list) else 0


def _normalization_warnings(records: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for record in records:
        source_id = record.get("source_id", "unknown")
        if not record.get("title"):
            warnings.append(f"{source_id}: missing title")
        if not record.get("description"):
            warnings.append(f"{source_id}: missing description")
        if not record.get("modalities"):
            warnings.append(f"{source_id}: no modalities extracted")
        if not record.get("tasks"):
            warnings.append(f"{source_id}: no tasks extracted")
    return warnings
