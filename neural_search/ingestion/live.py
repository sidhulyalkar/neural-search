"""Shared utilities for safe live ingestion connectors."""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from neural_search.db import Base
from neural_search.ingestion.demo_seed import DEFAULT_DATABASE_URL, stable_uuid
from neural_search.models import Dataset, Paper


RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def normalize_query_slug(query: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", query.strip().lower()).strip("-")
    return slug[:80] or "query"


def save_raw_response(source: str, query: str, payload: Any) -> Path:
    """Save a raw API payload without overwriting existing files."""

    source_dir = RAW_DATA_DIR / source
    source_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = source_dir / f"{timestamp}-{normalize_query_slug(query)}.json"
    path = base
    counter = 1
    while path.exists():
        path = base.with_name(f"{base.stem}-{counter}{base.suffix}")
        counter += 1
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def print_normalized_records(records: list[dict[str, Any]]) -> None:
    print(json.dumps(records, indent=2, sort_keys=True))


def print_cli_error(source: str, error: Exception) -> None:
    print(
        json.dumps(
            {
                "source": source,
                "status": "error",
                "error_type": type(error).__name__,
                "message": str(error),
            },
            indent=2,
            sort_keys=True,
        ),
        file=sys.stderr,
    )


def create_session(database_url: str = DEFAULT_DATABASE_URL) -> Session:
    from sqlalchemy import create_engine

    if database_url.startswith("sqlite:///"):
        db_path = Path(database_url.removeprefix("sqlite:///"))
        if db_path.parent != Path("."):
            db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return session_factory()


def dataset_exists(session: Session, source: str, source_id: str) -> bool:
    statement = select(Dataset).where(
        Dataset.source == source,
        Dataset.source_id == source_id,
    )
    return session.execute(statement).scalar_one_or_none() is not None


def paper_exists(session: Session, openalex_id: str | None, title: str) -> bool:
    if openalex_id:
        statement = select(Paper).where(Paper.openalex_id == openalex_id)
        if session.execute(statement).scalar_one_or_none() is not None:
            return True
    statement = select(Paper).where(Paper.title == title)
    return session.execute(statement).scalar_one_or_none() is not None


def save_dataset_records(
    records: list[dict[str, Any]],
    database_url: str = DEFAULT_DATABASE_URL,
    force: bool = False,
) -> dict[str, int]:
    """Save normalized dataset records, skipping existing records unless forced."""

    saved = 0
    skipped = 0
    with create_session(database_url) as session:
        for record in records:
            source = record["source"]
            source_id = record["source_id"]
            if dataset_exists(session, source, source_id) and not force:
                skipped += 1
                continue
            dataset = Dataset(
                id=stable_uuid(f"live-dataset:{source}:{source_id}"),
                source=source,
                source_id=source_id,
                title=record["title"],
                description=record.get("description"),
                url=record.get("url"),
                license=record.get("license"),
                species=record.get("species", []),
                modalities=record.get("modalities", []),
                brain_regions=record.get("brain_regions", []),
                tasks=record.get("tasks", []),
                behaviors=record.get("behaviors", []),
                data_standards=record.get("data_standards", []),
                has_behavior=record.get("has_behavior", False),
                has_trials=record.get("has_trials", False),
                has_raw_data=record.get("has_raw_data", False),
                has_processed_data=record.get("has_processed_data", False),
                metadata_json=record.get("metadata_json", {}),
            )
            session.merge(dataset)
            saved += 1
        session.commit()
    return {"saved": saved, "skipped": skipped}


def save_paper_records(
    records: list[dict[str, Any]],
    database_url: str = DEFAULT_DATABASE_URL,
    force: bool = False,
) -> dict[str, int]:
    """Save normalized paper records, skipping existing records unless forced."""

    saved = 0
    skipped = 0
    with create_session(database_url) as session:
        for record in records:
            openalex_id = record.get("openalex_id")
            title = record.get("title", "")
            if paper_exists(session, openalex_id, title) and not force:
                skipped += 1
                continue
            paper = Paper(
                id=stable_uuid(f"live-paper:{openalex_id or title}"),
                openalex_id=openalex_id,
                doi=record.get("doi"),
                title=title,
                abstract=record.get("abstract"),
                publication_year=record.get("publication_year"),
                authors_json=record.get("authors_json", []),
                url=record.get("url"),
                concepts=record.get("concepts", []),
                linked_dataset_ids=record.get("linked_dataset_ids", []),
                metadata_json=record.get("metadata_json", {}),
            )
            session.merge(paper)
            saved += 1
        session.commit()
    return {"saved": saved, "skipped": skipped}
