"""Deterministic fixture seed data for local demos, tests, and DB seeding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from neural_search.cards import generate_dataset_card_json
from neural_search.db import Base
from neural_search.extraction import extract_dataset_labels
from neural_search.models import Dataset, DatasetAsset, DatasetCard, Embedding, Paper


FIXTURE_PATH = Path(__file__).resolve().parents[2] / "data" / "seed" / "demo_fixtures.json"
UUID_NAMESPACE = UUID("5b50ee83-7c12-4d38-94a6-4d7ff7a0f21b")
DEFAULT_DATABASE_URL = "sqlite:///data/seed/demo_seed.db"


def stable_uuid(value: str) -> UUID:
    return uuid5(UUID_NAMESPACE, value)


def deterministic_embedding(text: str, dimensions: int = 16) -> list[float]:
    """Create a deterministic lightweight embedding for tests and demos."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for index in range(dimensions):
        byte = digest[index % len(digest)]
        values.append(round((byte / 255.0) * 2.0 - 1.0, 6))
    return values


def load_fixture_payload(path: str | Path = FIXTURE_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_demo_seed() -> list[dict]:
    payload = load_fixture_payload()
    records: list[dict] = []
    for fixture in payload["datasets"]:
        dataset = {
            key: value
            for key, value in fixture.items()
            if key not in {"assets", "papers"}
        }
        dataset_assets = [
            {**asset, "dataset_id": dataset["source_id"]} for asset in fixture.get("assets", [])
        ]
        linked_papers = [
            {**paper, "linked_dataset_ids": [dataset["source_id"]]}
            for paper in fixture.get("papers", [])
        ]
        extraction = extract_dataset_labels(
            title=dataset["title"],
            description=dataset["description"],
            file_paths=[asset["path"] for asset in dataset_assets],
            source_metadata=dataset,
            linked_paper_abstracts=[paper["abstract"] for paper in linked_papers],
        )
        records.append(
            {
                "dataset": dataset,
                "assets": dataset_assets,
                "papers": linked_papers,
                "extraction": extraction,
            }
        )
    return records


def _dataset_model(dataset: dict[str, Any]) -> Dataset:
    metadata = dict(dataset.get("metadata_json", {}))
    metadata["fixture"] = True
    return Dataset(
        id=stable_uuid(f"dataset:{dataset['source_id']}"),
        source=dataset["source"],
        source_id=dataset["source_id"],
        title=dataset["title"],
        description=dataset.get("description"),
        url=dataset.get("url"),
        license=dataset.get("license"),
        species=dataset.get("species", []),
        modalities=dataset.get("modalities", []),
        brain_regions=dataset.get("brain_regions", []),
        tasks=dataset.get("tasks", []),
        behaviors=dataset.get("behaviors", []),
        data_standards=dataset.get("data_standards", []),
        has_behavior=dataset.get("has_behavior", False),
        has_trials=dataset.get("has_trials", False),
        has_raw_data=dataset.get("has_raw_data", False),
        has_processed_data=dataset.get("has_processed_data", False),
        metadata_json=metadata,
    )


def _asset_model(asset: dict[str, Any], dataset_id: UUID) -> DatasetAsset:
    return DatasetAsset(
        id=stable_uuid(f"asset:{asset['id']}"),
        dataset_id=dataset_id,
        path=asset["path"],
        asset_type=asset.get("asset_type"),
        file_format=asset.get("file_format"),
        size_bytes=asset.get("size_bytes"),
        subject_id=asset.get("subject_id"),
        session_id=asset.get("session_id"),
        modality=asset.get("modality"),
        metadata_json={"fixture": True},
    )


def _paper_model(paper: dict[str, Any]) -> Paper:
    return Paper(
        id=stable_uuid(f"paper:{paper['id']}"),
        openalex_id=paper.get("openalex_id"),
        doi=paper.get("doi"),
        title=paper["title"],
        abstract=paper.get("abstract"),
        publication_year=paper.get("publication_year"),
        authors_json=paper.get("authors_json", []),
        url=paper.get("url"),
        concepts=paper.get("concepts", []),
        linked_dataset_ids=paper.get("linked_dataset_ids", []),
        metadata_json={"fixture": True, "fixture_id": paper["id"]},
    )


def _card_model(record: dict[str, Any], dataset_id: UUID) -> DatasetCard:
    card = generate_dataset_card_json(
        record["dataset"], record["extraction"], record.get("papers", [])
    )
    return DatasetCard(
        id=stable_uuid(f"card:{record['dataset']['source_id']}"),
        dataset_id=dataset_id,
        summary=card.summary,
        why_relevant=card.why_relevant,
        analysis_readiness_score=card.analysis_readiness.score,
        strengths=card.analysis_readiness.strengths,
        limitations=card.analysis_readiness.limitations,
        missing_fields=card.missing_fields,
        suggested_analyses=card.suggested_analyses,
        provenance_json=card.provenance,
        card_markdown=card.card_markdown or "",
    )


def _embedding_model(record: dict[str, Any], dataset_id: UUID) -> Embedding:
    dataset = record["dataset"]
    text = " ".join(
        str(part)
        for part in [
            dataset["title"],
            dataset.get("description", ""),
            dataset.get("tasks", []),
            dataset.get("behaviors", []),
            dataset.get("modalities", []),
        ]
    )
    return Embedding(
        id=stable_uuid(f"embedding:dataset:{dataset['source_id']}"),
        entity_type="dataset",
        entity_id=dataset_id,
        text_for_embedding=text,
        embedding=deterministic_embedding(text),
        embedding_model="deterministic-fixture-v1",
        metadata_json={"fixture": True},
    )


def seed_demo_database(database_url: str = DEFAULT_DATABASE_URL) -> dict[str, int | str]:
    if database_url.startswith("sqlite:///"):
        db_path = Path(database_url.removeprefix("sqlite:///"))
        if db_path.parent != Path("."):
            db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    records = build_demo_seed()

    with session_factory() as session:
        _merge_records(session, records)
        session.commit()

    return {
        "database_url": database_url,
        "datasets": len(records),
        "papers": sum(len(record["papers"]) for record in records),
        "cards": len(records),
        "embeddings": len(records),
    }


def _merge_records(session: Session, records: list[dict]) -> None:
    for record in records:
        dataset = _dataset_model(record["dataset"])
        session.merge(dataset)
        for asset in record["assets"]:
            session.merge(_asset_model(asset, dataset.id))
        for paper in record["papers"]:
            session.merge(_paper_model(paper))
        session.merge(_card_model(record, dataset.id))
        session.merge(_embedding_model(record, dataset.id))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.ingestion.demo_seed")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
        help="Database URL to populate. Defaults to a local SQLite fixture DB.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print fixture records instead of seeding the database.",
    )
    args = parser.parse_args(argv)

    if args.print_json:
        print(json.dumps(_serializable_records(), indent=2))
        return 0

    print(json.dumps(seed_demo_database(args.database_url), indent=2))
    return 0


def _serializable_records() -> list[dict[str, Any]]:
    payload = []
    for record in build_demo_seed():
        item = dict(record)
        item["extraction"] = record["extraction"].model_dump(mode="json")
        payload.append(item)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
