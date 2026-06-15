"""Deterministic fixture seed data for local demos, tests, and DB seeding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from neural_search.cards import generate_dataset_card_json
from neural_search.db import Base
from neural_search.extraction import extract_dataset_labels
from neural_search.models import Dataset, DatasetAsset, DatasetCard, Embedding, Paper

SEED_DIR = Path(__file__).resolve().parents[2] / "data" / "seed"
FIXTURE_PATH = SEED_DIR / "demo_datasets.yaml"
PAPERS_PATH = SEED_DIR / "demo_papers.yaml"
UUID_NAMESPACE = UUID("5b50ee83-7c12-4d38-94a6-4d7ff7a0f21b")
DEFAULT_DATABASE_URL = "sqlite:///data/seed/demo_seed.db"
REQUIRED_DATASET_FIELDS = {
    "source",
    "source_id",
    "title",
    "description",
    "url",
    "species",
    "modalities",
    "brain_regions",
    "tasks",
    "behaviors",
    "data_standards",
    "has_behavior",
    "has_trials",
    "license",
    "linked_paper_ids",
}


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


def _load_mapping(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    with fixture_path.open("r", encoding="utf-8") as handle:
        if fixture_path.suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle) or {}
        else:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Fixture root must be a mapping: {fixture_path}")
    return payload


def load_fixture_payload(
    datasets_path: str | Path = FIXTURE_PATH,
    papers_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_payload = _load_mapping(datasets_path)
    if "datasets" not in dataset_payload:
        raise ValueError(f"Dataset fixture must contain a datasets list: {datasets_path}")
    if papers_path is None:
        candidate = Path(datasets_path).with_name("demo_papers.yaml")
        papers_path = candidate if candidate.exists() else PAPERS_PATH
    paper_payload = _load_mapping(papers_path)
    if "papers" not in paper_payload:
        raise ValueError(f"Paper fixture must contain a papers list: {papers_path}")
    return {"datasets": dataset_payload["datasets"], "papers": paper_payload["papers"]}


def _validate_dataset_fixture(dataset: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_DATASET_FIELDS - set(dataset))
    if missing:
        source_id = dataset.get("source_id", "<missing source_id>")
        raise ValueError(f"Dataset fixture {source_id} missing fields: {missing}")
    for field in [
        "species",
        "modalities",
        "brain_regions",
        "tasks",
        "behaviors",
        "data_standards",
        "linked_paper_ids",
    ]:
        if not isinstance(dataset[field], list):
            raise ValueError(f"Dataset fixture {dataset['source_id']} field {field} must be a list")


def _synthetic_asset(dataset: dict[str, Any]) -> dict[str, Any]:
    source_id = dataset["source_id"]
    primary_modality = dataset.get("modalities", ["metadata"])[0]
    standard = next(iter(dataset.get("data_standards", [])), "metadata")
    extension = "nwb" if standard == "NWB" else "tsv"
    asset_type = "nwb" if standard == "NWB" else "events"
    return {
        "id": f"ASSET_{source_id}",
        "dataset_id": source_id,
        "path": f"data/seed/compiled/{source_id}.{extension}",
        "asset_type": asset_type,
        "file_format": extension,
        "modality": primary_modality,
    }


def build_demo_seed(
    datasets_path: str | Path = FIXTURE_PATH,
    papers_path: str | Path | None = None,
) -> list[dict]:
    payload = load_fixture_payload(datasets_path, papers_path)
    papers_by_id = {paper["id"]: paper for paper in payload["papers"]}
    records: list[dict] = []
    for fixture in payload["datasets"]:
        _validate_dataset_fixture(fixture)
        dataset = {
            key: value
            for key, value in fixture.items()
            if key not in {"assets", "papers"}
        }
        dataset["id"] = dataset.get("id", dataset["source_id"])
        dataset["qa_status"] = dataset.get("qa_status", "auto_generated")
        dataset["has_raw_data"] = dataset.get("has_raw_data", True)
        dataset["has_processed_data"] = dataset.get("has_processed_data", False)
        dataset["metadata_json"] = {
            **dataset.get("metadata_json", {}),
            "linked_paper_ids": dataset["linked_paper_ids"],
            "fixture_stage": "A",
        }
        dataset_assets = [
            {**asset, "dataset_id": dataset["source_id"]} for asset in fixture.get("assets", [])
        ] or [_synthetic_asset(dataset)]
        missing_paper_ids = [
            paper_id
            for paper_id in dataset["linked_paper_ids"]
            if paper_id not in papers_by_id
        ]
        if missing_paper_ids:
            raise ValueError(
                f"Dataset fixture {dataset['source_id']} references missing papers: {missing_paper_ids}"
            )
        linked_papers = [
            {**papers_by_id[paper_id], "linked_dataset_ids": [dataset["source_id"]]}
            for paper_id in dataset["linked_paper_ids"]
        ]
        extraction = extract_dataset_labels(
            title=dataset["title"],
            description=dataset["description"],
            file_paths=[asset["path"] for asset in dataset_assets],
            source_metadata=dataset,
            linked_paper_abstracts=[paper["abstract"] for paper in linked_papers],
        )
        dataset["recording_scales"] = [
            label.id for label in extraction.recording_scales
        ] or dataset.get("recording_scales", [])
        dataset["metadata_json"]["recording_scales"] = dataset["recording_scales"]
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
        qa_status=dataset.get("qa_status", "auto_generated"),
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
        qa_status=card.qa_status,
        task_labels_verified=card.task_labels_verified,
        modality_labels_verified=card.modality_labels_verified,
        behavior_labels_verified=card.behavior_labels_verified,
        brain_regions_verified=card.brain_regions_verified,
        linked_papers_verified=card.linked_papers_verified,
        notebook_tested=card.notebook_tested,
        reviewer_notes=card.reviewer_notes,
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


def seed_demo_database(
    database_url: str = DEFAULT_DATABASE_URL,
    datasets_path: str | Path = FIXTURE_PATH,
    papers_path: str | Path | None = None,
) -> dict[str, int | str]:
    if database_url.startswith("sqlite:///"):
        db_path = Path(database_url.removeprefix("sqlite:///"))
        if db_path.parent != Path("."):
            db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    records = build_demo_seed(datasets_path, papers_path)

    with session_factory() as session:
        _merge_records(session, records)
        session.commit()

    return {
        "database_url": database_url,
        "datasets_path": str(datasets_path),
        "papers_path": str(papers_path or Path(datasets_path).with_name("demo_papers.yaml")),
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


NORMALIZED_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus" / "normalized"


def _iter_normalized_jsonl_files(corpus_path: Path) -> list[Path]:
    """Return normalized JSONL files, tolerating nested artifact directories."""
    if corpus_path.is_file():
        return [corpus_path]
    if not corpus_path.is_dir():
        return []
    return sorted(path for path in corpus_path.rglob("*.jsonl") if path.is_file())


def build_combined_corpus(
    include_demo: bool = True,
    include_real: bool = True,
    datasets_path: str | Path = FIXTURE_PATH,
    papers_path: str | Path | None = None,
    corpus_dir: str | Path = NORMALIZED_CORPUS_DIR,
) -> list[dict]:
    """Build corpus combining demo fixtures and real normalized records.

    Args:
        include_demo: Include demo fixture datasets
        include_real: Include real normalized corpus
        datasets_path: Path to demo datasets YAML
        papers_path: Path to demo papers YAML
        corpus_dir: Directory containing normalized JSONL files

    Returns:
        List of dataset records ready for search
    """
    records: list[dict] = []
    seen_ids: set[str] = set()

    # Add demo fixtures
    if include_demo:
        for record in build_demo_seed(datasets_path, papers_path):
            dataset_id = record.get("dataset", {}).get("id", "")
            if dataset_id and dataset_id not in seen_ids:
                seen_ids.add(dataset_id)
                records.append(record)

    # Add real normalized corpus (only non-demo, non-duplicate records)
    if include_real:
        corpus_path = Path(corpus_dir)
        if corpus_path.exists():
            for jsonl_file in _iter_normalized_jsonl_files(corpus_path):
                # Skip demo-derived normalized files
                if "demo" in jsonl_file.name.lower():
                    continue
                with open(jsonl_file, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            # Skip demo datasets that might be in normalized corpus
                            source_id = data.get("source_id", "")
                            if source_id.startswith("DEMO_"):
                                continue
                            # Convert normalized record to search format
                            record = _normalized_to_search_record(data)
                            if record:
                                dataset_id = record.get("dataset", {}).get("id", "")
                                if dataset_id and dataset_id not in seen_ids:
                                    seen_ids.add(dataset_id)
                                    records.append(record)

    return records


def _normalized_to_search_record(data: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a normalized record to the search record format."""
    # Handle both NormalizedDatasetRecord format and legacy format
    dataset_id = data.get("dataset_id") or data.get("source_id")
    if not dataset_id:
        return None

    # Extract label values from EvidenceLabel format
    def extract_labels(items: list) -> list[str]:
        if not items:
            return []
        result = []
        for item in items:
            if isinstance(item, dict):
                label = item.get("label") or item.get("id")
                if label:
                    result.append(label)
            elif isinstance(item, str):
                result.append(item)
        return result

    # Build dataset record
    dataset = {
        "id": dataset_id,
        "source": data.get("source", "unknown"),
        "source_id": data.get("source_id", dataset_id),
        "title": data.get("title", "Untitled"),
        "description": data.get("description"),
        "url": data.get("url"),
        "species": extract_labels(data.get("species", [])),
        "modalities": extract_labels(data.get("modalities", [])),
        "brain_regions": extract_labels(data.get("brain_regions", [])),
        "tasks": extract_labels(data.get("tasks", [])),
        "behaviors": extract_labels(data.get("behavioral_events", [])),
        "data_standards": extract_labels(data.get("data_standards", [])),
        "analysis_goals": extract_labels(data.get("analysis_goals", [])),
        "linked_paper_ids": data.get("linked_papers", []),
        "has_raw_data": data.get("usability_flags", {}).get("has_raw_data", True),
        "has_processed_data": data.get("usability_flags", {}).get("has_processed_data", False),
        "has_trials": data.get("usability_flags", {}).get("has_trials", False),
        "has_behavior": data.get("usability_flags", {}).get("has_behavior", False),
        "metadata_json": {
            "source": "normalized_corpus",
            "created_at": data.get("created_at"),
        },
    }

    # Create extraction placeholder
    extraction = extract_dataset_labels(
        title=dataset["title"],
        description=dataset.get("description"),
        file_paths=[],
        source_metadata=dataset,
        linked_paper_abstracts=[],
    )

    return {
        "dataset": dataset,
        "assets": [],
        "papers": [],
        "extraction": extraction,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.ingestion.demo_seed")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
        help="Database URL to populate. Defaults to a local SQLite fixture DB.",
    )
    parser.add_argument(
        "--papers-path",
        default=None,
        help="Optional paper fixture YAML path. Defaults to demo_papers.yaml next to datasets.",
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

    print(
        json.dumps(
            seed_demo_database(args.database_url, FIXTURE_PATH, args.papers_path),
            indent=2,
        )
    )
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
