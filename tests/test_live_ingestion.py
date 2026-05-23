from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from neural_search.db import Base
from neural_search.ingestion.dandi import normalize_dandiset
from neural_search.ingestion.live import save_dataset_records, save_paper_records
from neural_search.ingestion.openalex import normalize_work
from neural_search.ingestion.openneuro import normalize_openneuro_dataset
from neural_search.models import Dataset, Paper


def test_dandi_normalization_from_raw_record():
    record = normalize_dandiset(
        {
            "identifier": "000001",
            "most_recent_published_version": {
                "version": "0.1.0",
                "metadata": {
                    "name": "Go NoGo calcium imaging",
                    "description": "Mouse NWB lick reward omission trials",
                    "license": "CC-BY-4.0",
                },
            },
        }
    )

    assert record["source"] == "dandi"
    assert record["source_id"] == "000001"
    assert "go_nogo" in record["tasks"]
    assert "NWB" in record["data_standards"]


def test_openneuro_normalization_from_raw_record():
    record = normalize_openneuro_dataset(
        {
            "id": "ds000001",
            "name": "Motor imagery EEG",
            "description": "BCI motor imagery task with EEG",
            "latestSnapshot": {
                "tag": "1.0.0",
                "summary": {
                    "subjects": 2,
                    "tasks": ["motor_imagery"],
                    "modalities": ["EEG"],
                },
            },
        }
    )

    assert record["source"] == "openneuro"
    assert record["source_id"] == "ds000001"
    assert "BIDS" in record["data_standards"]
    assert "eeg" in record["modalities"]


def test_openalex_normalization_from_raw_record():
    record = normalize_work(
        {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.0000/example",
            "title": "Reversal learning electrophysiology",
            "publication_year": 2024,
            "authorships": [{"author": {"display_name": "Demo Author"}}],
            "concepts": [{"display_name": "Neuroscience"}],
            "abstract_inverted_index": {"Reward": [0], "omission": [1]},
        }
    )

    assert record["source"] == "openalex"
    assert record["source_id"] == "W123"
    assert record["abstract"] == "Reward omission"


def test_save_dataset_records_does_not_overwrite_without_force(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'live.db'}"
    record = {
        "source": "dandi",
        "source_id": "000001",
        "title": "Original title",
        "description": "Original description",
        "url": "https://example.org",
        "license": "CC-BY-4.0",
        "species": [],
        "modalities": [],
        "brain_regions": [],
        "tasks": [],
        "behaviors": [],
        "data_standards": ["NWB"],
        "has_behavior": False,
        "has_trials": False,
        "has_raw_data": True,
        "has_processed_data": False,
        "metadata_json": {},
    }

    first = save_dataset_records([record], database_url)
    second = save_dataset_records([{**record, "title": "New title"}], database_url)

    assert first == {"saved": 1, "skipped": 0}
    assert second == {"saved": 0, "skipped": 1}
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Dataset)) == 1
        assert session.scalar(select(Dataset.title)) == "Original title"


def test_save_paper_records_does_not_overwrite_without_force(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'live.db'}"
    record = {
        "openalex_id": "https://openalex.org/W123",
        "title": "Original paper",
        "abstract": "Original",
        "publication_year": 2024,
        "authors_json": [],
        "url": "https://openalex.org/W123",
        "concepts": [],
        "linked_dataset_ids": [],
        "metadata_json": {},
    }

    first = save_paper_records([record], database_url)
    second = save_paper_records([{**record, "title": "New paper"}], database_url)

    assert first == {"saved": 1, "skipped": 0}
    assert second == {"saved": 0, "skipped": 1}
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Paper)) == 1
        assert session.scalar(select(Paper.title)) == "Original paper"

