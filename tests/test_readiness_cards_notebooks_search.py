import nbformat
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from neural_search.cards import generate_dataset_card_json
from neural_search.db import Base
from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.demo_seed import build_demo_seed, seed_demo_database
from neural_search.models import Dataset, DatasetCard, Embedding, Paper
from neural_search.notebooks import generate_nwb_starter_notebook
from neural_search.readiness import compute_analysis_readiness
from neural_search.search import parse_query, score_dataset_against_query, search_datasets


def _dataset():
    return {
        "id": "D1",
        "source": "demo",
        "source_id": "D1",
        "title": "Reversal learning Neuropixels dataset",
        "description": "Mouse OFC trials with choice, reward omission, and event timestamps.",
        "license": "CC-BY-4.0",
        "data_standards": ["NWB"],
        "has_behavior": True,
        "has_trials": True,
        "has_processed_data": True,
        "metadata_json": {"trial_columns": ["choice", "reward", "omission"]},
    }


def _extraction():
    return extract_dataset_labels(
        title="Reversal learning Neuropixels dataset",
        description="Mouse OFC trials with choice, reward omission, and event timestamps.",
        file_paths=["sub-01/session.nwb"],
        source_metadata=_dataset(),
        linked_paper_abstracts=["Probabilistic reversal learning in mouse OFC."],
    )


def test_readiness_scoring():
    readiness = compute_analysis_readiness(_dataset(), _extraction(), [{"title": "paper"}])

    assert readiness.score == 100
    assert any("NWB" in strength for strength in readiness.strengths)


def test_card_generation_includes_provenance_and_markdown():
    card = generate_dataset_card_json(_dataset(), _extraction(), [{"title": "paper"}])

    assert card.analysis_readiness.score == 100
    assert card.provenance["linked_paper_count"] == 1
    assert "## Why Matched" in card.card_markdown
    assert "reversal_learning" in card.card_markdown


def test_notebook_generation_writes_valid_ipynb(tmp_path):
    output_path = tmp_path / "starter.ipynb"
    response = generate_nwb_starter_notebook(
        _dataset(),
        {"id": "A1", "path": "sub-01/session.nwb"},
        output_path,
    )

    assert response.valid is True
    notebook = nbformat.read(output_path, as_version=4)
    nbformat.validate(notebook)
    assert any("Load NWB" in cell.source for cell in notebook.cells)


def test_search_scoring_matches_reversal_dataset():
    extraction = _extraction()
    card = generate_dataset_card_json(_dataset(), extraction, [{"title": "paper"}])
    parsed = parse_query("Find reversal learning datasets with reward omission")
    result = score_dataset_against_query(_dataset(), card, parsed)

    assert result.score > 50
    assert any("Task matched" in reason for reason in result.why_matched)
    assert any("Behavior matched" in reason for reason in result.why_matched)


def test_search_datasets_demo_seed_orders_relevant_results():
    response = search_datasets("Find reversal learning datasets with reward omission", {})

    assert response.results
    assert response.results[0].dataset_id == "DEMO_REVERSAL_EPHYS"


def test_demo_seed_contains_five_fixture_datasets_with_papers():
    records = build_demo_seed()

    assert len(records) == 5
    assert {record["dataset"]["source_id"] for record in records} == {
        "DEMO_GONOGO_CALCIUM",
        "DEMO_REVERSAL_EPHYS",
        "DEMO_DELAY_DISCOUNTING",
        "DEMO_REACHING_ECOG_IEEG",
        "DEMO_VISUAL_DECISION_NEUROPIXELS",
    }
    assert all(record["papers"] for record in records)


def test_demo_seed_populates_database(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'demo_seed.db'}"

    summary = seed_demo_database(database_url)

    assert summary["datasets"] == 5
    assert summary["papers"] == 5
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Dataset)) == 5
        assert session.scalar(select(func.count()).select_from(Paper)) == 5
        assert session.scalar(select(func.count()).select_from(DatasetCard)) == 5
        assert session.scalar(select(func.count()).select_from(Embedding)) == 5
