import nbformat

from neural_search.cards import generate_dataset_card_json
from neural_search.extraction import extract_dataset_labels
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
    assert response.results[0].dataset_id == "DEMO"

