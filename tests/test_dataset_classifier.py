"""Tests for dataset inclusion classifier."""
from neural_search.ingestion.dataset_classifier import is_valid_dataset, ClassificationResult


def test_rejects_paper_only():
    rec = {
        "title": "Neural correlates of attention: a review",
        "description": "Review article analyzing 50 fMRI studies on attention",
        "resource_type": "publication",
        "license": "CC-BY",
        "doi": "10.0001/review",
    }
    result = is_valid_dataset(rec)
    assert result.accepted is False
    assert "data" in result.failure_reason.lower()


def test_rejects_no_doi():
    rec = {
        "title": "Mouse hippocampus calcium imaging dataset",
        "description": "Calcium imaging data from mouse CA1 during navigation task",
        "resource_type": "dataset",
        "license": "CC-BY",
    }
    result = is_valid_dataset(rec)
    assert result.accepted is False
    assert "identifier" in result.failure_reason.lower() or "doi" in result.failure_reason.lower()


def test_accepts_valid_neuroscience_dataset():
    rec = {
        "title": "Mouse prefrontal cortex electrophysiology dataset",
        "description": (
            "Extracellular recordings from mouse prefrontal cortex during reversal learning task. "
            "Spike sorted single units, trial events, behavioral outcomes."
        ),
        "resource_type": "dataset",
        "license": "CC-BY",
        "doi": "10.1234/example",
        "subjects": ["Mus musculus"],
    }
    result = is_valid_dataset(rec)
    assert result.accepted is True


def test_rejects_code_only():
    rec = {
        "title": "Python analysis code for spike sorting",
        "description": "Python scripts and notebooks for spike sorting analysis",
        "resource_type": "software",
        "license": "MIT",
        "doi": "10.5678/code",
    }
    result = is_valid_dataset(rec)
    assert result.accepted is False


def test_rejects_no_species_or_modality_signal():
    rec = {
        "title": "General data file",
        "description": "Some data collected in 2020.",
        "resource_type": "dataset",
        "license": "CC-BY",
        "doi": "10.0001/generic",
    }
    result = is_valid_dataset(rec)
    assert result.accepted is False
