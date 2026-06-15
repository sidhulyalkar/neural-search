"""Tests for OSF ingestion adapter."""


def test_import():
    from neural_search.ingestion.osf import normalize_osf_project
    assert callable(normalize_osf_project)


def test_normalize_basic():
    from neural_search.ingestion.osf import normalize_osf_project
    raw = {
        "id": "abc12",
        "type": "nodes",
        "attributes": {
            "title": "Mouse hippocampus calcium imaging",
            "description": "Two-photon calcium imaging data from mouse CA1 during spatial navigation",
            "public": True,
            "date_created": "2022-01-01",
        },
    }
    rec = normalize_osf_project(raw)
    assert rec["source"] == "osf"
    assert rec["source_id"] == "abc12"


def test_classifier_gate_applied():
    from neural_search.ingestion.dataset_classifier import is_valid_dataset
    from neural_search.ingestion.osf import normalize_osf_project
    raw = {
        "id": "xyz99",
        "type": "nodes",
        "attributes": {
            "title": "Analysis code for attention study",
            "description": "Python scripts for statistical analysis",
            "public": True,
        },
    }
    rec = normalize_osf_project(raw)
    result = is_valid_dataset(rec)
    assert result.accepted is False
