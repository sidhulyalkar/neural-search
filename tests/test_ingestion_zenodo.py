"""Tests for zenodo ingestion adapter."""


def test_import():
    from neural_search.ingestion.zenodo import normalize_zenodo_record
    assert callable(normalize_zenodo_record)


def test_normalize_basic():
    from neural_search.ingestion.zenodo import normalize_zenodo_record
    raw = {
        "id": "7654321",
        "doi": "10.5281/zenodo.7654321",
        "metadata": {
            "title": "Human EEG resting state data",
            "description": "64-channel EEG recordings from 50 healthy adults during resting state",
            "resource_type": {"type": "dataset"},
            "license": {"id": "cc-by-4.0"},
            "keywords": ["eeg", "resting state", "human"],
        },
    }
    rec = normalize_zenodo_record(raw)
    assert rec["source"] == "zenodo"
    assert rec["doi"] == "10.5281/zenodo.7654321"


def test_registered_in_registry():
    import neural_search.ingestion.zenodo  # noqa: F401
    from neural_search.ingestion.registry import list_adapters
    assert "zenodo" in list_adapters()
