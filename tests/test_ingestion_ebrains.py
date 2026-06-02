"""Tests for EBRAINS Knowledge Graph ingestion adapter."""


def test_import():
    from neural_search.ingestion.ebrains import normalize_ebrains_dataset
    assert callable(normalize_ebrains_dataset)


def test_normalize_basic():
    from neural_search.ingestion.ebrains import normalize_ebrains_dataset
    raw = {
        "id": "abc-123",
        "fields": {
            "name": "Rat hippocampus electrophysiology",
            "description": "Extracellular ephys recordings from rat CA1 during spatial navigation",
            "license": "CC-BY",
        },
    }
    rec = normalize_ebrains_dataset(raw)
    assert rec["source"] == "ebrains"
    assert rec["source_id"] == "abc-123"


def test_normalize_url_style_id():
    from neural_search.ingestion.ebrains import normalize_ebrains_dataset
    raw = {"@id": "https://kg.ebrains.eu/api/instances/xyz-456"}
    rec = normalize_ebrains_dataset(raw)
    assert rec["source_id"] == "xyz-456"


def test_registered_in_registry():
    import neural_search.ingestion.ebrains  # noqa: F401
    from neural_search.ingestion.registry import list_adapters
    assert "ebrains" in list_adapters()
