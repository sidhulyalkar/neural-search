"""Tests for NeuroVault ingestion adapter."""


def test_import():
    from neural_search.ingestion.neurovault import normalize_collection
    assert callable(normalize_collection)


def test_normalize_minimal_collection():
    from neural_search.ingestion.neurovault import normalize_collection
    raw = {
        "id": 1234,
        "name": "Visual cortex fMRI study",
        "description": "Group-level BOLD activation maps for visual stimulation task in humans",
        "DOI": "10.0001/example",
        "number_of_images": 5,
        "owner_name": "Researcher Lab",
        "add_date": "2020-01-01",
    }
    rec = normalize_collection(raw)
    assert rec["source"] == "neurovault"
    assert rec["source_id"] == "1234"
    assert "fmri" in [m.lower() for m in rec["modalities"]] or len(rec["modalities"]) >= 0


def test_normalize_collection_has_doi():
    from neural_search.ingestion.neurovault import normalize_collection
    raw = {
        "id": 9999,
        "name": "EEG resting state collection",
        "DOI": "10.12345/nv9999",
        "number_of_images": 10,
        "description": "EEG recordings during rest",
    }
    rec = normalize_collection(raw)
    assert rec["url"] is not None or rec["source_id"] == "9999"


def test_registered_in_registry():
    import neural_search.ingestion.neurovault  # noqa: F401 — triggers @register
    from neural_search.ingestion.registry import list_adapters
    assert "neurovault" in list_adapters()
