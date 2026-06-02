"""Tests for figshare ingestion adapter."""


def test_import():
    from neural_search.ingestion.figshare import normalize_figshare_item
    assert callable(normalize_figshare_item)


def test_normalize_basic():
    from neural_search.ingestion.figshare import normalize_figshare_item
    raw = {
        "id": 55555,
        "title": "Rat barrel cortex spiking data",
        "description": "Extracellular recordings from rat S1 barrel cortex during whisker stimulation",
        "doi": "10.6084/m9.figshare.55555",
        "license": {"name": "CC BY 4.0"},
        "defined_type_name": "dataset",
        "categories": [{"title": "Neuroscience"}],
        "tags": ["electrophysiology", "rat", "barrel cortex"],
    }
    rec = normalize_figshare_item(raw)
    assert rec["source"] == "figshare"
    assert rec["source_id"] == "55555"
    assert rec.get("doi") or "doi" in str(rec.get("metadata_json", {})).lower()


def test_registered_in_registry():
    import neural_search.ingestion.figshare  # noqa: F401
    from neural_search.ingestion.registry import list_adapters
    assert "figshare" in list_adapters()
