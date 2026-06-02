"""Tests for G-Node GIN ingestion adapter."""


def test_import():
    from neural_search.ingestion.gin import normalize_gin_repo
    assert callable(normalize_gin_repo)


def test_normalize_basic():
    from neural_search.ingestion.gin import normalize_gin_repo
    raw = {
        "id": 111,
        "name": "mouse-ephys-study",
        "full_name": "lab/mouse-ephys-study",
        "description": "Neuropixels recordings in mouse visual cortex during drifting gratings",
        "html_url": "https://gin.g-node.org/lab/mouse-ephys-study",
        "updated": "2023-05-01T12:00:00Z",
    }
    rec = normalize_gin_repo(raw)
    assert rec["source"] == "gin"
    assert rec["source_id"] == "111"
    assert "neuropixels" in str(rec).lower() or len(rec["modalities"]) >= 0


def test_registered_in_registry():
    import neural_search.ingestion.gin  # noqa: F401
    from neural_search.ingestion.registry import list_adapters
    assert "gin" in list_adapters()
