"""Tests for HCP ingestion adapter."""


def test_fetch_returns_records():
    from neural_search.ingestion.hcp import fetch_hcp
    records = fetch_hcp(limit=3)
    assert len(records) == 3
    for rec in records:
        assert rec["source"] == "hcp"
        assert "human" in rec["species"]
        assert rec["metadata_json"]["auth_required"] is True


def test_registered_in_registry():
    import neural_search.ingestion.hcp  # noqa: F401
    from neural_search.ingestion.registry import list_adapters
    assert "hcp" in list_adapters()
