"""Tests for ingestion adapter registry."""


def test_registry_list_empty_initially():
    from neural_search.ingestion.registry import list_adapters
    adapters = list_adapters()
    assert isinstance(adapters, list)


def test_register_and_run():
    from neural_search.ingestion.registry import register, run_adapter

    @register("test_source", extra_kwarg="hello")
    def _fetch_test(limit: int = 10, extra_kwarg: str = "") -> list[dict]:
        return [{"id": f"rec{i}", "extra": extra_kwarg} for i in range(limit)]

    results = run_adapter("test_source", limit=3)
    assert len(results) == 3
    assert results[0]["extra"] == "hello"


def test_unknown_adapter_raises():
    from neural_search.ingestion.registry import run_adapter
    try:
        run_adapter("definitely_not_registered_xyz123", limit=1)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "definitely_not_registered_xyz123" in str(e)
