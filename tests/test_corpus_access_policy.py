import pytest

from neural_search.services.corpus_access import CorpusAccessService


def _missing_corpus(_artifact_id: str):
    return {
        "id": "full_corpus_v09",
        "usable": False,
        "state": "missing_generated_asset",
        "absolute_path": "/missing/full_corpus_v09.jsonl",
    }


def test_demo_profile_can_use_portable_fallback(monkeypatch):
    monkeypatch.setattr(
        "neural_search.services.corpus_access.artifact_status",
        _missing_corpus,
    )
    records, source = CorpusAccessService(profile="demo").load()

    assert records
    assert source == "demo_fallback"


def test_researcher_profile_never_silently_uses_demo_fallback(monkeypatch):
    monkeypatch.setattr(
        "neural_search.services.corpus_access.artifact_status",
        _missing_corpus,
    )

    with pytest.raises(ValueError, match="will not silently substitute demo data"):
        CorpusAccessService(profile="researcher").load()


def test_fallback_can_be_explicitly_requested_by_non_demo_caller(monkeypatch):
    monkeypatch.setattr(
        "neural_search.services.corpus_access.artifact_status",
        _missing_corpus,
    )
    records, source = CorpusAccessService(profile="researcher").load(
        allow_demo_fallback=True
    )

    assert records
    assert source == "demo_fallback"
