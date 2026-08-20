import pytest

from neural_search.services import RuntimeReadinessService


def test_runtime_service_defaults_to_demo(monkeypatch):
    monkeypatch.delenv("NEURAL_SEARCH_PROFILE", raising=False)
    service = RuntimeReadinessService()

    assert service.active_profile() == "demo"
    assert service.status()["profile"]["name"] == "demo"


def test_runtime_service_falls_back_from_invalid_environment_profile(monkeypatch):
    monkeypatch.setenv("NEURAL_SEARCH_PROFILE", "not-a-profile")
    service = RuntimeReadinessService()

    assert service.active_profile() == "demo"


def test_runtime_service_rejects_explicit_unknown_profile():
    service = RuntimeReadinessService()

    with pytest.raises(ValueError, match="Unknown execution profile"):
        service.status("not-a-profile")


def test_runtime_service_exposes_artifact_state():
    service = RuntimeReadinessService()

    payload = service.artifacts("demo_datasets")
    artifact = payload["artifacts"][0]
    assert artifact["id"] == "demo_datasets"
    assert artifact["exists"] is True
