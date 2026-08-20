from fastapi.testclient import TestClient

from apps.api.application import app


def test_v2_search_uses_service_backed_active_corpus():
    with TestClient(app) as client:
        response = client.post(
            "/api/v2/search",
            json={"query": "visual decision Neuropixels", "limit": 3},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "visual decision Neuropixels"
    assert payload["results"]
    assert payload["runtime_context"]["corpus_source"] in {
        "demo_fallback",
        "full_corpus_v09",
    }
    assert isinstance(payload["runtime_context"]["dataset_count"], int)


def test_v2_literature_distinguishes_missing_assets_from_zero_results():
    with TestClient(app) as client:
        response = client.post(
            "/api/v2/literature/search",
            json={"query": "reversal learning", "limit": 3},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "reversal learning"
    assert "papers" in payload
    assert "findings" in payload
    assert "source_state" in payload
    assert "warnings" in payload


def test_reanalysis_endpoint_operates_on_portable_demo_corpus():
    with TestClient(app) as client:
        response = client.get(
            "/api/reanalysis/DEMO_VISUAL_DECISION_NEUROPIXELS",
            params={"limit": 5},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"].lower().endswith(
        "demo_visual_decision_neuropixels"
    )
    assert payload["candidates"]
    assert all(
        candidate["requires_human_review"] for candidate in payload["candidates"]
    )
    assert "evidence_policy" in payload
    assert payload["corpus_source"] in {"demo_fallback", "full_corpus_v09"}


def test_adoption_event_endpoint_is_disabled_by_default(tmp_path, monkeypatch):
    path = tmp_path / "adoption-events.jsonl"
    monkeypatch.setenv("NEURAL_SEARCH_ADOPTION_EVENTS", str(path))
    monkeypatch.delenv("NEURAL_SEARCH_ADOPTION_STUDY", raising=False)

    with TestClient(app) as client:
        status = client.get("/api/adoption/status")
        recorded = client.post(
            "/api/adoption/events",
            json={
                "session_id": "test-session",
                "timestamp": "2026-08-19T12:00:00-07:00",
                "event_type": "workflow_complete",
                "success": True,
            },
        )

    assert status.status_code == 200
    assert status.json()["enabled"] is False
    assert recorded.status_code == 403
    assert not path.exists()


def test_adoption_event_endpoint_is_local_and_reportable(tmp_path, monkeypatch):
    path = tmp_path / "adoption-events.jsonl"
    monkeypatch.setenv("NEURAL_SEARCH_ADOPTION_EVENTS", str(path))
    monkeypatch.setenv("NEURAL_SEARCH_ADOPTION_STUDY", "1")
    event = {
        "session_id": "test-session",
        "timestamp": "2026-08-19T12:00:00-07:00",
        "event_type": "workflow_complete",
        "success": True,
    }

    with TestClient(app) as client:
        recorded = client.post("/api/adoption/events", json=event)
        report = client.get("/api/adoption/report")

    assert recorded.status_code == 200
    assert recorded.json()["recorded"] is True
    assert path.is_file()
    assert report.status_code == 200
    assert report.json()["sessions"] == 1
    assert report.json()["metrics"]["workflow_completion_rate"] == 1.0
