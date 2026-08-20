from fastapi.testclient import TestClient

from apps.api.application import app


def test_runtime_profile_status_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/runtime/status", params={"profile": "demo"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"]["name"] == "demo"
    assert payload["ready"] is True
    assert payload["health"] in {"ready", "degraded"}
    assert "compatibility" in payload


def test_runtime_artifact_endpoint_does_not_require_generated_assets():
    with TestClient(app) as client:
        response = client.get(
            "/api/runtime/artifacts",
            params={"artifact_id": "production_graph"},
        )

    assert response.status_code == 200
    artifact = response.json()["artifacts"][0]
    assert artifact["id"] == "production_graph"
    assert artifact["kind"] == "generated_local"
    assert "exists" in artifact
    assert "lineage_state" in artifact


def test_runtime_capabilities_surface_optional_scientific_substrates():
    with TestClient(app) as client:
        response = client.get(
            "/api/runtime/capabilities",
            params={"profile": "researcher"},
        )

    assert response.status_code == 200
    payload = response.json()
    names = {item["capability"] for item in payload["capabilities"]}
    assert "real_dataset_corpus" in names
    assert "knowledge_graph" in names
    assert "specter2_index" in names
    assert "neurosynth_enrichment" in names


def test_runtime_bundle_state_is_available_on_fresh_clone():
    with TestClient(app) as client:
        response = client.get("/api/runtime/bundles")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available_bundles"] == []
    assert payload["verification"]["valid"] is True


def test_runtime_unknown_profile_is_404():
    with TestClient(app) as client:
        response = client.get("/api/runtime/status", params={"profile": "unknown"})

    assert response.status_code == 404
