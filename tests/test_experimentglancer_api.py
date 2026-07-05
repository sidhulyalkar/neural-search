from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api import experimentglancer_router as eg_router
from neural_search.experimentglancer import persistence

CALCIUM_DATASET = {
    "id": "dandi:000100",
    "source": "dandi",
    "source_id": "000100",
    "title": "Two-photon calcium imaging during visual change detection",
    "description": (
        "Two-photon calcium imaging in visual cortex during a change-detection "
        "task with lick events and trial structure."
    ),
    "modalities": ["calcium_imaging"],
    "tasks": ["visual_change_detection"],
    "behaviors": ["licking"],
    "species": ["mouse"],
    "brain_regions": ["visual_cortex"],
    "data_standard": "NWB",
    "url": "https://dandiarchive.org/dandiset/000100",
    "has_raw_data": True,
    "has_processed_data": True,
    "has_trials": True,
}

RECORD = {"dataset": CALCIUM_DATASET, "papers": []}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(eg_router, "_demo_data", [RECORD])
    monkeypatch.setattr(persistence, "SCENES_DIR", tmp_path / "scenes")
    eg_router._SCENE_CACHE.clear()
    app = FastAPI()
    app.include_router(eg_router.router)
    return TestClient(app)


def test_dataset_scene_endpoint_works_for_demo_dataset(client):
    resp = client.get("/api/experimentglancer/datasets/dandi:000100/scene")
    assert resp.status_code == 200
    data = resp.json()
    assert data["scene"]["dataset"]["dataset_id"] == "dandi:000100"
    assert data["scene_url"] == f"/experimentglancer?scene_id={data['scene']['scene_id']}"
    kinds = {layer["kind"] for layer in data["scene"]["layers"]}
    assert "neural.calcium" in kinds


def test_dataset_scene_unknown_dataset_returns_404(client):
    resp = client.get("/api/experimentglancer/datasets/dandi:999999/scene")
    assert resp.status_code == 404


def test_dataset_introspection_endpoint(client):
    resp = client.get("/api/experimentglancer/datasets/dandi:000100/introspection")
    assert resp.status_code == 200
    data = resp.json()
    assert data["dataset_id"] == "dandi:000100"
    assert data["resolver"] == "metadata_only"


def test_introspection_unknown_dataset_returns_404(client):
    resp = client.get("/api/experimentglancer/datasets/dandi:999999/introspection")
    assert resp.status_code == 404


def test_from_search_result_endpoint(client):
    resp = client.post(
        "/api/experimentglancer/scenes/from-search-result",
        json={
            "query": "visual change detection with lick events",
            "dataset_id": "dandi:000100",
            "rank": 1,
            "retrieval_method": "hybrid_search",
            "score": 0.87,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["scene"]["source"]["kind"] == "search_result"
    assert data["scene"]["source"]["score"] == 0.87
    assert data["scene"]["source"]["rank"] == 1


def test_from_search_result_unknown_dataset_returns_404(client):
    resp = client.post(
        "/api/experimentglancer/scenes/from-search-result",
        json={"query": "test", "dataset_id": "dandi:999999"},
    )
    assert resp.status_code == 404


def test_from_search_result_respects_anchor_hint(client):
    resp = client.post(
        "/api/experimentglancer/scenes/from-search-result",
        json={
            "query": "confident model but failed trials",
            "dataset_id": "dandi:000100",
            "anchor_hint": {"kind": "event", "event_type": "lick_onset", "relative_time": -2.3},
        },
    )
    assert resp.status_code == 200
    anchor = resp.json()["scene"]["anchors"][0]
    assert anchor["event_type"] == "lick_onset"
    assert anchor["time"] is None


def test_from_search_result_excludes_probable_layers_when_requested(client):
    resp = client.post(
        "/api/experimentglancer/scenes/from-search-result",
        json={
            "query": "calcium imaging",
            "dataset_id": "dandi:000100",
            "include_probable_layers": False,
        },
    )
    assert resp.status_code == 200
    statuses = {layer["status"] for layer in resp.json()["scene"]["layers"]}
    assert "probable" not in statuses


def test_get_scene_by_id_round_trips(client):
    create_resp = client.post(
        "/api/experimentglancer/scenes/from-search-result",
        json={"query": "calcium imaging", "dataset_id": "dandi:000100"},
    )
    scene_id = create_resp.json()["scene"]["scene_id"]

    get_resp = client.get(f"/api/experimentglancer/scenes/{scene_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["scene"]["scene_id"] == scene_id


def test_get_unknown_scene_returns_404(client):
    resp = client.get("/api/experimentglancer/scenes/eg_does_not_exist")
    assert resp.status_code == 404


def test_scene_survives_in_memory_cache_eviction(client):
    """A shared URL must keep resolving after the in-memory cache entry is
    gone (TTL expiry or a process restart) -- the scene is rehydrated from
    disk. See apps/api/experimentglancer_router.py `_get_cached_scene`."""

    create_resp = client.post(
        "/api/experimentglancer/scenes/from-search-result",
        json={"query": "calcium imaging", "dataset_id": "dandi:000100"},
    )
    scene_id = create_resp.json()["scene"]["scene_id"]

    eg_router._SCENE_CACHE.clear()

    get_resp = client.get(f"/api/experimentglancer/scenes/{scene_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["scene"]["scene_id"] == scene_id


def test_different_requested_layers_get_different_scene_ids(client):
    """Two calls that differ only in `requested_layers`/`affordance_ids`
    must not collapse onto the same scene_id -- otherwise a shared URL could
    resolve to a scene the requester didn't ask for."""

    base = client.post(
        "/api/experimentglancer/scenes/from-search-result",
        json={"query": "calcium imaging", "dataset_id": "dandi:000100"},
    )
    with_layers = client.post(
        "/api/experimentglancer/scenes/from-search-result",
        json={
            "query": "calcium imaging",
            "dataset_id": "dandi:000100",
            "requested_layers": ["video.frames"],
            "affordance_ids": ["choice_decoding"],
        },
    )

    assert base.json()["scene"]["scene_id"] != with_layers.json()["scene"]["scene_id"]
