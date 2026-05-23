from fastapi.testclient import TestClient

from apps.api.main import app


def test_core_api_smoke_flow():
    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        search = client.post(
            "/api/search",
            json={"query": "go/no-go calcium imaging", "limit": 2},
        )
        assert search.status_code == 200
        search_payload = search.json()
        assert search_payload["total_count"] >= 1

        dataset_id = search_payload["results"][0]["dataset"]["source_id"]

        dataset = client.get(f"/api/datasets/{dataset_id}")
        assert dataset.status_code == 200
        assert dataset.json()["source_id"] == dataset_id

        card = client.get(f"/api/datasets/{dataset_id}/card")
        assert card.status_code == 200
        assert card.json()["dataset_id"] == dataset_id

        datasets = client.get("/api/datasets?limit=2")
        assert datasets.status_code == 200
        listed_ids = [item["source_id"] for item in datasets.json()["datasets"]]

        comparison = client.post(
            "/api/datasets/compare",
            json={"dataset_ids": listed_ids[:2]},
        )
        assert comparison.status_code == 200
        assert len(comparison.json()["dataset_ids"]) == 2

        report = client.get("/api/reports/compilation")
        assert report.status_code == 200
        assert report.json()["total_datasets"] >= 1
