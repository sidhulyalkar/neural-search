from fastapi.testclient import TestClient

from apps.api import main as api_main
from neural_search.ingestion.services import IngestionRunResult


def test_dandi_ingestion_endpoint_returns_service_result(monkeypatch):
    def fake_ingest(query: str, limit: int, *, save: bool, force: bool):
        assert query == "go no-go"
        assert limit == 2
        assert save is False
        assert force is False
        return IngestionRunResult(
            source="dandi",
            query=query,
            fetched=2,
            normalized=1,
            dataset_ids=["000001"],
        )

    monkeypatch.setattr(api_main.ingestion_services, "ingest_dandi", fake_ingest)

    with TestClient(api_main.app) as client:
        response = client.post(
            "/api/ingest/dandi",
            json={"query": "go no-go", "limit": 2},
        )

    assert response.status_code == 200
    assert response.json()["dataset_ids"] == ["000001"]
    assert response.json()["normalized"] == 1


def test_ingestion_endpoint_rejects_blank_query():
    with TestClient(api_main.app) as client:
        response = client.post(
            "/api/ingest/dandi",
            json={"query": "   ", "limit": 2},
        )

    assert response.status_code == 422
