from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

CLAIMS = [
    {
        "claim_id": "node:claim:hippocampus_increase_abc12345",
        "statement": "Theta oscillations increase during spatial navigation",
        "direction": "increase",
        "regions": ["hippocampus"],
        "species": ["mouse"],
        "consensus_confidence": 0.87,
        "n_supporting_findings": 5,
        "n_contradicting_findings": 0,
        "magnitude_summary": "r=0.7",
        "timescale": "millisecond",
        "evidence_strength": "direct",
        "status": "active",
        "supporting_datasets": ["dandi:000026"],
        "supporting_papers": ["paper:openalex:W123"],
        "contradicted_by": [],
        "synthesis_model": "claude-haiku-4-5-20251001",
        "synthesis_prompt_version": "synthesis_v1",
        "synthesized_at": "2026-06-21T00:00:00+00:00",
    },
    {
        "claim_id": "node:claim:pfc_correlation_def67890",
        "statement": "Prefrontal theta correlates with working memory load",
        "direction": "correlation",
        "regions": ["prefrontal cortex"],
        "species": ["human"],
        "consensus_confidence": 0.75,
        "n_supporting_findings": 8,
        "n_contradicting_findings": 1,
        "magnitude_summary": "r=0.5",
        "timescale": "second",
        "evidence_strength": "direct",
        "status": "contested",
        "supporting_datasets": ["openneuro:ds000120"],
        "supporting_papers": ["paper:openalex:W456"],
        "contradicted_by": ["node:claim:pfc_no_change_ghi11111"],
        "synthesis_model": "claude-haiku-4-5-20251001",
        "synthesis_prompt_version": "synthesis_v1",
        "synthesized_at": "2026-06-21T00:00:00+00:00",
    },
]


@pytest.fixture
def claims_file(tmp_path) -> Path:
    p = tmp_path / "claims_validated.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in CLAIMS), encoding="utf-8")
    return p


@pytest.fixture
def client(claims_file, monkeypatch):
    from apps.api import claims_router
    monkeypatch.setattr(claims_router, "CLAIMS_PATH", claims_file)
    claims_router._claims_cache = None  # reset cache
    from fastapi import FastAPI

    from apps.api.claims_router import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_get_claims_returns_all(client):
    resp = client.get("/api/claims")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["claims"]) == 2


def test_get_claims_filter_by_direction(client):
    resp = client.get("/api/claims?direction=increase")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["claims"][0]["direction"] == "increase"


def test_get_claims_filter_by_region(client):
    resp = client.get("/api/claims?regions=hippocampus")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1


def test_get_claims_filter_by_min_confidence(client):
    resp = client.get("/api/claims?min_confidence=0.80")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["claims"][0]["consensus_confidence"] >= 0.80


def test_get_claim_by_id(client):
    # Colons are valid URL path chars — no encoding needed
    resp = client.get("/api/claims/node%3Aclaim%3Ahippocampus_increase_abc12345")
    assert resp.status_code == 200
    data = resp.json()
    assert data["claim_id"] == "node:claim:hippocampus_increase_abc12345"


def test_get_claim_by_id_not_found(client):
    resp = client.get("/api/claims/node%3Aclaim%3Anonexistent")
    assert resp.status_code == 404


def test_get_claim_evidence(client):
    resp = client.get("/api/claims/node%3Aclaim%3Ahippocampus_increase_abc12345/evidence")
    assert resp.status_code == 200
    data = resp.json()
    assert "supporting_datasets" in data
    assert "supporting_papers" in data
    assert "dandi:000026" in data["supporting_datasets"]


def test_get_contradictions(client):
    resp = client.get("/api/claims/contradictions")
    assert resp.status_code == 200
    data = resp.json()
    assert "contested_claims" in data


def test_get_digest_returns_compact_objects(client):
    resp = client.get("/api/claims/digest")
    assert resp.status_code == 200
    data = resp.json()
    assert "claims" in data
    assert len(data["claims"]) == 2
    # digest objects must have agent_digest field
    for c in data["claims"]:
        assert "agent_digest" in c
        assert "claim_id" in c
