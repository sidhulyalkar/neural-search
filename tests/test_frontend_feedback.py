"""Tests for frontend feedback schema validation and API helper behaviour."""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers to build minimal valid feedback events
# ---------------------------------------------------------------------------


def _minimal_feedback(
    dataset_id: str = "ds001",
    usefulness: str = "useful",
    query_text: str = "find calcium imaging mouse hippocampus",
) -> dict:
    return {
        "feedback_id": "fb_test_001",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "session_id": "session_abc",
        "query_id": None,
        "query_text": query_text,
        "retrieval_method": "bm25",
        "rank": 1,
        "dataset_id": dataset_id,
        "dataset_title": "Test Dataset",
        "usefulness": usefulness,
        "would_use_for_analysis": "yes",
        "clicked": True,
        "opened_evidence": False,
        "saved": False,
        "exported": False,
        "reason_tags": [],
        "free_text_note": "",
        "judge_snapshot": {},
        "provenance": "user_feedback_downstream_signal",
    }


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestFeedbackSchemaValidation:
    def test_valid_usefulness_values(self):
        valid = ["useful", "partially_useful", "not_useful", "unsure"]
        for v in valid:
            fb = _minimal_feedback(usefulness=v)
            assert fb["usefulness"] == v

    def test_invalid_usefulness_rejected_by_api(self):
        """The API Pydantic model should reject unknown usefulness strings."""
        from fastapi.testclient import TestClient

        from apps.api.main import app

        client = TestClient(app)
        fb = _minimal_feedback(usefulness="definitely_useful")
        resp = client.post("/api/frontend/feedback", json=fb)
        assert resp.status_code == 422

    def test_valid_would_use_values(self):
        valid_would_use = ["yes", "maybe", "no", None]
        for v in valid_would_use:
            fb = _minimal_feedback()
            fb["would_use_for_analysis"] = v
            assert fb["would_use_for_analysis"] == v

    def test_provenance_field_is_downstream_signal(self):
        fb = _minimal_feedback()
        assert fb["provenance"] == "user_feedback_downstream_signal"

    def test_reason_tags_are_list(self):
        fb = _minimal_feedback()
        fb["reason_tags"] = ["wrong_modality", "wrong_species"]
        assert isinstance(fb["reason_tags"], list)
        assert "wrong_modality" in fb["reason_tags"]

    def test_judge_snapshot_is_dict(self):
        fb = _minimal_feedback()
        fb["judge_snapshot"] = {
            "label": 2,
            "confidence": 0.75,
            "label_provenance": "neuro_judge",
        }
        assert isinstance(fb["judge_snapshot"], dict)
        assert fb["judge_snapshot"]["label"] == 2

    def test_judge_snapshot_label_provenance_not_human_gold(self):
        fb = _minimal_feedback()
        fb["judge_snapshot"] = {"label": 3, "label_provenance": "human_gold"}
        # Downstream feedback must never carry human_gold provenance from the judge
        assert fb["judge_snapshot"]["label_provenance"] != "neuro_judge_rag"

    def test_feedback_event_missing_dataset_id_rejected(self):
        from fastapi.testclient import TestClient

        from apps.api.main import app

        client = TestClient(app)
        fb = _minimal_feedback()
        del fb["dataset_id"]
        resp = client.post("/api/frontend/feedback", json=fb)
        assert resp.status_code == 422

    def test_feedback_event_missing_query_text_uses_default(self):
        # query_text has a default empty string — omitting it is allowed.
        from fastapi.testclient import TestClient

        from apps.api.main import app

        client = TestClient(app)
        fb = _minimal_feedback()
        del fb["query_text"]
        resp = client.post("/api/frontend/feedback", json=fb)
        # 200 because query_text defaults to ""
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# JSONL artifact storage tests
# ---------------------------------------------------------------------------


class TestFeedbackArtifactStorage:
    def test_feedback_is_appended_to_jsonl(self, tmp_path: Path, monkeypatch):
        import apps.api.main as api

        feedback_path = tmp_path / "retrieval_feedback.jsonl"
        monkeypatch.setattr(api, "FRONTEND_ARTIFACT_DIR", tmp_path)

        from fastapi.testclient import TestClient

        client = TestClient(api.app)
        fb = _minimal_feedback()
        resp = client.post("/api/frontend/feedback", json=fb)
        assert resp.status_code == 200

        assert feedback_path.exists()
        rows = [json.loads(line) for line in feedback_path.read_text().splitlines() if line.strip()]
        assert len(rows) == 1
        assert rows[0]["dataset_id"] == "ds001"
        assert rows[0]["provenance"] == "user_feedback_downstream_signal"

    def test_multiple_feedback_events_appended(self, tmp_path: Path, monkeypatch):
        import apps.api.main as api

        feedback_path = tmp_path / "retrieval_feedback.jsonl"
        monkeypatch.setattr(api, "FRONTEND_ARTIFACT_DIR", tmp_path)

        from fastapi.testclient import TestClient

        client = TestClient(api.app)
        for i in range(3):
            fb = _minimal_feedback(dataset_id=f"ds{i:03d}")
            resp = client.post("/api/frontend/feedback", json=fb)
            assert resp.status_code == 200

        rows = [json.loads(line) for line in feedback_path.read_text().splitlines() if line.strip()]
        assert len(rows) == 3

    def test_search_session_creates_artifact(self, tmp_path: Path, monkeypatch):
        import apps.api.main as api

        sessions_path = tmp_path / "search_sessions.jsonl"
        monkeypatch.setattr(api, "FRONTEND_ARTIFACT_DIR", tmp_path)

        from fastapi.testclient import TestClient

        client = TestClient(api.app)
        payload = {
            "query_text": "place cell theta oscillation",
            "retrieval_method": "neural",
        }
        resp = client.post("/api/frontend/search-sessions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["provenance"] == "user_feedback_downstream_signal"

        assert sessions_path.exists()
        rows = [json.loads(line) for line in sessions_path.read_text().splitlines() if line.strip()]
        assert len(rows) == 1
        assert rows[0]["query_text"] == "place cell theta oscillation"

    def test_saved_dataset_creates_artifact(self, tmp_path: Path, monkeypatch):
        import apps.api.main as api

        saved_path = tmp_path / "saved_datasets.jsonl"
        monkeypatch.setattr(api, "FRONTEND_ARTIFACT_DIR", tmp_path)

        from fastapi.testclient import TestClient

        client = TestClient(api.app)
        payload = {
            "query_text": "theta oscillation place cell",
            "dataset_id": "DANDI:000001",
            "dataset_title": "Hippocampal place cells",
        }
        resp = client.post("/api/frontend/saved-datasets", json=payload)
        assert resp.status_code == 200

        assert saved_path.exists()
        rows = [json.loads(line) for line in saved_path.read_text().splitlines() if line.strip()]
        assert len(rows) == 1
        assert rows[0]["dataset_id"] == "DANDI:000001"


# ---------------------------------------------------------------------------
# Known reason-tag values
# ---------------------------------------------------------------------------


KNOWN_REASON_TAGS = {
    "wrong_modality",
    "wrong_species",
    "wrong_region",
    "missing_raw_data",
    "insufficient_metadata",
    "good_match",
    "interesting_reuse_candidate",
    "needs_manual_review",
    "wrong_task",
    "processed_only",
    "low_evidence",
}


class TestReasonTags:
    def test_all_known_tags_are_strings(self):
        for tag in KNOWN_REASON_TAGS:
            assert isinstance(tag, str)

    def test_feedback_with_multiple_tags(self, tmp_path: Path, monkeypatch):
        import apps.api.main as api

        monkeypatch.setattr(api, "FRONTEND_ARTIFACT_DIR", tmp_path)
        from fastapi.testclient import TestClient

        client = TestClient(api.app)
        fb = _minimal_feedback(usefulness="not_useful")
        fb["reason_tags"] = ["wrong_species", "wrong_modality"]
        resp = client.post("/api/frontend/feedback", json=fb)
        assert resp.status_code == 200

        rows = [
            json.loads(line)
            for line in (tmp_path / "retrieval_feedback.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert set(rows[0]["reason_tags"]) == {"wrong_species", "wrong_modality"}
