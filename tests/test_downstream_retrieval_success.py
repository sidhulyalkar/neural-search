"""Tests for compute_downstream_retrieval_success.py metrics and report generation."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _feedback_event(
    dataset_id: str = "ds001",
    usefulness: str = "useful",
    rank: int = 1,
    method: str = "bm25",
    judge_label: int | None = None,
    ec: float | None = None,
    saved: bool = False,
    exported: bool = False,
    would_use: str | None = None,
) -> dict:
    snap: dict = {}
    if judge_label is not None:
        snap["label"] = judge_label
    if ec is not None:
        snap["evidence_completeness"] = ec
    return {
        "feedback_id": f"fb_{dataset_id}",
        "query_text": "calcium imaging mouse hippocampus",
        "dataset_id": dataset_id,
        "dataset_title": f"Title {dataset_id}",
        "usefulness": usefulness,
        "rank": rank,
        "retrieval_method": method,
        "saved": saved,
        "exported": exported,
        "would_use_for_analysis": would_use,
        "reason_tags": [],
        "judge_snapshot": snap,
        "provenance": "user_feedback_downstream_signal",
    }


def _load_script():
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "compute_downstream_retrieval_success",
        Path(__file__).resolve().parents[1]
        / "scripts/eval/compute_downstream_retrieval_success.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# bucket helpers
# ---------------------------------------------------------------------------


class TestBucketHelpers:
    def setup_method(self):
        self.mod = _load_script()

    def test_bucket_rank_low(self):
        assert self.mod.bucket_rank(1) == "1-3"
        assert self.mod.bucket_rank(3) == "1-3"

    def test_bucket_rank_mid(self):
        assert self.mod.bucket_rank(5) == "4-10"
        assert self.mod.bucket_rank(10) == "4-10"

    def test_bucket_rank_high(self):
        assert self.mod.bucket_rank(11) == "11-50"
        assert self.mod.bucket_rank(51) == "51+"

    def test_bucket_rank_none(self):
        assert self.mod.bucket_rank(None) == "unknown"

    def test_bucket_completeness_low(self):
        assert self.mod.bucket_completeness(0.1) == "0.00-0.24"

    def test_bucket_completeness_mid(self):
        assert self.mod.bucket_completeness(0.5) == "0.50-0.74"

    def test_bucket_completeness_high(self):
        assert self.mod.bucket_completeness(0.9) == "0.75-1.00"

    def test_bucket_completeness_none(self):
        assert self.mod.bucket_completeness(None) == "unknown"

    def test_pct_zero_denominator(self):
        assert self.mod.pct(0, 0) == 0.0

    def test_pct_basic(self):
        assert self.mod.pct(3, 10) == 0.3


# ---------------------------------------------------------------------------
# Metrics computation tests
# ---------------------------------------------------------------------------


class TestMetricsComputation:
    def setup_method(self):
        self.mod = _load_script()

    def _run(
        self,
        feedback: list[dict],
        sessions: list[dict] | None = None,
        saved: list[dict] | None = None,
    ) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            fb_path = tmpdir / "feedback.jsonl"
            sess_path = tmpdir / "sessions.jsonl"
            saved_path = tmpdir / "saved.jsonl"
            json_out = tmpdir / "out.json"
            md_out = tmpdir / "out.md"

            _write_jsonl(fb_path, feedback)
            _write_jsonl(sess_path, sessions or [])
            _write_jsonl(saved_path, saved or [])

            self.mod.main(
                [
                    "--feedback", str(fb_path),
                    "--sessions", str(sess_path),
                    "--saved", str(saved_path),
                    "--out-json", str(json_out),
                    "--out-md", str(md_out),
                ]
            )
            return json.loads(json_out.read_text())

    def test_empty_feedback_produces_zero_metrics(self):
        report = self._run([])
        assert report["number_of_feedback_events"] == 0

    def test_usefulness_distribution(self):
        feedback = [
            _feedback_event("a", "useful"),
            _feedback_event("b", "not_useful"),
            _feedback_event("c", "unsure"),
        ]
        report = self._run(feedback)
        dist = report["usefulness_distribution"]
        assert dist.get("useful", 0) == 1
        assert dist.get("not_useful", 0) == 1

    def test_save_export_rate(self):
        feedback = [
            _feedback_event("a", "useful", saved=True),
            _feedback_event("b", "not_useful", saved=False),
        ]
        report = self._run(feedback)
        # save_export_rate = (save_events + export_events) / total
        assert report["save_export_rate"] == 0.5

    def test_would_use_rate(self):
        feedback = [
            _feedback_event("a", "useful", would_use="yes"),
            _feedback_event("b", "useful", would_use="maybe"),
            _feedback_event("c", "not_useful", would_use="no"),
        ]
        report = self._run(feedback)
        rate = report["would_use_for_analysis_rate"]
        assert rate == pytest.approx(2 / 3, abs=0.01)

    def test_false_high_feedback_detected(self):
        feedback = [
            _feedback_event("a", "not_useful", judge_label=3),
            _feedback_event("b", "useful", judge_label=0),
        ]
        report = self._run(feedback)
        assert report["false_high_feedback_count"] >= 1

    def test_false_low_feedback_detected(self):
        feedback = [
            _feedback_event("a", "useful", judge_label=0),
        ]
        report = self._run(feedback)
        assert report["false_low_feedback_count"] >= 1

    def test_usefulness_by_rank_bucket(self):
        feedback = [
            _feedback_event("a", "useful", rank=1),
            _feedback_event("b", "not_useful", rank=20),
        ]
        report = self._run(feedback)
        by_rank = report.get("usefulness_by_rank_bucket") or {}
        assert "1-3" in by_rank

    def test_usefulness_by_method(self):
        feedback = [
            _feedback_event("a", "useful", method="bm25"),
            _feedback_event("b", "not_useful", method="neural"),
        ]
        report = self._run(feedback)
        by_method = report.get("usefulness_by_retrieval_method") or {}
        assert "bm25" in by_method

    def test_markdown_has_disclaimer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            fb_path = tmpdir / "feedback.jsonl"
            _write_jsonl(fb_path, [_feedback_event("a", "useful")])
            md_out = tmpdir / "out.md"
            json_out = tmpdir / "out.json"
            self.mod.main([
                "--feedback", str(fb_path),
                "--sessions", str(tmpdir / "s.jsonl"),
                "--saved", str(tmpdir / "sv.jsonl"),
                "--out-json", str(json_out),
                "--out-md", str(md_out),
            ])
            md = md_out.read_text()
            assert "downstream" in md.lower() or "user_feedback" in md.lower()

    def test_report_provenance_label(self):
        feedback = [_feedback_event("a", "useful")]
        report = self._run(feedback)
        assert "downstream" in str(report.get("provenance") or "").lower()
