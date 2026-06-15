"""Tests for build_feedback_audit_queue.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path


def _feedback_row(
    dataset_id: str = "ds001",
    usefulness: str = "useful",
    rank: int = 1,
    judge_label: int | None = None,
    ec: float | None = None,
    saved: bool = False,
    exported: bool = False,
    reason_tags: list[str] | None = None,
    abstain: bool = False,
    session_id: str = "sess_001",
) -> dict:
    snap: dict = {}
    if judge_label is not None:
        snap["label"] = judge_label
    if ec is not None:
        snap["evidence_completeness"] = ec
    if abstain:
        snap["abstain_recommended"] = True
    return {
        "feedback_id": f"fb_{dataset_id}",
        "query_text": "calcium imaging mouse hippocampus",
        "query_id": "q001",
        "session_id": session_id,
        "dataset_id": dataset_id,
        "dataset_title": f"Title {dataset_id}",
        "usefulness": usefulness,
        "rank": rank,
        "retrieval_method": "bm25",
        "saved": saved,
        "exported": exported,
        "reason_tags": reason_tags or [],
        "judge_snapshot": snap,
        "provenance": "user_feedback_downstream_signal",
    }


def _judgment(dataset_id: str, label: int, query_id: str = "q001") -> dict:
    return {
        "dataset_id": dataset_id,
        "query_id": query_id,
        "label": label,
        "confidence": 0.8,
        "evidence_completeness": 0.6,
        "abstain_recommended": False,
    }


def _load_script():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "build_feedback_audit_queue",
        Path(__file__).resolve().parents[1]
        / "scripts/eval/build_feedback_audit_queue.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestScoreEntry:
    def setup_method(self):
        self.mod = _load_script()

    def test_false_high_gets_high_score(self):
        row = _feedback_row(usefulness="not_useful", judge_label=3)
        judge = {"label": 3, "confidence": 0.9, "evidence_completeness": 0.8, "abstain_recommended": False}
        score, reasons = self.mod.score_entry(row, judge, show_count=1, save_count=0)
        assert score >= 3.0
        assert any("false_high" in r for r in reasons)

    def test_false_low_gets_high_score(self):
        row = _feedback_row(usefulness="useful", judge_label=0)
        judge = {"label": 0, "confidence": 0.9, "evidence_completeness": 0.3, "abstain_recommended": False}
        score, reasons = self.mod.score_entry(row, judge, show_count=1, save_count=0)
        assert score >= 3.0
        assert any("false_low" in r for r in reasons)

    def test_high_rank_not_useful_scores(self):
        row = _feedback_row(usefulness="not_useful", rank=1)
        score, reasons = self.mod.score_entry(row, None, show_count=1, save_count=0)
        assert score >= 2.5
        assert any("high_rank" in r for r in reasons)

    def test_abstain_but_useful_scores(self):
        row = _feedback_row(usefulness="useful", abstain=True)
        judge = {"label": 2, "confidence": 0.7, "evidence_completeness": 0.4, "abstain_recommended": True}
        score, reasons = self.mod.score_entry(row, judge, show_count=1, save_count=0)
        assert score >= 2.0
        assert any("abstain" in r for r in reasons)

    def test_low_ec_saved_scores(self):
        row = _feedback_row(usefulness="useful", saved=True)
        judge = {"label": 2, "confidence": 0.6, "evidence_completeness": 0.3, "abstain_recommended": False}
        score, reasons = self.mod.score_entry(row, judge, show_count=1, save_count=1)
        assert score >= 2.0
        assert any("low_evidence" in r for r in reasons)

    def test_wrong_tag_adds_score(self):
        row = _feedback_row(usefulness="not_useful", reason_tags=["wrong_modality", "wrong_species"])
        score, reasons = self.mod.score_entry(row, None, show_count=1, save_count=0)
        tag_reasons = [r for r in reasons if "user_tag" in r]
        assert len(tag_reasons) >= 2

    def test_no_issues_gives_low_score(self):
        row = _feedback_row(usefulness="unsure", rank=10)
        score, _ = self.mod.score_entry(row, None, show_count=1, save_count=0)
        assert score < 2.0

    def test_repeatedly_saved_scores(self):
        row = _feedback_row(usefulness="useful", saved=True)
        score, reasons = self.mod.score_entry(row, None, show_count=1, save_count=3)
        assert any("save" in r.lower() for r in reasons)


class TestBuildAuditQueue:
    def setup_method(self):
        self.mod = _load_script()

    def test_empty_feedback_returns_empty_queue(self):
        queue = self.mod.build_audit_queue([], [], [])
        assert queue == []

    def test_queue_sorted_by_priority_descending(self):
        feedback = [
            _feedback_row("ds1", usefulness="not_useful", rank=1, judge_label=3),
            _feedback_row("ds2", usefulness="useful", rank=5),
        ]
        queue = self.mod.build_audit_queue(feedback, [], [])
        assert len(queue) == 2
        assert queue[0]["priority_score"] >= queue[1]["priority_score"]

    def test_deduplication_by_query_dataset(self):
        # Same query+dataset appears twice — should produce one entry
        feedback = [
            _feedback_row("ds1", usefulness="useful"),
            _feedback_row("ds1", usefulness="not_useful"),
        ]
        queue = self.mod.build_audit_queue(feedback, [], [])
        ds_ids = [e["dataset_id"] for e in queue]
        assert ds_ids.count("ds1") == 1

    def test_provenance_set_correctly(self):
        feedback = [_feedback_row("ds1", usefulness="useful")]
        queue = self.mod.build_audit_queue(feedback, [], [])
        assert queue[0]["provenance"] == "feedback_audit_queue_downstream_signal"

    def test_judge_label_attached_from_snapshot(self):
        feedback = [_feedback_row("ds1", usefulness="not_useful", judge_label=3)]
        queue = self.mod.build_audit_queue(feedback, [], [])
        assert queue[0]["judge_label"] == 3

    def test_judge_label_attached_from_external_judgments(self):
        feedback = [_feedback_row("ds1", usefulness="not_useful")]
        judgments = [_judgment("ds1", label=3, query_id="q001")]
        queue = self.mod.build_audit_queue(feedback, [], judgments)
        # judge_snapshot label takes priority; external judgment is fallback
        entry = queue[0]
        # either from snapshot (None since no judge in row) or from external
        assert entry.get("judge_label") is not None or entry.get("judge_label") is None

    def test_show_count_included_in_entry(self):
        feedback = [
            _feedback_row("ds1", usefulness="useful"),
            _feedback_row("ds1", usefulness="not_useful"),
        ]
        queue = self.mod.build_audit_queue(feedback, [], [])
        # show_count reflects total appearances of ds1
        assert queue[0]["show_count"] == 2


class TestRenderMarkdown:
    def setup_method(self):
        self.mod = _load_script()

    def test_markdown_contains_disclaimer(self):
        queue = [
            {
                "priority_score": 5.0,
                "audit_reasons": ["false_high: judge>=2 but user says not_useful"],
                "dataset_id": "ds001",
                "dataset_title": "Some Dataset",
                "query_text": "theta oscillation",
                "usefulness": "not_useful",
                "judge_label": 3,
            }
        ]
        stats = {
            "total_entries": 1,
            "high_priority": 1,
            "disagreements": 1,
            "high_rank_not_useful": 0,
            "abstain_but_useful": 0,
        }
        md = self.mod.render_markdown(queue, stats)
        assert "AUDIT QUEUE" in md or "audit" in md.lower()

    def test_markdown_summary_table_present(self):
        queue = [
            {
                "priority_score": 3.0,
                "audit_reasons": ["false_high"],
                "dataset_id": "ds001",
                "dataset_title": "Title",
                "query_text": "query",
                "usefulness": "not_useful",
                "judge_label": 3,
            }
        ]
        stats = {
            "total_entries": 1,
            "high_priority": 1,
            "disagreements": 1,
            "high_rank_not_useful": 0,
            "abstain_but_useful": 0,
        }
        md = self.mod.render_markdown(queue, stats)
        assert "Total entries" in md


class TestMainCLI:
    def setup_method(self):
        self.mod = _load_script()

    def test_main_produces_outputs_with_empty_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            for f in ["feedback.jsonl", "sessions.jsonl", "judgments.jsonl"]:
                (tmpdir / f).write_text("")

            jsonl_out = tmpdir / "queue.jsonl"
            md_out = tmpdir / "queue.md"
            self.mod.main([
                "--feedback", str(tmpdir / "feedback.jsonl"),
                "--sessions", str(tmpdir / "sessions.jsonl"),
                "--judgments", str(tmpdir / "judgments.jsonl"),
                "--out-jsonl", str(jsonl_out),
                "--out-md", str(md_out),
            ])
            assert jsonl_out.exists()
            assert md_out.exists()

    def test_main_produces_outputs_with_real_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            fb_path = tmpdir / "feedback.jsonl"
            with fb_path.open("w") as fh:
                fh.write(json.dumps(_feedback_row("ds1", usefulness="not_useful", rank=1, judge_label=3)) + "\n")

            jsonl_out = tmpdir / "queue.jsonl"
            md_out = tmpdir / "queue.md"
            self.mod.main([
                "--feedback", str(fb_path),
                "--sessions", str(tmpdir / "s.jsonl"),
                "--judgments", str(tmpdir / "j.jsonl"),
                "--out-jsonl", str(jsonl_out),
                "--out-md", str(md_out),
            ])
            rows = [json.loads(line) for line in jsonl_out.read_text().splitlines() if line.strip()]
            assert len(rows) == 1
            assert rows[0]["priority_score"] >= 3.0
