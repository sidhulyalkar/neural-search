"""Tests for the learning signal generator."""

from __future__ import annotations

import pytest

from neural_search.field_state.learning_signals import (
    VALID_SIGNAL_TYPES,
    LearningSignal,
    LearningSignalGenerator,
)


class TestLearningSignalValidation:
    def test_valid_signal_type(self) -> None:
        sig = LearningSignal(
            signal_type="false_high_candidate",
            dataset_id="dataset:dandi:000001",
        )
        assert sig.signal_type == "false_high_candidate"

    def test_invalid_signal_type_raises(self) -> None:
        with pytest.raises(ValueError):
            LearningSignal(signal_type="invalid_type", dataset_id="dataset:dandi:000001")

    def test_to_dict_has_required_fields(self) -> None:
        sig = LearningSignal(
            signal_type="missing_metadata_candidate",
            dataset_id="dataset:dandi:000001",
            evidence=["missing_modality"],
        )
        d = sig.to_dict()
        assert d["signal_type"] == "missing_metadata_candidate"
        assert d["provenance"] != "human_gold"
        assert "created_at" in d
        assert "signal_version" in d

    def test_all_valid_types_accepted(self) -> None:
        for stype in VALID_SIGNAL_TYPES:
            sig = LearningSignal(signal_type=stype, dataset_id="dataset:test:1")
            assert sig.signal_type == stype


class TestFeedbackProcessing:
    def _feedback_events(self, dataset_id: str, n_not_useful: int, n_useful: int) -> list[dict]:
        events = []
        for _ in range(n_not_useful):
            events.append({
                "feedback_id": f"fb_{len(events)}",
                "dataset_id": dataset_id,
                "usefulness": "not_useful",
                "reason_tags": ["wrong_modality"],
            })
        for _ in range(n_useful):
            events.append({
                "feedback_id": f"fb_{len(events)}",
                "dataset_id": dataset_id,
                "usefulness": "useful",
                "reason_tags": ["good_match"],
            })
        return events

    def test_high_not_useful_rate_generates_false_high(self) -> None:
        gen = LearningSignalGenerator()
        events = self._feedback_events("dataset:dandi:1", n_not_useful=4, n_useful=0)
        gen.process_feedback(events)
        types = [s.signal_type for s in gen.signals]
        assert "false_high_candidate" in types

    def test_high_useful_rate_generates_success_signal(self) -> None:
        gen = LearningSignalGenerator()
        events = self._feedback_events("dataset:dandi:2", n_not_useful=0, n_useful=5)
        gen.process_feedback(events)
        types = [s.signal_type for s in gen.signals]
        assert "dataset_reuse_success_signal" in types

    def test_too_few_events_skipped(self) -> None:
        gen = LearningSignalGenerator()
        events = self._feedback_events("dataset:dandi:3", n_not_useful=1, n_useful=0)
        gen.process_feedback(events)
        assert len(gen.signals) == 0

    def test_missing_tags_generate_missing_metadata(self) -> None:
        gen = LearningSignalGenerator()
        events = [
            {"feedback_id": "fb1", "dataset_id": "dataset:d:1", "usefulness": "not_useful", "reason_tags": ["missing_raw_data"]},
            {"feedback_id": "fb2", "dataset_id": "dataset:d:1", "usefulness": "not_useful", "reason_tags": ["missing_raw_data"]},
        ]
        gen.process_feedback(events)
        types = [s.signal_type for s in gen.signals]
        assert "missing_metadata_candidate" in types


class TestJudgmentProcessing:
    def test_human_gold_judgment_skipped(self) -> None:
        gen = LearningSignalGenerator()
        jmts = [{
            "query_id": "q1",
            "dataset_id": "dataset:dandi:000001",
            "label": 3,
            "label_provenance": "human_gold",
        }]
        gen.process_judgments(jmts)
        assert len(gen.signals) == 0

    def test_hard_negative_generates_signal(self) -> None:
        gen = LearningSignalGenerator()
        jmts = [{
            "query_id": "q1",
            "dataset_id": "dataset:dandi:000001",
            "label": 2,
            "label_provenance": "neuro_judge_silver",
            "hard_negative_detected": True,
            "confidence": 0.8,
            "evidence_completeness": 0.9,
        }]
        gen.process_judgments(jmts)
        types = [s.signal_type for s in gen.signals]
        assert "hard_negative_candidate" in types

    def test_low_evidence_completeness_generates_missing_metadata(self) -> None:
        gen = LearningSignalGenerator()
        jmts = [{
            "query_id": "q1",
            "dataset_id": "dataset:dandi:000001",
            "label": 1,
            "label_provenance": "neuro_judge_silver",
            "hard_negative_detected": False,
            "confidence": 0.6,
            "evidence_completeness": 0.2,
            "abstain_recommended": False,
        }]
        gen.process_judgments(jmts)
        types = [s.signal_type for s in gen.signals]
        assert "missing_metadata_candidate" in types

    def test_judge_user_disagreement_generates_false_low(self) -> None:
        gen = LearningSignalGenerator()
        jmts = [{
            "query_id": "q1",
            "dataset_id": "dataset:dandi:000001",
            "label": 0,
            "label_provenance": "neuro_judge_silver",
            "hard_negative_detected": False,
            "confidence": 0.7,
            "evidence_completeness": 0.9,
        }]
        feedback = [
            {"feedback_id": f"fb{i}", "dataset_id": "dataset:dandi:000001", "usefulness": "useful", "reason_tags": []}
            for i in range(3)
        ]
        gen.process_judgments(jmts, feedback_records=feedback)
        types = [s.signal_type for s in gen.signals]
        assert "false_low_candidate" in types


class TestAuditPriorityQueue:
    def test_priority_queue_ranked_by_score(self) -> None:
        gen = LearningSignalGenerator()
        gen.signals = [
            LearningSignal("false_high_candidate", "dataset:a:1", score=0.9),
            LearningSignal("false_high_candidate", "dataset:a:1", score=0.8),
            LearningSignal("missing_metadata_candidate", "dataset:b:2", score=0.3),
        ]
        queue = gen.compute_audit_priority_queue()
        assert queue[0]["dataset_id"] == "dataset:a:1"  # highest aggregate score
        assert queue[0]["priority_score"] > queue[1]["priority_score"]

    def test_signal_types_aggregated(self) -> None:
        gen = LearningSignalGenerator()
        gen.signals = [
            LearningSignal("false_high_candidate", "dataset:a:1"),
            LearningSignal("hard_negative_candidate", "dataset:a:1"),
        ]
        queue = gen.compute_audit_priority_queue()
        assert len(queue) == 1
        assert "false_high_candidate" in queue[0]["signal_types"]
        assert "hard_negative_candidate" in queue[0]["signal_types"]


class TestLearningSignalExport:
    def test_export_creates_file(self, tmp_path) -> None:
        gen = LearningSignalGenerator()
        gen.signals = [
            LearningSignal("false_high_candidate", "dataset:a:1", evidence=["wrong_modality"]),
        ]
        path = tmp_path / "signals.jsonl"
        count = gen.export_signals(path)
        assert count == 1
        assert path.exists()
        content = path.read_text()
        assert "false_high_candidate" in content
        assert "human_gold" not in content

    def test_report_renders(self) -> None:
        gen = LearningSignalGenerator()
        gen.signals = [
            LearningSignal("false_high_candidate", "dataset:a:1"),
            LearningSignal("dataset_reuse_success_signal", "dataset:b:2"),
        ]
        report = gen.render_report()
        assert "Learning Signals Report" in report
        assert "false_high_candidate" in report
        assert "provisional" in report.lower()
