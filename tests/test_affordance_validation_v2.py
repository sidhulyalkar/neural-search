"""Tests for affordance validation v2."""
import json
import tempfile
from pathlib import Path
import pytest
from neural_search.evaluation.affordance_validation_v2 import (
    AffordanceValidationV2,
    SyntheticCard,
    GroundTruthLabel,
    ValidationReport,
    run_validation,
)


def _make_cards():
    return [
        SyntheticCard(
            dataset_id="ds_001",
            predicted_affordances=["choice_decoding", "q_learning"],
            modalities=["neuropixels"],
            has_trials=True,
            has_timestamps=True,
        ),
        SyntheticCard(
            dataset_id="ds_002",
            predicted_affordances=["calcium_imaging"],
            modalities=["calcium_imaging"],
            has_trials=False,
            has_timestamps=False,
        ),
        SyntheticCard(
            dataset_id="ds_003",
            predicted_affordances=["choice_decoding"],
            modalities=["neuropixels"],
            has_trials=True,
            has_timestamps=True,
        ),
    ]


def _make_labels():
    return [
        GroundTruthLabel(dataset_id="ds_001", affordance="choice_decoding", supported=True),
        GroundTruthLabel(dataset_id="ds_001", affordance="q_learning", supported=True),
        GroundTruthLabel(dataset_id="ds_002", affordance="calcium_imaging", supported=False),
        GroundTruthLabel(dataset_id="ds_003", affordance="choice_decoding", supported=True),
    ]


class TestAffordanceValidationV2:
    def test_run_returns_report(self):
        cards = _make_cards()
        labels = _make_labels()
        report = run_validation(cards, labels)
        assert isinstance(report, ValidationReport)

    def test_report_has_precision_for_labeled_affordances(self):
        cards = _make_cards()
        labels = _make_labels()
        report = run_validation(cards, labels)
        assert "choice_decoding" in report.per_affordance_precision
        assert 0.0 <= report.per_affordance_precision["choice_decoding"] <= 1.0

    def test_confusion_table_populated(self):
        cards = _make_cards()
        labels = _make_labels()
        report = run_validation(cards, labels)
        assert isinstance(report.confusion_table, dict)
        for aff, table in report.confusion_table.items():
            assert "tp" in table
            assert "fp" in table
            assert "fn" in table

    def test_no_labels_produces_empty_precision(self):
        cards = _make_cards()
        labels = []
        report = run_validation(cards, labels)
        assert report.n_labeled == 0
        assert isinstance(report.per_affordance_precision, dict)

    def test_markdown_report_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cards = _make_cards()
            labels = _make_labels()
            out_path = Path(tmpdir) / "report.md"
            run_validation(cards, labels, out_path=out_path)
            assert out_path.exists()
            content = out_path.read_text()
            assert "Precision" in content or "precision" in content.lower()

    def test_json_report_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cards = _make_cards()
            labels = _make_labels()
            json_path = Path(tmpdir) / "results.json"
            run_validation(cards, labels, json_out_path=json_path)
            assert json_path.exists()
            data = json.loads(json_path.read_text())
            assert "per_affordance_precision" in data

    def test_true_positive_counted_correctly(self):
        cards = [SyntheticCard(dataset_id="d1", predicted_affordances=["choice_decoding"])]
        labels = [GroundTruthLabel(dataset_id="d1", affordance="choice_decoding", supported=True)]
        report = run_validation(cards, labels)
        ct = report.confusion_table.get("choice_decoding", {})
        assert ct.get("tp", 0) == 1
        assert ct.get("fp", 0) == 0

    def test_false_positive_counted_correctly(self):
        cards = [SyntheticCard(dataset_id="d1", predicted_affordances=["choice_decoding"])]
        labels = [GroundTruthLabel(dataset_id="d1", affordance="choice_decoding", supported=False)]
        report = run_validation(cards, labels)
        ct = report.confusion_table.get("choice_decoding", {})
        assert ct.get("tp", 0) == 0
        assert ct.get("fp", 0) == 1

    def test_false_negative_counted_correctly(self):
        cards = [SyntheticCard(dataset_id="d1", predicted_affordances=[])]
        labels = [GroundTruthLabel(dataset_id="d1", affordance="choice_decoding", supported=True)]
        report = run_validation(cards, labels)
        ct = report.confusion_table.get("choice_decoding", {})
        assert ct.get("fn", 0) == 1
        assert ct.get("tp", 0) == 0
