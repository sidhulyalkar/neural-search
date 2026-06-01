"""Tests for graded usefulness benchmark."""
import json
import math
import pytest
from neural_search.evaluation.usefulness_benchmark import (
    UsefulnessLabel,
    UsefulnessQuery,
    PairLabel,
    compute_ndcg_at_k,
    compute_mrr,
    compute_precision_at_k,
    hard_negative_violation_rate,
    run_usefulness_benchmark,
    BenchmarkReport,
    GAIN,
)


class TestGainValues:
    def test_gain_ordering(self):
        assert GAIN["not_useful"] < GAIN["weakly_useful"] < GAIN["useful"] < GAIN["highly_useful"]

    def test_gain_nonnegative(self):
        for v in GAIN.values():
            assert v >= 0


class TestNDCG:
    def _make_labels(self):
        return {
            "c1": "highly_useful",
            "c2": "useful",
            "c3": "weakly_useful",
            "c4": "not_useful",
        }

    def test_perfect_ranking(self):
        labels = self._make_labels()
        ranked = ["c1", "c2", "c3", "c4"]
        score = compute_ndcg_at_k(ranked, labels, k=4)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_worst_ranking_below_one(self):
        labels = self._make_labels()
        ranked = ["c4", "c3", "c2", "c1"]
        score = compute_ndcg_at_k(ranked, labels, k=4)
        assert score < 1.0

    def test_empty_ranked_returns_zero(self):
        assert compute_ndcg_at_k([], {"c1": "useful"}, k=5) == pytest.approx(0.0)

    def test_k_truncation(self):
        labels = {"c1": "highly_useful", "c2": "not_useful"}
        score_k1 = compute_ndcg_at_k(["c1", "c2"], labels, k=1)
        score_k2 = compute_ndcg_at_k(["c1", "c2"], labels, k=2)
        assert score_k1 == pytest.approx(1.0)
        assert score_k2 == pytest.approx(1.0)


class TestMRR:
    def test_first_position_returns_one(self):
        labels = {"c1": "useful"}
        assert compute_mrr(["c1"], labels) == pytest.approx(1.0)

    def test_second_position_returns_half(self):
        labels = {"c1": "not_useful", "c2": "useful"}
        assert compute_mrr(["c1", "c2"], labels) == pytest.approx(0.5)

    def test_no_relevant_returns_zero(self):
        labels = {"c1": "not_useful"}
        assert compute_mrr(["c1"], labels) == pytest.approx(0.0)

    def test_weakly_useful_not_counted_as_relevant(self):
        labels = {"c1": "weakly_useful", "c2": "useful"}
        assert compute_mrr(["c1", "c2"], labels) == pytest.approx(0.5)


class TestPrecisionAtK:
    def test_all_useful_returns_one(self):
        labels = {"c1": "useful", "c2": "highly_useful"}
        assert compute_precision_at_k(["c1", "c2"], labels, k=2) == pytest.approx(1.0)

    def test_none_useful_returns_zero(self):
        labels = {"c1": "not_useful", "c2": "weakly_useful"}
        assert compute_precision_at_k(["c1", "c2"], labels, k=2) == pytest.approx(0.0)


class TestHardNegativeViolation:
    def test_hard_negative_ranked_first_is_violation(self):
        labels = {"hn1": "not_useful", "c1": "useful"}
        hard_negatives = {"hn1"}
        rate = hard_negative_violation_rate(["hn1", "c1"], labels, hard_negatives)
        assert rate == pytest.approx(1.0)

    def test_no_violations_when_hard_negatives_ranked_last(self):
        labels = {"c1": "useful", "hn1": "not_useful"}
        hard_negatives = {"hn1"}
        rate = hard_negative_violation_rate(["c1", "hn1"], labels, hard_negatives)
        assert rate == pytest.approx(0.0)

    def test_no_hard_negatives_returns_zero(self):
        labels = {"c1": "useful"}
        rate = hard_negative_violation_rate(["c1"], labels, set())
        assert rate == pytest.approx(0.0)


class TestRunBenchmark:
    def test_report_has_expected_attributes(self):
        queries = [
            UsefulnessQuery(
                query_id="q1",
                query="test query",
                intent="strict_lookup",
                candidate_ids=["c1", "c2", "c3"],
            )
        ]
        labels = [
            PairLabel(query_id="q1", candidate_id="c1", usefulness_label="highly_useful", label_type="reusable"),
            PairLabel(query_id="q1", candidate_id="c2", usefulness_label="useful", label_type="reusable"),
            PairLabel(query_id="q1", candidate_id="c3", usefulness_label="not_useful", label_type="reusable"),
        ]
        run = {"q1": ["c1", "c2", "c3"]}
        report = run_usefulness_benchmark(queries, labels, run, k=3)
        assert isinstance(report, BenchmarkReport)
        assert 0.0 <= report.ndcg_at_k <= 1.0
        assert 0.0 <= report.mrr <= 1.0
        assert 0.0 <= report.precision_at_k <= 1.0
        assert isinstance(report.per_intent_metrics, dict)

    def test_empty_labels_raises(self):
        with pytest.raises(ValueError, match="No labels"):
            run_usefulness_benchmark([], [], {})

    def test_to_markdown_contains_table(self):
        queries = [
            UsefulnessQuery(query_id="q1", query="test", intent="replication", candidate_ids=["c1"])
        ]
        labels = [PairLabel(query_id="q1", candidate_id="c1", usefulness_label="useful", label_type="reusable")]
        run = {"q1": ["c1"]}
        report = run_usefulness_benchmark(queries, labels, run, k=1)
        md = report.to_markdown()
        assert "|" in md
        assert "NDCG" in md or "ndcg" in md.lower()
