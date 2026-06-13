"""Unit tests for IR metrics and calibration computations."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from scripts.eval.compute_ir_metrics import (
    dcg, ndcg_at_k, mrr, recall_at_k, precision_at_k, mean,
    hard_negative_violation_rate, source_skew_at_k, bootstrap_ci,
    aggregate_metrics, compute_query_metrics, load_qrels,
)
from scripts.eval.compute_calibration import compute_ece


# ---------------------------------------------------------------------------
# DCG / NDCG
# ---------------------------------------------------------------------------

class TestDCG:
    def test_zero_gains_is_zero(self):
        assert dcg([0, 0, 0]) == 0.0

    def test_perfect_gain_first(self):
        val = dcg([3, 0, 0])
        expected = (2**3 - 1) / math.log2(2)
        assert abs(val - expected) < 1e-9

    def test_gain_decreases_with_rank(self):
        val_first = dcg([3, 0])
        val_second = dcg([0, 3])
        assert val_first > val_second


class TestNDCGAtK:
    def test_perfect_ranking_is_one(self):
        qrel = {"a": 3, "b": 2, "c": 1}
        ranked = ["a", "b", "c"]
        assert abs(ndcg_at_k(qrel, ranked, k=3) - 1.0) < 1e-9

    def test_empty_qrel_is_zero(self):
        assert ndcg_at_k({}, ["a", "b"], k=10) == 0.0

    def test_all_irrelevant_is_zero(self):
        qrel = {"a": 3}
        ranked = ["x", "y", "z"]
        assert ndcg_at_k(qrel, ranked, k=10) == 0.0

    def test_reversed_ranking_lower_than_perfect(self):
        qrel = {"a": 3, "b": 1}
        perfect = ndcg_at_k(qrel, ["a", "b"], k=2)
        reversed_ = ndcg_at_k(qrel, ["b", "a"], k=2)
        assert perfect > reversed_

    def test_truncates_at_k(self):
        qrel = {"a": 3}
        # a is at position 11, beyond k=10
        ranked = ["x"] * 10 + ["a"]
        assert ndcg_at_k(qrel, ranked, k=10) == 0.0


class TestMRR:
    def test_first_relevant_rank_one(self):
        qrel = {"a": 2}
        assert abs(mrr(qrel, ["a", "b"]) - 1.0) < 1e-9

    def test_first_relevant_rank_two(self):
        qrel = {"a": 2}
        assert abs(mrr(qrel, ["b", "a"]) - 0.5) < 1e-9

    def test_no_relevant_is_zero(self):
        qrel = {"a": 0, "b": 1}
        assert mrr(qrel, ["a", "b"]) == 0.0

    def test_label_below_threshold_not_counted(self):
        qrel = {"a": 1}
        assert mrr(qrel, ["a"]) == 0.0


class TestRecallAtK:
    def test_all_relevant_returned(self):
        qrel = {"a": 2, "b": 3}
        assert abs(recall_at_k(qrel, ["a", "b", "c"], k=50) - 1.0) < 1e-9

    def test_none_returned_is_zero(self):
        qrel = {"a": 2}
        assert recall_at_k(qrel, ["b", "c"], k=50) == 0.0

    def test_partial_recall(self):
        qrel = {"a": 2, "b": 3}
        val = recall_at_k(qrel, ["a"], k=50)
        assert abs(val - 0.5) < 1e-9

    def test_empty_qrel_is_zero(self):
        assert recall_at_k({}, ["a", "b"], k=50) == 0.0

    def test_only_counts_relevant_threshold(self):
        qrel = {"a": 0, "b": 1, "c": 2}
        val = recall_at_k(qrel, ["c"], k=50)
        assert abs(val - 1.0) < 1e-9


class TestMean:
    def test_empty_list_is_zero(self):
        assert mean([]) == 0.0

    def test_single_value(self):
        assert abs(mean([0.7]) - 0.7) < 1e-9

    def test_average(self):
        assert abs(mean([0.2, 0.8]) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Calibration / ECE
# ---------------------------------------------------------------------------

class TestECE:
    def test_perfect_calibration_low_ece(self):
        # Score 0.9 always matches relevant label=3 → near-perfect
        n = 50
        scores = [0.9] * n
        labels = [3] * n
        result = compute_ece(scores, labels, n_bins=10)
        assert result["ece"] is not None
        assert result["ece"] < 0.15

    def test_empty_scores_returns_no_data(self):
        result = compute_ece([], [], n_bins=10)
        assert result["status"] == "no_data"
        assert result["ece"] is None

    def test_high_score_irrelevant_is_miscalibrated(self):
        # Score 0.95 but all labels irrelevant → high ECE
        n = 40
        scores = [0.95] * n
        labels = [0] * n
        result = compute_ece(scores, labels, n_bins=10)
        assert result["ece"] > 0.5

    def test_per_bin_count_sums_to_total(self):
        scores = [i / 20 for i in range(20)]
        labels = [2 if s > 0.5 else 0 for s in scores]
        result = compute_ece(scores, labels, n_bins=10)
        bin_total = sum(b["count"] for b in result["per_bin"])
        assert bin_total == len(scores)

    def test_relevant_threshold_applied(self):
        # label=1 should be below default threshold=2, treated as not relevant
        scores = [0.8] * 20
        labels = [1] * 20
        result = compute_ece(scores, labels, n_bins=10, relevant_threshold=2)
        assert result["ece"] > 0.5

    def test_custom_threshold(self):
        # With threshold=1, label=1 counts as relevant
        scores = [0.8] * 20
        labels = [1] * 20
        result = compute_ece(scores, labels, n_bins=10, relevant_threshold=1)
        assert result["ece"] < 0.3


# ---------------------------------------------------------------------------
# Hard-negative violation rate (via IR metrics script)
# ---------------------------------------------------------------------------

class TestHardNegativeViolationRate:
    """Verify hard_negative_violation_rate from compute_ir_metrics."""

    def test_no_violation_when_relevant_first(self):
        qrel = {"good": 3, "bad": 0}
        ranked = ["good", "bad"]
        assert hard_negative_violation_rate(qrel, ranked) == 0.0

    def test_violation_when_hard_neg_first(self):
        qrel = {"good": 3, "bad": 0}
        ranked = ["bad", "good"]
        rate = hard_negative_violation_rate(qrel, ranked)
        assert rate == 1.0

    def test_no_relevant_returns_zero(self):
        qrel = {"bad": 0}
        assert hard_negative_violation_rate(qrel, ["bad"]) == 0.0

    def test_no_hard_negatives_returns_zero(self):
        qrel = {"good": 3}
        assert hard_negative_violation_rate(qrel, ["good"]) == 0.0

    def test_partial_violation(self):
        qrel = {"good": 3, "hn1": 0, "hn2": 0}
        ranked = ["hn1", "good", "hn2"]
        rate = hard_negative_violation_rate(qrel, ranked)
        assert rate == 0.5  # 1 of 2 hard negatives before first relevant


class TestSourceSkewAtK:
    def test_single_source_dominates(self):
        ranked = ["dandi:001", "dandi:002", "dandi:003"]
        skew = source_skew_at_k(ranked, k=3)
        assert abs(skew["dandi"] - 1.0) < 1e-9

    def test_equal_distribution(self):
        ranked = ["dandi:001", "openneuro:002"]
        skew = source_skew_at_k(ranked, k=2)
        assert abs(skew["dandi"] - 0.5) < 1e-9
        assert abs(skew["openneuro"] - 0.5) < 1e-9

    def test_empty_ranking(self):
        assert source_skew_at_k([], k=10) == {}

    def test_truncates_at_k(self):
        ranked = ["a:1", "b:2", "c:3", "d:4"]
        skew = source_skew_at_k(ranked, k=2)
        assert sum(skew.values()) == 1.0
        assert "c" not in skew

    def test_unknown_source_for_non_compound_id(self):
        ranked = ["nodot"]
        skew = source_skew_at_k(ranked, k=1)
        assert "unknown" in skew


class TestBootstrapCI:
    def test_single_value_returns_same(self):
        lo, hi = bootstrap_ci([0.5])
        assert lo == hi == 0.5

    def test_ci_contains_true_mean(self):
        vals = [0.0] * 50 + [1.0] * 50
        lo, hi = bootstrap_ci(vals, n_bootstrap=500, ci=0.95, seed=42)
        assert lo <= 0.5 <= hi

    def test_ci_wider_with_more_variance(self):
        narrow = [0.5] * 100
        wide = [0.0] * 50 + [1.0] * 50
        lo_n, hi_n = bootstrap_ci(narrow, n_bootstrap=500, seed=42)
        lo_w, hi_w = bootstrap_ci(wide, n_bootstrap=500, seed=42)
        assert (hi_w - lo_w) > (hi_n - lo_n)

    def test_deterministic_with_seed(self):
        vals = [0.1, 0.5, 0.9, 0.3, 0.7]
        r1 = bootstrap_ci(vals, seed=42)
        r2 = bootstrap_ci(vals, seed=42)
        assert r1 == r2


class TestAggregateMetrics:
    def test_aggregate_includes_ci95(self):
        per_query = [
            {"ndcg_at_10": 0.8, "ndcg_at_20": 0.7, "mrr": 0.9,
             "precision_at_10": 0.5, "recall_at_50": 1.0,
             "hard_negative_violation_rate": 0.0},
            {"ndcg_at_10": 0.6, "ndcg_at_20": 0.5, "mrr": 0.5,
             "precision_at_10": 0.3, "recall_at_50": 0.8,
             "hard_negative_violation_rate": 0.2},
        ]
        agg = aggregate_metrics(per_query)
        assert "ndcg_at_10_ci95" in agg
        assert isinstance(agg["ndcg_at_10_ci95"], list)
        assert len(agg["ndcg_at_10_ci95"]) == 2

    def test_aggregate_mean_correct(self):
        per_query = [
            {"ndcg_at_10": 0.6, "ndcg_at_20": 0.5, "mrr": 0.5,
             "precision_at_10": 0.3, "recall_at_50": 0.8,
             "hard_negative_violation_rate": 0.0},
            {"ndcg_at_10": 0.8, "ndcg_at_20": 0.7, "mrr": 0.9,
             "precision_at_10": 0.5, "recall_at_50": 1.0,
             "hard_negative_violation_rate": 0.0},
        ]
        agg = aggregate_metrics(per_query)
        assert abs(agg["ndcg_at_10"] - 0.7) < 1e-3


class TestPrecisionAtK:
    def test_all_relevant(self):
        qrel = {"a": 2, "b": 3}
        assert abs(precision_at_k(qrel, ["a", "b"], k=2) - 1.0) < 1e-9

    def test_none_relevant(self):
        qrel = {"a": 0, "b": 1}
        assert precision_at_k(qrel, ["a", "b"], k=2) == 0.0

    def test_empty_ranking(self):
        assert precision_at_k({"a": 3}, [], k=10) == 0.0


class TestQrelsLoading:
    def test_load_qrels_accepts_dataset_id_relevance_shape(self, tmp_path):
        qrels_path = tmp_path / "neuro_qrels_consensus_for_metrics.jsonl"
        qrels_path.write_text(
            json.dumps(
                {
                    "query_id": "q1",
                    "dataset_id": "dandi:000001",
                    "relevance": 2,
                    "label": "partially_relevant",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        qrels = load_qrels(qrels_path)
        assert qrels["q1"]["dandi:000001"] == 2

    def test_neuro_judge_qrels_require_silver_acknowledgement(self, tmp_path):
        from scripts.eval.compute_ir_metrics import main as compute_main

        qrels_path = tmp_path / "neuro_qrels_consensus_for_metrics.jsonl"
        run_path = tmp_path / "run.jsonl"
        out_path = tmp_path / "report.json"
        qrels_path.write_text("", encoding="utf-8")
        run_path.write_text("", encoding="utf-8")

        result = compute_main(
            [
                "--qrels",
                str(qrels_path),
                "--run",
                str(run_path),
                "--out",
                str(out_path),
            ]
        )

        assert result == 2

    def test_neuro_judge_metric_report_is_watermarked(self, tmp_path):
        from scripts.eval.compute_ir_metrics import main as compute_main

        qrels_path = tmp_path / "neuro_qrels_consensus_for_metrics.jsonl"
        run_path = tmp_path / "run.jsonl"
        out_path = tmp_path / "report.json"
        qrels_path.write_text(
            json.dumps({"query_id": "q1", "dataset_id": "d1", "relevance": 3})
            + "\n",
            encoding="utf-8",
        )
        run_path.write_text(
            json.dumps({"query_id": "q1", "record_id": "d1", "rank": 1, "score": 0.9})
            + "\n",
            encoding="utf-8",
        )

        result = compute_main(
            [
                "--qrels",
                str(qrels_path),
                "--run",
                str(run_path),
                "--out",
                str(out_path),
                "--allow-silver",
            ]
        )

        assert result == 0
        report = json.loads(out_path.read_text(encoding="utf-8"))
        assert "neuro_judge_warning" in report
        assert "NOT HUMAN GOLD" in report["neuro_judge_warning"]


# ---------------------------------------------------------------------------
# Mini corpus qrels schema
# ---------------------------------------------------------------------------

MINI_CORPUS_DIR = Path("tests/fixtures/mini_corpus")


class TestMiniCorpusSchema:
    def test_records_are_valid_jsonl(self):
        records_path = MINI_CORPUS_DIR / "records.jsonl"
        assert records_path.exists(), "Mini corpus records.jsonl missing"
        records = []
        with records_path.open() as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        assert len(records) >= 5
        for record in records:
            assert "dataset_id" in record
            assert "source" in record

    def test_queries_are_valid_jsonl(self):
        queries_path = MINI_CORPUS_DIR / "benchmark_queries.jsonl"
        assert queries_path.exists(), "Mini corpus benchmark_queries.jsonl missing"
        queries = []
        with queries_path.open() as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))
        assert len(queries) >= 3
        for q in queries:
            assert "query_id" in q
            assert "query" in q
            assert "intent" in q

    def test_qrels_are_valid_jsonl(self):
        qrels_path = MINI_CORPUS_DIR / "qrels.jsonl"
        assert qrels_path.exists(), "Mini corpus qrels.jsonl missing"
        qrels = []
        with qrels_path.open() as f:
            for line in f:
                if line.strip():
                    qrels.append(json.loads(line))
        assert len(qrels) >= 5
        for qrel in qrels:
            assert "query_id" in qrel
            assert "record_id" in qrel
            assert "label" in qrel
            assert qrel["label"] in {0, 1, 2, 3}

    def test_qrel_record_ids_exist_in_corpus(self):
        records_path = MINI_CORPUS_DIR / "records.jsonl"
        qrels_path = MINI_CORPUS_DIR / "qrels.jsonl"
        record_ids = set()
        with records_path.open() as f:
            for line in f:
                if line.strip():
                    record_ids.add(json.loads(line)["dataset_id"])
        with qrels_path.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    assert row["record_id"] in record_ids, f"{row['record_id']} not in corpus"

    def test_qrel_query_ids_exist_in_queries(self):
        queries_path = MINI_CORPUS_DIR / "benchmark_queries.jsonl"
        qrels_path = MINI_CORPUS_DIR / "qrels.jsonl"
        query_ids = set()
        with queries_path.open() as f:
            for line in f:
                if line.strip():
                    query_ids.add(json.loads(line)["query_id"])
        with qrels_path.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    assert row["query_id"] in query_ids, f"{row['query_id']} not in queries"
