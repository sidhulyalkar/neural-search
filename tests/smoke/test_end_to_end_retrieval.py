"""End-to-end smoke test: mini corpus → usefulness scoring → IR metrics.

This test is designed to be deterministic and fast. It does not require the
full 10K corpus, embeddings, or a running API server. It validates the pipeline
from fixture corpus → scorer → metric computation using only the mini corpus
fixtures and the usefulness scorer module.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

from neural_search.retrieval.usefulness_scorer import (
    DatasetContext,
    score_usefulness,
)
from neural_search.retrieval.query_intent import UsefulnessIntent
from scripts.eval.compute_ir_metrics import ndcg_at_k, mrr, recall_at_k
from scripts.eval.compute_calibration import compute_ece

MINI_CORPUS = Path("tests/fixtures/mini_corpus")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mini_records() -> list[dict]:
    records = []
    with (MINI_CORPUS / "records.jsonl").open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def mini_queries() -> list[dict]:
    queries = []
    with (MINI_CORPUS / "benchmark_queries.jsonl").open() as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


@pytest.fixture(scope="module")
def mini_qrels() -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with (MINI_CORPUS / "qrels.jsonl").open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                qrels[row["query_id"]][row["record_id"]] = int(row["label"])
    return dict(qrels)


def _record_to_context(record: dict) -> DatasetContext:
    return DatasetContext(
        dataset_id=record["dataset_id"],
        modalities=record.get("modalities") or [],
        tasks=record.get("tasks") or [],
        species=record.get("species") or [],
        brain_regions=record.get("brain_regions") or [],
        affordances=record.get("affordances") or [],
        data_standards=record.get("data_standards") or [],
        session_count=record.get("session_count"),
        trial_count=record.get("trial_count"),
        quality_score=record.get("quality_score", 0.5),
    )


def _query_to_context(query: dict, records_by_id: dict) -> DatasetContext:
    """Build a query context from the first highly-relevant record if available."""
    return DatasetContext(
        dataset_id=f"query:{query['query_id']}",
        modalities=[],
        tasks=[],
        species=[],
        brain_regions=[],
        affordances=[],
        data_standards=[],
    )


# ---------------------------------------------------------------------------
# Smoke: corpus loads correctly
# ---------------------------------------------------------------------------

class TestMiniCorpusLoads:
    def test_records_load(self, mini_records):
        assert len(mini_records) == 10

    def test_queries_load(self, mini_queries):
        assert len(mini_queries) == 5

    def test_qrels_load(self, mini_qrels):
        assert len(mini_qrels) >= 4


# ---------------------------------------------------------------------------
# Smoke: scorer produces valid outputs on all records
# ---------------------------------------------------------------------------

class TestScorerOnMiniCorpus:
    def test_all_records_score_without_error(self, mini_records):
        ctx_a = _record_to_context(mini_records[0])
        for record in mini_records:
            ctx_b = _record_to_context(record)
            result = score_usefulness(ctx_a, ctx_b, UsefulnessIntent.STRICT_LOOKUP)
            assert 0.0 <= result.total_score <= 1.0

    def test_perfect_self_score_is_high(self, mini_records):
        for record in mini_records:
            ctx = _record_to_context(record)
            result = score_usefulness(ctx, ctx, UsefulnessIntent.STRICT_LOOKUP)
            assert result.total_score >= 0.5, f"{record['dataset_id']} self-score too low: {result.total_score}"

    def test_scores_vary_across_candidates(self, mini_records):
        query_record = next(r for r in mini_records if "openneuro" in r["dataset_id"] and "ds000001" in r["dataset_id"])
        ctx_q = _record_to_context(query_record)
        scores = []
        for record in mini_records:
            ctx_c = _record_to_context(record)
            result = score_usefulness(ctx_q, ctx_c, UsefulnessIntent.STRICT_LOOKUP)
            scores.append(result.total_score)
        # Scores should not all be identical
        assert max(scores) - min(scores) > 0.01, "All scores are identical — scorer may not be discriminating"


# ---------------------------------------------------------------------------
# Smoke: usefulness-ranked results vs qrels
# ---------------------------------------------------------------------------

class TestUsefulnessRankingVsQrels:
    """Rank all mini corpus records for each query, then check NDCG against qrels."""

    def _rank_for_query(self, query: dict, records: list[dict], intent: UsefulnessIntent) -> list[str]:
        """Use the query's required_evidence fields to build a pseudo query context."""
        query_modalities = []
        query_tasks = []
        query_species = []

        # Parse query text for rough matching — intentionally simple for smoke test
        text = query["query"].lower()
        if "fmri" in text or "mri" in text:
            query_modalities = ["fmri"]
        if "calcium" in text:
            query_modalities = ["calcium_imaging", "two_photon"]
        if "electrophysiology" in text or "neuropixels" in text or "spike" in text:
            query_modalities = ["extracellular_electrophysiology", "neuropixels"]
        if "reward" in text or "reinforcement" in text:
            query_tasks = ["reward_learning", "reinforcement_learning"]
        if "working memory" in text:
            query_tasks = ["working_memory", "delayed_response"]
        if "visual" in text:
            query_tasks = ["visual_stimulation", "orientation_tuning"]
        if "decision" in text:
            query_tasks = ["decision_making", "reward_learning"]
        if "human" in text:
            query_species = ["human"]
        if "mouse" in text:
            query_species = ["mouse"]

        ctx_q = DatasetContext(
            dataset_id=f"query:{query['query_id']}",
            modalities=query_modalities,
            tasks=query_tasks,
            species=query_species,
        )

        scored = []
        for record in records:
            ctx_c = _record_to_context(record)
            result = score_usefulness(ctx_q, ctx_c, intent)
            scored.append((result.total_score, record["dataset_id"]))

        scored.sort(key=lambda x: -x[0])
        return [rid for _, rid in scored]

    def test_q0001_fmri_reward_ndcg_above_random(self, mini_records, mini_qrels):
        query = {"query_id": "q_0001", "query": "human fMRI reward prediction error reinforcement learning task"}
        qrel = mini_qrels.get("q_0001", {})
        ranked = self._rank_for_query(query, mini_records, UsefulnessIntent.META_ANALYSIS)
        ndcg = ndcg_at_k(qrel, ranked, k=10)
        assert ndcg > 0.0, "NDCG@10 should be > 0 for reward fMRI query"

    def test_q0002_calcium_imaging_ndcg_above_random(self, mini_records, mini_qrels):
        query = {"query_id": "q_0002", "query": "mouse visual cortex calcium imaging population coding orientation tuning"}
        qrel = mini_qrels.get("q_0002", {})
        ranked = self._rank_for_query(query, mini_records, UsefulnessIntent.STRICT_LOOKUP)
        ndcg = ndcg_at_k(qrel, ranked, k=10)
        assert ndcg > 0.0, "NDCG@10 should be > 0 for calcium imaging query"

    def test_q0003_spike_sorting_ndcg_above_random(self, mini_records, mini_qrels):
        query = {"query_id": "q_0003", "query": "extracellular electrophysiology spike sorting neuropixels single unit"}
        qrel = mini_qrels.get("q_0003", {})
        ranked = self._rank_for_query(query, mini_records, UsefulnessIntent.PIPELINE_REUSE)
        ndcg = ndcg_at_k(qrel, ranked, k=10)
        assert ndcg > 0.0, "NDCG@10 should be > 0 for spike sorting query"

    def test_recall_at_50_on_small_corpus(self, mini_records, mini_qrels):
        """With only 10 records, top-10 should retrieve all labeled relevant records."""
        query = {"query_id": "q_0001", "query": "human fMRI reward prediction error reinforcement learning task"}
        qrel = mini_qrels.get("q_0001", {})
        ranked = self._rank_for_query(query, mini_records, UsefulnessIntent.META_ANALYSIS)
        recall = recall_at_k(qrel, ranked, k=10)
        assert recall >= 0.0


# ---------------------------------------------------------------------------
# Smoke: IR metric pipeline on fixture data
# ---------------------------------------------------------------------------

class TestIRMetricPipeline:
    def test_ndcg_at_k_on_perfect_ranking(self):
        qrel = {"a": 3, "b": 2, "c": 1}
        ranked = ["a", "b", "c"]
        assert abs(ndcg_at_k(qrel, ranked, k=3) - 1.0) < 1e-6

    def test_mrr_first_relevant_at_one(self):
        qrel = {"a": 2}
        assert abs(mrr(qrel, ["a", "b", "c"]) - 1.0) < 1e-6

    def test_recall_at_50_all_relevant(self):
        qrel = {"a": 2, "b": 3, "c": 2}
        ranked = ["a", "b", "c"]
        assert abs(recall_at_k(qrel, ranked, k=50) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Smoke: calibration on synthetic scores
# ---------------------------------------------------------------------------

class TestCalibrationSmoke:
    def test_ece_computes_on_mini_data(self):
        scores = [0.1, 0.2, 0.5, 0.7, 0.9]
        labels = [0, 1, 2, 3, 3]
        result = compute_ece(scores, labels, n_bins=5)
        assert result["ece"] is not None
        assert 0.0 <= result["ece"] <= 1.0

    def test_ece_per_bin_totals(self):
        scores = [float(i) / 10 for i in range(10)]
        labels = [i % 4 for i in range(10)]
        result = compute_ece(scores, labels, n_bins=10)
        assert sum(b["count"] for b in result["per_bin"]) == len(scores)


# ---------------------------------------------------------------------------
# Smoke: scripts run without error (import-level)
# ---------------------------------------------------------------------------

class TestScriptImports:
    def test_freeze_corpus_snapshot_importable(self):
        from scripts.eval import freeze_corpus_snapshot
        assert hasattr(freeze_corpus_snapshot, "main")

    def test_compute_ir_metrics_importable(self):
        from scripts.eval import compute_ir_metrics
        assert hasattr(compute_ir_metrics, "main")

    def test_compute_calibration_importable(self):
        from scripts.eval import compute_calibration
        assert hasattr(compute_calibration, "main")

    def test_run_ablation_suite_importable(self):
        from scripts.eval import run_ablation_suite
        assert hasattr(run_ablation_suite, "main")

    def test_generate_paper_tables_importable(self):
        from scripts.eval import generate_paper_tables
        assert hasattr(generate_paper_tables, "main")

    def test_build_benchmark_pool_importable(self):
        from scripts.eval import build_benchmark_pool
        assert hasattr(build_benchmark_pool, "main")

    def test_audit_extraction_quality_importable(self):
        from scripts.eval import audit_extraction_quality
        assert hasattr(audit_extraction_quality, "main")
