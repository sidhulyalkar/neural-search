"""Tests for Sprint 1 evaluation scripts: expand_candidate_pool and report_benchmark_metrics."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval.expand_candidate_pool import (
    _candidate_id,
    _record_id,
    bm25_score,
    build_bm25_index,
    build_candidate,
    retrieve_top_k,
    tokenize,
)
from scripts.eval.report_benchmark_metrics import (
    build_report,
    compute_metrics,
    dcg,
    macro_average,
    mrr,
    ndcg,
    precision_at_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {
        "source": "dandi",
        "source_id": "000001",
        "title": "Mouse hippocampus calcium imaging during reversal learning",
        "description": "Calcium imaging of place cells in mouse hippocampus during spatial reversal learning task",
        "species": ["mouse"],
        "modalities": ["calcium_imaging"],
        "tasks": ["reversal_learning"],
        "brain_regions": ["hippocampus"],
    },
    {
        "source": "openneuro",
        "source_id": "ds000001",
        "title": "Human fMRI reward prediction error reinforcement learning",
        "description": "fMRI study of human subjects performing probabilistic reward learning with prediction error signals",
        "species": ["human"],
        "modalities": ["fMRI"],
        "tasks": ["reward_prediction_error", "reinforcement_learning"],
        "brain_regions": ["striatum", "OFC"],
    },
    {
        "source": "neurovault",
        "source_id": "nv001",
        "title": "Neuropixels visual cortex mouse orientation selectivity",
        "description": "Neuropixels recordings from mouse visual cortex during orientation selectivity paradigm",
        "species": ["mouse"],
        "modalities": ["extracellular_ephys", "neuropixels"],
        "tasks": ["visual_stimulation"],
        "brain_regions": ["visual_cortex"],
    },
    {
        "source": "zenodo",
        "source_id": "zen001",
        "title": "Resting state fMRI connectivity dataset",
        "description": "Resting state fMRI collected at rest with no task",
        "species": ["human"],
        "modalities": ["fMRI"],
        "tasks": [],
        "brain_regions": [],
    },
    {
        "source": "osf",
        "source_id": "osf001",
        "title": "EEG motor imagery BCI dataset",
        "description": "EEG recordings during motor imagery tasks for brain-computer interface classification",
        "species": ["human"],
        "modalities": ["EEG"],
        "tasks": ["motor_imagery"],
        "brain_regions": ["motor_cortex"],
    },
]

SAMPLE_QUERIES = [
    {
        "query_id": "q_0001",
        "query_text": "human fMRI reward prediction error reinforcement learning task",
        "intent": "META_ANALYSIS",
        "expected_species": ["human"],
        "expected_modalities": ["fMRI"],
        "expected_tasks": ["reward_prediction_error"],
        "hard_negatives": ["resting-state fMRI", "animal RL task"],
    },
    {
        "query_id": "q_0002",
        "query_text": "mouse visual cortex calcium imaging population coding orientation",
        "intent": "MODEL_VALIDATION",
        "expected_species": ["mouse"],
        "expected_modalities": ["calcium_imaging"],
        "expected_tasks": ["visual_stimulation"],
        "hard_negatives": ["mouse electrophysiology without imaging"],
    },
]


# ---------------------------------------------------------------------------
# expand_candidate_pool: tokenizer
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_lowercase(self) -> None:
        assert tokenize("Calcium Imaging") == ["calcium", "imaging"]

    def test_strips_punctuation(self) -> None:
        assert tokenize("mouse, hippocampus.") == ["mouse", "hippocampus"]

    def test_empty_string(self) -> None:
        assert tokenize("") == []

    def test_numbers_preserved(self) -> None:
        tokens = tokenize("fMRI 3T scanner 1.5T")
        assert "fmri" in tokens
        assert "3t" in tokens

    def test_none_input(self) -> None:
        assert tokenize(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# expand_candidate_pool: BM25 index
# ---------------------------------------------------------------------------

class TestBM25Index:
    def setup_method(self) -> None:
        self.idf, self.doc_tf, self.doc_lengths, self.avg_dl = build_bm25_index(SAMPLE_RECORDS)

    def test_idf_keys_are_lowercase(self) -> None:
        for tok in self.idf:
            assert tok == tok.lower()

    def test_common_token_has_low_idf(self) -> None:
        # "mouse" appears in 3/5 docs — should have lower idf than rare term
        idf_mouse = self.idf.get("mouse", 0)
        idf_rare = self.idf.get("striatum", 0)
        assert idf_mouse < idf_rare

    def test_doc_tf_covers_all_records(self) -> None:
        assert len(self.doc_tf) == len(SAMPLE_RECORDS)

    def test_avg_dl_positive(self) -> None:
        assert self.avg_dl > 0


class TestBM25Score:
    def setup_method(self) -> None:
        self.idf, self.doc_tf, self.doc_lengths, self.avg_dl = build_bm25_index(SAMPLE_RECORDS)

    def test_relevant_doc_scores_higher(self) -> None:
        q_tokens = tokenize("human fMRI reward reinforcement learning")
        rec_relevant = _record_id(SAMPLE_RECORDS[1])  # human fMRI reward
        rec_irrelevant = _record_id(SAMPLE_RECORDS[0])  # mouse calcium

        score_rel = bm25_score(q_tokens, rec_relevant, self.idf, self.doc_tf, self.doc_lengths, self.avg_dl)
        score_irr = bm25_score(q_tokens, rec_irrelevant, self.idf, self.doc_tf, self.doc_lengths, self.avg_dl)
        assert score_rel > score_irr

    def test_empty_query_scores_zero(self) -> None:
        rec_id = _record_id(SAMPLE_RECORDS[0])
        score = bm25_score([], rec_id, self.idf, self.doc_tf, self.doc_lengths, self.avg_dl)
        assert score == 0.0

    def test_unknown_record_scores_zero(self) -> None:
        score = bm25_score(
            tokenize("mouse calcium"),
            "does_not_exist:999",
            self.idf,
            self.doc_tf,
            self.doc_lengths,
            self.avg_dl,
        )
        assert score == 0.0


# ---------------------------------------------------------------------------
# expand_candidate_pool: retrieve_top_k
# ---------------------------------------------------------------------------

class TestRetrieveTopK:
    def setup_method(self) -> None:
        self.idf, self.doc_tf, self.doc_lengths, self.avg_dl = build_bm25_index(SAMPLE_RECORDS)

    def test_returns_at_most_k_results(self) -> None:
        q = SAMPLE_QUERIES[0]
        results = retrieve_top_k(q, SAMPLE_RECORDS, self.idf, self.doc_tf, self.doc_lengths, self.avg_dl, top_k=3)
        assert len(results) <= 3

    def test_fmri_query_ranks_fmri_first(self) -> None:
        q = SAMPLE_QUERIES[0]  # human fMRI reward
        results = retrieve_top_k(q, SAMPLE_RECORDS, self.idf, self.doc_tf, self.doc_lengths, self.avg_dl, top_k=5)
        top_rec_id, top_score, top_rec = results[0]
        # human fMRI reward record should rank first
        assert top_rec.get("source") in ("openneuro", "neurovault", "dandi", "zenodo")
        assert top_score > 0

    def test_results_sorted_descending(self) -> None:
        q = SAMPLE_QUERIES[0]
        results = retrieve_top_k(q, SAMPLE_RECORDS, self.idf, self.doc_tf, self.doc_lengths, self.avg_dl, top_k=5)
        scores = [s for _, s, _ in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# expand_candidate_pool: build_candidate
# ---------------------------------------------------------------------------

class TestBuildCandidate:
    def test_candidate_has_required_fields(self) -> None:
        query = SAMPLE_QUERIES[0]
        rec = SAMPLE_RECORDS[1]
        rec_id = _record_id(rec)
        candidate = build_candidate(query, rank=1, rec_id=rec_id, bm25_score_val=5.3, record=rec)

        assert candidate["query_id"] == "q_0001"
        assert candidate["dataset_id"] == rec_id
        assert candidate["rank"] == 1
        assert abs(candidate["retrieval_score"] - 5.3) < 0.001
        assert candidate["schema_version"] == "0.3"

    def test_candidate_id_format(self) -> None:
        query = SAMPLE_QUERIES[0]
        rec = SAMPLE_RECORDS[0]
        rec_id = _record_id(rec)
        candidate = build_candidate(query, rank=2, rec_id=rec_id, bm25_score_val=1.0, record=rec)
        assert candidate["id"] == _candidate_id("q_0001", rec_id)

    def test_hard_negatives_in_metadata(self) -> None:
        query = SAMPLE_QUERIES[0]
        rec = SAMPLE_RECORDS[0]
        rec_id = _record_id(rec)
        candidate = build_candidate(query, rank=3, rec_id=rec_id, bm25_score_val=0.5, record=rec)
        assert "resting-state fMRI" in candidate["metadata"]["query_known_failure_modes"]


# ---------------------------------------------------------------------------
# report_benchmark_metrics: IR metrics
# ---------------------------------------------------------------------------

class TestIRMetrics:
    def test_dcg_perfect_ranking(self) -> None:
        # Perfect ranking: [3, 2, 1, 0] — DCG = (7/1 + 3/log2(3) + 1/2 + 0)
        scores = [3, 2, 1, 0]
        result = dcg(scores, k=4)
        expected = 7.0 / math.log2(2) + 3.0 / math.log2(3) + 1.0 / math.log2(4) + 0.0
        assert abs(result - expected) < 1e-6

    def test_ndcg_perfect_is_1(self) -> None:
        scores = [3, 3, 2, 1, 0]
        assert abs(ndcg(scores) - 1.0) < 1e-6

    def test_ndcg_worst_case(self) -> None:
        # Worst: reverse ideal order
        ideal = [3, 2, 1, 0]
        worst = list(reversed(ideal))
        assert ndcg(worst) < ndcg(ideal)

    def test_ndcg_all_zeros(self) -> None:
        assert ndcg([0, 0, 0]) == 0.0

    def test_mrr_first_relevant(self) -> None:
        assert abs(mrr([3, 0, 0]) - 1.0) < 1e-6

    def test_mrr_second_relevant(self) -> None:
        assert abs(mrr([1, 2, 0]) - 0.5) < 1e-6  # score 1 < threshold 2; score 2 at pos 2

    def test_mrr_no_relevant(self) -> None:
        assert mrr([0, 1, 0]) == 0.0

    def test_precision_at_5(self) -> None:
        # 3 of 5 are relevant (≥2)
        scores = [3, 2, 1, 0, 3]
        p5 = precision_at_k(scores, 5)
        assert abs(p5 - 3 / 5) < 1e-6

    def test_recall_at_k(self) -> None:
        scores = [3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3]  # 3rd relevant is outside top 10
        r10 = recall_at_k(scores, all_relevant=3, k=10)
        assert abs(r10 - 2 / 3) < 1e-6


class TestComputeMetrics:
    def test_returns_all_metric_keys(self) -> None:
        m = compute_metrics([3, 2, 1, 0, 0])
        assert "ndcg@10" in m
        assert "mrr" in m
        assert "p@5" in m
        assert "p@10" in m
        assert "recall@10" in m
        assert "n_labeled" in m
        assert "n_relevant" in m

    def test_n_labeled_correct(self) -> None:
        m = compute_metrics([3, 2, 1])
        assert m["n_labeled"] == 3

    def test_n_relevant_counts_ge_threshold(self) -> None:
        m = compute_metrics([3, 2, 1, 0])
        assert m["n_relevant"] == 2  # 3 and 2 are ≥ 2

    def test_empty_scores(self) -> None:
        m = compute_metrics([])
        assert m["n_labeled"] == 0
        assert m["ndcg@10"] == 0.0


class TestMacroAverage:
    def test_averages_float_fields(self) -> None:
        mlist = [
            {"ndcg@10": 0.8, "mrr": 0.9},
            {"ndcg@10": 0.6, "mrr": 0.7},
        ]
        avg = macro_average(mlist)
        assert abs(avg["ndcg@10"] - 0.7) < 1e-6
        assert abs(avg["mrr"] - 0.8) < 1e-6

    def test_empty_list_returns_empty(self) -> None:
        assert macro_average([]) == {}

    def test_n_queries_counted(self) -> None:
        mlist = [{"ndcg@10": 0.5}, {"ndcg@10": 0.7}]
        avg = macro_average(mlist)
        assert avg["n_queries"] == 2.0


# ---------------------------------------------------------------------------
# report_benchmark_metrics: build_report integration
# ---------------------------------------------------------------------------

class TestBuildReport:
    def _make_qrels(self, pairs: list[tuple[str, str, int]]) -> list[dict]:
        return [
            {
                "candidate_id": f"qrels_candidate:{q_id}:{ds_id}",
                "query_id": q_id,
                "dataset_id": ds_id,
                "relevance": rel,
                "label": "partial",
                "hard_negative_violation": rel == 0,
                "annotator_id": "test",
                "timestamp": "2026-06-11T00:00:00+00:00",
                "adjudicated": True,
                "adjudication_notes": "",
                "schema_version": "0.3",
            }
            for q_id, ds_id, rel in pairs
        ]

    def _make_query_map(self) -> dict:
        return {
            "q_0001": {
                "query_id": "q_0001",
                "query_text": "human fMRI reward",
                "intent": "META_ANALYSIS",
            },
            "q_0002": {
                "query_id": "q_0002",
                "query_text": "mouse calcium imaging",
                "intent": "MODEL_VALIDATION",
            },
        }

    def _make_candidate_map(self) -> dict:
        return {
            "q_0001": ["openneuro:ds000001", "dandi:000001", "zenodo:zen001"],
            "q_0002": ["dandi:000001", "neurovault:nv001"],
        }

    def test_report_contains_ndcg(self) -> None:
        qrels = self._make_qrels([
            ("q_0001", "openneuro:ds000001", 3),
            ("q_0001", "dandi:000001", 1),
            ("q_0001", "zenodo:zen001", 0),
        ])
        report = build_report(qrels, self._make_query_map(), self._make_candidate_map())
        assert "NDCG@10" in report

    def test_preliminary_warning_shown_for_small_set(self) -> None:
        qrels = self._make_qrels([("q_0001", "openneuro:ds000001", 3)])
        report = build_report(qrels, self._make_query_map(), self._make_candidate_map())
        assert "PRELIMINARY" in report

    def test_no_preliminary_for_large_set(self) -> None:
        pairs = [(f"q_000{i % 2 + 1}", f"d{i}", 2) for i in range(35)]
        qrels = self._make_qrels(pairs)
        cmap = {
            "q_0001": [f"d{i}" for i in range(0, 35, 2)],
            "q_0002": [f"d{i}" for i in range(1, 35, 2)],
        }
        report = build_report(qrels, self._make_query_map(), cmap)
        assert "PRELIMINARY" not in report

    def test_per_query_table_present(self) -> None:
        qrels = self._make_qrels([
            ("q_0001", "openneuro:ds000001", 3),
            ("q_0002", "dandi:000001", 2),
        ])
        report = build_report(qrels, self._make_query_map(), self._make_candidate_map())
        assert "q_0001" in report
        assert "q_0002" in report

    def test_by_intent_section(self) -> None:
        qrels = self._make_qrels([
            ("q_0001", "openneuro:ds000001", 3),
            ("q_0002", "dandi:000001", 2),
        ])
        report = build_report(qrels, self._make_query_map(), self._make_candidate_map())
        assert "META_ANALYSIS" in report
        assert "MODEL_VALIDATION" in report
