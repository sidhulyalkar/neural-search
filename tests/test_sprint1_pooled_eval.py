"""Tests for Sprint 1 pooled evaluation scripts.

Covers:
  - build_pooled_qrels_candidates: pooling, deduplication, multi-system tracking
  - evaluate_known_item_lookup: known-item metrics, alias boost, source dedup
  - report_gold_qrels_metrics: per-system metric computation, PRELIMINARY watermark
  - verify_demo_dataset: corpus presence, URL, loading route, suitability checks
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval.build_pooled_qrels_candidates import (
    _add_to_pool,
    _candidate_id,
    _hard_negative_match,
    _normalize,
    _vec_add,
    build_pooled_candidate,
    dense_retrieve_prf,
    hybrid_rrf,
    tokenize,
)
from scripts.eval.evaluate_known_item_lookup import (
    compute_system_metrics,
    known_item_boost,
    mrr,
    recall_at_k,
    source_dedup_rerank,
)
from scripts.eval.report_gold_qrels_metrics import (
    _compute_metrics,
    _macro_avg,
    _ndcg,
    _mrr,
    _p_at_k,
    _recall_at_k,
    _build_system_rankings,
    build_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_record(source: str, source_id: str, **kwargs) -> dict:
    return {
        "source": source,
        "source_id": source_id,
        "title": kwargs.get("title", f"{source} {source_id} dataset"),
        "description": kwargs.get("description", ""),
        "species": kwargs.get("species", []),
        "modalities": kwargs.get("modalities", []),
        "tasks": kwargs.get("tasks", []),
        "brain_regions": kwargs.get("brain_regions", []),
        "has_behavior": kwargs.get("has_behavior", False),
        "has_trials": kwargs.get("has_trials", False),
        "has_raw_data": kwargs.get("has_raw_data", False),
        "has_processed_data": kwargs.get("has_processed_data", False),
        "url": kwargs.get("url", f"https://example.com/{source}/{source_id}"),
    }


def _make_qrel(query_id: str, dataset_id: str, relevance: int, hn: bool = False) -> dict:
    return {
        "candidate_id": f"qrels_candidate:{query_id}:{dataset_id}",
        "query_id": query_id,
        "dataset_id": dataset_id,
        "relevance": relevance,
        "label": "partial",
        "hard_negative_violation": hn,
        "annotator_id": "test",
        "timestamp": "2026-06-11T00:00:00+00:00",
        "adjudicated": True,
        "adjudication_notes": "",
        "schema_version": "0.3",
    }


def _make_pooled_candidate(
    query_id: str,
    dataset_id: str,
    sources: list[str],
    ranks: dict[str, int],
) -> dict:
    src, sid = dataset_id.split(":", 1)
    return {
        "id": f"qrels_candidate:{query_id}:{dataset_id}",
        "query_id": query_id,
        "dataset_id": dataset_id,
        "dataset_title": f"Dataset {sid}",
        "dataset_source": src,
        "retrieval_sources": sources,
        "ranks_by_system": ranks,
        "usefulness_score": 0.5,
        "affordance_matches": [],
        "hard_negative_warning": False,
        "retrieval_score": 10.0,
        "schema_version": "0.3",
    }


# ---------------------------------------------------------------------------
# build_pooled_qrels_candidates: pool deduplication
# ---------------------------------------------------------------------------

class TestPoolDeduplication:
    def test_add_new_dataset(self) -> None:
        pool: dict = {}
        rec = _make_record("dandi", "000001")
        _add_to_pool(pool, "dandi:000001", rec, "bm25", 1, 15.0)
        assert "dandi:000001" in pool
        assert pool["dandi:000001"]["sources"] == ["bm25"]

    def test_multiple_systems_tracked(self) -> None:
        pool: dict = {}
        rec = _make_record("dandi", "000001")
        _add_to_pool(pool, "dandi:000001", rec, "bm25", 3, 12.0)
        _add_to_pool(pool, "dandi:000001", rec, "dense_prf", 5, 0.92)
        _add_to_pool(pool, "dandi:000001", rec, "hybrid_rrf", 2, 0.5)
        entry = pool["dandi:000001"]
        assert set(entry["sources"]) == {"bm25", "dense_prf", "hybrid_rrf"}
        assert entry["ranks"]["bm25"] == 3
        assert entry["ranks"]["dense_prf"] == 5

    def test_best_score_updated(self) -> None:
        pool: dict = {}
        rec = _make_record("dandi", "000001")
        _add_to_pool(pool, "dandi:000001", rec, "bm25", 1, 5.0)
        _add_to_pool(pool, "dandi:000001", rec, "dense_prf", 2, 20.0)
        assert pool["dandi:000001"]["best_score"] == 20.0

    def test_no_duplicate_sources(self) -> None:
        pool: dict = {}
        rec = _make_record("dandi", "000001")
        _add_to_pool(pool, "dandi:000001", rec, "bm25", 1, 10.0)
        _add_to_pool(pool, "dandi:000001", rec, "bm25", 3, 8.0)  # same system again
        assert pool["dandi:000001"]["sources"].count("bm25") == 1


# ---------------------------------------------------------------------------
# build_pooled_qrels_candidates: build_pooled_candidate
# ---------------------------------------------------------------------------

class TestBuildPooledCandidate:
    def _make_query(self, **kwargs) -> dict:
        return {
            "query_id": "q_0001",
            "query_text": kwargs.get("query_text", "mouse calcium imaging"),
            "intent": kwargs.get("intent", "PIPELINE_REUSE"),
            "known_failure_modes": kwargs.get("known_failure_modes", []),
            "required_evidence": kwargs.get("required_evidence", []),
        }

    def test_candidate_has_required_fields(self) -> None:
        query = self._make_query()
        rec = _make_record("dandi", "000001", species=["mouse"], modalities=["calcium_imaging"])
        info = {"sources": ["bm25", "dense_prf"], "ranks": {"bm25": 1}, "usefulness_score": 0.7, "best_score": 15.0}
        cand = build_pooled_candidate(query, "dandi:000001", rec, info)
        assert cand["query_id"] == "q_0001"
        assert cand["dataset_id"] == "dandi:000001"
        assert "bm25" in cand["retrieval_sources"]
        assert "dense_prf" in cand["retrieval_sources"]
        assert cand["ranks_by_system"]["bm25"] == 1
        assert cand["schema_version"] == "0.3"

    def test_source_url_preserved(self) -> None:
        query = self._make_query()
        rec = _make_record("dandi", "000001", url="https://dandiarchive.org/dandiset/000001")
        info = {"sources": ["bm25"], "ranks": {}, "usefulness_score": 0.0, "best_score": 1.0}
        cand = build_pooled_candidate(query, "dandi:000001", rec, info)
        assert cand["dataset_source_url"] == "https://dandiarchive.org/dandiset/000001"

    def test_hard_negative_warning_triggered(self) -> None:
        query = self._make_query(known_failure_modes=["resting state without task"])
        rec = _make_record("zenodo", "z001", title="Resting state fMRI without task", description="collected at rest")
        info = {"sources": ["bm25"], "ranks": {}, "usefulness_score": 0.0, "best_score": 1.0}
        cand = build_pooled_candidate(query, "zenodo:z001", rec, info)
        assert cand["hard_negative_warning"] is True

    def test_affordance_matches_behavioral(self) -> None:
        query = self._make_query(required_evidence=["behavioral_metadata"])
        rec = _make_record("dandi", "000005", has_behavior=True)
        info = {"sources": ["bm25"], "ranks": {}, "usefulness_score": 0.0, "best_score": 1.0}
        cand = build_pooled_candidate(query, "dandi:000005", rec, info)
        assert "behavioral_metadata" in cand["affordance_matches"]

    def test_candidate_id_format(self) -> None:
        query = self._make_query()
        rec = _make_record("dandi", "000001")
        info = {"sources": [], "ranks": {}, "usefulness_score": 0.0, "best_score": 0.0}
        cand = build_pooled_candidate(query, "dandi:000001", rec, info)
        assert cand["id"] == _candidate_id("q_0001", "dandi:000001")


# ---------------------------------------------------------------------------
# build_pooled_qrels_candidates: vector helpers
# ---------------------------------------------------------------------------

class TestVectorHelpers:
    def test_normalize_unit_vector(self) -> None:
        v = [3.0, 4.0]
        normalized = _normalize(v)
        norm = math.sqrt(sum(x ** 2 for x in normalized))
        assert abs(norm - 1.0) < 1e-6

    def test_normalize_zero_vector_unchanged(self) -> None:
        v = [0.0, 0.0, 0.0]
        result = _normalize(v)
        # Should return original (no divide by zero)
        assert result == v

    def test_vec_add(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = _vec_add(a, b)
        assert result == [5.0, 7.0, 9.0]


# ---------------------------------------------------------------------------
# build_pooled_qrels_candidates: dense PRF
# ---------------------------------------------------------------------------

class TestDensePRF:
    def _make_emb_index(self, n: int = 5, dim: int = 4) -> dict[str, list[float]]:
        """Make n normalized random-ish embeddings."""
        index = {}
        for i in range(n):
            v = [float((i + 1) % 3), float((i + 2) % 3), float((i + 3) % 3), 1.0]
            norm = math.sqrt(sum(x * x for x in v))
            index[f"dandi:{i:06d}"] = [x / norm for x in v]
        return index

    def test_returns_top_k(self) -> None:
        emb = self._make_emb_index(10)
        # BM25 top results that have embeddings
        bm25_top = [(f"dandi:{i:06d}", float(10 - i), {}) for i in range(5)]
        results = dense_retrieve_prf(bm25_top, emb, top_k=3)
        assert len(results) <= 3

    def test_empty_emb_index_returns_empty(self) -> None:
        results = dense_retrieve_prf([("dandi:000001", 5.0, {})], {}, top_k=5)
        assert results == []

    def test_empty_bm25_returns_empty(self) -> None:
        emb = self._make_emb_index(5)
        results = dense_retrieve_prf([], emb, top_k=5)
        assert results == []

    def test_scores_are_finite(self) -> None:
        emb = self._make_emb_index(10)
        bm25_top = [(f"dandi:{i:06d}", 10.0, {}) for i in range(5)]
        results = dense_retrieve_prf(bm25_top, emb, top_k=5)
        for _, score in results:
            assert math.isfinite(score)


# ---------------------------------------------------------------------------
# build_pooled_qrels_candidates: hybrid RRF
# ---------------------------------------------------------------------------

class TestHybridRRF:
    def _make_corpus(self, n: int = 5) -> dict[str, dict]:
        return {f"src:{i}": _make_record("src", str(i)) for i in range(n)}

    def test_returns_at_most_top_k(self) -> None:
        corpus = self._make_corpus(5)
        bm25 = [(f"src:{i}", float(10 - i), corpus[f"src:{i}"]) for i in range(5)]
        dense = [(f"src:{i}", float(1 - i * 0.1)) for i in range(5)]
        results = hybrid_rrf(bm25, dense, corpus, top_k=3)
        assert len(results) <= 3

    def test_dataset_in_both_systems_ranks_high(self) -> None:
        corpus = self._make_corpus(5)
        shared_id = "src:0"
        bm25 = [(shared_id, 10.0, corpus[shared_id])] + [
            (f"src:{i}", float(5 - i), corpus[f"src:{i}"]) for i in range(1, 5)
        ]
        dense = [(shared_id, 0.95)] + [(f"src:{i}", 0.5 - i * 0.1) for i in range(1, 5)]
        results = hybrid_rrf(bm25, dense, corpus, top_k=5)
        top_id = results[0][0]
        assert top_id == shared_id

    def test_unknown_dataset_id_excluded(self) -> None:
        corpus = {"src:0": _make_record("src", "0")}
        bm25 = [("src:0", 10.0, corpus["src:0"]), ("src:unknown", 5.0, {})]
        dense = [("src:0", 0.9)]
        results = hybrid_rrf(bm25, dense, corpus, top_k=5)
        ids = [rid for rid, _, _ in results]
        assert "src:unknown" not in ids


# ---------------------------------------------------------------------------
# evaluate_known_item_lookup: source dedup
# ---------------------------------------------------------------------------

class TestSourceDedupRerank:
    def _allen_rec(self, eid: str) -> dict:
        return _make_record("allen", eid, title=f"Allen Visual Coding Neuropixels: session_{eid}")

    def _dandi_rec(self, sid: str) -> dict:
        return _make_record("dandi", sid, title=f"Neuropixels recordings in mouse visual system")

    def test_collapses_allen_sessions(self) -> None:
        records = [
            ("allen:001", 20.0, self._allen_rec("001")),
            ("allen:002", 19.0, self._allen_rec("002")),
            ("allen:003", 18.0, self._allen_rec("003")),
            ("dandi:000040", 13.0, self._dandi_rec("000040")),
        ]
        result = source_dedup_rerank(records)
        allen_count = sum(1 for rid, _, _ in result if rid.startswith("allen:"))
        assert allen_count <= 1

    def test_non_session_sources_preserved(self) -> None:
        records = [
            ("allen:001", 20.0, self._allen_rec("001")),
            ("dandi:000040", 13.0, self._dandi_rec("000040")),
            ("openneuro:ds001", 12.0, _make_record("openneuro", "ds001")),
        ]
        result = source_dedup_rerank(records)
        result_ids = [rid for rid, _, _ in result]
        assert "dandi:000040" in result_ids
        assert "openneuro:ds001" in result_ids


# ---------------------------------------------------------------------------
# evaluate_known_item_lookup: alias boost
# ---------------------------------------------------------------------------

class TestKnownItemBoost:
    def _make_corpus(self) -> dict[str, dict]:
        return {
            "dandi:000040": _make_record("dandi", "000040", title="Neuropixels recordings in mouse visual system"),
            "dandi:000022": _make_record("dandi", "000022", title="Allen Visual Coding Neuropixels FC"),
        }

    def test_steinmetz_2019_injected_at_top(self) -> None:
        corpus = self._make_corpus()
        initial = [("dandi:000022", 20.0, corpus["dandi:000022"])]
        boosted = known_item_boost("Steinmetz 2019 Neuropixels visual coding", initial, corpus)
        top_id = boosted[0][0]
        assert top_id == "dandi:000040"

    def test_no_alias_match_unchanged(self) -> None:
        corpus = self._make_corpus()
        initial = [("dandi:000022", 20.0, corpus["dandi:000022"])]
        result = known_item_boost("Allen Institute calcium imaging visual cortex", initial, corpus)
        assert result[0][0] == "dandi:000022"

    def test_alias_already_present_moved_to_top(self) -> None:
        corpus = self._make_corpus()
        initial = [
            ("dandi:000022", 20.0, corpus["dandi:000022"]),
            ("dandi:000040", 13.0, corpus["dandi:000040"]),
        ]
        boosted = known_item_boost("Steinmetz 2019 Neuropixels", initial, corpus)
        assert boosted[0][0] == "dandi:000040"


# ---------------------------------------------------------------------------
# evaluate_known_item_lookup: metrics
# ---------------------------------------------------------------------------

class TestKnownItemMetrics:
    def test_recall_at_1_found(self) -> None:
        assert recall_at_k(1, 1) == 1.0

    def test_recall_at_1_not_found(self) -> None:
        assert recall_at_k(2, 1) == 0.0

    def test_recall_at_k_none(self) -> None:
        assert recall_at_k(None, 10) == 0.0

    def test_mrr_rank_1(self) -> None:
        assert mrr(1) == 1.0

    def test_mrr_rank_3(self) -> None:
        assert abs(mrr(3) - 1 / 3) < 1e-6

    def test_mrr_not_found(self) -> None:
        assert mrr(None) == 0.0

    def test_compute_system_metrics_all_found(self) -> None:
        ranks = [1, 2, 3]
        m = compute_system_metrics(ranks)
        assert m["recall@1"] == pytest.approx(1 / 3)
        assert m["recall@3"] == pytest.approx(1.0)
        assert m["recall@10"] == pytest.approx(1.0)
        assert m["mrr"] == pytest.approx((1.0 + 0.5 + 1 / 3) / 3)

    def test_compute_system_metrics_none_found(self) -> None:
        ranks = [None, None]
        m = compute_system_metrics(ranks)
        assert m["recall@1"] == 0.0
        assert m["mrr"] == 0.0

    def test_compute_system_metrics_empty(self) -> None:
        assert compute_system_metrics([]) == {}


# ---------------------------------------------------------------------------
# report_gold_qrels_metrics: IR metric functions
# ---------------------------------------------------------------------------

class TestGoldMetricFunctions:
    def test_ndcg_perfect(self) -> None:
        assert abs(_ndcg([3, 2, 1, 0]) - 1.0) < 1e-6

    def test_ndcg_all_zeros(self) -> None:
        assert _ndcg([0, 0, 0]) == 0.0

    def test_mrr_first_relevant(self) -> None:
        assert abs(_mrr([3, 0, 0]) - 1.0) < 1e-6

    def test_mrr_no_relevant(self) -> None:
        assert _mrr([0, 1, 0]) == 0.0  # score 1 < threshold 2

    def test_p_at_5(self) -> None:
        # 2 of 5 are ≥2
        assert abs(_p_at_k([3, 0, 0, 2, 0], 5) - 2 / 5) < 1e-6

    def test_recall_at_10(self) -> None:
        # 2 relevant in top 10, 3 total
        assert abs(_recall_at_k([3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3], n_rel=3, k=10) - 2 / 3) < 1e-6

    def test_compute_metrics_keys(self) -> None:
        m = _compute_metrics([3, 2, 1, 0])
        assert all(k in m for k in ("ndcg@10", "mrr", "p@5", "recall@10", "n_labeled", "n_relevant"))

    def test_macro_avg_correct(self) -> None:
        mlist = [{"ndcg@10": 0.8, "mrr": 0.9}, {"ndcg@10": 0.6, "mrr": 0.7}]
        avg = _macro_avg(mlist)
        assert abs(avg["ndcg@10"] - 0.7) < 1e-6

    def test_macro_avg_empty(self) -> None:
        assert _macro_avg([]) == {}


# ---------------------------------------------------------------------------
# report_gold_qrels_metrics: system rankings
# ---------------------------------------------------------------------------

class TestBuildSystemRankings:
    def test_extracts_per_system_ranks(self) -> None:
        candidates = [
            _make_pooled_candidate("q_0001", "dandi:000001", ["bm25", "dense_prf"], {"bm25": 1, "dense_prf": 5}),
            _make_pooled_candidate("q_0001", "dandi:000002", ["bm25"], {"bm25": 2}),
        ]
        rankings = _build_system_rankings(candidates, ["bm25", "dense_prf"])
        assert rankings["bm25"]["q_0001"]["dandi:000001"] == 1
        assert rankings["bm25"]["q_0001"]["dandi:000002"] == 2
        assert rankings["dense_prf"]["q_0001"]["dandi:000001"] == 5
        assert "dandi:000002" not in rankings["dense_prf"]["q_0001"]

    def test_unknown_system_ignored(self) -> None:
        candidates = [
            _make_pooled_candidate("q_0001", "dandi:000001", ["bm25"], {"bm25": 1, "mystery_system": 3}),
        ]
        rankings = _build_system_rankings(candidates, ["bm25"])
        assert "mystery_system" not in rankings


# ---------------------------------------------------------------------------
# report_gold_qrels_metrics: build_report
# ---------------------------------------------------------------------------

class TestGoldBuildReport:
    def _make_qrels_and_candidates(
        self, n_pairs: int = 5
    ) -> tuple[list[dict], list[dict], dict[str, dict]]:
        qrels = []
        candidates = []
        for i in range(n_pairs):
            qid = f"q_000{(i % 2) + 1}"
            did = f"dandi:{i:06d}"
            qrels.append(_make_qrel(qid, did, relevance=2 if i % 3 else 0))
            candidates.append(
                _make_pooled_candidate(qid, did, ["bm25", "hybrid_rrf"], {"bm25": i + 1, "hybrid_rrf": i + 1})
            )
        query_map = {
            "q_0001": {"query_id": "q_0001", "intent": "META_ANALYSIS", "query_text": "fMRI reward"},
            "q_0002": {"query_id": "q_0002", "intent": "PIPELINE_REUSE", "query_text": "calcium imaging"},
        }
        return qrels, candidates, query_map

    def test_preliminary_shown_below_100(self) -> None:
        qrels, candidates, query_map = self._make_qrels_and_candidates(10)
        report = build_report(qrels, candidates, query_map)
        assert "PRELIMINARY" in report

    def test_per_system_table_has_bm25(self) -> None:
        qrels, candidates, query_map = self._make_qrels_and_candidates(10)
        report = build_report(qrels, candidates, query_map)
        assert "bm25" in report

    def test_per_system_table_has_hybrid(self) -> None:
        qrels, candidates, query_map = self._make_qrels_and_candidates(10)
        report = build_report(qrels, candidates, query_map)
        assert "hybrid_rrf" in report

    def test_query_ids_in_report(self) -> None:
        qrels, candidates, query_map = self._make_qrels_and_candidates(10)
        report = build_report(qrels, candidates, query_map)
        assert "q_0001" in report
        assert "q_0002" in report

    def test_hn_violation_rate_displayed(self) -> None:
        qrels = [_make_qrel("q_0001", "dandi:000001", 0, hn=True)]
        candidates = [_make_pooled_candidate("q_0001", "dandi:000001", ["bm25"], {"bm25": 1})]
        query_map = {"q_0001": {"query_id": "q_0001", "intent": "META_ANALYSIS", "query_text": "fMRI"}}
        report = build_report(qrels, candidates, query_map)
        assert "HN violation" in report or "violation" in report.lower()


# ---------------------------------------------------------------------------
# verify_demo_dataset: individual checks
# ---------------------------------------------------------------------------

from scripts.eval.verify_demo_dataset import (
    check_corpus_presence,
    check_loading_route,
    check_modality_metadata,
    check_notebook_suitability,
    check_source_url,
    check_species_metadata,
    list_candidates,
    verify_dataset,
)


class TestVerifyDemoDataset:
    def _nwb_record(self, **kwargs) -> dict:
        return {
            "source": kwargs.get("source", "dandi"),
            "source_id": kwargs.get("source_id", "000039"),
            "title": kwargs.get("title", "Allen calcium imaging visual cortex contrast tuning"),
            "description": kwargs.get("description", "Two-photon recordings of V1 neurons during drifting grating stimuli at multiple contrasts."),
            "species": kwargs.get("species", ["mouse"]),
            "modalities": kwargs.get("modalities", ["calcium_imaging"]),
            "tasks": [],
            "brain_regions": ["visual_cortex"],
            "data_standards": kwargs.get("data_standards", ["NWB"]),
            "has_raw_data": kwargs.get("has_raw_data", True),
            "has_processed_data": False,
            "has_behavior": False,
            "has_trials": False,
            "url": kwargs.get("url", "https://dandiarchive.org/dandiset/000039"),
        }

    def test_corpus_presence_found(self) -> None:
        rec = self._nwb_record()
        passed, msg = check_corpus_presence(rec, "dandi:000039")
        assert passed
        assert "Found" in msg

    def test_corpus_presence_not_found(self) -> None:
        passed, msg = check_corpus_presence(None, "dandi:999999")
        assert not passed
        assert "not found" in msg

    def test_source_url_present(self) -> None:
        rec = self._nwb_record()
        passed, msg = check_source_url(rec)
        assert passed
        assert "dandiarchive" in msg

    def test_source_url_missing(self) -> None:
        rec = self._nwb_record(url="")
        passed, _ = check_source_url(rec)
        assert not passed

    def test_modality_present(self) -> None:
        rec = self._nwb_record()
        passed, msg = check_modality_metadata(rec)
        assert passed
        assert "calcium_imaging" in msg

    def test_modality_missing(self) -> None:
        rec = self._nwb_record(modalities=[])
        passed, _ = check_modality_metadata(rec)
        assert not passed

    def test_species_present(self) -> None:
        rec = self._nwb_record()
        passed, _ = check_species_metadata(rec)
        assert passed

    def test_species_missing(self) -> None:
        rec = self._nwb_record(species=[])
        passed, _ = check_species_metadata(rec)
        assert not passed

    def test_loading_route_nwb(self) -> None:
        rec = self._nwb_record(data_standards=["NWB"])
        passed, msg = check_loading_route(rec)
        assert passed
        assert "NWB" in msg

    def test_loading_route_bids(self) -> None:
        rec = self._nwb_record(data_standards=["BIDS"])
        passed, msg = check_loading_route(rec)
        assert passed
        assert "BIDS" in msg

    def test_loading_route_from_archive_url(self) -> None:
        rec = self._nwb_record(data_standards=[], source="openneuro", url="https://openneuro.org/ds000001")
        passed, _ = check_loading_route(rec)
        assert passed

    def test_loading_route_missing(self) -> None:
        rec = self._nwb_record(data_standards=[], source="unknown", url="")
        passed, _ = check_loading_route(rec)
        assert not passed

    def test_notebook_suitability_high_quality_passes(self) -> None:
        rec = self._nwb_record()
        passed, msg = check_notebook_suitability(rec)
        assert passed
        assert "PASS" in msg

    def test_notebook_suitability_sparse_record_fails(self) -> None:
        rec = {
            "source": "zenodo",
            "source_id": "sparse",
            "title": "t",
            "description": None,
            "species": [],
            "modalities": [],
            "data_standards": [],
            "has_raw_data": False,
            "has_processed_data": False,
            "has_behavior": False,
            "has_trials": False,
            "url": "",
        }
        passed, msg = check_notebook_suitability(rec)
        assert not passed
        assert "FAIL" in msg

    def test_verify_dataset_pass(self) -> None:
        rec = self._nwb_record(source_id="000039")
        corpus = [rec]
        result = verify_dataset("dandi:000039", corpus)
        assert result["overall"] == "PASS"
        assert result["title"] is not None

    def test_verify_dataset_not_in_corpus(self) -> None:
        result = verify_dataset("dandi:999999", [])
        assert result["overall"] == "FAIL"
        first_check = result["checks"][0]
        assert first_check["check"] == "corpus_presence"
        assert not first_check["passed"]

    def test_list_candidates_returns_nwb_datasets(self) -> None:
        corpus = [
            self._nwb_record(source="dandi", source_id=f"{i:06d}") for i in range(5)
        ] + [
            {  # sparse record that should fail
                "source": "zenodo", "source_id": "bad", "title": "x",
                "description": None, "species": [], "modalities": [],
                "data_standards": [], "has_raw_data": False, "has_processed_data": False,
                "has_behavior": False, "has_trials": False, "url": "",
            }
        ]
        candidates = list_candidates(corpus)
        ids = [c["dataset_id"] for c in candidates]
        assert "zenodo:bad" not in ids
        assert len(candidates) <= 5
