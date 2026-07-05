"""Tests for Neural Search Benchmark v0.6 protocol modules.

Covers:
- BenchmarkQueryV1 and QrelsEntryV1 schema models (aliases, validators)
- validate_benchmark_queries: valid file, duplicate IDs, empty file, warnings
- validate_qrels: valid file, hn_violation on non-zero, missing rationale, unknown qid
- sample_candidate_pool: deduplication, concept_rerank strategy presence/absence
- analyze_failures: placeholder on empty qrels, real FP/FN detection
- build_result_card_gallery: placeholder on empty qrels, card collection
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.eval.analyze_failures import _analyze_variant, analyze_failures
from scripts.eval.benchmark_schema import (
    BenchmarkQueryV1,
    QrelsEntryV1,
)
from scripts.eval.build_result_card_gallery import _collect_cards, build_gallery
from scripts.eval.sample_candidate_pool import build_pool
from scripts.eval.validate_benchmark_queries import validate_queries
from scripts.eval.validate_qrels import validate_qrels

FIXTURE_DIR = Path(__file__).parent / "fixtures"
QUERIES_FIXTURE = FIXTURE_DIR / "benchmark_queries_small.jsonl"
QRELS_FIXTURE = FIXTURE_DIR / "qrels_small.jsonl"
RUN_FIXTURE = FIXTURE_DIR / "run_small.jsonl"


# ---------------------------------------------------------------------------
# BenchmarkQueryV1 schema
# ---------------------------------------------------------------------------


class TestBenchmarkQueryV1:
    def test_valid_v1_fields(self):
        q = BenchmarkQueryV1.model_validate(
            {
                "query_id": "q_0001",
                "query_text": "human fMRI resting state",
                "intent": "META_ANALYSIS",
                "scientific_goal": "Find resting-state datasets.",
                "must_have": ["modality:fMRI"],
                "hard_negatives": ["task-based fMRI"],
            }
        )
        assert q.query_id == "q_0001"
        assert q.query_text == "human fMRI resting state"

    def test_legacy_query_alias(self):
        q = BenchmarkQueryV1.model_validate(
            {
                "query_id": "q_leg",
                "query": "EEG motor imagery",
                "intent": "MODEL_VALIDATION",
                "scientific_goal": "Validate BCI model.",
                "required_evidence": ["modality:EEG"],
                "known_failure_modes": ["resting-state EEG"],
            }
        )
        assert q.query_text == "EEG motor imagery"
        assert q.must_have == ["modality:EEG"]
        assert q.hard_negatives == ["resting-state EEG"]

    def test_legacy_intent_model_validation(self):
        q = BenchmarkQueryV1.model_validate(
            {
                "query_id": "q_x",
                "query_text": "test",
                "intent": "MODEL_VALIDATION",
                "scientific_goal": "test",
                "must_have": ["x"],
                "hard_negatives": ["y"],
            }
        )
        assert q.canonical_intent() == "REPLICATION"

    def test_legacy_intent_reanalysis_feasibility(self):
        q = BenchmarkQueryV1.model_validate(
            {
                "query_id": "q_x",
                "query_text": "test",
                "intent": "REANALYSIS_FEASIBILITY",
                "scientific_goal": "test",
                "must_have": ["x"],
                "hard_negatives": ["y"],
            }
        )
        assert q.canonical_intent() == "PIPELINE_REUSE"

    def test_invalid_intent_raises(self):
        with pytest.raises(Exception, match="intent"):
            BenchmarkQueryV1.model_validate(
                {
                    "query_id": "q_x",
                    "query_text": "test",
                    "intent": "NONEXISTENT",
                    "scientific_goal": "test",
                    "must_have": ["x"],
                    "hard_negatives": ["y"],
                }
            )

    def test_invalid_split_raises(self):
        with pytest.raises(Exception, match="split"):
            BenchmarkQueryV1.model_validate(
                {
                    "query_id": "q_x",
                    "query_text": "test",
                    "intent": "EXPLORATION",
                    "scientific_goal": "test",
                    "must_have": ["x"],
                    "hard_negatives": ["y"],
                    "split": "invalid_split",
                }
            )

    def test_smoke_fixture_loads_all_5_queries(self):
        queries = []
        for line in QUERIES_FIXTURE.read_text().splitlines():
            line = line.strip()
            if line:
                queries.append(BenchmarkQueryV1.model_validate(json.loads(line)))
        assert len(queries) == 5

    def test_smoke_fixture_legacy_query_parses(self):
        queries = []
        for line in QUERIES_FIXTURE.read_text().splitlines():
            line = line.strip()
            if line:
                queries.append(BenchmarkQueryV1.model_validate(json.loads(line)))
        legacy = next(q for q in queries if q.query_id == "q_smoke_legacy")
        assert legacy.query_text == "EEG motor imagery BCI"
        assert legacy.canonical_intent() == "REPLICATION"


# ---------------------------------------------------------------------------
# QrelsEntryV1 schema
# ---------------------------------------------------------------------------


class TestQrelsEntryV1:
    def test_valid_entry(self):
        e = QrelsEntryV1.model_validate(
            {
                "query_id": "q_0001",
                "dataset_id": "openneuro:ds000001",
                "relevance": 3,
                "rationale": "Perfect match.",
                "hard_negative_violation": False,
                "annotator_id": "ann_01",
                "timestamp": "2026-06-10T00:00:00Z",
                "adjudicated": True,
            }
        )
        assert e.relevance == 3
        assert e.requires_rationale() is True

    def test_relevance_0_requires_rationale_flag(self):
        e = QrelsEntryV1.model_validate(
            {"query_id": "q", "dataset_id": "d", "relevance": 0}
        )
        assert e.requires_rationale() is True

    def test_relevance_1_2_no_rationale_required(self):
        for rel in (1, 2):
            e = QrelsEntryV1.model_validate(
                {"query_id": "q", "dataset_id": "d", "relevance": rel}
            )
            assert e.requires_rationale() is False

    def test_invalid_relevance_raises(self):
        with pytest.raises(ValidationError):
            QrelsEntryV1.model_validate(
                {"query_id": "q", "dataset_id": "d", "relevance": 4}
            )


# ---------------------------------------------------------------------------
# validate_benchmark_queries
# ---------------------------------------------------------------------------


class TestValidateBenchmarkQueries:
    def test_fixture_passes(self):
        queries, result = validate_queries(QUERIES_FIXTURE)
        assert result.ok, f"Validation failed: {result.errors}"
        assert len(queries) == 5

    def test_duplicate_ids_fail(self, tmp_path):
        path = tmp_path / "queries.jsonl"
        row = {
            "query_id": "q_dup",
            "query_text": "test",
            "intent": "EXPLORATION",
            "scientific_goal": "test",
            "must_have": ["x"],
            "hard_negatives": ["y"],
        }
        path.write_text(
            json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
        )
        _, result = validate_queries(path)
        assert not result.ok
        assert any("Duplicate" in e.message for e in result.errors)

    def test_empty_file_fails(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        _, result = validate_queries(path)
        assert not result.ok

    def test_missing_file_fails(self, tmp_path):
        _, result = validate_queries(tmp_path / "nonexistent.jsonl")
        assert not result.ok

    def test_empty_query_text_fails(self, tmp_path):
        path = tmp_path / "q.jsonl"
        path.write_text(
            json.dumps(
                {
                    "query_id": "q_1",
                    "query_text": "  ",
                    "intent": "EXPLORATION",
                    "scientific_goal": "test",
                    "must_have": ["x"],
                    "hard_negatives": ["y"],
                }
            )
            + "\n"
        )
        _, result = validate_queries(path)
        assert not result.ok

    def test_missing_must_have_warns(self, tmp_path):
        path = tmp_path / "q.jsonl"
        path.write_text(
            json.dumps(
                {
                    "query_id": "q_1",
                    "query_text": "human fMRI",
                    "intent": "EXPLORATION",
                    "scientific_goal": "test.",
                    "must_have": [],
                    "hard_negatives": ["y"],
                }
            )
            + "\n"
        )
        _, result = validate_queries(path, strict=False)
        assert any("must_have" in w.field for w in result.warnings)

    def test_strict_missing_must_have_fails(self, tmp_path):
        path = tmp_path / "q.jsonl"
        path.write_text(
            json.dumps(
                {
                    "query_id": "q_1",
                    "query_text": "human fMRI",
                    "intent": "EXPLORATION",
                    "scientific_goal": "test.",
                    "must_have": [],
                    "hard_negatives": ["y"],
                }
            )
            + "\n"
        )
        _, result = validate_queries(path, strict=True)
        assert not result.ok

    def test_intent_diversity_warning_when_missing_intents(self, tmp_path):
        path = tmp_path / "q.jsonl"
        rows = [
            {
                "query_id": f"q_{i}",
                "query_text": "test",
                "intent": "EXPLORATION",
                "scientific_goal": "test.",
                "must_have": ["x"],
                "hard_negatives": ["y"],
            }
            for i in range(3)
        ]
        # Give unique IDs
        for i, row in enumerate(rows):
            row["query_id"] = f"q_{i:03d}"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        _, result = validate_queries(path)
        # Should warn about missing intents
        assert any("intent_diversity" in w.field for w in result.warnings)


# ---------------------------------------------------------------------------
# validate_qrels
# ---------------------------------------------------------------------------


class TestValidateQrels:
    def test_fixture_passes(self):
        entries, result = validate_qrels(QRELS_FIXTURE)
        assert result.ok, f"Errors: {result.errors}"
        assert len(entries) == 7

    def test_missing_file_fails(self, tmp_path):
        _, result = validate_qrels(tmp_path / "no.jsonl")
        assert not result.ok

    def test_hn_violation_on_nonzero_fails(self, tmp_path):
        path = tmp_path / "qrels.jsonl"
        path.write_text(
            json.dumps(
                {
                    "query_id": "q_1",
                    "dataset_id": "openneuro:ds0001",
                    "relevance": 2,
                    "hard_negative_violation": True,
                    "annotator_id": "ann",
                    "timestamp": "2026-06-10T00:00:00Z",
                    "adjudicated": False,
                }
            )
            + "\n"
        )
        _, result = validate_qrels(path)
        assert not result.ok
        assert any("hard_negative_violation" in e.field for e in result.errors)

    def test_missing_rationale_warns_for_extreme_scores(self, tmp_path):
        path = tmp_path / "qrels.jsonl"
        path.write_text(
            json.dumps(
                {
                    "query_id": "q_1",
                    "dataset_id": "openneuro:ds0001",
                    "relevance": 3,
                    "rationale": "",
                    "hard_negative_violation": False,
                    "annotator_id": "ann",
                    "timestamp": "2026-06-10T00:00:00Z",
                    "adjudicated": False,
                }
            )
            + "\n"
        )
        _, result = validate_qrels(path, strict=False)
        assert any("rationale" in w.field for w in result.warnings)

    def test_unknown_query_id_when_queries_provided(self, tmp_path):
        qrels_path = tmp_path / "qrels.jsonl"
        queries_path = tmp_path / "queries.jsonl"
        queries_path.write_text(
            json.dumps(
                {
                    "query_id": "q_known",
                    "query_text": "test",
                    "intent": "EXPLORATION",
                    "scientific_goal": "t",
                    "must_have": ["x"],
                    "hard_negatives": ["y"],
                }
            )
            + "\n"
        )
        qrels_path.write_text(
            json.dumps(
                {
                    "query_id": "q_unknown",
                    "dataset_id": "d:1",
                    "relevance": 2,
                    "hard_negative_violation": False,
                    "annotator_id": "ann",
                    "timestamp": "2026-06-10T00:00:00Z",
                    "adjudicated": False,
                }
            )
            + "\n"
        )
        _, result = validate_qrels(qrels_path, queries_path=queries_path)
        assert not result.ok
        assert any("query_id" in e.field for e in result.errors)

    def test_duplicate_annotator_triple_fails(self, tmp_path):
        path = tmp_path / "qrels.jsonl"
        entry = {
            "query_id": "q_1",
            "dataset_id": "d:1",
            "relevance": 2,
            "hard_negative_violation": False,
            "annotator_id": "ann",
            "timestamp": "2026-06-10T00:00:00Z",
            "adjudicated": False,
        }
        path.write_text(json.dumps(entry) + "\n" + json.dumps(entry) + "\n")
        _, result = validate_qrels(path)
        assert not result.ok
        assert any("duplicate" in e.field for e in result.errors)


# ---------------------------------------------------------------------------
# sample_candidate_pool (pool deduplication)
# ---------------------------------------------------------------------------


class TestSampleCandidatePool:
    def _make_run_dir(self, tmp_path: Path, rows: list[dict]) -> Path:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        run_path = runs_dir / "bm25.jsonl"
        run_path.write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        return runs_dir

    def test_basic_pool_builds(self, tmp_path):
        runs_dir = self._make_run_dir(
            tmp_path,
            [
                {"query_id": "q_1", "record_id": "d:1", "rank": 1, "score": 0.9},
                {"query_id": "q_1", "record_id": "d:2", "rank": 2, "score": 0.8},
            ],
        )
        out = tmp_path / "pool.jsonl"
        queries_path = tmp_path / "queries.jsonl"
        queries_path.write_text("")  # empty — still runs
        summary = build_pool(
            queries_path=queries_path,
            runs_dir=runs_dir,
            out_path=out,
            depth=50,
            skip_concept=True,
        )
        assert out.exists()
        assert summary["total_pairs"] == 2

    def test_deduplication_across_strategies(self, tmp_path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        # Two run files with overlapping results
        (runs_dir / "bm25.jsonl").write_text(
            json.dumps({"query_id": "q_1", "record_id": "d:1", "rank": 1, "score": 0.9})
            + "\n"
        )
        (runs_dir / "usefulness.jsonl").write_text(
            json.dumps({"query_id": "q_1", "record_id": "d:1", "rank": 2, "score": 0.85})  # noqa: E501
            + "\n"
        )
        out = tmp_path / "pool.jsonl"
        summary = build_pool(
            queries_path=tmp_path / "empty.jsonl",
            runs_dir=runs_dir,
            out_path=out,
            depth=50,
            skip_concept=True,
        )
        # Only 1 pair total (deduplicated)
        assert summary["total_pairs"] == 1
        # pooled_from should have both strategies
        record = json.loads(out.read_text().splitlines()[0])
        assert sorted(record["pooled_from"]) == ["bm25", "usefulness"]
        assert record["priority"] == 2  # appears in 2 strategies

    def test_pool_uses_minimum_rank(self, tmp_path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        (runs_dir / "bm25.jsonl").write_text(
            json.dumps({"query_id": "q_1", "record_id": "d:1", "rank": 5, "score": 0.7})
            + "\n"
        )
        (runs_dir / "usefulness.jsonl").write_text(
            json.dumps({"query_id": "q_1", "record_id": "d:1", "rank": 2, "score": 0.85})  # noqa: E501
            + "\n"
        )
        out = tmp_path / "pool.jsonl"
        build_pool(
            queries_path=tmp_path / "empty.jsonl",
            runs_dir=runs_dir,
            out_path=out,
            depth=50,
            skip_concept=True,
        )
        record = json.loads(out.read_text().splitlines()[0])
        assert record["min_rank"] == 2

    def test_per_intent_slice_via_fixture_queries(self, tmp_path):
        """Fixture queries cover multiple intents — verify they all load."""
        queries, result = validate_queries(QUERIES_FIXTURE)
        intents = {q.canonical_intent() for q in queries}
        # Fixture covers: META_ANALYSIS, PIPELINE_REUSE, REPLICATION, EXPLORATION
        assert "META_ANALYSIS" in intents
        assert "PIPELINE_REUSE" in intents
        assert "EXPLORATION" in intents


# ---------------------------------------------------------------------------
# analyze_failures
# ---------------------------------------------------------------------------


class TestAnalyzeFailures:
    def test_placeholder_on_missing_qrels(self, tmp_path):
        out = tmp_path / "failures.md"
        code = analyze_failures(
            qrels_path=tmp_path / "no_qrels.jsonl",
            queries_path=QUERIES_FIXTURE,
            runs_dir=tmp_path / "runs",
            out_path=out,
        )
        assert code == 0
        assert out.exists()
        assert "No adjudicated qrels" in out.read_text()

    def test_false_positive_detection(self):

        qrel_scores = {
            "q_smoke_001": {"openneuro:ds000003": 0, "openneuro:ds000001": 3}
        }
        qrel_meta = {
            "q_smoke_001": {
                "openneuro:ds000003": {
                    "rationale": "Task-based fMRI",
                    "hard_negative_violation": True,
                },
                "openneuro:ds000001": {
                    "rationale": "Perfect match",
                    "hard_negative_violation": False,
                },
            }
        }
        ranked = {
            "q_smoke_001": [
                (1, "openneuro:ds000003", 0.88),
                (2, "openneuro:ds000001", 0.92),
            ]
        }
        report = _analyze_variant(
            variant="bm25",
            ranked=ranked,
            qrel_scores=qrel_scores,
            qrel_meta=qrel_meta,
            judgment_meta={},
            query_intent={"q_smoke_001": "META_ANALYSIS"},
            top_k=10,
        )
        assert len(report.false_positives) == 1
        assert report.false_positives[0].record_id == "openneuro:ds000003"
        assert len(report.hn_violations) == 1

    def test_false_negative_detection(self):
        qrel_scores = {"q_1": {"relevant_d": 3, "irrelevant_d": 0}}
        qrel_meta = {
            "q_1": {"relevant_d": {"rationale": "", "hard_negative_violation": False}}
        }
        # relevant_d is NOT in top-K
        ranked = {"q_1": [(1, "other_d", 0.5)]}
        report = _analyze_variant(
            variant="bm25",
            ranked=ranked,
            qrel_scores=qrel_scores,
            qrel_meta=qrel_meta,
            judgment_meta={},
            query_intent={"q_1": "EXPLORATION"},
            top_k=10,
        )
        assert any(fn.record_id == "relevant_d" for fn in report.false_negatives)

    def test_real_fixture_runs(self, tmp_path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        import shutil
        shutil.copy(RUN_FIXTURE, runs_dir / "bm25.jsonl")
        out = tmp_path / "failures.md"
        code = analyze_failures(
            qrels_path=QRELS_FIXTURE,
            queries_path=QUERIES_FIXTURE,
            runs_dir=runs_dir,
            out_path=out,
        )
        assert code == 0
        assert out.exists()
        content = out.read_text()
        assert "bm25" in content


# ---------------------------------------------------------------------------
# build_result_card_gallery
# ---------------------------------------------------------------------------


class TestBuildResultCardGallery:
    def test_placeholder_on_missing_qrels(self, tmp_path):
        out = tmp_path / "gallery.md"
        code = build_gallery(
            qrels_path=tmp_path / "no_qrels.jsonl",
            queries_path=QUERIES_FIXTURE,
            runs_dir=tmp_path / "runs",
            corpus_path=None,
            out_path=out,
        )
        assert code == 0
        assert out.exists()
        assert "No adjudicated qrels" in out.read_text()

    def test_card_collection_success_and_failure(self):
        queries = {
            "q_smoke_001": {
                "query_text": "human fMRI resting state",
                "intent": "META_ANALYSIS",
            }
        }
        qrels = {
            "q_smoke_001": {
                "openneuro:ds000001": {
                    "relevance": 3,
                    "label": "highly_relevant",
                    "rationale": "Perfect.",
                    "hard_negative_violation": False,
                },
                "openneuro:ds000003": {
                    "relevance": 0,
                    "label": "not_relevant",
                    "rationale": "Wrong.",
                    "hard_negative_violation": True,
                },
            }
        }
        run = {
            "q_smoke_001": [
                (1, "openneuro:ds000003", 0.88),
                (2, "openneuro:ds000001", 0.75),
            ]
        }
        successes, failures, hn_examples = _collect_cards(
            run=run,
            queries=queries,
            qrels=qrels,
            corpus_index={},
            variant="bm25",
            top_k=10,
        )
        assert len(successes) == 1
        assert successes[0].record_id == "openneuro:ds000001"
        assert len(failures) == 0  # relevance=0 but HN violation → goes to hn bucket
        assert len(hn_examples) == 1
        assert hn_examples[0].record_id == "openneuro:ds000003"

    def test_gallery_runs_with_fixture(self, tmp_path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        import shutil
        shutil.copy(RUN_FIXTURE, runs_dir / "bm25.jsonl")
        out = tmp_path / "gallery.md"
        code = build_gallery(
            qrels_path=QRELS_FIXTURE,
            queries_path=QUERIES_FIXTURE,
            runs_dir=runs_dir,
            corpus_path=None,
            out_path=out,
        )
        assert code == 0
        content = out.read_text()
        assert "bm25" in content
        has_cards = any(kw in content for kw in ("SUCCESS", "FAILURE", "HARD_NEGATIVE"))
        assert has_cards
