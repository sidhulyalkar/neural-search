"""Tests for Sprint 3 benchmark infrastructure."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


class TestExpandQuerySet:
    def test_generates_100_queries(self) -> None:
        from scripts.eval.expand_query_set import generate_queries, load_existing

        base = Path("data/eval/benchmark_queries.yaml")
        if not base.exists():
            pytest.skip("Base queries not available")
        existing = load_existing(base)
        queries = generate_queries(existing)
        assert len(queries) == 100

    def test_all_queries_have_required_fields(self) -> None:
        from scripts.eval.expand_query_set import generate_queries, load_existing

        base = Path("data/eval/benchmark_queries.yaml")
        if not base.exists():
            pytest.skip("Base queries not available")
        queries = generate_queries(load_existing(base))
        for q in queries:
            assert "id" in q
            assert "query" in q
            assert isinstance(q["id"], str)
            assert isinstance(q["query"], str)
            assert len(q["query"]) > 10

    def test_query_ids_are_unique(self) -> None:
        from scripts.eval.expand_query_set import generate_queries, load_existing

        base = Path("data/eval/benchmark_queries.yaml")
        if not base.exists():
            pytest.skip("Base queries not available")
        queries = generate_queries(load_existing(base))
        ids = [q["id"] for q in queries]
        assert len(ids) == len(set(ids)), "Duplicate query IDs found"

    def test_has_adversarial_queries(self) -> None:
        from scripts.eval.expand_query_set import generate_queries, load_existing

        base = Path("data/eval/benchmark_queries.yaml")
        if not base.exists():
            pytest.skip("Base queries not available")
        queries = generate_queries(load_existing(base))
        adversarial = [
            q for q in queries
            if any(k in q for k in ["hard_negative_modalities", "hard_negative_tasks", "hard_negative_sources"])
        ]
        assert len(adversarial) >= 5, "Expected at least 5 adversarial queries"

    def test_output_file_is_valid_yaml(self, tmp_path: Path) -> None:
        from scripts.eval.expand_query_set import generate_queries, load_existing

        base = Path("data/eval/benchmark_queries.yaml")
        if not base.exists():
            pytest.skip("Base queries not available")
        queries = generate_queries(load_existing(base))
        out = tmp_path / "queries_v2.yaml"
        out.write_text(yaml.dump({"benchmark_queries": queries}, default_flow_style=False, allow_unicode=True))
        loaded = yaml.safe_load(out.read_text())
        assert "benchmark_queries" in loaded
        assert len(loaded["benchmark_queries"]) == 100


class TestBootstrapCI:
    def _make_qrels(self, tmp_path: Path) -> Path:
        lines = [
            json.dumps({"query_id": f"q{i:03d}", "dataset_id": f"d{j}", "relevance": 1 if j <= 2 else 0})
            for i in range(1, 6)
            for j in range(1, 11)
        ]
        p = tmp_path / "qrels.jsonl"
        p.write_text("\n".join(lines))
        return p

    def _make_run(self, tmp_path: Path, name: str, shuffle: bool = False) -> Path:
        import random
        lines = []
        for i in range(1, 6):
            docs = [f"d{j}" for j in range(1, 21)]
            if shuffle:
                random.Random(i + 42).shuffle(docs)
            for rank, did in enumerate(docs, 1):
                lines.append(json.dumps({
                    "query_id": f"q{i:03d}",
                    "dataset_id": did,
                    "rank": rank,
                    "score": 1.0 / rank,
                }))
        p = tmp_path / f"{name}.jsonl"
        p.write_text("\n".join(lines))
        return p

    def test_produces_output_json(self, tmp_path: Path) -> None:
        from scripts.eval.compute_bootstrap_ci import main

        qrels = self._make_qrels(tmp_path)
        run_a = self._make_run(tmp_path, "system_a")
        out = tmp_path / "report.json"
        rc = main(["--qrels", str(qrels), "--runs", str(run_a), "--out", str(out), "--n-bootstrap", "100"])
        assert rc == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "systems" in data
        assert "system_a" in data["systems"]

    def test_metrics_in_valid_range(self, tmp_path: Path) -> None:
        from scripts.eval.compute_bootstrap_ci import main

        qrels = self._make_qrels(tmp_path)
        run_a = self._make_run(tmp_path, "system_a")
        out = tmp_path / "report.json"
        main(["--qrels", str(qrels), "--runs", str(run_a), "--out", str(out), "--n-bootstrap", "50"])
        data = json.loads(out.read_text())
        for metric_name, ci in data["systems"]["system_a"]["ci"].items():
            assert 0.0 <= ci["mean"] <= 1.0, f"{metric_name} mean out of range"
            assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"], f"{metric_name} CI inverted"

    def test_two_system_comparison(self, tmp_path: Path) -> None:
        from scripts.eval.compute_bootstrap_ci import main

        qrels = self._make_qrels(tmp_path)
        run_a = self._make_run(tmp_path, "system_a")
        run_b = self._make_run(tmp_path, "system_b", shuffle=True)
        out = tmp_path / "report.json"
        rc = main(["--qrels", str(qrels), "--runs", str(run_a), str(run_b),
                   "--out", str(out), "--n-bootstrap", "50"])
        assert rc == 0
        data = json.loads(out.read_text())
        assert "system_a" in data["systems"]
        assert "system_b" in data["systems"]
        assert len(data["pairwise_significance"]) > 0

    def test_load_run_accepts_record_id(self, tmp_path: Path) -> None:
        from scripts.eval.compute_bootstrap_ci import load_run

        run = tmp_path / "run.jsonl"
        run.write_text(
            json.dumps({"query_id": "q001", "record_id": "source:item 1", "rank": 2}) + "\n"
            + json.dumps({"query_id": "q001", "record_id": "source:item 0", "rank": 1}) + "\n"
        )
        assert load_run(run) == {"q001": ["source:item 0", "source:item 1"]}


class TestComputeNdcgFromQrels:
    def test_trec_loader_preserves_space_bearing_dataset_ids(self, tmp_path: Path) -> None:
        from scripts.eval.compute_ndcg_from_qrels import _load_qrels

        qrels = tmp_path / "qrels.trec"
        qrels.write_text("q001 0 neuromorpho:Physio Lab - Medical Faculty - UoI 2\n")
        assert _load_qrels(qrels) == {
            "q001": {"neuromorpho:Physio Lab - Medical Faculty - UoI": 2}
        }


class TestIntentStratification:
    def test_reports_metrics_by_intent(self, tmp_path: Path) -> None:
        from scripts.eval.report_intent_stratification import main

        queries = tmp_path / "queries.yaml"
        queries.write_text(
            yaml.safe_dump(
                {
                    "benchmark_queries": [
                        {"id": "q001", "query": "find datasets", "intent": "EXPLORATION"},
                        {"id": "q002", "query": "decode choice", "intent": "REANALYSIS_FEASIBILITY"},
                    ]
                }
            )
        )
        qrels = tmp_path / "qrels.trec"
        qrels.write_text("q001 0 d1 2\nq001 0 d2 0\nq002 0 d3 1\nq002 0 d4 0\n")
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        (runs_dir / "bm25.jsonl").write_text(
            json.dumps({"query_id": "q001", "record_id": "d1", "rank": 1}) + "\n"
            + json.dumps({"query_id": "q002", "record_id": "d4", "rank": 1}) + "\n"
            + json.dumps({"query_id": "q002", "record_id": "d3", "rank": 2}) + "\n"
        )
        out = tmp_path / "intent.json"
        md = tmp_path / "intent.md"

        rc = main([
            "--queries", str(queries),
            "--qrels", str(qrels),
            "--runs-dir", str(runs_dir),
            "--out", str(out),
            "--md", str(md),
        ])

        assert rc == 0
        report = json.loads(out.read_text())
        assert report["systems"]["bm25"]["EXPLORATION"]["n_queries"] == 1
        assert report["systems"]["bm25"]["EXPLORATION"]["ndcg@10"] == pytest.approx(1.0)
        assert report["systems"]["bm25"]["REANALYSIS_FEASIBILITY"]["mrr"] == pytest.approx(0.5)
        assert "REANALYSIS_FEASIBILITY" in md.read_text()


class TestDualJudgeReliability:
    def test_reports_pairwise_qwk_for_overlapping_judges(self) -> None:
        from scripts.eval.report_dual_judge_reliability import compute_reliability

        records = [
            {"query_id": "q1", "dataset_id": "d1", "label": 0, "judge_model": "judge_a"},
            {"query_id": "q1", "dataset_id": "d1", "label": 0, "judge_model": "judge_b"},
            {"query_id": "q1", "dataset_id": "d2", "label": 1, "judge_model": "judge_a"},
            {"query_id": "q1", "dataset_id": "d2", "label": 2, "judge_model": "judge_b"},
        ]

        report = compute_reliability(records)

        stats = report["pairwise"]["judge_a :: judge_b"]
        assert report["estimable"] is True
        assert stats["n_overlap"] == 2
        assert stats["exact_agreement"] == pytest.approx(0.5)
        assert -1.0 <= stats["quadratic_weighted_kappa"] <= 1.0

    def test_marks_qwk_not_estimable_without_overlap(self) -> None:
        from scripts.eval.report_dual_judge_reliability import compute_reliability

        report = compute_reliability([
            {"query_id": "q1", "dataset_id": "d1", "label": 0, "judge_model": "judge_a"},
            {"query_id": "q1", "dataset_id": "d2", "label": 1, "judge_model": "judge_b"},
        ])

        assert report["estimable"] is False
        assert report["pairs_with_two_or_more_judges"] == 0
        assert report["pairwise"] == {}


class TestEvalClaimLedgerAndGate:
    def test_claim_ledger_marks_hybrid_claims_and_qwk_caveat(self) -> None:
        from scripts.eval.build_eval_claim_ledger import build_ledger

        ndcg = {
            "bm25": {"judged_queries": 317, "ndcg@10": 0.60, "mrr": 0.80, "recall@50": 0.50},
            "dense_bge": {"judged_queries": 317, "ndcg@10": 0.50, "mrr": 0.70, "recall@50": 0.40},
            "hybrid_rrf": {"judged_queries": 317, "ndcg@10": 0.70, "mrr": 0.90, "recall@50": 0.60},
        }
        bootstrap = {
            "n_labeled_pairs": 1000,
            "pairwise_significance": [
                {
                    "system_a": "dense_bge",
                    "system_b": "hybrid_rrf",
                    "metric": "ndcg@10",
                    "a_wins": 1,
                    "b_wins": 10,
                    "significant_at_05": True,
                },
                {
                    "system_a": "dense_bge",
                    "system_b": "hybrid_rrf",
                    "metric": "mrr",
                    "a_wins": 1,
                    "b_wins": 10,
                    "significant_at_05": True,
                },
            ],
        }

        ledger = {row["claim_id"]: row for row in build_ledger(ndcg, bootstrap, {"systems": {"bm25": {}}}, {"estimable": False})}

        assert ledger["claim_hybrid_rrf_beats_bm25"]["evidence_level"] == "partially_supported"
        assert ledger["claim_hybrid_rrf_beats_dense_bge"]["evidence_level"] == "supported"
        assert ledger["claim_dual_judge_qwk"]["evidence_level"] == "not_estimable"

    def test_regression_gate_passes_current_shape_and_fails_hybrid_regression(self) -> None:
        from scripts.eval.check_eval_regression_gate import check_gate

        ndcg = {
            "bm25": {"judged_queries": 317, "ndcg@10": 0.60, "mrr": 0.80, "recall@50": 0.50},
            "bm25_structured": {"judged_queries": 317},
            "dense_bge": {"judged_queries": 317},
            "hybrid_rrf": {"judged_queries": 317, "ndcg@10": 0.70, "mrr": 0.90, "recall@50": 0.60},
        }

        passed = check_gate(ndcg, {"n_labeled_pairs": 12000}, {"estimable": False}, 300, 10000)
        assert passed["passed"] is True
        assert passed["warnings"]

        ndcg["hybrid_rrf"]["ndcg@10"] = 0.50
        failed = check_gate(ndcg, {"n_labeled_pairs": 12000}, {"estimable": True}, 300, 10000)
        assert failed["passed"] is False


class TestFailureAnalysis:
    def test_canonical_failure_breakdowns_use_judge_metadata(self, tmp_path: Path) -> None:
        from scripts.eval.analyze_failures import analyze_failures

        qrels = tmp_path / "qrels.jsonl"
        qrels.write_text(
            json.dumps({"query_id": "q001", "dataset_id": "d_bad", "label": 0}) + "\n"
            + json.dumps({"query_id": "q001", "dataset_id": "d_good", "label": 2}) + "\n"
        )
        queries = tmp_path / "queries.yaml"
        queries.write_text(
            yaml.safe_dump(
                {
                    "benchmark_queries": [
                        {"id": "q001", "query": "mouse ephys task", "intent": "EXPLORATION"}
                    ]
                }
            )
        )
        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "hybrid_rrf.jsonl").write_text(
            json.dumps({"query_id": "q001", "record_id": "d_bad", "rank": 1, "score": 1.0}) + "\n"
        )
        judgments = tmp_path / "judgments.jsonl"
        judgments.write_text(
            json.dumps(
                {
                    "query_id": "q001",
                    "dataset_id": "d_bad",
                    "label": 0,
                    "rationale_short": "wrong species and missing task",
                    "failure_modes": ["wrong_species", "missing_task"],
                    "required_dimensions_missing": ["species", "tasks"],
                    "missing_information": ["task details"],
                    "hard_negative_detected": True,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "query_id": "q001",
                    "dataset_id": "d_good",
                    "label": 2,
                    "rationale_short": "useful but missing raw data",
                    "failure_modes": [],
                    "required_dimensions_missing": ["raw_data"],
                    "missing_information": ["raw data availability"],
                    "hard_negative_detected": False,
                }
            )
            + "\n"
        )
        out = tmp_path / "failures.md"
        json_out = tmp_path / "failures.json"

        rc = analyze_failures(
            qrels_path=qrels,
            queries_path=queries,
            runs_dir=runs,
            out_path=out,
            json_out_path=json_out,
            judgments_path=judgments,
            top_k=1,
        )

        assert rc == 0
        report = json.loads(json_out.read_text())
        variant = report["variants"]["hybrid_rrf"]
        assert variant["false_positive_count"] == 1
        assert variant["false_negative_count"] == 1
        assert variant["intent_fp_counts"]["EXPLORATION"] == 1
        assert variant["source_fp_counts"]["unknown"] == 1
        assert variant["fp_mismatch_counts"]["species_mismatch"] == 1
        assert variant["fp_mismatch_counts"]["task_mismatch"] == 1
        assert variant["fp_metadata_missing_counts"]["species"] == 1
        assert variant["fn_metadata_missing_counts"]["raw_data"] == 1
        assert "False-Positive Mismatch Breakdown" in out.read_text()


class TestReanalysisReports:
    def test_reanalysis_affordance_report_ranks_candidates(self) -> None:
        from scripts.eval.build_reanalysis_affordance_report import build_report

        records = [
            {
                "dataset_id": "d1",
                "source": "dandi",
                "source_id": "000001",
                "title": "Mouse Neuropixels choice reward task",
                "modalities": ["Neuropixels"],
                "tasks": ["choice reward"],
                "brain_regions": ["OFC"],
                "data_standards": ["NWB"],
                "usability_flags": {"has_behavior": True, "has_raw_data": True},
            },
            {"dataset_id": "d2", "source": "zenodo", "source_id": "z2", "title": "metadata sparse"},
        ]

        report = build_report(records, top_n=2)

        assert report["dataset_count"] == 2
        assert report["top_datasets"][0]["record_id"] == "dandi:000001"
        assert report["metadata_gap_counts"]["species"] >= 1

    def test_new_method_matcher_flags_old_or_unknown_date_datasets(self) -> None:
        from scripts.eval.match_new_methods_to_old_datasets import build_matches

        records = [
            {
                "dataset_id": "d1",
                "source": "openneuro",
                "source_id": "ds1",
                "title": "2017 fMRI reward choice task",
                "modalities": ["fMRI"],
                "tasks": ["reward choice"],
                "brain_regions": ["PFC"],
                "data_standards": ["BIDS"],
                "usability_flags": {"has_behavior": True, "has_raw_data": True},
            }
        ]

        report = build_matches(records, min_score=0.5, top_n=10)

        assert report["match_count"] >= 1
        assert report["matches"][0]["date_status"] == "older_than_method"

    def test_metadata_priorities_use_failure_counts(self) -> None:
        from scripts.eval.prioritize_metadata_enrichment import build_priorities

        report = build_priorities(
            {
                "variants": {
                    "hybrid_rrf": {
                        "fp_metadata_missing_counts": {"species": 3},
                        "fn_metadata_missing_counts": {"raw_data": 2},
                        "fp_mismatch_counts": {"species_mismatch": 4},
                        "source_fp_counts": {"zenodo": 2},
                        "intent_fp_counts": {"EXPLORATION": 2},
                    }
                }
            }
        )

        fields = [row["field"] for row in report["priorities"]]
        assert fields[:2] == ["species", "raw_data"]
        assert report["top_false_positive_sources"]["zenodo"] == 2


class TestGraphCalibrationAndEdgeQuality:
    def test_graph_calibration_keeps_rrf_when_graph_has_no_signal(self) -> None:
        from neural_search.graph.schema import KnowledgeGraph, KnowledgeGraphNode
        from scripts.eval.calibrate_graph_rerank import calibrate

        graph = KnowledgeGraph(
            nodes={
                "node:dataset:d1": KnowledgeGraphNode(
                    node_id="node:dataset:d1",
                    node_type="dataset",
                    label="Dataset 1",
                )
            }
        )
        report = calibrate(
            qrels={"q1": {"d1": 2}},
            queries={"q1": {"id": "q1", "query": "dataset"}},
            run={"q1": [("d1", 1.0)]},
            graph=graph,
            profiles={"empty": {"degree": 0.0}},
            global_weights=[0.0, 0.1],
        )

        assert report["baseline_metrics"]["ndcg@10"] == pytest.approx(1.0)
        assert report["best"]["ndcg_delta_vs_rrf"] == pytest.approx(0.0)
        assert report["recommendation"] == "keep_hybrid_rrf_as_quality_baseline"

    def test_relationship_edge_quality_counts_helpful_and_harmful_promotions(self) -> None:
        from neural_search.graph.schema import (
            KnowledgeGraph,
            KnowledgeGraphEdge,
            KnowledgeGraphNode,
            make_edge_id,
        )
        from scripts.eval.analyze_relationship_edge_quality import analyze_edge_quality

        d1 = KnowledgeGraphNode(node_id="node:dataset:d1", node_type="dataset", label="Dataset 1")
        d2 = KnowledgeGraphNode(node_id="node:dataset:d2", node_type="dataset", label="Dataset 2")
        edge = KnowledgeGraphEdge(
            edge_id=make_edge_id(d1.node_id, "dataset_reanalysis_bridge_dataset", d2.node_id),
            source_node_id=d1.node_id,
            target_node_id=d2.node_id,
            edge_type="dataset_reanalysis_bridge_dataset",
            confidence=0.8,
        )
        graph = KnowledgeGraph(
            nodes={d1.node_id: d1, d2.node_id: d2},
            edges={edge.edge_id: edge},
        )

        report = analyze_edge_quality(
            qrels={"q1": {"d2": 2}, "q2": {"d2": 0}},
            base_run={
                "q1": [(1, "d1"), (2, "d2")],
                "q2": [(1, "d1"), (2, "d2")],
            },
            graph_run={
                "q1": [(1, "d2"), (2, "d1")],
                "q2": [(1, "d2"), (2, "d1")],
            },
            graph=graph,
            top_k=1,
        )

        by_edge = {row["edge_type"]: row for row in report["edge_quality"]}
        stats = by_edge["dataset_reanalysis_bridge_dataset"]
        assert stats["helpful_promotions"] == 1
        assert stats["harmful_promotions"] == 1
        assert stats["helpful_rate"] == pytest.approx(0.5)


class TestAcquisitionPlan:
    def test_plan_generates_items(self) -> None:
        from scripts.coverage.generate_acquisition_plan import generate_plan

        db = Path("data/coverage/ledger.duckdb")
        if not db.exists():
            pytest.skip("DuckDB ledger not available")
        items = generate_plan(db)
        assert len(items) > 0

    def test_all_items_have_required_fields(self) -> None:
        from scripts.coverage.generate_acquisition_plan import generate_plan

        db = Path("data/coverage/ledger.duckdb")
        if not db.exists():
            pytest.skip("DuckDB ledger not available")
        items = generate_plan(db)
        for item in items:
            assert "type" in item
            assert "impact_score" in item
            assert "priority" in item
            assert 0.0 <= item["impact_score"] <= 1.0
            assert item["priority"] in ("P0", "P1", "P2", "P3")

    def test_items_sorted_by_impact(self) -> None:
        from scripts.coverage.generate_acquisition_plan import generate_plan

        db = Path("data/coverage/ledger.duckdb")
        if not db.exists():
            pytest.skip("DuckDB ledger not available")
        items = generate_plan(db)
        scores = [i["impact_score"] for i in items]
        assert scores == sorted(scores, reverse=True)

    def test_covers_multiple_gap_types(self) -> None:
        from scripts.coverage.generate_acquisition_plan import generate_plan

        db = Path("data/coverage/ledger.duckdb")
        if not db.exists():
            pytest.skip("DuckDB ledger not available")
        items = generate_plan(db)
        gap_types = {i["gap_type"] for i in items}
        assert "uncovered_region" in gap_types
        assert "dark_region_modality_pair" in gap_types
        assert "low_coverage_source" in gap_types
