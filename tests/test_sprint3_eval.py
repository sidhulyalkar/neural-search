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
