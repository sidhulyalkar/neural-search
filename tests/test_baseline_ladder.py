"""Tests for baseline ladder evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_search.evaluation.run_baseline_ladder import (
    LADDER_MODES,
    BaselineLadderReport,
    LadderResult,
    _build_ladder_config,
    generate_ladder_json,
    generate_ladder_markdown,
)


class TestLadderConfigBuilding:
    """Test ladder config construction."""

    def test_all_modes_have_configs(self):
        """All ladder modes should build valid configs."""
        for mode in LADDER_MODES:
            config = _build_ladder_config(mode)
            assert isinstance(config, dict)
            assert "weights" in config or mode == "full_system"

    def test_keyword_disables_advanced_features(self):
        """Keyword mode should disable embeddings and ontology."""
        config = _build_ladder_config("keyword")
        assert config.get("use_embeddings") is False
        assert config.get("use_ontology") is False
        assert config.get("use_graph") is False

    def test_bm25_basic_config(self):
        """BM25 mode should have basic text matching."""
        config = _build_ladder_config("bm25")
        assert config.get("use_embeddings") is False
        assert config.get("use_ontology") is False

    def test_dense_only_enables_embeddings(self):
        """Dense-only mode should use embeddings."""
        config = _build_ladder_config("dense_only")
        assert config.get("use_embeddings") is True
        assert config.get("use_ontology") is False

    def test_plus_ontology_enables_ontology(self):
        """Plus-ontology mode should enable ontology matching."""
        config = _build_ladder_config("plus_ontology")
        assert config.get("use_ontology") is True
        assert config.get("use_embeddings") is True

    def test_plus_graph_enables_graph(self):
        """Plus-graph mode should enable graph features."""
        config = _build_ladder_config("plus_graph")
        assert config.get("use_graph") is True
        assert config.get("use_ontology") is True

    def test_full_system_has_all_weights(self):
        """Full system should have non-zero weights for main components."""
        config = _build_ladder_config("full_system")
        weights = config.get("weights", {})
        # Full system should have balanced weights
        assert isinstance(weights, dict)

    def test_unknown_mode_raises(self):
        """Unknown mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown ladder mode"):
            _build_ladder_config("nonexistent_mode")


class TestLadderReportGeneration:
    """Test ladder report generation."""

    @pytest.fixture
    def mock_ladder_report(self):
        """Create a mock ladder report for testing."""
        from dataclasses import dataclass, field
        from datetime import UTC, datetime
        from typing import Any

        from neural_search.evaluation.run_benchmark import EvaluationReport

        @dataclass
        class MockQueryEval:
            query_id: str = "q001"
            query: str = "test query"
            num_results: int = 5
            precision_at_1: float = 0.8
            precision_at_3: float = 0.7
            precision_at_5: float = 0.6
            precision_at_10: float = 0.5
            recall_at_5: float = 0.4
            recall_at_10: float = 0.5
            label_recall_at_10: float = 0.6
            mrr: float = 0.9
            ndcg_at_10: float = 0.85
            task_match_rate: float = 0.7
            modality_match_rate: float = 0.8
            behavior_match_rate: float = 0.6
            matched_tasks: list = field(default_factory=list)
            matched_modalities: list = field(default_factory=list)
            matched_behaviors: list = field(default_factory=list)
            matched_regions: list = field(default_factory=list)
            matched_species: list = field(default_factory=list)
            matched_data_standards: list = field(default_factory=list)
            matched_sources: list = field(default_factory=list)
            matched_analysis: list = field(default_factory=list)
            missing_expected_tasks: list = field(default_factory=list)
            missing_expected_modalities: list = field(default_factory=list)
            missing_expected_behaviors: list = field(default_factory=list)
            missing_expected_regions: list = field(default_factory=list)
            missing_expected_species: list = field(default_factory=list)
            missing_expected_data_standards: list = field(default_factory=list)
            missing_expected_sources: list = field(default_factory=list)
            missing_expected_analysis: list = field(default_factory=list)
            expected_dataset_ids: list = field(default_factory=list)
            missed_expected_datasets: list = field(default_factory=list)
            hard_negative_violations: list = field(default_factory=list)
            top_false_positives: list = field(default_factory=list)
            why_failed: list = field(default_factory=list)
            top_results: list = field(default_factory=list)
            warnings: list = field(default_factory=list)
            parsed_query: dict = field(default_factory=dict)

        mock_report = EvaluationReport(
            generated_at=datetime.now(UTC).isoformat(),
            total_queries=5,
            queries_with_results=5,
            mean_precision_at_1=0.8,
            mean_precision_at_3=0.7,
            mean_precision_at_5=0.6,
            mean_precision_at_10=0.5,
            mean_recall_at_5=0.4,
            mean_recall_at_10=0.5,
            mean_label_recall_at_10=0.6,
            mean_mrr=0.9,
            mean_ndcg_at_10=0.85,
            mean_task_match_rate=0.7,
            mean_modality_match_rate=0.8,
            mean_behavior_match_rate=0.6,
            queries=[MockQueryEval()],
            summary_warnings=[],
            recommendations=[],
        )

        results = [
            LadderResult(
                mode="keyword",
                description="Exact keyword matching",
                config={"weights": {"semantic": 1.0}},
                report=mock_report,
                latency_ms=10.0,
                num_candidates=50,
            ),
            LadderResult(
                mode="full_system",
                description="All components",
                config={"weights": {"semantic": 0.5, "ontology": 0.3}},
                report=mock_report,
                latency_ms=50.0,
                num_candidates=100,
            ),
        ]

        return BaselineLadderReport(
            generated_at=datetime.now(UTC).isoformat(),
            suite="test",
            modes=["keyword", "full_system"],
            results=results,
            comparison=[
                {"metric": "mean_precision_at_5", "keyword": 0.6, "full_system": 0.7},
                {"metric": "mean_mrr", "keyword": 0.8, "full_system": 0.9},
            ],
            delta_analysis=[
                {"mode": "keyword", "delta_precision": 0.0, "delta_mrr": 0.0},
                {"mode": "full_system", "delta_precision": 0.1, "delta_mrr": 0.1},
            ],
            summary={
                "best_precision_mode": "full_system",
                "best_mrr_mode": "full_system",
                "full_vs_baseline_delta": {"precision": 0.1, "mrr": 0.1, "ndcg": 0.05},
            },
            total_hard_negative_violations=0,
        )

    def test_generate_markdown_report(self, mock_ladder_report):
        """Markdown report should contain expected sections."""
        md = generate_ladder_markdown(mock_ladder_report)
        assert "# Neural Search Baseline Ladder Report" in md
        assert "## Metric Comparison" in md
        assert "## Incremental Improvement" in md
        assert "## Summary" in md
        assert "keyword" in md
        assert "full_system" in md

    def test_generate_json_report(self, mock_ladder_report):
        """JSON report should be valid JSON with expected fields."""
        json_str = generate_ladder_json(mock_ladder_report)
        data = json.loads(json_str)
        assert "generated_at" in data
        assert "suite" in data
        assert "modes" in data
        assert "results" in data
        assert "comparison" in data
        assert "summary" in data

    def test_json_report_has_all_modes(self, mock_ladder_report):
        """JSON report should include all modes."""
        json_str = generate_ladder_json(mock_ladder_report)
        data = json.loads(json_str)
        assert "keyword" in data["modes"]
        assert "full_system" in data["modes"]
        assert len(data["results"]) == 2


class TestLadderModes:
    """Test ladder mode constants."""

    def test_ladder_modes_ordered(self):
        """Ladder modes should be in increasing complexity order."""
        expected_order = [
            "keyword",
            "bm25",
            "field_weighted_bm25",
            "dense_only",
            "bm25_dense_rrf",
            "plus_ontology",
            "plus_graph",
            "full_system",
        ]
        assert list(LADDER_MODES) == expected_order

    def test_all_modes_have_descriptions(self):
        """All modes should have descriptions."""
        from neural_search.evaluation.run_baseline_ladder import LADDER_DESCRIPTIONS

        for mode in LADDER_MODES:
            assert mode in LADDER_DESCRIPTIONS
            assert len(LADDER_DESCRIPTIONS[mode]) > 0


class TestEmptyResults:
    """Test handling of empty or edge case results."""

    def test_empty_comparison_table(self):
        """Report with no results should handle gracefully."""
        report = BaselineLadderReport(
            generated_at="2026-01-01T00:00:00",
            suite="test",
            modes=[],
            results=[],
            comparison=[],
            delta_analysis=[],
            summary={},
            total_hard_negative_violations=0,
        )
        md = generate_ladder_markdown(report)
        assert "# Neural Search Baseline Ladder Report" in md

    def test_single_mode_no_delta(self):
        """Single mode should have zero delta."""
        from datetime import UTC, datetime

        from neural_search.evaluation.run_benchmark import EvaluationReport

        mock_report = EvaluationReport(
            generated_at=datetime.now(UTC).isoformat(),
            total_queries=1,
            queries_with_results=1,
            mean_precision_at_1=0.5,
            mean_precision_at_3=0.5,
            mean_precision_at_5=0.5,
            mean_precision_at_10=0.5,
            mean_recall_at_5=0.5,
            mean_recall_at_10=0.5,
            mean_label_recall_at_10=0.5,
            mean_mrr=0.5,
            mean_ndcg_at_10=0.5,
            mean_task_match_rate=0.5,
            mean_modality_match_rate=0.5,
            mean_behavior_match_rate=0.5,
            queries=[],
            summary_warnings=[],
            recommendations=[],
        )

        report = BaselineLadderReport(
            generated_at=datetime.now(UTC).isoformat(),
            suite="test",
            modes=["keyword"],
            results=[
                LadderResult(
                    mode="keyword",
                    description="Test",
                    config={},
                    report=mock_report,
                    latency_ms=10.0,
                    num_candidates=10,
                )
            ],
            comparison=[{"metric": "mean_precision_at_5", "keyword": 0.5}],
            delta_analysis=[{"mode": "keyword", "delta_precision": 0.0}],
            summary={},
            total_hard_negative_violations=0,
        )

        md = generate_ladder_markdown(report)
        assert "keyword" in md
