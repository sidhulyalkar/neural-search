"""Tests for ablation evaluation runner."""

import pytest

from neural_search.evaluation.run_ablation import (
    ABLATION_MODES,
    COMPONENT_ABLATION_MODES,
    AblationReport,
    AblationResult,
    ComponentImpact,
    _build_config,
    compute_component_impact,
    generate_ablation_json,
    generate_ablation_markdown,
    run_ablation,
)


def test_ablation_modes_are_defined():
    assert len(ABLATION_MODES) >= 4
    assert "lexical_only" in ABLATION_MODES
    assert "ontology_only" in ABLATION_MODES
    assert "semantic_only" in ABLATION_MODES
    assert "hybrid_default" in ABLATION_MODES


def test_build_config_returns_valid_weights():
    config = _build_config("lexical_only")

    assert "weights" in config
    assert config["weights"]["semantic"] > 0


def test_build_config_hybrid_default_preserves_weights():
    config = _build_config("hybrid_default")

    weights = config.get("weights", {})
    assert weights.get("ontology", 0) > 0
    assert weights.get("semantic", 0) > 0


def test_build_config_ontology_only_zeros_non_ontology():
    config = _build_config("ontology_only")

    weights = config.get("weights", {})
    assert weights.get("ontology", 0) > 0
    assert weights.get("behavior", 0) > 0
    assert weights.get("readiness", 0) == 0
    assert weights.get("paper_confidence", 0) == 0


def test_build_config_no_negative_constraints_zeros_penalties():
    config = _build_config("hybrid_no_negative_constraints")

    penalties = config.get("penalties", {})
    assert penalties.get("modality_mismatch", 1) == 0
    assert penalties.get("exclusion_violation", 1) == 0


def test_run_ablation_produces_valid_report():
    # Run with minimal modes to reduce test time
    report = run_ablation("demo_v02", modes=["hybrid_default"])

    assert isinstance(report, AblationReport)
    assert report.suite == "demo_v02"
    assert len(report.results) == 1
    assert report.results[0].mode == "hybrid_default"
    assert report.results[0].report.total_queries > 0


def test_ablation_comparison_table_has_all_modes():
    report = run_ablation("demo_v02", modes=["hybrid_default", "lexical_only"])

    assert len(report.comparison) > 0
    for row in report.comparison:
        assert "metric" in row
        assert "hybrid_default" in row
        assert "lexical_only" in row


def test_ablation_summary_identifies_best_mode():
    report = run_ablation("demo_v02", modes=["hybrid_default"])

    assert "best_by_metric" in report.summary
    assert "mean_precision_at_5" in report.summary["best_by_metric"]


def test_generate_ablation_markdown_produces_output():
    report = run_ablation("demo_v02", modes=["hybrid_default"])

    markdown = generate_ablation_markdown(report)

    assert "# Neural Search Ablation Report" in markdown
    assert "demo_v02" in markdown
    assert "hybrid_default" in markdown


def test_generate_ablation_json_roundtrips():
    import json

    report = run_ablation("demo_v02", modes=["hybrid_default"])

    json_str = generate_ablation_json(report)
    data = json.loads(json_str)

    assert data["suite"] == "demo_v02"
    assert "hybrid_default" in data["modes"]
    assert len(data["results"]) == 1


# Component ablation tests
def test_component_ablation_modes_defined():
    """Test component ablation modes are properly defined."""
    assert len(COMPONENT_ABLATION_MODES) >= 5
    assert "no_ontology" in COMPONENT_ABLATION_MODES
    assert "no_behavior" in COMPONENT_ABLATION_MODES
    assert "no_semantic" in COMPONENT_ABLATION_MODES


def test_build_config_component_ablation_zeroes_target():
    """Test component ablation configs zero the target component."""
    config = _build_config("no_ontology")
    weights = config.get("weights", {})
    assert weights.get("ontology", 1) == 0

    config = _build_config("no_semantic")
    weights = config.get("weights", {})
    assert weights.get("semantic", 1) == 0


def test_build_config_component_ablation_preserves_others():
    """Test component ablation preserves other components."""
    config = _build_config("no_ontology")
    weights = config.get("weights", {})

    # Other weights should be preserved (non-zero)
    non_zero_weights = [k for k, v in weights.items() if v > 0 and k != "ontology"]
    assert len(non_zero_weights) > 0


class TestComponentImpact:
    """Tests for ComponentImpact computation."""

    @pytest.fixture
    def mock_baseline_result(self):
        """Create mock baseline result."""
        from neural_search.evaluation.run_benchmark import EvaluationReport

        report = EvaluationReport(
            generated_at="2024-01-01",
            suite="test",
            benchmark_path="test.yaml",
            queries=[],
            total_queries=10,
            mean_precision_at_5=0.65,
            mean_precision_at_10=0.60,
            mean_label_recall_at_10=0.55,
            mean_mrr=0.70,
            mean_ndcg_at_10=0.68,
            hard_negative_violations=0,
            query_category_breakdown={},
        )

        return AblationResult(mode="hybrid_default", config={}, report=report)

    @pytest.fixture
    def mock_critical_ablation(self):
        """Create mock result with critical impact."""
        from neural_search.evaluation.run_benchmark import EvaluationReport

        report = EvaluationReport(
            generated_at="2024-01-01",
            suite="test",
            benchmark_path="test.yaml",
            queries=[],
            total_queries=10,
            mean_precision_at_5=0.45,  # -20% from baseline
            mean_precision_at_10=0.40,
            mean_label_recall_at_10=0.35,
            mean_mrr=0.50,
            mean_ndcg_at_10=0.48,
            hard_negative_violations=0,
            query_category_breakdown={},
        )

        return AblationResult(mode="no_ontology", config={}, report=report)

    @pytest.fixture
    def mock_marginal_ablation(self):
        """Create mock result with marginal impact."""
        from neural_search.evaluation.run_benchmark import EvaluationReport

        report = EvaluationReport(
            generated_at="2024-01-01",
            suite="test",
            benchmark_path="test.yaml",
            queries=[],
            total_queries=10,
            mean_precision_at_5=0.63,  # -2% from baseline
            mean_precision_at_10=0.58,
            mean_label_recall_at_10=0.53,
            mean_mrr=0.68,
            mean_ndcg_at_10=0.66,
            hard_negative_violations=0,
            query_category_breakdown={},
        )

        return AblationResult(mode="no_readiness", config={}, report=report)

    def test_compute_critical_impact(self, mock_baseline_result, mock_critical_ablation):
        """Test computing impact for critical component."""
        impact = compute_component_impact(mock_baseline_result, mock_critical_ablation)

        assert impact.component == "ontology"
        assert impact.delta_precision == pytest.approx(-0.20, abs=0.01)
        assert impact.is_critical is True
        assert "KEEP" in impact.recommendation

    def test_compute_marginal_impact(self, mock_baseline_result, mock_marginal_ablation):
        """Test computing impact for marginal component."""
        impact = compute_component_impact(mock_baseline_result, mock_marginal_ablation)

        assert impact.component == "readiness"
        assert impact.delta_precision == pytest.approx(-0.02, abs=0.01)
        assert impact.is_critical is False
        assert "OPTIONAL" in impact.recommendation


class TestAblationReportWithComponentImpacts:
    """Tests for ablation reports with component impacts."""

    @pytest.fixture
    def sample_report_with_impacts(self):
        """Create sample report with component impacts."""
        impacts = [
            ComponentImpact(
                component="ontology",
                baseline_precision=0.65,
                ablated_precision=0.45,
                delta_precision=-0.20,
                baseline_mrr=0.70,
                ablated_mrr=0.50,
                delta_mrr=-0.20,
                is_critical=True,
                recommendation="KEEP - critical",
            ),
            ComponentImpact(
                component="readiness",
                baseline_precision=0.65,
                ablated_precision=0.63,
                delta_precision=-0.02,
                baseline_mrr=0.70,
                ablated_mrr=0.68,
                delta_mrr=-0.02,
                is_critical=False,
                recommendation="OPTIONAL - marginal",
            ),
        ]

        return AblationReport(
            generated_at="2024-01-15T12:00:00Z",
            suite="test_suite",
            modes=["hybrid_default", "no_ontology", "no_readiness"],
            results=[],
            comparison=[],
            summary={},
            component_impacts=impacts,
        )

    def test_markdown_includes_component_impacts(self, sample_report_with_impacts):
        """Test markdown includes component impact section."""
        markdown = generate_ablation_markdown(sample_report_with_impacts)

        assert "Component Impact Analysis" in markdown
        assert "ontology" in markdown
        assert "readiness" in markdown

    def test_markdown_includes_critical_markers(self, sample_report_with_impacts):
        """Test markdown marks critical components."""
        markdown = generate_ablation_markdown(sample_report_with_impacts)

        # Should have critical components section
        assert "Critical Components" in markdown
