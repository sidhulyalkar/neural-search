"""Tests for ablation evaluation runner."""

from neural_search.evaluation.run_ablation import (
    ABLATION_MODES,
    AblationReport,
    _build_config,
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
