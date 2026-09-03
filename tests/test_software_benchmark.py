"""Tests for historical scientific software audit benchmarking."""

import pytest

from neural_search.software.benchmark import (
    AuditBenchmarkPrediction,
    HistoricalAuditCase,
    compute_audit_benchmark_metrics,
)


def test_historical_benchmark_computes_component_and_verification_metrics() -> None:
    cases = [
        HistoricalAuditCase(
            case_id="kilosort-known-fix",
            repository_url="https://github.com/MouseLand/Kilosort",
            pre_fix_revision="abc",
            fix_revision="def",
            affected_component_ids=["kilosort:cluster"],
            expected_finding_summary="known historical behavior",
            scientific_domain="electrophysiology",
        ),
        HistoricalAuditCase(
            case_id="suite2p-known-fix",
            repository_url="https://github.com/MouseLand/suite2p",
            pre_fix_revision="123",
            fix_revision="456",
            affected_component_ids=["suite2p:phase"],
            expected_finding_summary="known historical behavior",
            scientific_domain="calcium_imaging",
        ),
    ]
    predictions = [
        AuditBenchmarkPrediction(
            case_id="kilosort-known-fix",
            ranked_component_ids=["kilosort:cluster"],
            finding_detected=True,
            reproducer_succeeded=True,
            patch_correct=True,
            regression_test_adequate=True,
        ),
        AuditBenchmarkPrediction(
            case_id="suite2p-known-fix",
            ranked_component_ids=["other", "suite2p:phase"],
            finding_detected=False,
            reproducer_succeeded=False,
            patch_correct=False,
            regression_test_adequate=True,
        ),
    ]

    metrics = compute_audit_benchmark_metrics(cases, predictions)

    assert metrics.component_recall_at_1 == 0.5
    assert metrics.component_recall_at_5 == 1.0
    assert metrics.finding_recall == 0.5
    assert metrics.reproducer_success_rate == 0.5
    assert metrics.patch_correctness_rate == 0.5
    assert metrics.regression_test_adequacy_rate == 1.0


def test_historical_benchmark_rejects_missing_predictions() -> None:
    cases = [
        HistoricalAuditCase(
            case_id="case-a",
            repository_url="https://example.test/repo",
            pre_fix_revision="abc",
            fix_revision="def",
            affected_component_ids=["component"],
            expected_finding_summary="known fix",
            scientific_domain="neuroimaging",
        )
    ]

    with pytest.raises(ValueError, match="missing benchmark predictions"):
        compute_audit_benchmark_metrics(cases, [])
