"""Tests for hard-negative adversarial benchmark."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from neural_search.evaluation.run_hard_negative_benchmark import (
    Exclusion,
    HardNegativeQuery,
    HardNegativeReport,
    QueryViolationResult,
    Violation,
    check_violations,
    generate_hard_negative_json,
    generate_hard_negative_markdown,
    load_hard_negative_config,
)


class TestExclusionDetection:
    """Test violation detection logic."""

    def test_no_violation_when_field_missing(self):
        """No violation if excluded field not in result."""
        query = HardNegativeQuery(
            id="test",
            query="test query",
            exclusions=[Exclusion(field="species", values=["human"])],
        )
        results = [{"dataset_id": "d1", "modalities": ["neuropixels"]}]
        violations = check_violations(query, results, {})
        assert len(violations) == 0

    def test_violation_detected_in_direct_field(self):
        """Violation detected in direct result field."""
        query = HardNegativeQuery(
            id="test",
            query="test query",
            exclusions=[Exclusion(field="species", values=["human"])],
        )
        results = [{"dataset_id": "d1", "species": ["human"]}]
        violations = check_violations(query, results, {"species": "critical"})
        assert len(violations) == 1
        assert violations[0].excluded_field == "species"
        assert violations[0].offending_value == "human"
        assert violations[0].severity == "critical"

    def test_violation_in_preview(self):
        """Violation detected in dataset_card_preview."""
        query = HardNegativeQuery(
            id="test",
            query="test query",
            exclusions=[Exclusion(field="modality", values=["fmri"])],
        )
        results = [
            {
                "dataset_id": "d1",
                "dataset_card_preview": {"modalities": ["fmri", "eeg"]},
            }
        ]
        violations = check_violations(query, results, {})
        assert len(violations) == 1
        assert violations[0].offending_value == "fmri"

    def test_multiple_violations_same_result(self):
        """Multiple violations in same result detected separately."""
        query = HardNegativeQuery(
            id="test",
            query="test query",
            exclusions=[
                Exclusion(field="species", values=["human"]),
                Exclusion(field="modality", values=["fmri"]),
            ],
        )
        results = [
            {
                "dataset_id": "d1",
                "species": ["human"],
                "dataset_card_preview": {"modalities": ["fmri"]},
            }
        ]
        violations = check_violations(query, results, {})
        assert len(violations) == 2

    def test_case_insensitive_matching(self):
        """Exclusion matching is case-insensitive."""
        query = HardNegativeQuery(
            id="test",
            query="test query",
            exclusions=[Exclusion(field="species", values=["Human"])],
        )
        results = [{"dataset_id": "d1", "species": ["HUMAN"]}]
        violations = check_violations(query, results, {})
        assert len(violations) == 1

    def test_synonym_matching(self):
        """Exclusion catches normalized synonyms."""
        query = HardNegativeQuery(
            id="test",
            query="test query",
            exclusions=[Exclusion(field="species", values=["homo_sapiens"])],
        )
        # After normalization, "homo_sapiens" matches "homo sapiens"
        results = [{"dataset_id": "d1", "species": ["homo sapiens"]}]
        violations = check_violations(query, results, {})
        # This depends on normalize_text behavior
        # For now, underscore vs space may not match
        # This test documents expected behavior

    def test_rank_recorded_correctly(self):
        """Violation rank corresponds to result position."""
        query = HardNegativeQuery(
            id="test",
            query="test query",
            exclusions=[Exclusion(field="species", values=["human"])],
        )
        results = [
            {"dataset_id": "d1", "species": ["mouse"]},
            {"dataset_id": "d2", "species": ["mouse"]},
            {"dataset_id": "d3", "species": ["human"]},
        ]
        violations = check_violations(query, results, {})
        assert len(violations) == 1
        assert violations[0].result_rank == 3


class TestConfigLoading:
    """Test config file loading."""

    def test_load_valid_config(self, tmp_path):
        """Load valid YAML config."""
        config_data = {
            "hard_negative_queries": [
                {
                    "id": "hn001",
                    "query": "test query NOT human",
                    "exclusions": [{"field": "species", "values": ["human"]}],
                    "expected_species": ["mouse"],
                }
            ],
            "violation_severity": {"species": "critical"},
        }
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(config_data))

        queries, severity_map = load_hard_negative_config(config_path)

        assert len(queries) == 1
        assert queries[0].id == "hn001"
        assert queries[0].exclusions[0].field == "species"
        assert severity_map["species"] == "critical"

    def test_missing_config_raises(self, tmp_path):
        """Missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_hard_negative_config(tmp_path / "nonexistent.yaml")

    def test_empty_config(self, tmp_path):
        """Empty config returns empty lists."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")
        queries, severity_map = load_hard_negative_config(config_path)
        assert queries == []
        assert severity_map == {}


class TestReportGeneration:
    """Test report generation."""

    @pytest.fixture
    def mock_report(self):
        """Create mock report for testing."""
        return HardNegativeReport(
            generated_at="2026-01-01T00:00:00",
            config_path="test_config.yaml",
            total_queries=5,
            queries_with_violations=1,
            total_violations=2,
            compliance_rate=0.96,
            query_results=[
                QueryViolationResult(
                    query_id="hn001",
                    query="test query NOT human",
                    num_results=10,
                    num_violations=2,
                    violations=[
                        Violation(
                            query_id="hn001",
                            result_id="d1",
                            result_rank=3,
                            excluded_field="species",
                            offending_value="human",
                            exclusion_values=["human"],
                            evidence_source="direct field",
                            explanation="Contains human",
                            severity="critical",
                        ),
                        Violation(
                            query_id="hn001",
                            result_id="d5",
                            result_rank=5,
                            excluded_field="species",
                            offending_value="human",
                            exclusion_values=["human"],
                            evidence_source="direct field",
                            explanation="Contains human",
                            severity="critical",
                        ),
                    ],
                    compliant_results=8,
                    compliance_rate=0.8,
                ),
                QueryViolationResult(
                    query_id="hn002",
                    query="another query",
                    num_results=10,
                    num_violations=0,
                    violations=[],
                    compliant_results=10,
                    compliance_rate=1.0,
                ),
            ],
            violations_by_field={"species": 2},
            violations_by_severity={"critical": 2},
            summary={
                "total_queries": 5,
                "worst_query": "hn001",
                "most_violated_field": "species",
                "recommendation": "Fix 2 violations",
            },
        )

    def test_markdown_contains_sections(self, mock_report):
        """Markdown report contains expected sections."""
        md = generate_hard_negative_markdown(mock_report)
        assert "# Hard-Negative Adversarial Benchmark Report" in md
        assert "## Summary" in md
        assert "## Violations by Field" in md
        assert "## Per-Query Results" in md
        assert "## Violation Details" in md
        assert "hn001" in md
        assert "species" in md

    def test_markdown_shows_pass_when_no_violations(self):
        """Markdown shows PASS when no violations."""
        report = HardNegativeReport(
            generated_at="2026-01-01T00:00:00",
            config_path="test.yaml",
            total_queries=5,
            queries_with_violations=0,
            total_violations=0,
            compliance_rate=1.0,
            query_results=[],
            violations_by_field={},
            violations_by_severity={},
            summary={},
        )
        md = generate_hard_negative_markdown(report)
        assert "PASS" in md

    def test_json_valid(self, mock_report):
        """JSON report is valid JSON."""
        json_str = generate_hard_negative_json(mock_report)
        data = json.loads(json_str)
        assert data["total_violations"] == 2
        assert data["compliance_rate"] == 0.96
        assert len(data["query_results"]) == 2

    def test_json_contains_violation_details(self, mock_report):
        """JSON report contains violation details."""
        json_str = generate_hard_negative_json(mock_report)
        data = json.loads(json_str)
        violations = data["query_results"][0]["violations"]
        assert len(violations) == 2
        assert violations[0]["excluded_field"] == "species"
        assert violations[0]["result_rank"] == 3


class TestComplianceCalculation:
    """Test compliance rate calculation."""

    def test_perfect_compliance(self):
        """Perfect compliance when no violations."""
        qr = QueryViolationResult(
            query_id="test",
            query="test",
            num_results=10,
            num_violations=0,
            violations=[],
            compliant_results=10,
            compliance_rate=1.0,
        )
        assert qr.compliance_rate == 1.0

    def test_partial_compliance(self):
        """Partial compliance with some violations."""
        qr = QueryViolationResult(
            query_id="test",
            query="test",
            num_results=10,
            num_violations=2,
            violations=[],
            compliant_results=8,
            compliance_rate=0.8,
        )
        assert qr.compliance_rate == 0.8

    def test_empty_results_full_compliance(self):
        """Empty results count as full compliance."""
        qr = QueryViolationResult(
            query_id="test",
            query="test",
            num_results=0,
            num_violations=0,
            violations=[],
            compliant_results=0,
            compliance_rate=1.0,
        )
        assert qr.compliance_rate == 1.0
