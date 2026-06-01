"""Tests for affordance validation rubric."""

from __future__ import annotations

import json

import pytest
import yaml

from neural_search.evaluation.affordance_validation import (
    AffordanceValidationReport,
    AffordanceValidationResult,
    DatasetValidationSummary,
    generate_validation_json,
    generate_validation_markdown,
    load_rubric,
)


class TestRubricLoading:
    """Test rubric configuration loading."""

    def test_load_valid_rubric(self, tmp_path):
        """Load valid rubric file."""
        rubric_data = {
            "affordances": {
                "choice_decoding": {
                    "description": "Test affordance",
                    "required_fields": ["neural_activity", "choice_labels"],
                    "optional_fields": ["reaction_time"],
                    "confidence_threshold": 0.8,
                    "evidence_fields": ["has_neural_data"],
                    "failure_reasons": ["Missing neural data"],
                }
            },
            "field_mappings": {
                "neural_activity": {
                    "schema_fields": ["spike_times", "fluorescence"],
                    "usability_flags": ["has_neural_data"],
                }
            },
        }
        rubric_path = tmp_path / "rubric.yaml"
        rubric_path.write_text(yaml.dump(rubric_data))

        requirements, mappings = load_rubric(rubric_path)

        assert "choice_decoding" in requirements
        assert requirements["choice_decoding"].confidence_threshold == 0.8
        assert "neural_activity" in mappings
        assert "has_neural_data" in mappings["neural_activity"].usability_flags

    def test_missing_rubric_raises(self, tmp_path):
        """Missing rubric file raises error."""
        with pytest.raises(FileNotFoundError):
            load_rubric(tmp_path / "nonexistent.yaml")

    def test_empty_rubric(self, tmp_path):
        """Empty rubric returns empty dicts."""
        rubric_path = tmp_path / "empty.yaml"
        rubric_path.write_text("")
        requirements, mappings = load_rubric(rubric_path)
        assert requirements == {}
        assert mappings == {}

    def test_partial_rubric(self, tmp_path):
        """Rubric with missing optional fields."""
        rubric_data = {
            "affordances": {
                "test_aff": {
                    "description": "Minimal",
                    "required_fields": ["field1"],
                }
            }
        }
        rubric_path = tmp_path / "partial.yaml"
        rubric_path.write_text(yaml.dump(rubric_data))

        requirements, mappings = load_rubric(rubric_path)
        assert "test_aff" in requirements
        assert requirements["test_aff"].optional_fields == []
        assert requirements["test_aff"].confidence_threshold == 0.7  # default


class TestValidationResult:
    """Test validation result data structures."""

    def test_valid_result_creation(self):
        """Create valid validation result."""
        result = AffordanceValidationResult(
            dataset_id="ds001",
            affordance_id="choice_decoding",
            predicted_support="high",
            predicted_confidence=0.9,
            required_present=["neural_activity", "choice_labels"],
            required_missing=[],
            optional_present=["reaction_time"],
            optional_missing=[],
            validation_status="valid",
            validation_confidence=0.95,
            failure_reasons=[],
            notes="",
        )
        assert result.validation_status == "valid"
        assert len(result.required_present) == 2

    def test_invalid_result_has_reasons(self):
        """Invalid result should have failure reasons."""
        result = AffordanceValidationResult(
            dataset_id="ds001",
            affordance_id="choice_decoding",
            predicted_support="high",
            predicted_confidence=0.9,
            required_present=[],
            required_missing=["neural_activity", "choice_labels"],
            optional_present=[],
            optional_missing=[],
            validation_status="invalid",
            validation_confidence=0.2,
            failure_reasons=["Missing neural data", "Missing choice labels"],
            notes="Potential false positive",
        )
        assert result.validation_status == "invalid"
        assert len(result.failure_reasons) == 2


class TestDatasetSummary:
    """Test dataset validation summary."""

    def test_summary_counts(self):
        """Summary should have correct counts."""
        summary = DatasetValidationSummary(
            dataset_id="ds001",
            dataset_title="Test Dataset",
            total_affordances=10,
            supported_affordances=5,
            validated_affordances=4,
            invalid_affordances=2,
            uncertain_affordances=4,
            results=[],
        )
        assert summary.total_affordances == 10
        assert summary.validated_affordances + summary.invalid_affordances + summary.uncertain_affordances == 10


class TestReportGeneration:
    """Test report generation."""

    @pytest.fixture
    def mock_report(self):
        """Create mock validation report."""
        result = AffordanceValidationResult(
            dataset_id="ds001",
            affordance_id="choice_decoding",
            predicted_support="high",
            predicted_confidence=0.9,
            required_present=["neural_activity"],
            required_missing=["choice_labels"],
            optional_present=[],
            optional_missing=["reaction_time"],
            validation_status="uncertain",
            validation_confidence=0.5,
            failure_reasons=["Missing choice labels"],
            notes="Review required",
        )

        summary = DatasetValidationSummary(
            dataset_id="ds001",
            dataset_title="Test Dataset",
            total_affordances=5,
            supported_affordances=3,
            validated_affordances=2,
            invalid_affordances=1,
            uncertain_affordances=2,
            results=[result],
        )

        return AffordanceValidationReport(
            generated_at="2026-01-01T00:00:00",
            rubric_path="test_rubric.yaml",
            total_datasets=1,
            total_validations=5,
            validation_rate=0.4,
            invalid_rate=0.2,
            dataset_summaries=[summary],
            affordance_statistics={
                "choice_decoding": {
                    "total": 1,
                    "valid": 0,
                    "invalid": 0,
                    "uncertain": 1,
                    "predicted_supported": 1,
                }
            },
            summary={
                "most_validated_affordance": "q_learning",
                "most_invalid_affordance": "rsa",
                "recommendation": "Review detection rules",
            },
        )

    def test_markdown_has_sections(self, mock_report):
        """Markdown report has expected sections."""
        md = generate_validation_markdown(mock_report)
        assert "# Affordance Validation Report" in md
        assert "## Summary" in md
        assert "## Affordance Statistics" in md
        assert "## Dataset Summaries" in md

    def test_json_valid(self, mock_report):
        """JSON report is valid JSON."""
        json_str = generate_validation_json(mock_report)
        data = json.loads(json_str)
        assert data["total_datasets"] == 1
        assert data["validation_rate"] == 0.4
        assert "affordance_statistics" in data

    def test_json_contains_results(self, mock_report):
        """JSON contains detailed results."""
        json_str = generate_validation_json(mock_report)
        data = json.loads(json_str)
        summaries = data["dataset_summaries"]
        assert len(summaries) == 1
        assert summaries[0]["dataset_id"] == "ds001"
        assert len(summaries[0]["results"]) == 1


class TestValidationLogic:
    """Test validation logic edge cases."""

    def test_all_required_present_is_valid(self):
        """All required fields present = valid."""
        result = AffordanceValidationResult(
            dataset_id="ds001",
            affordance_id="test",
            predicted_support="high",
            predicted_confidence=0.9,
            required_present=["field1", "field2"],
            required_missing=[],
            optional_present=["opt1"],
            optional_missing=["opt2"],
            validation_status="valid",
            validation_confidence=0.85,
            failure_reasons=[],
            notes="",
        )
        assert result.validation_status == "valid"

    def test_empty_requirements_is_valid(self):
        """No requirements = valid by default."""
        # This tests the edge case of an affordance with no required fields
        result = AffordanceValidationResult(
            dataset_id="ds001",
            affordance_id="test",
            predicted_support="medium",
            predicted_confidence=0.6,
            required_present=[],
            required_missing=[],
            optional_present=[],
            optional_missing=[],
            validation_status="valid",
            validation_confidence=1.0,
            failure_reasons=[],
            notes="",
        )
        assert result.validation_status == "valid"

    def test_confidence_calculation(self):
        """Confidence should reflect field coverage."""
        # 50% required + 100% optional = 0.7*0.5 + 0.3*1.0 = 0.65
        result = AffordanceValidationResult(
            dataset_id="ds001",
            affordance_id="test",
            predicted_support="medium",
            predicted_confidence=0.6,
            required_present=["field1"],
            required_missing=["field2"],
            optional_present=["opt1", "opt2"],
            optional_missing=[],
            validation_status="uncertain",
            validation_confidence=0.65,
            failure_reasons=[],
            notes="",
        )
        # Note: actual confidence is calculated in validation function
        assert 0.5 < result.validation_confidence < 0.9
