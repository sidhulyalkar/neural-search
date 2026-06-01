"""Tests for calibration metrics module."""

from __future__ import annotations

import pytest

from neural_search.evaluation.calibration import (
    CalibrationConfig,
    CalibrationCurve,
    CalibrationResult,
    ReliabilityBin,
    calibrate_from_labels,
    compute_brier_score,
    compute_calibration_metrics,
    compute_ece,
    compute_reliability_bins,
    explain_calibration,
)


class TestComputeReliabilityBins:
    """Tests for reliability bin computation."""

    def test_basic_binning(self):
        """Test basic reliability bin computation."""
        confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

        bins = compute_reliability_bins(confidences, labels)

        assert len(bins) == 10
        assert all(isinstance(b, ReliabilityBin) for b in bins)
        # Each sample falls in its own bin
        assert sum(b.count for b in bins) == 10

    def test_custom_num_bins(self):
        """Test custom number of bins."""
        confidences = [0.1, 0.5, 0.9]
        labels = [0, 1, 1]

        config = CalibrationConfig(num_bins=3)
        bins = compute_reliability_bins(confidences, labels, config)

        assert len(bins) == 3
        assert bins[0].lower_bound == 0.0
        assert abs(bins[0].upper_bound - 1/3) < 0.01

    def test_empty_input(self):
        """Test with empty input."""
        bins = compute_reliability_bins([], [])
        assert bins == ()

    def test_perfect_calibration(self):
        """Test perfectly calibrated predictions."""
        # Confidence = accuracy in each bin
        confidences = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        # Labels match confidence levels (approximately)
        labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        bins = compute_reliability_bins(confidences, labels)

        # Low bins should have low accuracy, high bins should have high accuracy
        low_bins = [b for b in bins if b.mean_confidence < 0.5 and b.count > 0]
        high_bins = [b for b in bins if b.mean_confidence >= 0.5 and b.count > 0]

        if low_bins:
            avg_low_acc = sum(b.mean_accuracy for b in low_bins) / len(low_bins)
            assert avg_low_acc < 0.5
        if high_bins:
            avg_high_acc = sum(b.mean_accuracy for b in high_bins) / len(high_bins)
            assert avg_high_acc >= 0.5

    def test_overconfident_predictions(self):
        """Test detection of overconfident predictions."""
        # High confidence but low accuracy
        confidences = [0.9, 0.9, 0.9, 0.9, 0.9]
        labels = [0, 0, 0, 0, 1]  # Only 20% actually relevant

        bins = compute_reliability_bins(confidences, labels)

        high_conf_bin = [b for b in bins if b.count > 0][-1]
        assert high_conf_bin.mean_confidence > 0.8
        assert high_conf_bin.mean_accuracy < 0.5
        assert high_conf_bin.calibration_error > 0.3


class TestComputeCalibrationMetrics:
    """Tests for full calibration analysis."""

    def test_basic_calibration(self):
        """Test basic calibration computation."""
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [0, 0, 1, 1, 1]

        result = compute_calibration_metrics(confidences, labels)

        assert isinstance(result, CalibrationResult)
        assert result.total_samples == 5
        assert result.positive_samples == 3
        assert result.negative_samples == 2
        assert 0 <= result.ece <= 1
        assert 0 <= result.mce <= 1
        assert 0 <= result.brier_score <= 1

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        confidences = [0.0, 0.0, 1.0, 1.0]
        labels = [0, 0, 1, 1]

        result = compute_calibration_metrics(confidences, labels)

        assert result.brier_score == 0.0
        assert result.mean_accuracy == 0.5

    def test_worst_predictions(self):
        """Test with worst possible predictions."""
        confidences = [1.0, 1.0, 0.0, 0.0]
        labels = [0, 0, 1, 1]

        result = compute_calibration_metrics(confidences, labels)

        assert result.brier_score == 1.0

    def test_calibration_curve(self):
        """Test calibration curve data."""
        confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        result = compute_calibration_metrics(confidences, labels)

        assert isinstance(result.curve, CalibrationCurve)
        assert len(result.curve.bin_edges) > 0
        assert len(result.curve.bin_accuracies) == len(result.bins)
        assert len(result.curve.bin_confidences) == len(result.bins)

    def test_empty_calibration(self):
        """Test calibration with empty data."""
        result = compute_calibration_metrics([], [])

        assert result.total_samples == 0
        assert result.ece == 0.0
        assert result.mce == 0.0
        assert result.brier_score == 0.0

    def test_calibration_slope(self):
        """Test calibration slope computation."""
        # Well-calibrated predictions
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [0, 0, 1, 1, 1]

        result = compute_calibration_metrics(confidences, labels)

        # Slope should be close to 1 for well-calibrated
        assert 0 < result.calibration_slope < 3

    def test_overconfidence_rate(self):
        """Test overconfidence rate computation."""
        # All overconfident
        confidences = [0.9, 0.9, 0.9, 0.9]
        labels = [0, 0, 0, 0]

        result = compute_calibration_metrics(confidences, labels)

        assert result.curve.overconfidence_rate == 1.0
        assert result.curve.underconfidence_rate == 0.0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_ece(self):
        """Test ECE convenience function."""
        confidences = [0.1, 0.5, 0.9]
        labels = [0, 1, 1]

        ece = compute_ece(confidences, labels)

        assert isinstance(ece, float)
        assert 0 <= ece <= 1

    def test_compute_ece_custom_bins(self):
        """Test ECE with custom bin count."""
        confidences = [0.1, 0.5, 0.9]
        labels = [0, 1, 1]

        ece_5 = compute_ece(confidences, labels, num_bins=5)
        ece_20 = compute_ece(confidences, labels, num_bins=20)

        # Both should be valid
        assert 0 <= ece_5 <= 1
        assert 0 <= ece_20 <= 1

    def test_compute_brier_score(self):
        """Test Brier score computation."""
        confidences = [0.0, 1.0]
        labels = [0, 1]

        brier = compute_brier_score(confidences, labels)

        assert brier == 0.0  # Perfect predictions

    def test_compute_brier_score_imperfect(self):
        """Test Brier score with imperfect predictions."""
        confidences = [0.6, 0.6]
        labels = [0, 1]

        brier = compute_brier_score(confidences, labels)

        # (0.6 - 0)^2 + (0.6 - 1)^2 = 0.36 + 0.16 = 0.52, /2 = 0.26
        assert abs(brier - 0.26) < 0.001


class TestCalibrateFromLabels:
    """Tests for calibrating from prediction dicts and label mapping."""

    def test_basic_calibration_from_labels(self):
        """Test calibration from label mapping."""
        predictions = [
            {"dataset_id": "d1", "score": 0.9},
            {"dataset_id": "d2", "score": 0.3},
            {"dataset_id": "d3", "score": 0.7},
        ]
        labels = {"d1": 1, "d2": 0, "d3": 1}

        result = calibrate_from_labels(predictions, labels)

        assert result.total_samples == 3
        assert result.positive_samples == 2

    def test_missing_labels(self):
        """Test that missing labels are skipped."""
        predictions = [
            {"dataset_id": "d1", "score": 0.9},
            {"dataset_id": "d2", "score": 0.3},
            {"dataset_id": "d3", "score": 0.7},
        ]
        labels = {"d1": 1}  # Only one label

        result = calibrate_from_labels(predictions, labels)

        assert result.total_samples == 1

    def test_custom_keys(self):
        """Test with custom field keys."""
        predictions = [
            {"id": "item1", "confidence": 0.8},
            {"id": "item2", "confidence": 0.2},
        ]
        labels = {"item1": True, "item2": False}

        result = calibrate_from_labels(
            predictions,
            labels,
            confidence_key="confidence",
            id_key="id",
        )

        assert result.total_samples == 2


class TestExplainCalibration:
    """Tests for calibration explanation."""

    def test_explain_basic(self):
        """Test basic explanation generation."""
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [0, 0, 1, 1, 1]

        result = compute_calibration_metrics(confidences, labels)
        explanation = explain_calibration(result)

        assert "summary" in explanation
        assert "interpretation" in explanation
        assert "bin_analysis" in explanation
        assert "recommendations" in explanation

    def test_explain_summary_fields(self):
        """Test that summary contains required fields."""
        confidences = [0.5, 0.5]
        labels = [0, 1]

        result = compute_calibration_metrics(confidences, labels)
        explanation = explain_calibration(result)

        summary = explanation["summary"]
        assert "total_samples" in summary
        assert "ece" in summary
        assert "mce" in summary
        assert "brier_score" in summary

    def test_explain_good_calibration(self):
        """Test explanation for well-calibrated model."""
        # Near-perfect calibration
        confidences = [0.0, 0.0, 1.0, 1.0]
        labels = [0, 0, 1, 1]

        result = compute_calibration_metrics(confidences, labels)
        explanation = explain_calibration(result)

        # Should note good calibration
        assert any("calibration" in i.lower() for i in explanation["interpretation"])

    def test_explain_poor_calibration(self):
        """Test explanation for poorly-calibrated model."""
        # Very overconfident
        confidences = [0.95, 0.95, 0.95, 0.95]
        labels = [0, 0, 0, 0]

        result = compute_calibration_metrics(confidences, labels)
        explanation = explain_calibration(result)

        # Should have recommendations
        assert len(explanation["interpretation"]) > 0


class TestCalibrationConfig:
    """Tests for CalibrationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CalibrationConfig()

        assert config.num_bins == 10
        assert config.adaptive_bins is False
        assert config.confidence_threshold == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = CalibrationConfig(
            num_bins=20,
            adaptive_bins=True,
            confidence_threshold=0.7,
        )

        assert config.num_bins == 20
        assert config.adaptive_bins is True
        assert config.confidence_threshold == 0.7


class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_same_confidence(self):
        """Test when all confidences are the same."""
        confidences = [0.5, 0.5, 0.5, 0.5]
        labels = [0, 0, 1, 1]

        result = compute_calibration_metrics(confidences, labels)

        assert result.total_samples == 4
        assert result.mean_confidence == 0.5

    def test_all_same_label(self):
        """Test when all labels are the same."""
        confidences = [0.1, 0.5, 0.9]
        labels = [1, 1, 1]

        result = compute_calibration_metrics(confidences, labels)

        assert result.positive_samples == 3
        assert result.negative_samples == 0
        assert result.mean_accuracy == 1.0

    def test_single_sample(self):
        """Test with single sample."""
        confidences = [0.7]
        labels = [1]

        result = compute_calibration_metrics(confidences, labels)

        assert result.total_samples == 1
        # Use approximate comparison due to floating point
        assert abs(result.brier_score - (0.7 - 1) ** 2) < 0.001

    def test_length_mismatch_raises(self):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError, match="same length"):
            compute_calibration_metrics([0.5, 0.5], [1])

    def test_boolean_labels(self):
        """Test with boolean labels."""
        confidences = [0.3, 0.7]
        labels = [False, True]

        result = compute_calibration_metrics(confidences, labels)

        assert result.total_samples == 2
        assert result.positive_samples == 1

    def test_adaptive_bins(self):
        """Test adaptive (quantile-based) binning."""
        # Clustered confidences
        confidences = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
        labels = [0, 0, 0, 1, 1, 1]

        config = CalibrationConfig(num_bins=3, adaptive_bins=True)
        result = compute_calibration_metrics(confidences, labels, config)

        # Should still compute valid results
        assert result.total_samples == 6
        assert 0 <= result.ece <= 1


class TestReliabilityBinProperties:
    """Tests for ReliabilityBin dataclass."""

    def test_bin_counts_sum(self):
        """Test that bin counts sum to total samples."""
        confidences = list(range(100))
        confidences = [c / 100 for c in confidences]
        labels = [1 if c > 50 else 0 for c in range(100)]

        bins = compute_reliability_bins(confidences, labels)

        assert sum(b.count for b in bins) == 100

    def test_bin_positive_negative_sum(self):
        """Test that positive + negative = count per bin."""
        confidences = [0.1, 0.2, 0.3, 0.4, 0.5]
        labels = [0, 1, 0, 1, 1]

        bins = compute_reliability_bins(confidences, labels)

        for bin_ in bins:
            assert bin_.positive_count + bin_.negative_count == bin_.count

    def test_bin_bounds_coverage(self):
        """Test that bins cover [0, 1] range."""
        confidences = [0.5]
        labels = [1]

        config = CalibrationConfig(num_bins=5)
        bins = compute_reliability_bins(confidences, labels, config)

        assert bins[0].lower_bound == 0.0
        assert bins[-1].upper_bound == 1.0

        # No gaps
        for i in range(len(bins) - 1):
            assert bins[i].upper_bound == bins[i + 1].lower_bound
