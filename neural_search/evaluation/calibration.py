"""Calibration metrics for search result confidence scoring.

This module computes reliability bins and confidence curves from human
relevance labels to evaluate how well system confidence corresponds to
actual relevance.

Mathematical formalization:
    Given a set of search results with confidence scores c_i and binary
    relevance labels r_i (from human judgments):

    1. Reliability Diagram: Group results into B bins by confidence, compute
       accuracy in each bin: acc(B_k) = (1/|B_k|) * sum_{i in B_k} r_i

    2. Expected Calibration Error (ECE):
       ECE = sum_k (|B_k|/N) * |acc(B_k) - conf(B_k)|
       where conf(B_k) = (1/|B_k|) * sum_{i in B_k} c_i

    3. Maximum Calibration Error (MCE):
       MCE = max_k |acc(B_k) - conf(B_k)|

    4. Brier Score: mean((c_i - r_i)^2)
       Measures overall calibration and refinement
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class ReliabilityBin:
    """Statistics for one calibration bin."""

    bin_index: int
    lower_bound: float  # Inclusive
    upper_bound: float  # Exclusive (except last bin)
    count: int
    mean_confidence: float
    mean_accuracy: float  # Proportion of positive labels
    calibration_error: float  # |accuracy - confidence|
    positive_count: int
    negative_count: int


@dataclass
class CalibrationCurve:
    """Reliability diagram data for plotting."""

    # Bin boundaries
    bin_edges: tuple[float, ...]

    # Per-bin statistics
    bin_accuracies: tuple[float, ...]
    bin_confidences: tuple[float, ...]
    bin_counts: tuple[int, ...]

    # Summary metrics
    expected_calibration_error: float  # ECE
    maximum_calibration_error: float  # MCE
    brier_score: float
    overconfidence_rate: float  # Fraction of predictions more confident than accurate
    underconfidence_rate: float  # Fraction less confident than accurate


@dataclass
class CalibrationResult:
    """Complete calibration analysis result."""

    total_samples: int
    positive_samples: int
    negative_samples: int

    # Binned calibration
    bins: tuple[ReliabilityBin, ...]
    curve: CalibrationCurve

    # Summary metrics
    ece: float
    mce: float
    brier_score: float

    # Additional statistics
    mean_confidence: float
    mean_accuracy: float
    calibration_slope: float  # Slope of calibration curve (1.0 = perfect)
    calibration_intercept: float


@dataclass
class CalibrationConfig:
    """Configuration for calibration computation."""

    num_bins: int = 10
    adaptive_bins: bool = False  # Use equal-count bins instead of equal-width
    confidence_threshold: float = 0.5  # Threshold for binary classification
    min_samples_per_bin: int = 1


def compute_reliability_bins(
    confidences: Sequence[float],
    labels: Sequence[int | bool],
    config: CalibrationConfig | None = None,
) -> tuple[ReliabilityBin, ...]:
    """Compute reliability bins for calibration analysis.

    Args:
        confidences: Model confidence scores (0-1).
        labels: Binary relevance labels (0/False=irrelevant, 1/True=relevant).
        config: Optional configuration.

    Returns:
        Tuple of ReliabilityBin objects.
    """
    config = config or CalibrationConfig()

    if len(confidences) != len(labels):
        raise ValueError("confidences and labels must have same length")

    if not confidences:
        return ()

    # Convert to lists for processing
    conf_list = [float(c) for c in confidences]
    label_list = [int(bool(lbl)) for lbl in labels]

    # Determine bin edges
    if config.adaptive_bins:
        # Equal-count bins (quantile-based)
        sorted_confs = sorted(conf_list)
        n = len(sorted_confs)
        bin_edges = [0.0]
        for i in range(1, config.num_bins):
            idx = int(i * n / config.num_bins)
            bin_edges.append(sorted_confs[min(idx, n - 1)])
        bin_edges.append(1.0)
        # Remove duplicates while preserving order
        seen = set()
        unique_edges = []
        for edge in bin_edges:
            if edge not in seen:
                seen.add(edge)
                unique_edges.append(edge)
        bin_edges = unique_edges
    else:
        # Equal-width bins
        bin_edges = [i / config.num_bins for i in range(config.num_bins + 1)]

    # Build bins
    bins: list[ReliabilityBin] = []

    for bin_idx in range(len(bin_edges) - 1):
        lower = bin_edges[bin_idx]
        upper = bin_edges[bin_idx + 1]

        # Last bin is inclusive on upper bound
        is_last = bin_idx == len(bin_edges) - 2

        # Collect samples in this bin
        bin_confs: list[float] = []
        bin_labels: list[int] = []

        for conf, label in zip(conf_list, label_list, strict=False):
            if is_last:
                in_bin = lower <= conf <= upper
            else:
                in_bin = lower <= conf < upper

            if in_bin:
                bin_confs.append(conf)
                bin_labels.append(label)

        count = len(bin_confs)
        positive_count = sum(bin_labels)
        negative_count = count - positive_count

        if count > 0:
            mean_conf = sum(bin_confs) / count
            mean_acc = positive_count / count
            cal_error = abs(mean_acc - mean_conf)
        else:
            mean_conf = (lower + upper) / 2
            mean_acc = 0.0
            cal_error = 0.0

        bins.append(ReliabilityBin(
            bin_index=bin_idx,
            lower_bound=lower,
            upper_bound=upper,
            count=count,
            mean_confidence=round(mean_conf, 6),
            mean_accuracy=round(mean_acc, 6),
            calibration_error=round(cal_error, 6),
            positive_count=positive_count,
            negative_count=negative_count,
        ))

    return tuple(bins)


def compute_calibration_metrics(
    confidences: Sequence[float],
    labels: Sequence[int | bool],
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    """Compute full calibration analysis.

    Args:
        confidences: Model confidence scores (0-1).
        labels: Binary relevance labels.
        config: Optional configuration.

    Returns:
        CalibrationResult with all metrics and binned statistics.
    """
    config = config or CalibrationConfig()

    if len(confidences) != len(labels):
        raise ValueError("confidences and labels must have same length")

    n = len(confidences)
    if n == 0:
        return CalibrationResult(
            total_samples=0,
            positive_samples=0,
            negative_samples=0,
            bins=(),
            curve=CalibrationCurve(
                bin_edges=(0.0, 1.0),
                bin_accuracies=(),
                bin_confidences=(),
                bin_counts=(),
                expected_calibration_error=0.0,
                maximum_calibration_error=0.0,
                brier_score=0.0,
                overconfidence_rate=0.0,
                underconfidence_rate=0.0,
            ),
            ece=0.0,
            mce=0.0,
            brier_score=0.0,
            mean_confidence=0.0,
            mean_accuracy=0.0,
            calibration_slope=1.0,
            calibration_intercept=0.0,
        )

    conf_list = [float(c) for c in confidences]
    label_list = [int(bool(lbl)) for lbl in labels]

    # Compute bins
    bins = compute_reliability_bins(conf_list, label_list, config)

    # Global statistics
    total_positive = sum(label_list)
    total_negative = n - total_positive
    mean_conf = sum(conf_list) / n
    mean_acc = total_positive / n

    # Brier score
    brier = sum((c - lbl) ** 2 for c, lbl in zip(conf_list, label_list, strict=False)) / n

    # ECE and MCE from bins
    ece = 0.0
    mce = 0.0
    for bin_ in bins:
        if bin_.count > 0:
            weight = bin_.count / n
            ece += weight * bin_.calibration_error
            mce = max(mce, bin_.calibration_error)

    # Over/underconfidence rates
    overconf = sum(1 for c, lbl in zip(conf_list, label_list, strict=False) if c > lbl) / n
    underconf = sum(1 for c, lbl in zip(conf_list, label_list, strict=False) if c < lbl) / n

    # Calibration curve data
    bin_edges = tuple([b.lower_bound for b in bins] + [bins[-1].upper_bound] if bins else [0.0, 1.0])
    bin_accuracies = tuple(b.mean_accuracy for b in bins)
    bin_confidences = tuple(b.mean_confidence for b in bins)
    bin_counts = tuple(b.count for b in bins)

    # Simple linear regression for calibration slope
    slope, intercept = _compute_calibration_slope(conf_list, label_list)

    curve = CalibrationCurve(
        bin_edges=bin_edges,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
        expected_calibration_error=round(ece, 6),
        maximum_calibration_error=round(mce, 6),
        brier_score=round(brier, 6),
        overconfidence_rate=round(overconf, 6),
        underconfidence_rate=round(underconf, 6),
    )

    return CalibrationResult(
        total_samples=n,
        positive_samples=total_positive,
        negative_samples=total_negative,
        bins=bins,
        curve=curve,
        ece=round(ece, 6),
        mce=round(mce, 6),
        brier_score=round(brier, 6),
        mean_confidence=round(mean_conf, 6),
        mean_accuracy=round(mean_acc, 6),
        calibration_slope=round(slope, 6),
        calibration_intercept=round(intercept, 6),
    )


def _compute_calibration_slope(
    confidences: list[float],
    labels: list[int],
) -> tuple[float, float]:
    """Compute slope and intercept of calibration curve using linear regression.

    Returns (slope, intercept) where perfect calibration has slope=1, intercept=0.
    """
    n = len(confidences)
    if n < 2:
        return 1.0, 0.0

    # Simple OLS: labels ~ confidences
    sum_x = sum(confidences)
    sum_y = sum(labels)
    sum_xy = sum(c * lbl for c, lbl in zip(confidences, labels, strict=False))
    sum_x2 = sum(c * c for c in confidences)

    mean_x = sum_x / n
    mean_y = sum_y / n

    denom = sum_x2 - n * mean_x * mean_x
    if abs(denom) < 1e-10:
        return 1.0, mean_y - mean_x

    slope = (sum_xy - n * mean_x * mean_y) / denom
    intercept = mean_y - slope * mean_x

    return slope, intercept


def calibrate_from_labels(
    predictions: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int | bool],
    confidence_key: str = "score",
    id_key: str = "dataset_id",
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    """Compute calibration from predictions and label mapping.

    Args:
        predictions: List of prediction dicts with confidence scores.
        labels: Mapping from item ID to binary relevance label.
        confidence_key: Key for confidence score in prediction dicts.
        id_key: Key for item ID in prediction dicts.
        config: Optional configuration.

    Returns:
        CalibrationResult with calibration analysis.
    """
    confidences: list[float] = []
    binary_labels: list[int] = []

    for pred in predictions:
        item_id = pred.get(id_key, "")
        if item_id in labels:
            conf = float(pred.get(confidence_key, 0.0))
            label = int(bool(labels[item_id]))
            confidences.append(conf)
            binary_labels.append(label)

    return compute_calibration_metrics(confidences, binary_labels, config)


def explain_calibration(result: CalibrationResult) -> dict[str, Any]:
    """Generate human-readable calibration explanation.

    Args:
        result: CalibrationResult to explain.

    Returns:
        Dictionary with explanation and recommendations.
    """
    explanation: dict[str, Any] = {
        "summary": {},
        "interpretation": [],
        "bin_analysis": [],
        "recommendations": [],
    }

    # Summary
    explanation["summary"] = {
        "total_samples": result.total_samples,
        "positive_rate": round(result.mean_accuracy, 4),
        "mean_confidence": round(result.mean_confidence, 4),
        "ece": round(result.ece, 4),
        "mce": round(result.mce, 4),
        "brier_score": round(result.brier_score, 4),
    }

    # Interpretation
    if result.ece < 0.05:
        explanation["interpretation"].append("Excellent calibration (ECE < 0.05)")
    elif result.ece < 0.10:
        explanation["interpretation"].append("Good calibration (ECE < 0.10)")
    elif result.ece < 0.20:
        explanation["interpretation"].append("Moderate calibration (ECE < 0.20)")
    else:
        explanation["interpretation"].append("Poor calibration (ECE >= 0.20)")

    slope_diff = abs(result.calibration_slope - 1.0)
    if slope_diff < 0.1:
        explanation["interpretation"].append("Confidence scores are well-calibrated across range")
    elif result.calibration_slope < 0.9:
        explanation["interpretation"].append("System is overconfident (slope < 0.9)")
    elif result.calibration_slope > 1.1:
        explanation["interpretation"].append("System is underconfident (slope > 1.1)")

    # Bin analysis
    for bin_ in result.bins:
        if bin_.count > 0:
            status = "well-calibrated"
            if bin_.calibration_error > 0.15:
                if bin_.mean_confidence > bin_.mean_accuracy:
                    status = "overconfident"
                else:
                    status = "underconfident"

            explanation["bin_analysis"].append({
                "range": f"[{bin_.lower_bound:.1f}, {bin_.upper_bound:.1f})",
                "count": bin_.count,
                "accuracy": round(bin_.mean_accuracy, 3),
                "confidence": round(bin_.mean_confidence, 3),
                "status": status,
            })

    # Recommendations
    if result.ece > 0.10:
        if result.curve.overconfidence_rate > result.curve.underconfidence_rate:
            explanation["recommendations"].append(
                "Consider temperature scaling to reduce overconfidence"
            )
        else:
            explanation["recommendations"].append(
                "Consider confidence boosting for underconfident predictions"
            )

    if result.mce > 0.20:
        worst_bins = sorted(result.bins, key=lambda b: -b.calibration_error)[:2]
        ranges = [f"[{b.lower_bound:.1f}, {b.upper_bound:.1f})" for b in worst_bins if b.count > 0]
        if ranges:
            explanation["recommendations"].append(
                f"Focus calibration efforts on confidence ranges: {', '.join(ranges)}"
            )

    return explanation


def compute_ece(
    confidences: Sequence[float],
    labels: Sequence[int | bool],
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Convenience function for just the ECE metric.

    Args:
        confidences: Model confidence scores (0-1).
        labels: Binary relevance labels.
        num_bins: Number of bins for calibration.

    Returns:
        ECE value (lower is better).
    """
    config = CalibrationConfig(num_bins=num_bins)
    result = compute_calibration_metrics(confidences, labels, config)
    return result.ece


def compute_brier_score(
    confidences: Sequence[float],
    labels: Sequence[int | bool],
) -> float:
    """Compute Brier score.

    Brier score = mean((confidence - label)^2)
    Lower is better. Range [0, 1].

    Args:
        confidences: Model confidence scores (0-1).
        labels: Binary relevance labels.

    Returns:
        Brier score.
    """
    if len(confidences) != len(labels):
        raise ValueError("confidences and labels must have same length")

    n = len(confidences)
    if n == 0:
        return 0.0

    return sum(
        (float(c) - int(bool(lbl))) ** 2
        for c, lbl in zip(confidences, labels, strict=False)
    ) / n


# =============================================================================
# Calibration-Based Confidence Adjustment
# =============================================================================


@dataclass
class CalibrationAdjustment:
    """Calibration adjustment parameters for score correction."""

    temperature: float = 1.0  # Platt scaling temperature
    slope: float = 1.0  # Linear calibration slope
    intercept: float = 0.0  # Linear calibration intercept
    method: str = "temperature"  # "temperature", "linear", "isotonic"

    @classmethod
    def from_calibration_result(
        cls,
        result: CalibrationResult,
        method: str = "linear",
    ) -> CalibrationAdjustment:
        """Create adjustment parameters from calibration analysis.

        Args:
            result: CalibrationResult from calibration analysis
            method: Adjustment method to use

        Returns:
            CalibrationAdjustment with computed parameters
        """
        if method == "linear":
            # Use the calibration slope and intercept directly
            return cls(
                slope=result.calibration_slope,
                intercept=result.calibration_intercept,
                method="linear",
            )
        elif method == "temperature":
            # Estimate temperature from calibration slope
            # Higher slope = underconfident = lower temperature
            # Lower slope = overconfident = higher temperature
            if result.calibration_slope > 0.1:
                temperature = 1.0 / result.calibration_slope
            else:
                temperature = 1.0
            return cls(
                temperature=min(max(temperature, 0.5), 2.0),
                method="temperature",
            )
        else:
            return cls()


def adjust_confidence(
    score: float,
    adjustment: CalibrationAdjustment,
) -> float:
    """Adjust a confidence score using calibration parameters.

    Args:
        score: Original confidence score (0-100 or 0-1)
        adjustment: Calibration adjustment parameters

    Returns:
        Adjusted confidence score in same scale as input
    """
    # Normalize to 0-1 if needed
    is_percentage = score > 1.0
    normalized = score / 100.0 if is_percentage else score

    if adjustment.method == "temperature":
        # Platt scaling: calibrated = sigmoid(logit(score) / T)
        # Approximate without log for stability
        shifted = (normalized - 0.5) / adjustment.temperature
        adjusted = 0.5 + shifted * min(0.5, 0.5 / adjustment.temperature)
    elif adjustment.method == "linear":
        # Linear calibration: calibrated = slope * score + intercept
        adjusted = adjustment.slope * normalized + adjustment.intercept
    else:
        adjusted = normalized

    # Clamp to valid range
    adjusted = max(0.0, min(1.0, adjusted))

    return adjusted * 100.0 if is_percentage else adjusted


def adjust_search_results(
    results: list[dict[str, Any]],
    adjustment: CalibrationAdjustment,
    score_key: str = "score",
) -> list[dict[str, Any]]:
    """Adjust confidence scores in search results using calibration.

    Args:
        results: List of search result dicts
        adjustment: Calibration adjustment parameters
        score_key: Key for score in result dicts

    Returns:
        Results with adjusted scores (new list, original unchanged)
    """
    adjusted_results = []
    for result in results:
        new_result = dict(result)
        original_score = result.get(score_key, 0)
        new_result[score_key] = adjust_confidence(original_score, adjustment)
        new_result["original_score"] = original_score
        new_result["calibration_method"] = adjustment.method
        adjusted_results.append(new_result)
    return adjusted_results


def compute_calibration_adjustment(
    existing_labels: dict[str, Any],
    search_results: list[dict[str, Any]],
    method: str = "linear",
) -> CalibrationAdjustment:
    """Compute calibration adjustment from labeled data.

    Args:
        existing_labels: Relevance labels mapping query_id -> labels
        search_results: Search results with scores
        method: Adjustment method

    Returns:
        CalibrationAdjustment parameters
    """
    # Collect confidences and labels
    confidences: list[float] = []
    labels: list[int] = []

    for result in search_results:
        query_id = result.get("query_id", "")
        dataset_id = result.get("dataset_id", "")
        score = result.get("score", 0)

        # Normalize score to 0-1
        normalized_score = score / 100.0 if score > 1.0 else score

        # Check if we have a label
        if query_id in existing_labels:
            label_set = existing_labels[query_id]
            # Get relevance for this dataset
            if hasattr(label_set, "get_judgment_for_dataset"):
                judgment = label_set.get_judgment_for_dataset(dataset_id)
                if judgment:
                    # Convert relevance to binary
                    is_relevant = judgment.relevance_score > 0
                    confidences.append(normalized_score)
                    labels.append(int(is_relevant))

    if len(confidences) < 10:
        # Not enough data for calibration
        return CalibrationAdjustment()

    # Compute calibration metrics
    result = compute_calibration_metrics(confidences, labels)

    return CalibrationAdjustment.from_calibration_result(result, method)
