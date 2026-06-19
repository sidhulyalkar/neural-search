#!/usr/bin/env python3
"""Run calibration analysis against human relevance judgments.

This script:
1. Loads benchmark queries and runs search
2. Loads human relevance labels
3. Computes calibration metrics (ECE, MCE, Brier score)
4. Generates a calibration report with reliability diagram data

Usage:
    python scripts/run_calibration_analysis.py
    python scripts/run_calibration_analysis.py --suite demo_v02 --output data/reports/calibration_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from neural_search.evaluation import load_benchmark_queries
from neural_search.evaluation.calibration import (
    CalibrationConfig,
    CalibrationResult,
    compute_calibration_metrics,
    explain_calibration,
)

# Relevance string → numeric score mapping (migrated from deleted evaluation.relevance)
RELEVANCE_SCORES: dict[str, int] = {
    "exact": 3,
    "highly_relevant": 2,
    "partially": 1,
    "not_relevant": 0,
    "hard_negative": -1,
}
from neural_search.search import search_datasets

# Paths
BENCHMARK_DIR = project_root / "data" / "eval"
LABELS_PATH = project_root / "data" / "eval" / "relevance_labels_v01.jsonl"
OUTPUT_DIR = project_root / "data" / "reports"

# Relevance threshold for binary calibration
# exact=3, highly_relevant=2, partially=1, not_relevant=0, hard_negative=-1
RELEVANCE_THRESHOLD = 1  # At least "partially" relevant


def load_all_labels(path: Path) -> dict[str, dict[str, int]]:
    """Load relevance labels grouped by query_id.

    Returns:
        Dict mapping query_id -> {dataset_id: relevance_score}
    """
    labels_by_query: dict[str, dict[str, int]] = {}

    if not path.exists():
        return labels_by_query

    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            query_id = record.get("query_id", "")
            dataset_id = record.get("dataset_id", "")
            relevance_str = record.get("relevance", "not_relevant")

            # Convert relevance string to score
            score = RELEVANCE_SCORES.get(relevance_str, 0)

            if query_id not in labels_by_query:
                labels_by_query[query_id] = {}
            labels_by_query[query_id][dataset_id] = score

    return labels_by_query


def run_calibration_from_labels(
    labels_path: Path,
    relevance_threshold: int = 1,
    top_k: int = 10,
) -> tuple[CalibrationResult, list[dict]]:
    """Run calibration analysis using queries from relevance labels file.

    Args:
        labels_path: Path to relevance labels file
        relevance_threshold: Minimum score for "relevant" (default: 1 = partially)
        top_k: Number of results to evaluate per query

    Returns:
        Tuple of (CalibrationResult, list of per-query details)
    """
    # Load labels and extract unique queries
    all_labels = load_all_labels(labels_path)

    # Also extract query texts from the labels file
    query_texts: dict[str, str] = {}
    if labels_path.exists():
        with open(labels_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                query_id = record.get("query_id", "")
                query_text = record.get("query_text", "")
                if query_id and query_text:
                    query_texts[query_id] = query_text

    if not query_texts:
        print("No queries found in labels file")
        return None, []

    # Collect predictions and labels
    all_confidences: list[float] = []
    all_binary_labels: list[int] = []
    per_query_details: list[dict] = []

    for query_id, query_text in query_texts.items():
        # Get labels for this query
        query_labels = all_labels.get(query_id, {})
        if not query_labels:
            continue

        print(f"  Processing: {query_id} -> {query_text[:50]}...")

        # Run search
        try:
            response = search_datasets(query_text, limit=top_k)
            results = response.results  # Access results from SearchResponse
        except Exception as e:
            print(f"Error searching for {query_id}: {e}")
            continue

        query_confidences: list[float] = []
        query_binary_labels: list[int] = []
        query_matches: list[dict] = []

        for result in results:
            # SearchResult is a pydantic model, access attributes directly
            dataset_id = str(result.dataset_id)
            score = float(result.score)

            # Check if we have a label for this dataset
            if dataset_id in query_labels:
                relevance_score = query_labels[dataset_id]
                binary_label = 1 if relevance_score >= relevance_threshold else 0

                all_confidences.append(score)
                all_binary_labels.append(binary_label)
                query_confidences.append(score)
                query_binary_labels.append(binary_label)

                query_matches.append({
                    "dataset_id": dataset_id,
                    "score": round(score, 4),
                    "relevance_score": relevance_score,
                    "binary_label": binary_label,
                })

        if query_matches:
            per_query_details.append({
                "query_id": query_id,
                "query_text": query_text,
                "num_labeled_results": len(query_matches),
                "matches": query_matches,
            })

    if not all_confidences:
        print("No labeled results found")
        return None, []

    # Normalize scores to 0-1 range using min-max scaling
    min_score = min(all_confidences)
    max_score = max(all_confidences)
    score_range = max_score - min_score if max_score > min_score else 1.0

    normalized_confidences = [
        (score - min_score) / score_range for score in all_confidences
    ]

    print(f"  Score range: {min_score:.2f} - {max_score:.2f}")
    print("  Normalized to 0-1 for calibration analysis")

    # Update per-query details with normalized scores
    idx = 0
    for detail in per_query_details:
        for match in detail["matches"]:
            match["normalized_score"] = round(normalized_confidences[idx], 4)
            idx += 1

    # Compute calibration on normalized scores
    config = CalibrationConfig(num_bins=10, adaptive_bins=False)
    result = compute_calibration_metrics(normalized_confidences, all_binary_labels, config)

    return result, per_query_details


def run_calibration_for_suite(
    suite: str,
    labels_path: Path,
    relevance_threshold: int = 1,
    top_k: int = 10,
) -> tuple[CalibrationResult, list[dict]]:
    """Run calibration analysis for a benchmark suite.

    Args:
        suite: Benchmark suite name (or "from_labels" to use labels file directly)
        labels_path: Path to relevance labels file
        relevance_threshold: Minimum score for "relevant" (default: 1 = partially)
        top_k: Number of results to evaluate per query

    Returns:
        Tuple of (CalibrationResult, list of per-query details)
    """
    # If using labels file directly
    if suite == "from_labels":
        return run_calibration_from_labels(labels_path, relevance_threshold, top_k)

    # Load benchmark queries
    suite_path = BENCHMARK_DIR / f"benchmark_queries_{suite}.yaml"
    if not suite_path.exists():
        # Try without prefix for backwards compatibility
        suite_path = BENCHMARK_DIR / f"{suite}.yaml"
        if not suite_path.exists():
            print(f"Warning: Suite {suite} not found, using labels file directly...")
            return run_calibration_from_labels(labels_path, relevance_threshold, top_k)

    queries = load_benchmark_queries(str(suite_path))

    # Load labels
    all_labels = load_all_labels(labels_path)

    # Collect predictions and labels
    all_confidences: list[float] = []
    all_binary_labels: list[int] = []
    per_query_details: list[dict] = []

    for query in queries:
        query_id = query.id
        query_text = query.query

        # Get labels for this query
        query_labels = all_labels.get(query_id, {})
        if not query_labels:
            continue

        # Run search
        try:
            response = search_datasets(query_text, limit=top_k)
            results = response.results  # Access results from SearchResponse
        except Exception as e:
            print(f"Error searching for {query_id}: {e}")
            continue

        query_confidences: list[float] = []
        query_binary_labels: list[int] = []
        query_matches: list[dict] = []

        for result in results:
            # SearchResult is a pydantic model, access attributes directly
            dataset_id = str(result.dataset_id)
            score = float(result.score)

            # Check if we have a label for this dataset
            if dataset_id in query_labels:
                relevance_score = query_labels[dataset_id]
                binary_label = 1 if relevance_score >= relevance_threshold else 0

                all_confidences.append(score)
                all_binary_labels.append(binary_label)
                query_confidences.append(score)
                query_binary_labels.append(binary_label)

                query_matches.append({
                    "dataset_id": dataset_id,
                    "score": round(score, 4),
                    "relevance_score": relevance_score,
                    "binary_label": binary_label,
                })

        if query_matches:
            per_query_details.append({
                "query_id": query_id,
                "query_text": query_text,
                "num_labeled_results": len(query_matches),
                "matches": query_matches,
            })

    if not all_confidences:
        print("No labeled results in benchmark suite, using labels file directly...")
        return run_calibration_from_labels(labels_path, relevance_threshold, top_k)

    # Normalize scores to 0-1 range using min-max scaling
    min_score = min(all_confidences)
    max_score = max(all_confidences)
    score_range = max_score - min_score if max_score > min_score else 1.0

    normalized_confidences = [
        (score - min_score) / score_range for score in all_confidences
    ]

    # Update per-query details with normalized scores
    idx = 0
    for detail in per_query_details:
        for match in detail["matches"]:
            match["normalized_score"] = round(normalized_confidences[idx], 4)
            idx += 1

    # Compute calibration on normalized scores
    config = CalibrationConfig(num_bins=10, adaptive_bins=False)
    result = compute_calibration_metrics(normalized_confidences, all_binary_labels, config)

    return result, per_query_details


def generate_calibration_report(
    result: CalibrationResult,
    per_query_details: list[dict],
    suite: str,
) -> dict:
    """Generate comprehensive calibration report."""
    explanation = explain_calibration(result)

    # Build reliability diagram data
    reliability_diagram = []
    for bin_ in result.bins:
        if bin_.count > 0:
            reliability_diagram.append({
                "bin_range": f"[{bin_.lower_bound:.1f}, {bin_.upper_bound:.1f})",
                "count": bin_.count,
                "mean_confidence": round(bin_.mean_confidence, 4),
                "mean_accuracy": round(bin_.mean_accuracy, 4),
                "calibration_error": round(bin_.calibration_error, 4),
                "positive_count": bin_.positive_count,
                "negative_count": bin_.negative_count,
            })

    report = {
        "suite": suite,
        "summary": {
            "total_samples": result.total_samples,
            "positive_samples": result.positive_samples,
            "negative_samples": result.negative_samples,
            "ece": round(result.ece, 4),
            "mce": round(result.mce, 4),
            "brier_score": round(result.brier_score, 4),
            "mean_confidence": round(result.mean_confidence, 4),
            "mean_accuracy": round(result.mean_accuracy, 4),
            "calibration_slope": round(result.calibration_slope, 4),
            "calibration_intercept": round(result.calibration_intercept, 4),
        },
        "over_under_confidence": {
            "overconfidence_rate": round(result.curve.overconfidence_rate, 4),
            "underconfidence_rate": round(result.curve.underconfidence_rate, 4),
        },
        "reliability_diagram": reliability_diagram,
        "interpretation": explanation["interpretation"],
        "recommendations": explanation["recommendations"],
        "per_query_details": per_query_details,
    }

    return report


def format_markdown_report(report: dict) -> str:
    """Format calibration report as markdown."""
    lines = []
    lines.append("# Calibration Analysis Report")
    lines.append("")
    lines.append(f"**Suite:** {report['suite']}")
    lines.append("")

    summary = report["summary"]
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Samples | {summary['total_samples']} |")
    lines.append(f"| Positive Samples | {summary['positive_samples']} |")
    lines.append(f"| Negative Samples | {summary['negative_samples']} |")
    lines.append(f"| **ECE** | {summary['ece']:.4f} |")
    lines.append(f"| **MCE** | {summary['mce']:.4f} |")
    lines.append(f"| **Brier Score** | {summary['brier_score']:.4f} |")
    lines.append(f"| Mean Confidence | {summary['mean_confidence']:.4f} |")
    lines.append(f"| Mean Accuracy | {summary['mean_accuracy']:.4f} |")
    lines.append(f"| Calibration Slope | {summary['calibration_slope']:.4f} |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    for interp in report["interpretation"]:
        lines.append(f"- {interp}")
    lines.append("")

    ouc = report["over_under_confidence"]
    lines.append("## Over/Under Confidence")
    lines.append("")
    lines.append(f"- Overconfidence Rate: {ouc['overconfidence_rate']:.1%}")
    lines.append(f"- Underconfidence Rate: {ouc['underconfidence_rate']:.1%}")
    lines.append("")

    lines.append("## Reliability Diagram")
    lines.append("")
    lines.append("| Bin Range | Count | Confidence | Accuracy | Error |")
    lines.append("|-----------|-------|------------|----------|-------|")
    for bin_data in report["reliability_diagram"]:
        lines.append(
            f"| {bin_data['bin_range']} | {bin_data['count']} | "
            f"{bin_data['mean_confidence']:.3f} | {bin_data['mean_accuracy']:.3f} | "
            f"{bin_data['calibration_error']:.3f} |"
        )
    lines.append("")

    if report["recommendations"]:
        lines.append("## Recommendations")
        lines.append("")
        for rec in report["recommendations"]:
            lines.append(f"- {rec}")
        lines.append("")

    # Quality assessment
    lines.append("## Calibration Quality Assessment")
    lines.append("")
    ece = summary["ece"]
    if ece < 0.05:
        quality = "✅ **Excellent** (ECE < 0.05)"
    elif ece < 0.10:
        quality = "✅ **Good** (ECE < 0.10)"
    elif ece < 0.15:
        quality = "⚠️ **Moderate** (ECE < 0.15)"
    elif ece < 0.20:
        quality = "⚠️ **Fair** (ECE < 0.20)"
    else:
        quality = "❌ **Poor** (ECE >= 0.20)"

    lines.append(f"Overall Quality: {quality}")
    lines.append("")

    slope = summary["calibration_slope"]
    if 0.8 <= slope <= 1.2:
        lines.append("Confidence scores are well-distributed across the range.")
    elif slope < 0.8:
        lines.append("⚠️ System tends to be **overconfident** (high scores for irrelevant results).")
    else:
        lines.append("⚠️ System tends to be **underconfident** (low scores for relevant results).")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run calibration analysis")
    parser.add_argument(
        "--suite",
        type=str,
        default="demo_v02",
        help="Benchmark suite to analyze",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=str(LABELS_PATH),
        help="Path to relevance labels file",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="Minimum relevance score for 'relevant' (default: 1=partially)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to evaluate per query",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        help="Output path for markdown report",
    )
    args = parser.parse_args()

    print(f"Running calibration analysis for suite: {args.suite}")
    print(f"Using labels from: {args.labels}")
    print(f"Relevance threshold: {args.threshold}")
    print()

    result, per_query_details = run_calibration_for_suite(
        suite=args.suite,
        labels_path=Path(args.labels),
        relevance_threshold=args.threshold,
        top_k=args.top_k,
    )

    if result is None:
        print("No calibration data available")
        return 1

    report = generate_calibration_report(result, per_query_details, args.suite)

    # Print summary
    print("=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    summary = report["summary"]
    print(f"Total labeled results: {summary['total_samples']}")
    print(f"  - Relevant: {summary['positive_samples']}")
    print(f"  - Not relevant: {summary['negative_samples']}")
    print()
    print(f"Expected Calibration Error (ECE): {summary['ece']:.4f}")
    print(f"Maximum Calibration Error (MCE):  {summary['mce']:.4f}")
    print(f"Brier Score:                      {summary['brier_score']:.4f}")
    print()
    print(f"Mean Confidence: {summary['mean_confidence']:.4f}")
    print(f"Mean Accuracy:   {summary['mean_accuracy']:.4f}")
    print(f"Calibration Slope: {summary['calibration_slope']:.4f}")
    print()

    print("Interpretation:")
    for interp in report["interpretation"]:
        print(f"  - {interp}")
    print()

    if report["recommendations"]:
        print("Recommendations:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")
        print()

    # Save reports
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_json = args.output or str(OUTPUT_DIR / f"calibration_{args.suite}.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"JSON report saved to: {output_json}")

    output_md = args.markdown or str(OUTPUT_DIR / f"calibration_{args.suite}.md")
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(format_markdown_report(report))
    print(f"Markdown report saved to: {output_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
