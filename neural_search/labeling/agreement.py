"""Multi-annotator agreement computation for relevance labels.

This module provides inter-annotator agreement metrics including
Cohen's kappa, Fleiss' kappa, and Krippendorff's alpha.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from neural_search.labeling.session import RelevanceLabel


@dataclass
class AnnotatorStats:
    """Statistics for a single annotator."""

    annotator_id: str
    n_labels: int
    grade_distribution: dict[int, int]
    mean_grade: float
    median_grade: float


@dataclass
class PairwiseAgreement:
    """Agreement between two annotators."""

    annotator_1: str
    annotator_2: str
    n_shared_items: int
    exact_agreement_rate: float
    within_one_agreement_rate: float
    cohens_kappa: float
    correlation: float


@dataclass
class MultiAnnotatorReport:
    """Complete multi-annotator agreement report."""

    timestamp: str
    n_annotators: int
    n_items: int
    n_labels: int
    items_with_multiple_annotations: int

    # Overall metrics
    fleiss_kappa: float
    krippendorff_alpha: float
    mean_pairwise_kappa: float
    overall_agreement_rate: float

    # Per-annotator stats
    annotator_stats: dict[str, AnnotatorStats] = field(default_factory=dict)

    # Pairwise comparisons
    pairwise_agreements: list[PairwiseAgreement] = field(default_factory=list)

    # Disagreement analysis
    high_disagreement_items: list[dict[str, Any]] = field(default_factory=list)
    agreement_by_grade: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "n_annotators": self.n_annotators,
            "n_items": self.n_items,
            "n_labels": self.n_labels,
            "items_with_multiple_annotations": self.items_with_multiple_annotations,
            "fleiss_kappa": self.fleiss_kappa,
            "krippendorff_alpha": self.krippendorff_alpha,
            "mean_pairwise_kappa": self.mean_pairwise_kappa,
            "overall_agreement_rate": self.overall_agreement_rate,
            "annotator_stats": {
                k: {
                    "annotator_id": v.annotator_id,
                    "n_labels": v.n_labels,
                    "grade_distribution": v.grade_distribution,
                    "mean_grade": v.mean_grade,
                    "median_grade": v.median_grade,
                }
                for k, v in self.annotator_stats.items()
            },
            "pairwise_agreements": [
                {
                    "annotator_1": p.annotator_1,
                    "annotator_2": p.annotator_2,
                    "n_shared_items": p.n_shared_items,
                    "exact_agreement_rate": p.exact_agreement_rate,
                    "cohens_kappa": p.cohens_kappa,
                }
                for p in self.pairwise_agreements
            ],
            "high_disagreement_items": self.high_disagreement_items[:20],
            "agreement_by_grade": self.agreement_by_grade,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Multi-Annotator Agreement Report",
            "",
            f"**Generated:** {self.timestamp}",
            "",
            "## Summary",
            "",
            f"- **Annotators:** {self.n_annotators}",
            f"- **Total Items:** {self.n_items}",
            f"- **Total Labels:** {self.n_labels}",
            f"- **Items with 2+ annotations:** {self.items_with_multiple_annotations}",
            "",
            "## Overall Agreement Metrics",
            "",
            "| Metric | Value | Interpretation |",
            "|--------|-------|----------------|",
            f"| Fleiss' Kappa | {self.fleiss_kappa:.3f} | {_interpret_kappa(self.fleiss_kappa)} |",
            f"| Krippendorff's Alpha | {self.krippendorff_alpha:.3f} | {_interpret_alpha(self.krippendorff_alpha)} |",
            f"| Mean Pairwise Kappa | {self.mean_pairwise_kappa:.3f} | {_interpret_kappa(self.mean_pairwise_kappa)} |",
            f"| Exact Agreement Rate | {self.overall_agreement_rate:.1%} | |",
            "",
        ]

        if self.annotator_stats:
            lines.extend([
                "## Per-Annotator Statistics",
                "",
                "| Annotator | Labels | Mean Grade | Median |",
                "|-----------|--------|------------|--------|",
            ])
            for ann_id, stats in self.annotator_stats.items():
                lines.append(
                    f"| {ann_id} | {stats.n_labels} | "
                    f"{stats.mean_grade:.2f} | {stats.median_grade:.1f} |"
                )
            lines.append("")

        if self.pairwise_agreements:
            lines.extend([
                "## Pairwise Agreement",
                "",
                "| Annotator 1 | Annotator 2 | Shared | Agreement | Kappa |",
                "|-------------|-------------|--------|-----------|-------|",
            ])
            for pair in self.pairwise_agreements:
                lines.append(
                    f"| {pair.annotator_1} | {pair.annotator_2} | "
                    f"{pair.n_shared_items} | {pair.exact_agreement_rate:.1%} | "
                    f"{pair.cohens_kappa:.3f} |"
                )
            lines.append("")

        if self.high_disagreement_items:
            lines.extend([
                "## High Disagreement Items (Top 10)",
                "",
                "| Item | Grades | Range |",
                "|------|--------|-------|",
            ])
            for item in self.high_disagreement_items[:10]:
                grades_str = ", ".join(str(g) for g in item.get("grades", []))
                lines.append(
                    f"| {item.get('item_id', 'unknown')[:30]} | "
                    f"{grades_str} | {item.get('range', 0)} |"
                )
            lines.append("")

        return "\n".join(lines)


def _interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's/Fleiss' kappa value."""
    if kappa < 0:
        return "Poor (worse than chance)"
    elif kappa < 0.20:
        return "Slight agreement"
    elif kappa < 0.40:
        return "Fair agreement"
    elif kappa < 0.60:
        return "Moderate agreement"
    elif kappa < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def _interpret_alpha(alpha: float) -> str:
    """Interpret Krippendorff's alpha value."""
    if alpha < 0.667:
        return "Tentative conclusions"
    elif alpha < 0.800:
        return "Acceptable for exploratory"
    else:
        return "Good reliability"


def compute_cohens_kappa(
    grades_1: list[int],
    grades_2: list[int],
    n_categories: int = 4,
) -> float:
    """Compute Cohen's kappa for two annotators.

    Args:
        grades_1: Grades from annotator 1
        grades_2: Grades from annotator 2
        n_categories: Number of grade categories (default 4: 0-3)

    Returns:
        Cohen's kappa coefficient
    """
    if len(grades_1) != len(grades_2) or len(grades_1) == 0:
        return 0.0

    n = len(grades_1)

    # Build confusion matrix
    confusion = [[0] * n_categories for _ in range(n_categories)]
    for g1, g2 in zip(grades_1, grades_2):
        if 0 <= g1 < n_categories and 0 <= g2 < n_categories:
            confusion[g1][g2] += 1

    # Observed agreement
    po = sum(confusion[i][i] for i in range(n_categories)) / n

    # Expected agreement
    row_sums = [sum(confusion[i]) for i in range(n_categories)]
    col_sums = [sum(confusion[i][j] for i in range(n_categories)) for j in range(n_categories)]
    pe = sum(row_sums[i] * col_sums[i] for i in range(n_categories)) / (n * n)

    # Kappa
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def compute_fleiss_kappa(
    annotations: list[list[int | None]],
    n_categories: int = 4,
) -> float:
    """Compute Fleiss' kappa for multiple annotators.

    Args:
        annotations: List of items, each containing list of grades from each annotator
                    (None for missing annotations)
        n_categories: Number of grade categories (default 4: 0-3)

    Returns:
        Fleiss' kappa coefficient
    """
    # Filter items with at least 2 annotations
    valid_items = [
        [g for g in item if g is not None]
        for item in annotations
        if sum(1 for g in item if g is not None) >= 2
    ]

    if not valid_items:
        return 0.0

    n_items = len(valid_items)
    n_raters_per_item = [len(item) for item in valid_items]

    # Count category assignments per item
    category_counts = []
    for item in valid_items:
        counts = [0] * n_categories
        for g in item:
            if 0 <= g < n_categories:
                counts[g] += 1
        category_counts.append(counts)

    # Pi for each item (agreement within item)
    pi_values = []
    for i, counts in enumerate(category_counts):
        n = n_raters_per_item[i]
        if n <= 1:
            continue
        pi = (sum(c * (c - 1) for c in counts)) / (n * (n - 1))
        pi_values.append(pi)

    if not pi_values:
        return 0.0

    # P-bar (mean observed agreement)
    p_bar = sum(pi_values) / len(pi_values)

    # P-bar-e (expected agreement by chance)
    total_ratings = sum(n_raters_per_item)
    category_proportions = [
        sum(cc[j] for cc in category_counts) / total_ratings
        for j in range(n_categories)
    ]
    p_bar_e = sum(p ** 2 for p in category_proportions)

    # Kappa
    if p_bar_e == 1.0:
        return 1.0 if p_bar == 1.0 else 0.0
    return (p_bar - p_bar_e) / (1 - p_bar_e)


def compute_krippendorff_alpha(
    annotations: list[list[int | None]],
    n_categories: int = 4,
) -> float:
    """Compute Krippendorff's alpha for ordinal data.

    Args:
        annotations: List of items, each containing list of grades from each annotator
        n_categories: Number of grade categories (default 4: 0-3)

    Returns:
        Krippendorff's alpha coefficient
    """
    # Filter to items with at least 2 annotations
    valid_items = []
    for item in annotations:
        valid_grades = [g for g in item if g is not None]
        if len(valid_grades) >= 2:
            valid_items.append(valid_grades)

    if not valid_items:
        return 0.0

    # Collect all pairs of ratings within items
    pairs = []
    for item in valid_items:
        for i in range(len(item)):
            for j in range(i + 1, len(item)):
                pairs.append((item[i], item[j]))

    if not pairs:
        return 0.0

    # Observed disagreement (ordinal distance)
    def ordinal_distance(a: int, b: int) -> float:
        return (a - b) ** 2

    do = sum(ordinal_distance(a, b) for a, b in pairs) / len(pairs)

    # Expected disagreement
    all_values = [g for item in valid_items for g in item]
    n = len(all_values)
    de = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            de += ordinal_distance(all_values[i], all_values[j])
    de = 2 * de / (n * (n - 1)) if n > 1 else 0.0

    # Alpha
    if de == 0:
        return 1.0 if do == 0 else 0.0
    return 1 - (do / de)


def compute_agreement_report(
    labels: list[RelevanceLabel],
) -> MultiAnnotatorReport:
    """Compute comprehensive multi-annotator agreement report.

    Args:
        labels: List of relevance labels with annotator information

    Returns:
        MultiAnnotatorReport with all agreement metrics
    """
    # Group by annotator
    by_annotator: dict[str, list[RelevanceLabel]] = {}
    for label in labels:
        ann_id = label.annotator_id or "anonymous"
        if ann_id not in by_annotator:
            by_annotator[ann_id] = []
        by_annotator[ann_id].append(label)

    # Group by item (query, result_id pair)
    by_item: dict[str, dict[str, int]] = {}
    for label in labels:
        item_key = f"{label.query}::{label.result_id}"
        ann_id = label.annotator_id or "anonymous"
        if item_key not in by_item:
            by_item[item_key] = {}
        by_item[item_key][ann_id] = label.relevance_grade

    # Compute annotator stats
    annotator_stats: dict[str, AnnotatorStats] = {}
    for ann_id, ann_labels in by_annotator.items():
        grades = [l.relevance_grade for l in ann_labels]
        grade_dist = {}
        for g in grades:
            grade_dist[g] = grade_dist.get(g, 0) + 1

        sorted_grades = sorted(grades)
        median_idx = len(sorted_grades) // 2
        median = sorted_grades[median_idx] if sorted_grades else 0

        annotator_stats[ann_id] = AnnotatorStats(
            annotator_id=ann_id,
            n_labels=len(ann_labels),
            grade_distribution=grade_dist,
            mean_grade=sum(grades) / len(grades) if grades else 0,
            median_grade=float(median),
        )

    # Compute pairwise agreement
    annotator_ids = list(by_annotator.keys())
    pairwise_agreements: list[PairwiseAgreement] = []

    for i in range(len(annotator_ids)):
        for j in range(i + 1, len(annotator_ids)):
            ann1, ann2 = annotator_ids[i], annotator_ids[j]

            # Find shared items
            shared_grades_1 = []
            shared_grades_2 = []
            for item_key, item_grades in by_item.items():
                if ann1 in item_grades and ann2 in item_grades:
                    shared_grades_1.append(item_grades[ann1])
                    shared_grades_2.append(item_grades[ann2])

            if shared_grades_1:
                # Exact agreement
                exact = sum(1 for g1, g2 in zip(shared_grades_1, shared_grades_2) if g1 == g2)
                exact_rate = exact / len(shared_grades_1)

                # Within-one agreement
                within_one = sum(
                    1 for g1, g2 in zip(shared_grades_1, shared_grades_2)
                    if abs(g1 - g2) <= 1
                )
                within_one_rate = within_one / len(shared_grades_1)

                # Cohen's kappa
                kappa = compute_cohens_kappa(shared_grades_1, shared_grades_2)

                # Correlation
                mean1 = sum(shared_grades_1) / len(shared_grades_1)
                mean2 = sum(shared_grades_2) / len(shared_grades_2)
                cov = sum(
                    (g1 - mean1) * (g2 - mean2)
                    for g1, g2 in zip(shared_grades_1, shared_grades_2)
                )
                var1 = sum((g - mean1) ** 2 for g in shared_grades_1)
                var2 = sum((g - mean2) ** 2 for g in shared_grades_2)
                corr = cov / ((var1 * var2) ** 0.5) if var1 > 0 and var2 > 0 else 0

                pairwise_agreements.append(PairwiseAgreement(
                    annotator_1=ann1,
                    annotator_2=ann2,
                    n_shared_items=len(shared_grades_1),
                    exact_agreement_rate=exact_rate,
                    within_one_agreement_rate=within_one_rate,
                    cohens_kappa=kappa,
                    correlation=corr,
                ))

    # Build annotations matrix for Fleiss/Krippendorff
    annotations_matrix: list[list[int | None]] = []
    for item_key, item_grades in by_item.items():
        row = [item_grades.get(ann_id) for ann_id in annotator_ids]
        annotations_matrix.append(row)

    # Compute overall metrics
    fleiss_kappa = compute_fleiss_kappa(annotations_matrix)
    kripp_alpha = compute_krippendorff_alpha(annotations_matrix)

    mean_pairwise = (
        sum(p.cohens_kappa for p in pairwise_agreements) / len(pairwise_agreements)
        if pairwise_agreements else 0.0
    )

    # Overall agreement rate
    items_with_multiple = [
        (key, grades) for key, grades in by_item.items()
        if len(grades) >= 2
    ]
    if items_with_multiple:
        agreements = 0
        total_pairs = 0
        for _, grades in items_with_multiple:
            grade_values = list(grades.values())
            for idx in range(len(grade_values)):
                for jdx in range(idx + 1, len(grade_values)):
                    total_pairs += 1
                    if grade_values[idx] == grade_values[jdx]:
                        agreements += 1
        overall_agreement = agreements / total_pairs if total_pairs > 0 else 0.0
    else:
        overall_agreement = 0.0

    # Find high disagreement items
    high_disagreement: list[dict[str, Any]] = []
    for item_key, grades in by_item.items():
        if len(grades) >= 2:
            grade_values = list(grades.values())
            grade_range = max(grade_values) - min(grade_values)
            if grade_range >= 2:  # Disagreement of 2+ grades
                high_disagreement.append({
                    "item_id": item_key,
                    "grades": grade_values,
                    "range": grade_range,
                    "annotators": list(grades.keys()),
                })

    high_disagreement.sort(key=lambda x: x["range"], reverse=True)

    # Agreement by grade
    agreement_by_grade: dict[int, float] = {}
    for grade in range(4):  # 0-3
        grade_items = [
            (key, grades) for key, grades in by_item.items()
            if any(g == grade for g in grades.values()) and len(grades) >= 2
        ]
        if grade_items:
            agreements = 0
            total = 0
            for _, grades in grade_items:
                grade_values = list(grades.values())
                for idx in range(len(grade_values)):
                    for jdx in range(idx + 1, len(grade_values)):
                        if grade_values[idx] == grade or grade_values[jdx] == grade:
                            total += 1
                            if grade_values[idx] == grade_values[jdx]:
                                agreements += 1
            agreement_by_grade[grade] = agreements / total if total > 0 else 0.0

    return MultiAnnotatorReport(
        timestamp=datetime.utcnow().isoformat(),
        n_annotators=len(annotator_ids),
        n_items=len(by_item),
        n_labels=len(labels),
        items_with_multiple_annotations=len(items_with_multiple),
        fleiss_kappa=fleiss_kappa,
        krippendorff_alpha=kripp_alpha,
        mean_pairwise_kappa=mean_pairwise,
        overall_agreement_rate=overall_agreement,
        annotator_stats=annotator_stats,
        pairwise_agreements=pairwise_agreements,
        high_disagreement_items=high_disagreement,
        agreement_by_grade=agreement_by_grade,
    )
