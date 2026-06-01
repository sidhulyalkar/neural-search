"""Human relevance judgments for search result evaluation.

This module provides infrastructure for collecting, storing, and using
human relevance labels to validate search quality.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# Relevance levels from most to least relevant
RELEVANCE_LEVELS = [
    "exact",           # Perfect match for query intent
    "highly_relevant", # Good match, minor gaps
    "relevant",        # Matches query, some limitations
    "partially",       # Some relevance, significant gaps
    "not_relevant",    # Wrong domain/type/species
    "hard_negative",   # Explicitly should NOT match
]

# Numeric mapping for relevance levels
RELEVANCE_SCORES = {
    "exact": 3,
    "highly_relevant": 2,
    "relevant": 1,
    "partially": 0,
    "not_relevant": -1,
    "hard_negative": -2,
}


@dataclass
class RelevanceJudgment:
    """Human relevance judgment for a search result.

    Attributes:
        judgment_id: Unique identifier for this judgment
        query_id: Identifier for the benchmark query
        query_text: The original query string
        dataset_id: ID of the dataset being judged
        dataset_title: Title of the dataset for reference
        relevance: Overall relevance level
        task_match: Task match score (0-3)
        modality_match: Modality match score (0-3)
        species_match: Species match score (0-3)
        analysis_fit: Analysis affordance fit score (0-3)
        reviewer_id: Identifier for the human reviewer
        review_timestamp: ISO timestamp of the review
        review_notes: Optional notes from the reviewer
        confidence: Reviewer confidence in judgment (0-1)
    """

    judgment_id: str
    query_id: str
    query_text: str
    dataset_id: str
    dataset_title: str

    # Core relevance level
    relevance: Literal[
        "exact",
        "highly_relevant",
        "relevant",
        "partially",
        "not_relevant",
        "hard_negative",
    ]

    # Dimension-specific scores (0=wrong, 1=related, 2=compatible, 3=exact)
    task_match: int = 0
    modality_match: int = 0
    species_match: int = 0
    analysis_fit: int = 0

    # Metadata
    reviewer_id: str = "unknown"
    review_timestamp: str = ""
    review_notes: str = ""
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate scores are in valid range."""
        for score_name in ["task_match", "modality_match", "species_match", "analysis_fit"]:
            score = getattr(self, score_name)
            if not 0 <= score <= 3:
                raise ValueError(f"{score_name} must be between 0 and 3, got {score}")

        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")

        if self.relevance not in RELEVANCE_LEVELS:
            raise ValueError(f"relevance must be one of {RELEVANCE_LEVELS}, got {self.relevance}")

    @property
    def relevance_score(self) -> int:
        """Numeric score for relevance level."""
        return RELEVANCE_SCORES.get(self.relevance, 0)

    @property
    def dimension_score_total(self) -> int:
        """Sum of dimension-specific scores (0-12)."""
        return self.task_match + self.modality_match + self.species_match + self.analysis_fit

    def is_relevant(self, min_level: str = "relevant") -> bool:
        """Check if judgment meets minimum relevance threshold.

        Args:
            min_level: Minimum relevance level to consider relevant

        Returns:
            True if relevance meets or exceeds min_level
        """
        min_idx = RELEVANCE_LEVELS.index(min_level)
        current_idx = RELEVANCE_LEVELS.index(self.relevance)
        return current_idx <= min_idx

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelevanceJudgment:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RelevanceLabelSet:
    """Collection of relevance judgments for a query."""

    query_id: str
    query_text: str
    judgments: list[RelevanceJudgment] = field(default_factory=list)

    def add_judgment(self, judgment: RelevanceJudgment) -> None:
        """Add a judgment to the set."""
        if judgment.query_id != self.query_id:
            raise ValueError(
                f"Judgment query_id {judgment.query_id} does not match set query_id {self.query_id}"
            )
        self.judgments.append(judgment)

    def get_judgment_for_dataset(self, dataset_id: str) -> RelevanceJudgment | None:
        """Get judgment for a specific dataset."""
        for j in self.judgments:
            if j.dataset_id == dataset_id:
                return j
        return None

    @property
    def relevant_dataset_ids(self) -> set[str]:
        """Get IDs of datasets judged as relevant or better."""
        return {j.dataset_id for j in self.judgments if j.is_relevant()}

    @property
    def exact_match_ids(self) -> set[str]:
        """Get IDs of datasets judged as exact matches."""
        return {j.dataset_id for j in self.judgments if j.relevance == "exact"}

    @property
    def hard_negative_ids(self) -> set[str]:
        """Get IDs of datasets that should NOT match."""
        return {j.dataset_id for j in self.judgments if j.relevance == "hard_negative"}


def create_judgment(
    query_id: str,
    query_text: str,
    dataset_id: str,
    dataset_title: str,
    relevance: str,
    reviewer_id: str,
    task_match: int = 0,
    modality_match: int = 0,
    species_match: int = 0,
    analysis_fit: int = 0,
    notes: str = "",
    confidence: float = 1.0,
) -> RelevanceJudgment:
    """Create a new relevance judgment with auto-generated ID and timestamp.

    Args:
        query_id: Query identifier
        query_text: Original query string
        dataset_id: Dataset being judged
        dataset_title: Dataset title
        relevance: Relevance level
        reviewer_id: Reviewer identifier
        task_match: Task match score (0-3)
        modality_match: Modality match score (0-3)
        species_match: Species match score (0-3)
        analysis_fit: Analysis fit score (0-3)
        notes: Optional reviewer notes
        confidence: Confidence in judgment (0-1)

    Returns:
        New RelevanceJudgment instance
    """
    return RelevanceJudgment(
        judgment_id=f"j_{uuid.uuid4().hex[:12]}",
        query_id=query_id,
        query_text=query_text,
        dataset_id=dataset_id,
        dataset_title=dataset_title,
        relevance=relevance,  # type: ignore
        task_match=task_match,
        modality_match=modality_match,
        species_match=species_match,
        analysis_fit=analysis_fit,
        reviewer_id=reviewer_id,
        review_timestamp=datetime.now(timezone.utc).isoformat(),
        review_notes=notes,
        confidence=confidence,
    )


def load_relevance_labels(
    path: str | Path,
) -> dict[str, RelevanceLabelSet]:
    """Load relevance labels from JSONL file.

    Args:
        path: Path to JSONL file containing judgments

    Returns:
        Dict mapping query_id to RelevanceLabelSet
    """
    path = Path(path)
    if not path.exists():
        return {}

    label_sets: dict[str, RelevanceLabelSet] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            judgment = RelevanceJudgment.from_dict(data)

            if judgment.query_id not in label_sets:
                label_sets[judgment.query_id] = RelevanceLabelSet(
                    query_id=judgment.query_id,
                    query_text=judgment.query_text,
                )

            label_sets[judgment.query_id].add_judgment(judgment)

    return label_sets


def save_relevance_labels(
    judgments: list[RelevanceJudgment],
    path: str | Path,
    append: bool = True,
) -> int:
    """Save relevance labels to JSONL file.

    Args:
        judgments: List of judgments to save
        path: Path to JSONL file
        append: If True, append to existing file; otherwise overwrite

    Returns:
        Number of judgments written
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for judgment in judgments:
            f.write(json.dumps(judgment.to_dict()) + "\n")

    return len(judgments)


def compute_human_precision(
    result_ids: list[str],
    label_set: RelevanceLabelSet,
    k: int = 5,
    min_relevance: str = "relevant",
) -> float:
    """Compute precision@k using human relevance labels.

    Args:
        result_ids: List of returned dataset IDs in rank order
        label_set: RelevanceLabelSet with human judgments
        k: Number of results to consider
        min_relevance: Minimum relevance level to count as relevant

    Returns:
        Precision@k based on human judgments
    """
    if not result_ids:
        return 0.0

    top_k = result_ids[:k]
    relevant_count = 0

    for dataset_id in top_k:
        judgment = label_set.get_judgment_for_dataset(dataset_id)
        if judgment and judgment.is_relevant(min_relevance):
            relevant_count += 1

    return relevant_count / len(top_k)


def compute_human_recall(
    result_ids: list[str],
    label_set: RelevanceLabelSet,
    k: int | None = None,
    min_relevance: str = "relevant",
) -> float:
    """Compute recall using human relevance labels.

    Args:
        result_ids: List of returned dataset IDs
        label_set: RelevanceLabelSet with human judgments
        k: If provided, only consider top-k results
        min_relevance: Minimum relevance level to count as relevant

    Returns:
        Recall based on human judgments
    """
    if k is not None:
        result_ids = result_ids[:k]

    result_set = set(result_ids)

    # Get all judged relevant datasets
    relevant_ids = {
        j.dataset_id for j in label_set.judgments if j.is_relevant(min_relevance)
    }

    if not relevant_ids:
        return 0.0

    found_relevant = result_set & relevant_ids
    return len(found_relevant) / len(relevant_ids)


def compute_hard_negative_violations(
    result_ids: list[str],
    label_set: RelevanceLabelSet,
    k: int = 10,
) -> list[str]:
    """Check for hard negative violations in results.

    Args:
        result_ids: List of returned dataset IDs
        label_set: RelevanceLabelSet with human judgments
        k: Number of results to check

    Returns:
        List of dataset IDs that are hard negatives appearing in results
    """
    top_k = set(result_ids[:k])
    hard_negatives = label_set.hard_negative_ids
    return list(top_k & hard_negatives)


def compute_ndcg(
    result_ids: list[str],
    label_set: RelevanceLabelSet,
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain.

    Args:
        result_ids: List of returned dataset IDs in rank order
        label_set: RelevanceLabelSet with human judgments
        k: Number of results to consider

    Returns:
        NDCG@k score (0-1)
    """
    import math

    def relevance_gain(dataset_id: str) -> float:
        """Get relevance gain for a dataset."""
        judgment = label_set.get_judgment_for_dataset(dataset_id)
        if judgment is None:
            return 0.0
        # Use relevance score as gain (convert from -2..3 to 0..5 scale)
        return max(0, judgment.relevance_score + 2)

    # Compute DCG
    dcg = 0.0
    for i, dataset_id in enumerate(result_ids[:k]):
        gain = relevance_gain(dataset_id)
        dcg += gain / math.log2(i + 2)  # log2(rank+1) where rank starts at 1

    # Compute ideal DCG (IDCG)
    # Sort all judgments by relevance score descending
    ideal_gains = sorted(
        [max(0, j.relevance_score + 2) for j in label_set.judgments],
        reverse=True,
    )[:k]

    idcg = 0.0
    for i, gain in enumerate(ideal_gains):
        idcg += gain / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_mrr(
    result_ids: list[str],
    label_set: RelevanceLabelSet,
    min_relevance: str = "relevant",
) -> float:
    """Compute Mean Reciprocal Rank for first relevant result.

    Args:
        result_ids: List of returned dataset IDs in rank order
        label_set: RelevanceLabelSet with human judgments
        min_relevance: Minimum relevance level

    Returns:
        Reciprocal rank of first relevant result (0 if none found)
    """
    for rank, dataset_id in enumerate(result_ids, start=1):
        judgment = label_set.get_judgment_for_dataset(dataset_id)
        if judgment and judgment.is_relevant(min_relevance):
            return 1.0 / rank

    return 0.0


@dataclass
class HumanEvaluationMetrics:
    """Comprehensive metrics using human relevance labels."""

    query_id: str
    query_text: str

    # Core metrics
    precision_at_5: float
    precision_at_10: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float

    # Hard negative checking
    hard_negative_violations: int
    hard_negative_ids: list[str]

    # Counts
    total_judgments: int
    relevant_judgments: int
    exact_match_judgments: int


def compute_human_evaluation_metrics(
    result_ids: list[str],
    label_set: RelevanceLabelSet,
) -> HumanEvaluationMetrics:
    """Compute comprehensive evaluation metrics using human labels.

    Args:
        result_ids: List of returned dataset IDs in rank order
        label_set: RelevanceLabelSet with human judgments

    Returns:
        HumanEvaluationMetrics with all computed metrics
    """
    violations = compute_hard_negative_violations(result_ids, label_set, k=10)

    return HumanEvaluationMetrics(
        query_id=label_set.query_id,
        query_text=label_set.query_text,
        precision_at_5=compute_human_precision(result_ids, label_set, k=5),
        precision_at_10=compute_human_precision(result_ids, label_set, k=10),
        recall_at_10=compute_human_recall(result_ids, label_set, k=10),
        mrr=compute_mrr(result_ids, label_set),
        ndcg_at_10=compute_ndcg(result_ids, label_set, k=10),
        hard_negative_violations=len(violations),
        hard_negative_ids=violations,
        total_judgments=len(label_set.judgments),
        relevant_judgments=len(label_set.relevant_dataset_ids),
        exact_match_judgments=len(label_set.exact_match_ids),
    )


# =============================================================================
# Active Learning for Efficient Relevance Labeling
# =============================================================================


@dataclass
class SamplePriority:
    """Priority scoring for active learning sample selection."""

    dataset_id: str
    query_id: str
    uncertainty_score: float  # Higher = more uncertain
    diversity_score: float  # Higher = more different from labeled
    calibration_score: float  # Higher = more useful for calibration
    overall_priority: float

    @classmethod
    def compute(
        cls,
        dataset_id: str,
        query_id: str,
        search_score: float,
        labeled_ids: set[str],
        score_range: tuple[float, float],
    ) -> SamplePriority:
        """Compute priority scores for a sample.

        Args:
            dataset_id: Dataset identifier
            query_id: Query identifier
            search_score: Search system confidence score
            labeled_ids: Already labeled dataset IDs
            score_range: (min, max) scores across all results

        Returns:
            SamplePriority with computed scores
        """
        # Uncertainty: prioritize scores near decision boundary (0.5)
        min_score, max_score = score_range
        score_range_val = max_score - min_score if max_score > min_score else 1.0
        normalized = (search_score - min_score) / score_range_val
        uncertainty = 1.0 - abs(normalized - 0.5) * 2  # Peak at 0.5

        # Diversity: prioritize unlabeled samples
        diversity = 0.0 if dataset_id in labeled_ids else 1.0

        # Calibration: prioritize extreme scores for calibration
        calibration = abs(normalized - 0.5) * 2  # Peak at 0 and 1

        # Weighted combination
        overall = (
            0.4 * uncertainty
            + 0.4 * diversity
            + 0.2 * calibration
        )

        return cls(
            dataset_id=dataset_id,
            query_id=query_id,
            uncertainty_score=round(uncertainty, 4),
            diversity_score=round(diversity, 4),
            calibration_score=round(calibration, 4),
            overall_priority=round(overall, 4),
        )


def select_samples_for_labeling(
    search_results: list[dict[str, Any]],
    existing_labels: dict[str, RelevanceLabelSet],
    max_samples: int = 10,
    strategy: str = "uncertainty",
) -> list[SamplePriority]:
    """Select most informative samples for human labeling.

    Implements active learning strategies:
    - uncertainty: Select samples where model is least confident
    - diversity: Select samples most different from already labeled
    - hybrid: Balanced combination of uncertainty and diversity

    Args:
        search_results: List of search result dicts with dataset_id, query_id, score
        existing_labels: Already collected relevance labels
        max_samples: Maximum number of samples to select
        strategy: Selection strategy (uncertainty, diversity, hybrid)

    Returns:
        List of SamplePriority sorted by priority (highest first)
    """
    # Get already labeled dataset IDs
    labeled_ids: set[str] = set()
    for label_set in existing_labels.values():
        for judgment in label_set.judgments:
            labeled_ids.add(judgment.dataset_id)

    # Compute score range
    scores = [r.get("score", 0) for r in search_results]
    score_range = (min(scores) if scores else 0, max(scores) if scores else 100)

    # Compute priority for each sample
    priorities: list[SamplePriority] = []
    for result in search_results:
        dataset_id = result.get("dataset_id", "")
        query_id = result.get("query_id", "unknown")
        score = result.get("score", 0)

        priority = SamplePriority.compute(
            dataset_id=dataset_id,
            query_id=query_id,
            search_score=score,
            labeled_ids=labeled_ids,
            score_range=score_range,
        )

        # Adjust based on strategy
        if strategy == "uncertainty":
            priority.overall_priority = priority.uncertainty_score
        elif strategy == "diversity":
            priority.overall_priority = priority.diversity_score
        # hybrid uses the default weighted combination

        priorities.append(priority)

    # Sort by priority and return top samples
    priorities.sort(key=lambda p: p.overall_priority, reverse=True)
    return priorities[:max_samples]


def compute_labeling_coverage(
    existing_labels: dict[str, RelevanceLabelSet],
    total_queries: int,
    min_labels_per_query: int = 10,
) -> dict[str, Any]:
    """Compute coverage statistics for relevance labeling.

    Args:
        existing_labels: Collected relevance labels
        total_queries: Total number of benchmark queries
        min_labels_per_query: Minimum labels needed per query

    Returns:
        Coverage statistics dict
    """
    queries_with_labels = len(existing_labels)
    total_labels = sum(len(ls.judgments) for ls in existing_labels.values())
    avg_labels_per_query = total_labels / queries_with_labels if queries_with_labels else 0

    fully_labeled = sum(
        1 for ls in existing_labels.values()
        if len(ls.judgments) >= min_labels_per_query
    )

    return {
        "total_queries": total_queries,
        "queries_with_labels": queries_with_labels,
        "query_coverage_pct": round(queries_with_labels / total_queries * 100, 1) if total_queries else 0,
        "total_labels": total_labels,
        "avg_labels_per_query": round(avg_labels_per_query, 1),
        "fully_labeled_queries": fully_labeled,
        "fully_labeled_pct": round(fully_labeled / total_queries * 100, 1) if total_queries else 0,
        "labels_needed": max(0, total_queries * min_labels_per_query - total_labels),
    }
