"""Dataset linkage benchmark for evaluating dataset-to-dataset relatedness.

This module provides schemas and evaluation infrastructure for the
dataset linkage task: given a source dataset, find related datasets.

Linkage types:
- same_task: Datasets studying the same behavioral task
- same_modality: Datasets using the same recording modality
- same_species: Datasets from the same species
- same_brain_region: Datasets recording from overlapping regions
- topical: Datasets studying related scientific questions
- reusable: Datasets that could use similar analysis pipelines

Usage:
    from neural_search.evaluation.dataset_linkage import (
        DatasetPair,
        LinkageLabel,
        LinkageBenchmark,
        evaluate_linkage,
    )

    # Create labeled pairs
    pair = DatasetPair(
        source_id="dandi:000003",
        target_id="dandi:000005",
        linkage_type="same_task",
        relatedness_score=3,
    )

    # Run evaluation
    results = evaluate_linkage(retrieval_fn, benchmark)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field


class LinkageType(StrEnum):
    """Types of dataset-to-dataset relationships."""

    SAME_TASK = "same_task"
    SAME_MODALITY = "same_modality"
    SAME_SPECIES = "same_species"
    SAME_BRAIN_REGION = "same_brain_region"
    TOPICAL = "topical"
    REUSABLE = "reusable"
    UNRELATED = "unrelated"


class RelatednessScore(StrEnum):
    """Graded relatedness scale for dataset pairs."""

    UNRELATED = "0"       # No meaningful relationship
    TOPICAL = "1"         # Same general topic area
    COMPARABLE = "2"      # Could be compared in a meta-analysis
    REUSABLE = "3"        # Analysis pipeline could transfer


class AnnotatorLabel(BaseModel):
    """Label from a single annotator."""

    annotator_id: str
    linkage_type: LinkageType
    relatedness_score: int  # 0-3 scale
    confidence: float = 1.0  # 0-1
    notes: str | None = None
    labeled_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class DatasetPair(BaseModel):
    """A pair of datasets with relatedness labels."""

    pair_id: str | None = None
    source_id: str
    target_id: str

    # Consensus labels (after adjudication)
    linkage_type: LinkageType = LinkageType.UNRELATED
    relatedness_score: int = 0  # 0-3 scale

    # Individual annotator labels
    annotator_labels: list[AnnotatorLabel] = Field(default_factory=list)

    # Evidence for the relationship
    evidence: list[str] = Field(default_factory=list)
    shared_attributes: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def model_post_init(self, __context: Any) -> None:
        """Generate pair_id if not provided."""
        if not self.pair_id:
            # Canonical ordering for consistent IDs
            ids = sorted([self.source_id, self.target_id])
            self.pair_id = f"{ids[0]}::{ids[1]}"

    @property
    def n_annotators(self) -> int:
        return len(self.annotator_labels)

    def compute_agreement(self) -> float | None:
        """Compute pairwise agreement among annotators."""
        if len(self.annotator_labels) < 2:
            return None

        scores = [l.relatedness_score for l in self.annotator_labels]
        agreements = 0
        total = 0

        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                total += 1
                if scores[i] == scores[j]:
                    agreements += 1

        return agreements / total if total > 0 else None


class LinkageBenchmark(BaseModel):
    """Complete benchmark for dataset linkage evaluation."""

    benchmark_id: str
    version: str = "1.0"
    description: str = ""

    # Labeled pairs
    pairs: list[DatasetPair] = Field(default_factory=list)

    # Target distribution
    target_by_type: dict[str, int] = Field(default_factory=dict)

    # Metadata
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    annotator_ids: list[str] = Field(default_factory=list)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def get_pairs_by_type(self, linkage_type: LinkageType) -> list[DatasetPair]:
        """Get all pairs of a specific linkage type."""
        return [p for p in self.pairs if p.linkage_type == linkage_type]

    def get_source_ids(self) -> set[str]:
        """Get all unique source dataset IDs."""
        return {p.source_id for p in self.pairs}

    def get_target_ids(self) -> set[str]:
        """Get all unique target dataset IDs."""
        return {p.target_id for p in self.pairs}

    def to_relevance_labels(self) -> dict[str, set[str]]:
        """Convert to query->relevant_ids format for evaluation."""
        labels: dict[str, set[str]] = {}
        for pair in self.pairs:
            if pair.relatedness_score >= 2:  # Comparable or better
                if pair.source_id not in labels:
                    labels[pair.source_id] = set()
                labels[pair.source_id].add(pair.target_id)
        return labels


@dataclass
class LinkageMetrics:
    """Evaluation metrics for dataset linkage."""

    # Retrieval metrics
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0

    # Per-type breakdown
    metrics_by_type: dict[str, dict[str, float]] = field(default_factory=dict)

    # Coverage
    n_queries: int = 0
    n_pairs_evaluated: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision_at_5": self.precision_at_5,
            "precision_at_10": self.precision_at_10,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
            "ndcg_at_10": self.ndcg_at_10,
            "metrics_by_type": self.metrics_by_type,
            "n_queries": self.n_queries,
            "n_pairs_evaluated": self.n_pairs_evaluated,
        }


def evaluate_linkage(
    retrieval_fn: Callable[[str], list[str]],
    benchmark: LinkageBenchmark,
    top_k: int = 10,
) -> LinkageMetrics:
    """Evaluate a retrieval function on the linkage benchmark.

    Args:
        retrieval_fn: Function that takes a dataset ID and returns
            ranked list of similar dataset IDs
        benchmark: LinkageBenchmark with labeled pairs
        top_k: Number of results to evaluate

    Returns:
        LinkageMetrics with evaluation results
    """
    import math

    relevance_labels = benchmark.to_relevance_labels()

    precisions_5: list[float] = []
    precisions_10: list[float] = []
    recalls_5: list[float] = []
    recalls_10: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []

    type_metrics: dict[str, list[dict[str, float]]] = {}

    for source_id, relevant_ids in relevance_labels.items():
        # Get retrieval results
        results = retrieval_fn(source_id)[:top_k]

        # Compute metrics
        hits_5 = sum(1 for r in results[:5] if r in relevant_ids)
        hits_10 = sum(1 for r in results[:10] if r in relevant_ids)

        p_5 = hits_5 / 5 if results else 0.0
        p_10 = hits_10 / 10 if results else 0.0
        r_5 = hits_5 / len(relevant_ids) if relevant_ids else 0.0
        r_10 = hits_10 / len(relevant_ids) if relevant_ids else 0.0

        precisions_5.append(p_5)
        precisions_10.append(p_10)
        recalls_5.append(r_5)
        recalls_10.append(r_10)

        # MRR
        mrr = 0.0
        for i, r in enumerate(results):
            if r in relevant_ids:
                mrr = 1.0 / (i + 1)
                break
        mrrs.append(mrr)

        # NDCG
        dcg = 0.0
        for i, r in enumerate(results[:top_k]):
            if r in relevant_ids:
                dcg += 1.0 / math.log2(i + 2)

        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), top_k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    n = len(relevance_labels)
    return LinkageMetrics(
        precision_at_5=sum(precisions_5) / n if n else 0.0,
        precision_at_10=sum(precisions_10) / n if n else 0.0,
        recall_at_5=sum(recalls_5) / n if n else 0.0,
        recall_at_10=sum(recalls_10) / n if n else 0.0,
        mrr=sum(mrrs) / n if n else 0.0,
        ndcg_at_10=sum(ndcgs) / n if n else 0.0,
        n_queries=n,
        n_pairs_evaluated=len(benchmark.pairs),
    )


def create_sample_benchmark() -> LinkageBenchmark:
    """Create a sample linkage benchmark for testing."""
    pairs = [
        DatasetPair(
            source_id="dandi:000003",
            target_id="dandi:000005",
            linkage_type=LinkageType.SAME_TASK,
            relatedness_score=3,
            evidence=["Both study decision-making in mice"],
        ),
        DatasetPair(
            source_id="dandi:000003",
            target_id="dandi:000020",
            linkage_type=LinkageType.SAME_MODALITY,
            relatedness_score=2,
            evidence=["Both use Neuropixels recordings"],
        ),
        DatasetPair(
            source_id="dandi:000003",
            target_id="openneuro:ds001",
            linkage_type=LinkageType.UNRELATED,
            relatedness_score=0,
            evidence=["Different species, modality, and task"],
        ),
    ]

    return LinkageBenchmark(
        benchmark_id="linkage_v1_sample",
        version="1.0",
        description="Sample linkage benchmark for testing",
        pairs=pairs,
        target_by_type={
            "same_task": 100,
            "same_modality": 100,
            "same_species": 100,
            "same_brain_region": 100,
            "unrelated": 100,
        },
    )


def save_benchmark(benchmark: LinkageBenchmark, path: Path) -> None:
    """Save benchmark to JSON file."""
    with open(path, "w") as f:
        json.dump(benchmark.model_dump(), f, indent=2)


def load_benchmark(path: Path) -> LinkageBenchmark:
    """Load benchmark from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return LinkageBenchmark(**data)
