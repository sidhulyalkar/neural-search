"""Automated dataset relatedness scoring based on graph structure.

This module computes empirical relatedness between datasets based on
their structural relationships in the knowledge graph, NOT surface similarity.

The key question: How do we retrieve objects based on latent future usefulness
rather than surface similarity?

Dimensions of relatedness:
1. Modality alignment - Can the same analysis pipeline process both?
2. Task compatibility - Do they study compatible scientific questions?
3. Species/model alignment - Are findings transferable?
4. Brain region overlap - Do they cover comparable neural circuits?
5. Affordance compatibility - Do they support similar analyses?
6. Experimental design similarity - Are they methodologically comparable?
7. Cross-dataset comparability - Could they be combined in meta-analysis?
8. Provenance quality - Are both well-documented and trustworthy?
9. Statistical power proxy - Do both have sufficient data?
10. Pipeline transferability - Would code written for one work on the other?

Usage:
    from neural_search.evaluation.relatedness_scorer import (
        RelatednessScorer,
        score_dataset_pair,
        compute_corpus_relatedness_matrix,
    )

    scorer = RelatednessScorer.from_graph(graph)
    score = scorer.score_pair("dandi:000003", "dandi:000005")
    print(score.total_score)  # 0.0 - 1.0
    print(score.dimension_scores)  # Per-dimension breakdown
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class RelatednessDimension(StrEnum):
    """Dimensions of dataset relatedness."""

    MODALITY_ALIGNMENT = "modality_alignment"
    TASK_COMPATIBILITY = "task_compatibility"
    SPECIES_ALIGNMENT = "species_alignment"
    REGION_OVERLAP = "region_overlap"
    AFFORDANCE_COMPATIBILITY = "affordance_compatibility"
    DESIGN_SIMILARITY = "design_similarity"
    PROVENANCE_QUALITY = "provenance_quality"
    STATISTICAL_POWER = "statistical_power"
    PIPELINE_TRANSFERABILITY = "pipeline_transferability"
    GRAPH_PROXIMITY = "graph_proximity"


@dataclass
class DimensionScore:
    """Score for a single dimension of relatedness."""

    dimension: RelatednessDimension
    score: float  # 0.0 - 1.0
    evidence: list[str] = field(default_factory=list)
    shared_items: list[str] = field(default_factory=list)
    weight: float = 1.0

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class RelatednessScore:
    """Complete relatedness score between two datasets."""

    source_id: str
    target_id: str
    dimension_scores: dict[RelatednessDimension, DimensionScore] = field(
        default_factory=dict
    )

    @property
    def total_score(self) -> float:
        """Weighted average of all dimension scores."""
        if not self.dimension_scores:
            return 0.0

        total_weight = sum(d.weight for d in self.dimension_scores.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(d.weighted_score for d in self.dimension_scores.values())
        return weighted_sum / total_weight

    @property
    def reusability_score(self) -> float:
        """Score focused on analysis pipeline transferability."""
        key_dims = [
            RelatednessDimension.MODALITY_ALIGNMENT,
            RelatednessDimension.AFFORDANCE_COMPATIBILITY,
            RelatednessDimension.PIPELINE_TRANSFERABILITY,
        ]
        scores = [
            self.dimension_scores[d].score
            for d in key_dims
            if d in self.dimension_scores
        ]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def comparability_score(self) -> float:
        """Score focused on meta-analysis compatibility."""
        key_dims = [
            RelatednessDimension.TASK_COMPATIBILITY,
            RelatednessDimension.SPECIES_ALIGNMENT,
            RelatednessDimension.DESIGN_SIMILARITY,
        ]
        scores = [
            self.dimension_scores[d].score
            for d in key_dims
            if d in self.dimension_scores
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def to_grade(self) -> int:
        """Convert to 0-3 grade scale for benchmark compatibility."""
        score = self.total_score
        if score >= 0.7:
            return 3  # Highly reusable
        elif score >= 0.5:
            return 2  # Comparable
        elif score >= 0.25:
            return 1  # Topically related
        else:
            return 0  # Unrelated

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "total_score": self.total_score,
            "reusability_score": self.reusability_score,
            "comparability_score": self.comparability_score,
            "grade": self.to_grade(),
            "dimensions": {
                str(k): {
                    "score": v.score,
                    "weight": v.weight,
                    "evidence": v.evidence,
                    "shared_items": v.shared_items,
                }
                for k, v in self.dimension_scores.items()
            },
        }


class RelatednessScorer:
    """Computes empirical relatedness between datasets using graph structure."""

    # Default weights for each dimension
    DEFAULT_WEIGHTS = {
        RelatednessDimension.MODALITY_ALIGNMENT: 2.0,  # Critical for pipeline transfer
        RelatednessDimension.TASK_COMPATIBILITY: 1.5,
        RelatednessDimension.SPECIES_ALIGNMENT: 1.0,
        RelatednessDimension.REGION_OVERLAP: 1.2,
        RelatednessDimension.AFFORDANCE_COMPATIBILITY: 2.0,  # Key for reusability
        RelatednessDimension.DESIGN_SIMILARITY: 1.0,
        RelatednessDimension.PROVENANCE_QUALITY: 0.5,
        RelatednessDimension.STATISTICAL_POWER: 0.8,
        RelatednessDimension.PIPELINE_TRANSFERABILITY: 1.5,
        RelatednessDimension.GRAPH_PROXIMITY: 1.0,
    }

    def __init__(
        self,
        nodes: dict[str, dict],
        edges: dict[str, dict],
        weights: dict[RelatednessDimension, float] | None = None,
    ):
        self.nodes = nodes
        self.edges = edges
        self.weights = weights or self.DEFAULT_WEIGHTS

        # Build indices for fast lookup
        self._dataset_neighbors: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._build_indices()

    def _build_indices(self) -> None:
        """Build graph indices for efficient scoring."""
        for _edge_id, edge in self.edges.items():
            source = edge.get("source_node_id", "")
            target = edge.get("target_node_id", "")
            edge_type = edge.get("edge_type", "")

            if not source or not target:
                continue

            # Index dataset -> {edge_type -> set of targets}
            self._dataset_neighbors[source][edge_type].add(target)

    @classmethod
    def from_graph(cls, graph: dict, **kwargs) -> RelatednessScorer:
        """Create scorer from a knowledge graph dictionary."""
        return cls(
            nodes=graph.get("nodes", {}),
            edges=graph.get("edges", {}),
            **kwargs,
        )

    @classmethod
    def from_graph_file(cls, path: Path, **kwargs) -> RelatednessScorer:
        """Create scorer from a graph JSON file."""
        with open(path) as f:
            graph = json.load(f)
        return cls.from_graph(graph, **kwargs)

    def score_pair(self, source_id: str, target_id: str) -> RelatednessScore:
        """Compute full relatedness score between two datasets."""
        result = RelatednessScore(source_id=source_id, target_id=target_id)

        # Compute each dimension
        result.dimension_scores[RelatednessDimension.MODALITY_ALIGNMENT] = (
            self._score_modality_alignment(source_id, target_id)
        )
        result.dimension_scores[RelatednessDimension.TASK_COMPATIBILITY] = (
            self._score_task_compatibility(source_id, target_id)
        )
        result.dimension_scores[RelatednessDimension.SPECIES_ALIGNMENT] = (
            self._score_species_alignment(source_id, target_id)
        )
        result.dimension_scores[RelatednessDimension.REGION_OVERLAP] = (
            self._score_region_overlap(source_id, target_id)
        )
        result.dimension_scores[RelatednessDimension.AFFORDANCE_COMPATIBILITY] = (
            self._score_affordance_compatibility(source_id, target_id)
        )
        result.dimension_scores[RelatednessDimension.GRAPH_PROXIMITY] = (
            self._score_graph_proximity(source_id, target_id)
        )
        result.dimension_scores[RelatednessDimension.PROVENANCE_QUALITY] = (
            self._score_provenance_quality(source_id, target_id)
        )
        result.dimension_scores[RelatednessDimension.STATISTICAL_POWER] = (
            self._score_statistical_power(source_id, target_id)
        )
        result.dimension_scores[RelatednessDimension.PIPELINE_TRANSFERABILITY] = (
            self._score_pipeline_transferability(source_id, target_id)
        )

        return result

    def _jaccard_similarity(self, set_a: set, set_b: set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _dice_similarity(self, set_a: set, set_b: set) -> float:
        """Compute Dice coefficient (emphasizes overlap more than Jaccard)."""
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        return 2 * intersection / (len(set_a) + len(set_b)) if (set_a or set_b) else 0.0

    def _get_neighbors(self, node_id: str, edge_type: str) -> set[str]:
        """Get neighbor nodes of a specific edge type."""
        return self._dataset_neighbors.get(node_id, {}).get(edge_type, set())

    def _score_modality_alignment(
        self, source_id: str, target_id: str
    ) -> DimensionScore:
        """Score modality alignment (critical for pipeline transfer)."""
        source_mods = self._get_neighbors(source_id, "dataset_has_modality")
        target_mods = self._get_neighbors(target_id, "dataset_has_modality")

        shared = source_mods & target_mods
        # Use Dice for modality (exact match matters more)
        score = self._dice_similarity(source_mods, target_mods)

        # Boost for exact modality match
        if source_mods and source_mods == target_mods:
            score = min(1.0, score + 0.2)

        return DimensionScore(
            dimension=RelatednessDimension.MODALITY_ALIGNMENT,
            score=score,
            weight=self.weights[RelatednessDimension.MODALITY_ALIGNMENT],
            shared_items=[self._get_label(n) for n in shared],
            evidence=[f"Shared modalities: {len(shared)}/{len(source_mods | target_mods)}"],
        )

    def _score_task_compatibility(
        self, source_id: str, target_id: str
    ) -> DimensionScore:
        """Score task/paradigm compatibility."""
        source_tasks = self._get_neighbors(source_id, "dataset_has_task")
        target_tasks = self._get_neighbors(target_id, "dataset_has_task")

        shared = source_tasks & target_tasks
        score = self._jaccard_similarity(source_tasks, target_tasks)

        return DimensionScore(
            dimension=RelatednessDimension.TASK_COMPATIBILITY,
            score=score,
            weight=self.weights[RelatednessDimension.TASK_COMPATIBILITY],
            shared_items=[self._get_label(n) for n in shared],
            evidence=[f"Shared tasks: {len(shared)}"],
        )

    def _score_species_alignment(
        self, source_id: str, target_id: str
    ) -> DimensionScore:
        """Score species/model organism alignment."""
        source_species = self._get_neighbors(source_id, "dataset_has_species")
        target_species = self._get_neighbors(target_id, "dataset_has_species")

        shared = source_species & target_species
        score = self._dice_similarity(source_species, target_species)

        return DimensionScore(
            dimension=RelatednessDimension.SPECIES_ALIGNMENT,
            score=score,
            weight=self.weights[RelatednessDimension.SPECIES_ALIGNMENT],
            shared_items=[self._get_label(n) for n in shared],
            evidence=[f"Shared species: {len(shared)}"],
        )

    def _score_region_overlap(self, source_id: str, target_id: str) -> DimensionScore:
        """Score brain region overlap."""
        source_regions = self._get_neighbors(source_id, "dataset_records_region")
        target_regions = self._get_neighbors(target_id, "dataset_records_region")

        shared = source_regions & target_regions
        score = self._jaccard_similarity(source_regions, target_regions)

        return DimensionScore(
            dimension=RelatednessDimension.REGION_OVERLAP,
            score=score,
            weight=self.weights[RelatednessDimension.REGION_OVERLAP],
            shared_items=[self._get_label(n) for n in shared],
            evidence=[f"Shared regions: {len(shared)}"],
        )

    def _score_affordance_compatibility(
        self, source_id: str, target_id: str
    ) -> DimensionScore:
        """Score affordance compatibility (can same analyses be run?)."""
        # Get behavioral events as proxy for affordances
        source_events = self._get_neighbors(source_id, "dataset_has_behavioral_event")
        target_events = self._get_neighbors(target_id, "dataset_has_behavioral_event")

        shared = source_events & target_events
        score = self._jaccard_similarity(source_events, target_events)

        # Also check data standards
        source_standards = self._get_neighbors(source_id, "dataset_uses_standard")
        target_standards = self._get_neighbors(target_id, "dataset_uses_standard")
        shared_standards = source_standards & target_standards

        # Boost if same data standard (NWB, BIDS)
        if shared_standards:
            score = min(1.0, score + 0.3)

        return DimensionScore(
            dimension=RelatednessDimension.AFFORDANCE_COMPATIBILITY,
            score=score,
            weight=self.weights[RelatednessDimension.AFFORDANCE_COMPATIBILITY],
            shared_items=[self._get_label(n) for n in shared | shared_standards],
            evidence=[
                f"Shared behavioral events: {len(shared)}",
                f"Shared standards: {len(shared_standards)}",
            ],
        )

    def _score_graph_proximity(self, source_id: str, target_id: str) -> DimensionScore:
        """Score based on graph proximity (shared neighbors of all types)."""
        # Count total shared neighbors across all edge types
        source_all = set()
        target_all = set()

        for _edge_type, neighbors in self._dataset_neighbors.get(source_id, {}).items():
            source_all.update(neighbors)
        for _edge_type, neighbors in self._dataset_neighbors.get(target_id, {}).items():
            target_all.update(neighbors)

        shared = source_all & target_all
        score = self._jaccard_similarity(source_all, target_all)

        # Penalize datasets with no graph connections
        if not source_all or not target_all:
            score *= 0.5

        return DimensionScore(
            dimension=RelatednessDimension.GRAPH_PROXIMITY,
            score=score,
            weight=self.weights[RelatednessDimension.GRAPH_PROXIMITY],
            shared_items=[],  # Too many to list
            evidence=[f"Total shared neighbors: {len(shared)}/{len(source_all | target_all)}"],
        )

    def _score_provenance_quality(
        self, source_id: str, target_id: str
    ) -> DimensionScore:
        """Score based on provenance/documentation quality."""
        source_node = self.nodes.get(source_id, {})
        target_node = self.nodes.get(target_id, {})

        # Count evidence items as quality proxy
        source_evidence = len(source_node.get("evidence", []))
        target_evidence = len(target_node.get("evidence", []))

        # Normalize (assume 10 evidence items is "high quality")
        source_quality = min(1.0, source_evidence / 10)
        target_quality = min(1.0, target_evidence / 10)

        # Combined quality (geometric mean)
        score = math.sqrt(source_quality * target_quality) if source_quality and target_quality else 0.0

        return DimensionScore(
            dimension=RelatednessDimension.PROVENANCE_QUALITY,
            score=score,
            weight=self.weights[RelatednessDimension.PROVENANCE_QUALITY],
            shared_items=[],
            evidence=[
                f"Source evidence items: {source_evidence}",
                f"Target evidence items: {target_evidence}",
            ],
        )

    def _score_statistical_power(
        self, source_id: str, target_id: str
    ) -> DimensionScore:
        """Score based on statistical power (data richness)."""
        source_node = self.nodes.get(source_id, {})
        target_node = self.nodes.get(target_id, {})

        # Use usability flags as power proxy
        source_props = source_node.get("properties", {})
        target_props = target_node.get("properties", {})

        source_flags = source_props.get("usability_flags", {})
        target_flags = target_props.get("usability_flags", {})

        # Count positive flags
        source_power = sum(1 for v in source_flags.values() if v is True)
        target_power = sum(1 for v in target_flags.values() if v is True)

        # Normalize (assume 5 flags is "high power")
        source_norm = min(1.0, source_power / 5)
        target_norm = min(1.0, target_power / 5)

        # Take minimum (both need power for combined analysis)
        score = min(source_norm, target_norm)

        return DimensionScore(
            dimension=RelatednessDimension.STATISTICAL_POWER,
            score=score,
            weight=self.weights[RelatednessDimension.STATISTICAL_POWER],
            shared_items=[],
            evidence=[
                f"Source power flags: {source_power}",
                f"Target power flags: {target_power}",
            ],
        )

    def _score_pipeline_transferability(
        self, source_id: str, target_id: str
    ) -> DimensionScore:
        """Score pipeline transferability (composite of modality + standard + events)."""
        # This is a composite score
        mod_score = self._score_modality_alignment(source_id, target_id).score
        aff_score = self._score_affordance_compatibility(source_id, target_id).score

        # Pipeline transferability requires both modality match AND affordance match
        score = math.sqrt(mod_score * aff_score)  # Geometric mean

        return DimensionScore(
            dimension=RelatednessDimension.PIPELINE_TRANSFERABILITY,
            score=score,
            weight=self.weights[RelatednessDimension.PIPELINE_TRANSFERABILITY],
            shared_items=[],
            evidence=[
                f"Modality alignment: {mod_score:.2f}",
                f"Affordance compatibility: {aff_score:.2f}",
            ],
        )

    def _get_label(self, node_id: str) -> str:
        """Get human-readable label for a node."""
        node = self.nodes.get(node_id, {})
        return node.get("label", node_id.split(":")[-1])


def score_dataset_pair(
    graph_path: Path,
    source_id: str,
    target_id: str,
) -> RelatednessScore:
    """Convenience function to score a single pair."""
    scorer = RelatednessScorer.from_graph_file(graph_path)
    return scorer.score_pair(source_id, target_id)


def compute_corpus_relatedness_matrix(
    graph_path: Path,
    dataset_ids: list[str] | None = None,
    min_score: float = 0.1,
) -> dict[str, dict[str, float]]:
    """Compute pairwise relatedness for all datasets.

    Args:
        graph_path: Path to knowledge graph
        dataset_ids: Optional subset of datasets (default: all)
        min_score: Minimum score to include in matrix

    Returns:
        Nested dict: {source_id: {target_id: score}}
    """
    scorer = RelatednessScorer.from_graph_file(graph_path)

    if dataset_ids is None:
        # Get all dataset nodes
        dataset_ids = [
            nid for nid, n in scorer.nodes.items()
            if n.get("node_type") == "dataset"
        ]

    matrix: dict[str, dict[str, float]] = defaultdict(dict)

    total = len(dataset_ids) * (len(dataset_ids) - 1) // 2
    computed = 0

    for i, source_id in enumerate(dataset_ids):
        for target_id in dataset_ids[i + 1:]:
            score = scorer.score_pair(source_id, target_id)

            if score.total_score >= min_score:
                matrix[source_id][target_id] = score.total_score
                matrix[target_id][source_id] = score.total_score

            computed += 1
            if computed % 1000 == 0:
                print(f"  Computed {computed}/{total} pairs...")

    return dict(matrix)


def auto_label_linkage_benchmark(
    graph_path: Path,
    benchmark_path: Path,
    output_path: Path,
) -> None:
    """Automatically label a linkage benchmark using graph-based scoring."""
    from neural_search.evaluation.dataset_linkage import (
        load_benchmark,
        save_benchmark,
    )

    print(f"Loading benchmark from {benchmark_path}...")
    benchmark = load_benchmark(benchmark_path)

    print(f"Loading scorer from {graph_path}...")
    scorer = RelatednessScorer.from_graph_file(graph_path)

    print(f"Scoring {benchmark.n_pairs} pairs...")
    for i, pair in enumerate(benchmark.pairs):
        score = scorer.score_pair(pair.source_id, pair.target_id)
        pair.relatedness_score = score.to_grade()

        # Store detailed scores as evidence
        pair.evidence = [
            f"total={score.total_score:.3f}",
            f"reusability={score.reusability_score:.3f}",
            f"comparability={score.comparability_score:.3f}",
        ]
        pair.shared_attributes = score.to_dict()["dimensions"]

        if (i + 1) % 100 == 0:
            print(f"  Scored {i + 1}/{benchmark.n_pairs}")

    # Save updated benchmark
    save_benchmark(benchmark, output_path)
    print(f"Saved auto-labeled benchmark to {output_path}")

    # Print summary
    grades = [p.relatedness_score for p in benchmark.pairs]
    print("\nGrade distribution:")
    for g in range(4):
        count = grades.count(g)
        print(f"  Grade {g}: {count} ({count/len(grades)*100:.1f}%)")
