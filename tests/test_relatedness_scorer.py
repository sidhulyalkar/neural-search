"""Tests for automated relatedness scoring."""

from __future__ import annotations

import pytest

from neural_search.evaluation.relatedness_scorer import (
    DimensionScore,
    RelatednessDimension,
    RelatednessScore,
    RelatednessScorer,
)


class TestRelatednessScorer:
    """Tests for RelatednessScorer."""

    @pytest.fixture
    def sample_graph(self):
        """Create a minimal test graph."""
        return {
            "nodes": {
                "node:dataset:test:001": {
                    "node_type": "dataset",
                    "label": "Test Dataset 1",
                    "properties": {
                        "usability_flags": {"has_neural_data": True, "has_trials": True}
                    },
                    "evidence": [{"id": "e1"}, {"id": "e2"}],
                },
                "node:dataset:test:002": {
                    "node_type": "dataset",
                    "label": "Test Dataset 2",
                    "properties": {
                        "usability_flags": {"has_neural_data": True}
                    },
                    "evidence": [{"id": "e1"}],
                },
                "node:dataset:test:003": {
                    "node_type": "dataset",
                    "label": "Test Dataset 3",
                    "properties": {},
                },
                "node:modality:ephys": {"label": "Electrophysiology"},
                "node:modality:calcium": {"label": "Calcium Imaging"},
                "node:task:decision": {"label": "Decision Making"},
                "node:species:mouse": {"label": "Mouse"},
            },
            "edges": {
                "e1": {
                    "source_node_id": "node:dataset:test:001",
                    "target_node_id": "node:modality:ephys",
                    "edge_type": "dataset_has_modality",
                },
                "e2": {
                    "source_node_id": "node:dataset:test:002",
                    "target_node_id": "node:modality:ephys",
                    "edge_type": "dataset_has_modality",
                },
                "e3": {
                    "source_node_id": "node:dataset:test:003",
                    "target_node_id": "node:modality:calcium",
                    "edge_type": "dataset_has_modality",
                },
                "e4": {
                    "source_node_id": "node:dataset:test:001",
                    "target_node_id": "node:task:decision",
                    "edge_type": "dataset_has_task",
                },
                "e5": {
                    "source_node_id": "node:dataset:test:002",
                    "target_node_id": "node:task:decision",
                    "edge_type": "dataset_has_task",
                },
                "e6": {
                    "source_node_id": "node:dataset:test:001",
                    "target_node_id": "node:species:mouse",
                    "edge_type": "dataset_has_species",
                },
                "e7": {
                    "source_node_id": "node:dataset:test:002",
                    "target_node_id": "node:species:mouse",
                    "edge_type": "dataset_has_species",
                },
            },
        }

    def test_scorer_creation(self, sample_graph):
        """Test scorer can be created from graph."""
        scorer = RelatednessScorer.from_graph(sample_graph)
        assert scorer is not None

    def test_score_identical_datasets(self, sample_graph):
        """Test scoring same dataset returns high score."""
        scorer = RelatednessScorer.from_graph(sample_graph)
        score = scorer.score_pair(
            "node:dataset:test:001",
            "node:dataset:test:001"
        )
        # Same dataset should have perfect scores on shared dimensions
        assert score.total_score > 0.5

    def test_score_similar_datasets(self, sample_graph):
        """Test datasets with shared attributes score higher."""
        scorer = RelatednessScorer.from_graph(sample_graph)
        score = scorer.score_pair(
            "node:dataset:test:001",
            "node:dataset:test:002"
        )
        # Both have ephys, decision-making, mouse
        assert score.total_score > 0.3
        assert RelatednessDimension.MODALITY_ALIGNMENT in score.dimension_scores
        mod_score = score.dimension_scores[RelatednessDimension.MODALITY_ALIGNMENT]
        assert mod_score.score > 0.5  # Should have high modality alignment

    def test_score_different_modality(self, sample_graph):
        """Test datasets with different modalities score lower."""
        scorer = RelatednessScorer.from_graph(sample_graph)
        score = scorer.score_pair(
            "node:dataset:test:001",  # ephys
            "node:dataset:test:003"   # calcium
        )
        mod_score = score.dimension_scores[RelatednessDimension.MODALITY_ALIGNMENT]
        assert mod_score.score < 0.5  # Different modalities

    def test_to_grade(self, sample_graph):
        """Test grade conversion."""
        scorer = RelatednessScorer.from_graph(sample_graph)

        # Similar datasets should get higher grade
        score_similar = scorer.score_pair(
            "node:dataset:test:001",
            "node:dataset:test:002"
        )
        # Different datasets should get lower grade
        score_different = scorer.score_pair(
            "node:dataset:test:001",
            "node:dataset:test:003"
        )

        assert score_similar.to_grade() >= score_different.to_grade()

    def test_reusability_score(self, sample_graph):
        """Test reusability score computation."""
        scorer = RelatednessScorer.from_graph(sample_graph)
        score = scorer.score_pair(
            "node:dataset:test:001",
            "node:dataset:test:002"
        )
        # Reusability should focus on modality and affordances
        assert 0 <= score.reusability_score <= 1

    def test_comparability_score(self, sample_graph):
        """Test comparability score computation."""
        scorer = RelatednessScorer.from_graph(sample_graph)
        score = scorer.score_pair(
            "node:dataset:test:001",
            "node:dataset:test:002"
        )
        # Comparability should focus on task and species
        assert 0 <= score.comparability_score <= 1

    def test_to_dict(self, sample_graph):
        """Test serialization."""
        scorer = RelatednessScorer.from_graph(sample_graph)
        score = scorer.score_pair(
            "node:dataset:test:001",
            "node:dataset:test:002"
        )
        d = score.to_dict()

        assert "source_id" in d
        assert "target_id" in d
        assert "total_score" in d
        assert "dimensions" in d
        assert "grade" in d


class TestDimensionScore:
    """Tests for DimensionScore."""

    def test_creation(self):
        """Test dimension score creation."""
        score = DimensionScore(
            dimension=RelatednessDimension.MODALITY_ALIGNMENT,
            score=0.8,
            weight=2.0,
            shared_items=["ephys"],
        )

        assert score.score == 0.8
        assert score.weight == 2.0
        assert score.weighted_score == 1.6


class TestRelatednessScore:
    """Tests for RelatednessScore."""

    def test_empty_score(self):
        """Test empty score returns 0."""
        score = RelatednessScore(
            source_id="test:001",
            target_id="test:002"
        )
        assert score.total_score == 0.0

    def test_grade_thresholds(self):
        """Test grade threshold logic."""
        # Create scores with known values
        for target_score, expected_grade in [
            (0.0, 0),
            (0.2, 0),
            (0.3, 1),
            (0.5, 2),
            (0.7, 3),
            (1.0, 3),
        ]:
            score = RelatednessScore(
                source_id="test:001",
                target_id="test:002",
                dimension_scores={
                    RelatednessDimension.MODALITY_ALIGNMENT: DimensionScore(
                        dimension=RelatednessDimension.MODALITY_ALIGNMENT,
                        score=target_score,
                        weight=1.0,
                    )
                }
            )
            assert score.to_grade() == expected_grade, f"Failed for score {target_score}"
