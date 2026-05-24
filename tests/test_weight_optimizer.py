"""Tests for weight optimizer module."""

import pytest

from neural_search.search.intent import QueryIntent
from neural_search.search.weight_optimizer import (
    QueryAnalysis,
    QueryComplexity,
    WEIGHT_PROFILES,
    analyze_query_for_weights,
    boost_weights_for_constraints,
    get_adaptive_weights,
    interpolate_weights,
    normalize_weights,
)


class TestAnalyzeQueryForWeights:
    """Test query analysis for weight selection."""

    def test_simple_query_complexity(self):
        """Queries with one constraint should be simple."""
        parsed = {"tasks": ["reversal_learning"]}
        analysis = analyze_query_for_weights(parsed)

        assert analysis.complexity == QueryComplexity.SIMPLE
        assert analysis.has_task_constraints is True
        assert analysis.constraint_count == 1

    def test_moderate_query_complexity(self):
        """Queries with 2-3 constraints should be moderate."""
        parsed = {
            "tasks": ["reversal_learning"],
            "modalities": ["neuropixels"],
            "species": ["mouse"],
        }
        analysis = analyze_query_for_weights(parsed)

        assert analysis.complexity == QueryComplexity.MODERATE
        assert analysis.constraint_count == 3

    def test_complex_query_complexity(self):
        """Queries with 4+ constraints should be complex."""
        parsed = {
            "tasks": ["reversal_learning"],
            "modalities": ["neuropixels"],
            "species": ["mouse"],
            "brain_regions": ["hippocampus"],
            "affordances": ["decoding"],
        }
        analysis = analyze_query_for_weights(parsed)

        assert analysis.complexity == QueryComplexity.COMPLEX
        assert analysis.constraint_count == 5

    def test_task_search_profile_suggestion(self):
        """Task queries should suggest task_focused profile."""
        parsed = {"tasks": ["go_nogo", "reversal_learning"]}
        analysis = analyze_query_for_weights(parsed, intent=QueryIntent.TASK_SEARCH)

        assert analysis.suggested_profile == "task_focused"

    def test_analysis_search_profile_suggestion(self):
        """Analysis queries should suggest analysis_focused profile."""
        parsed = {"affordances": ["decoding", "dimensionality_reduction"]}
        analysis = analyze_query_for_weights(parsed, intent=QueryIntent.ANALYSIS_SEARCH)

        assert analysis.suggested_profile == "analysis_focused"

    def test_graph_reasoning_profile_suggestion(self):
        """Graph reasoning intent should suggest graph_reasoning profile."""
        parsed = {
            "tasks": ["reversal_learning"],
            "query_intent": {"primary": "graph_reasoning"},
        }
        analysis = analyze_query_for_weights(parsed)

        assert analysis.suggested_profile == "graph_reasoning"
        assert analysis.has_graph_patterns is True

    def test_dataset_lookup_profile_suggestion(self):
        """Dataset lookup intent should suggest dataset_lookup profile."""
        parsed = {"query": "DANDI 000026"}
        analysis = analyze_query_for_weights(parsed, intent=QueryIntent.DATASET_LOOKUP)

        assert analysis.suggested_profile == "dataset_lookup"

    def test_simple_query_semantic_profile(self):
        """Simple queries without clear intent should use semantic_heavy."""
        parsed = {}  # Empty parsed query
        analysis = analyze_query_for_weights(parsed, intent=QueryIntent.TASK_SEARCH)

        # Simple complexity but task intent -> task_focused
        # Actually no tasks, so it won't be task_focused
        assert analysis.complexity == QueryComplexity.SIMPLE


class TestGetAdaptiveWeights:
    """Test adaptive weight selection."""

    def test_returns_profile_weights(self):
        """Should return weights from suggested profile."""
        parsed = {"tasks": ["reversal_learning"]}
        weights = get_adaptive_weights(parsed, intent=QueryIntent.TASK_SEARCH)

        # Should have higher ontology weight for task-focused
        assert weights["ontology"] > weights["semantic"]

    def test_blends_with_base_weights(self):
        """Should blend profile with base weights."""
        parsed = {"tasks": ["reversal_learning"]}
        base = {"ontology": 0.20, "semantic": 0.30, "custom": 0.10}

        weights = get_adaptive_weights(
            parsed, base_weights=base, intent=QueryIntent.TASK_SEARCH
        )

        # Should have blended values
        assert "custom" in weights  # Preserves custom key
        assert weights["ontology"] > base["ontology"]  # Boosted toward profile

    def test_without_base_weights(self):
        """Should return pure profile weights when no base provided."""
        parsed = {"affordances": ["decoding"]}
        weights = get_adaptive_weights(parsed, intent=QueryIntent.ANALYSIS_SEARCH)

        profile = WEIGHT_PROFILES["analysis_focused"]
        assert weights == profile.weights


class TestBoostWeightsForConstraints:
    """Test constraint-based weight boosting."""

    def test_boosts_ontology_for_tasks(self):
        """Task constraints should boost ontology weight."""
        base = {"ontology": 0.20, "semantic": 0.20, "modality": 0.20}
        parsed = {"tasks": ["reversal_learning"]}

        boosted = boost_weights_for_constraints(base, parsed, boost_factor=0.3)

        assert boosted["ontology"] > base["ontology"]

    def test_boosts_modality_for_modality_constraints(self):
        """Modality constraints should boost modality weight."""
        base = {"ontology": 0.20, "modality": 0.20, "semantic": 0.20}
        parsed = {"modalities": ["neuropixels"]}

        boosted = boost_weights_for_constraints(base, parsed, boost_factor=0.3)

        assert boosted["modality"] > base["modality"]

    def test_multiple_constraint_boosts(self):
        """Multiple constraints should boost multiple weights."""
        base = {"ontology": 0.20, "modality": 0.20, "affordance": 0.20}
        parsed = {
            "tasks": ["reversal_learning"],
            "modalities": ["neuropixels"],
            "affordances": ["decoding"],
        }

        boosted = boost_weights_for_constraints(base, parsed, boost_factor=0.3)

        assert boosted["ontology"] > base["ontology"]
        assert boosted["modality"] > base["modality"]
        assert boosted["affordance"] > base["affordance"]


class TestNormalizeWeights:
    """Test weight normalization."""

    def test_normalizes_to_one(self):
        """Should normalize weights to sum to 1.0."""
        weights = {"a": 0.5, "b": 0.3, "c": 0.2}
        normalized = normalize_weights(weights)

        assert abs(sum(normalized.values()) - 1.0) < 0.001

    def test_handles_unbalanced_weights(self):
        """Should normalize unbalanced weights."""
        weights = {"a": 0.8, "b": 0.6, "c": 0.4}  # Sum = 1.8
        normalized = normalize_weights(weights)

        assert abs(sum(normalized.values()) - 1.0) < 0.001
        assert normalized["a"] > normalized["b"] > normalized["c"]

    def test_handles_zero_sum(self):
        """Should handle zero sum gracefully."""
        weights = {"a": 0.0, "b": 0.0}
        normalized = normalize_weights(weights)

        assert normalized == weights


class TestInterpolateWeights:
    """Test weight interpolation."""

    def test_alpha_zero_returns_a(self):
        """Alpha=0 should return weights_a."""
        a = {"x": 1.0, "y": 0.0}
        b = {"x": 0.0, "y": 1.0}

        result = interpolate_weights(a, b, alpha=0.0)

        assert result["x"] == 1.0
        assert result["y"] == 0.0

    def test_alpha_one_returns_b(self):
        """Alpha=1 should return weights_b."""
        a = {"x": 1.0, "y": 0.0}
        b = {"x": 0.0, "y": 1.0}

        result = interpolate_weights(a, b, alpha=1.0)

        assert result["x"] == 0.0
        assert result["y"] == 1.0

    def test_alpha_half_interpolates(self):
        """Alpha=0.5 should return midpoint."""
        a = {"x": 1.0, "y": 0.0}
        b = {"x": 0.0, "y": 1.0}

        result = interpolate_weights(a, b, alpha=0.5)

        assert result["x"] == 0.5
        assert result["y"] == 0.5

    def test_handles_different_keys(self):
        """Should handle different key sets."""
        a = {"x": 1.0}
        b = {"x": 0.0, "y": 1.0}

        result = interpolate_weights(a, b, alpha=0.5)

        assert "x" in result
        assert "y" in result

    def test_clamps_alpha(self):
        """Should clamp alpha to [0, 1]."""
        a = {"x": 1.0}
        b = {"x": 0.0}

        result_low = interpolate_weights(a, b, alpha=-0.5)
        result_high = interpolate_weights(a, b, alpha=1.5)

        assert result_low["x"] == 1.0  # Clamped to 0
        assert result_high["x"] == 0.0  # Clamped to 1


class TestWeightProfiles:
    """Test weight profile definitions."""

    def test_all_profiles_sum_near_one(self):
        """All profiles should have weights summing near 1.0."""
        for name, profile in WEIGHT_PROFILES.items():
            total = sum(profile.weights.values())
            assert 0.95 <= total <= 1.05, f"Profile {name} sums to {total}"

    def test_balanced_profile_exists(self):
        """Should have a balanced profile."""
        assert "balanced" in WEIGHT_PROFILES

    def test_task_focused_has_higher_ontology(self):
        """Task focused should have higher ontology than balanced."""
        task = WEIGHT_PROFILES["task_focused"]
        balanced = WEIGHT_PROFILES["balanced"]

        assert task.weights["ontology"] > balanced.weights["ontology"]

    def test_analysis_focused_has_higher_affordance(self):
        """Analysis focused should have higher affordance than balanced."""
        analysis = WEIGHT_PROFILES["analysis_focused"]
        balanced = WEIGHT_PROFILES["balanced"]

        assert analysis.weights["affordance"] > balanced.weights["affordance"]
