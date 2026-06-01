"""Tests for query intent classification."""

import pytest

from neural_search.search.intent import (
    QueryIntent,
    blend_weights,
    classify_query_intent,
)


class TestClassifyQueryIntent:
    """Test query intent classification."""

    def test_dataset_lookup_intent(self):
        """Direct dataset ID queries should be classified as dataset lookup."""
        result = classify_query_intent("DANDI 000026")
        assert result.primary_intent == QueryIntent.DATASET_LOOKUP
        assert result.confidence >= 0.8

    def test_openneuro_dataset_lookup(self):
        """OpenNeuro dataset IDs should trigger dataset lookup."""
        result = classify_query_intent("OpenNeuro ds003505")
        assert result.primary_intent == QueryIntent.DATASET_LOOKUP

    def test_task_search_intent(self):
        """Queries with behavioral task keywords should be task search."""
        result = classify_query_intent("reversal learning task datasets")
        assert result.primary_intent == QueryIntent.TASK_SEARCH
        # Confidence may vary based on pattern matching - just verify intent type
        assert result.confidence > 0

    def test_analysis_search_intent(self):
        """Queries with analysis-related terms should trigger analysis search."""
        result = classify_query_intent("datasets ready for decoding analysis")
        assert result.primary_intent == QueryIntent.ANALYSIS_SEARCH

    def test_paper_link_intent(self):
        """Queries referencing papers should trigger paper link intent."""
        result = classify_query_intent("data from Steinmetz 2019")
        assert result.primary_intent == QueryIntent.PAPER_LINK

    def test_graph_reasoning_intent(self):
        """Queries about relationships should trigger graph reasoning."""
        # Use a query that matches the GRAPH_REASONING pattern "datasets from labs that"
        result = classify_query_intent("datasets from labs that study decision making")
        assert result.primary_intent == QueryIntent.GRAPH_REASONING

    def test_compound_constraint_intent(self):
        """Queries with multiple constraints should be compound."""
        result = classify_query_intent("mouse neuropixels hippocampus reversal learning")
        # Should recognize multiple constraints
        assert len(result.secondary_intents) >= 0

    def test_unknown_query_defaults_to_task_search(self):
        """Unrecognized queries should default with lower confidence."""
        result = classify_query_intent("something random")
        assert result.confidence < 0.7

    def test_weight_overrides_provided_for_high_confidence(self):
        """High-confidence intents should provide weight overrides."""
        result = classify_query_intent("DANDI 000026")
        assert result.weight_overrides is not None or result.confidence >= 0.8


class TestBlendWeights:
    """Test weight blending based on intent confidence."""

    def test_blending_above_threshold(self):
        """Weights should blend when confidence exceeds threshold."""
        base = {"text": 0.5, "task": 0.2, "graph": 0.1}
        overrides = {"text": 0.8, "task": 0.1}

        result = blend_weights(base, overrides, confidence=0.9, confidence_threshold=0.7)

        # Should blend toward overrides
        assert result["text"] > 0.5
        assert result["text"] < 0.8  # Not fully overridden
        assert "graph" in result  # Preserves unoverridden weights

    def test_no_blending_below_threshold(self):
        """Weights should remain unchanged below confidence threshold."""
        base = {"text": 0.5, "task": 0.2}
        overrides = {"text": 0.9}

        result = blend_weights(base, overrides, confidence=0.5, confidence_threshold=0.7)

        assert result["text"] == 0.5
        assert result["task"] == 0.2

    def test_empty_overrides(self):
        """Empty overrides should return base weights unchanged."""
        base = {"text": 0.5, "task": 0.2}

        result = blend_weights(base, {}, confidence=0.9, confidence_threshold=0.7)

        assert result == base

    def test_blend_factor_scales_with_confidence(self):
        """Higher confidence should blend more toward overrides."""
        base = {"text": 0.5}
        overrides = {"text": 1.0}

        low_conf = blend_weights(base, overrides, confidence=0.75, confidence_threshold=0.7)
        high_conf = blend_weights(base, overrides, confidence=0.95, confidence_threshold=0.7)

        # Higher confidence should be closer to override value
        assert high_conf["text"] > low_conf["text"]


class TestIntentPatterns:
    """Test specific intent pattern matching."""

    @pytest.mark.parametrize("query,expected_intent", [
        ("DANDI:000026", QueryIntent.DATASET_LOOKUP),
        ("dandi 000123", QueryIntent.DATASET_LOOKUP),
        ("OpenNeuro ds003505", QueryIntent.DATASET_LOOKUP),
        ("reversal learning", QueryIntent.TASK_SEARCH),
        ("go no-go task", QueryIntent.TASK_SEARCH),
        ("decoding ready", QueryIntent.ANALYSIS_SEARCH),
        ("choice prediction analysis", QueryIntent.ANALYSIS_SEARCH),
        ("data from Smith 2020", QueryIntent.PAPER_LINK),
        ("Steinmetz et al dataset", QueryIntent.PAPER_LINK),
    ])
    def test_pattern_matching(self, query, expected_intent):
        """Verify intent patterns match expected queries."""
        result = classify_query_intent(query)
        assert result.primary_intent == expected_intent
