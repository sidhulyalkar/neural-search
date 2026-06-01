"""Tests for UsefulnessIntent classification."""
import pytest
from neural_search.retrieval.query_intent import (
    UsefulnessIntent,
    IntentClassification,
    classify_query_intent,
)


class TestUsefulnessIntentEnum:
    def test_all_intents_exist(self):
        values = {i.value for i in UsefulnessIntent}
        assert "strict_lookup" in values
        assert "replication" in values
        assert "meta_analysis" in values
        assert "pipeline_reuse" in values
        assert "cross_dataset_comparison" in values
        assert "exploration" in values
        assert "method_transfer" in values


class TestClassifyQueryIntent:
    def test_pipeline_reuse_detected(self):
        result = classify_query_intent("datasets like DANDI:000123")
        assert result.intent == UsefulnessIntent.PIPELINE_REUSE
        assert 0.0 <= result.confidence <= 1.0

    def test_replication_detected(self):
        result = classify_query_intent("replicate this choice decoding result")
        assert result.intent == UsefulnessIntent.REPLICATION

    def test_cross_dataset_comparison_detected(self):
        result = classify_query_intent("compare mouse and primate decision making")
        assert result.intent == UsefulnessIntent.CROSS_DATASET_COMPARISON

    def test_method_transfer_detected(self):
        result = classify_query_intent("datasets for Q-learning model fitting")
        assert result.intent == UsefulnessIntent.METHOD_TRANSFER

    def test_strict_lookup_fallback(self):
        result = classify_query_intent("visual cortex calcium imaging mouse")
        assert result.intent == UsefulnessIntent.STRICT_LOOKUP

    def test_exploration_detected(self):
        result = classify_query_intent("find surprising related datasets")
        assert result.intent == UsefulnessIntent.EXPLORATION

    def test_confidence_bounded(self):
        for q in ["foo", "DANDI:000001", "replicate mouse study", "compare species"]:
            r = classify_query_intent(q)
            assert 0.0 <= r.confidence <= 1.0

    def test_matched_patterns_populated(self):
        result = classify_query_intent("datasets like DANDI:000123")
        assert len(result.matched_patterns) >= 1

    def test_explanation_nonempty(self):
        result = classify_query_intent("replicate this study")
        assert len(result.explanation) > 0
