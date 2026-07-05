"""Tests for neural_search.ontology.cognitive_atlas."""

from __future__ import annotations

from neural_search.ontology.cognitive_atlas import get_cogat_coverage, get_cogat_match


class TestGetCogatMatch:
    def test_exact_match_returns_real_label(self):
        match = get_cogat_match("stroop_task")
        assert match is not None
        assert match.cogat_label == "Stroop task"
        assert match.match_type == "exact"

    def test_substring_match_returns_real_label(self):
        match = get_cogat_match("reversal_learning")
        assert match is not None
        assert match.cogat_label == "reversal learning task"

    def test_unknown_task_returns_none(self):
        assert get_cogat_match("not_a_real_task_id") is None

    def test_placeholder_concept_matches_are_filtered_out(self):
        # delay_discounting is mapped in task_atlas.yaml only to the empty
        # placeholder concept (trm_4f244f46ebf58, name="") — must not surface.
        assert get_cogat_match("delay_discounting") is None


class TestGetCogatCoverage:
    def test_coverage_reflects_filtered_matches_not_raw_file_claim(self):
        coverage = get_cogat_coverage()
        assert coverage["total_tasks"] == 87
        # The file's own _meta block claims 87/87 matched; only mappings with
        # a genuine (non-empty) Cognitive Atlas label should count here.
        assert 0 < coverage["validated_matches"] < coverage["total_tasks"]
