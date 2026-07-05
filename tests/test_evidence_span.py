"""Tests for neural_search.literature.evidence_span."""

from __future__ import annotations

from neural_search.literature.evidence_span import locate_evidence_span


class TestExactMatch:
    def test_exact_substring_match(self):
        source = "Theta power increased in hippocampus during navigation. Other text follows."
        span = locate_evidence_span(source, "Theta power increased in hippocampus during navigation.")
        assert span is not None
        assert span.match_method == "exact"
        assert span.match_score == 1.0
        assert source[span.char_start : span.char_end].lower() == (
            "theta power increased in hippocampus during navigation."
        )

    def test_case_insensitive_match(self):
        source = "THETA POWER INCREASED in hippocampus during navigation."
        span = locate_evidence_span(source, "theta power increased in hippocampus")
        assert span is not None
        assert span.match_method == "exact"

    def test_sentence_id_for_second_sentence(self):
        source = "First sentence here. Theta power increased in hippocampus. Third sentence."
        span = locate_evidence_span(source, "Theta power increased in hippocampus.")
        assert span is not None
        assert span.sentence_id == 1


class TestFuzzyMatch:
    def test_paraphrased_finding_falls_back_to_fuzzy(self):
        source = "We observed a marked elevation of theta oscillatory power within the hippocampal formation during the spatial navigation task."
        finding = "Theta power increased in hippocampus during navigation."
        span = locate_evidence_span(source, finding, min_score=0.2)
        assert span is not None
        assert span.match_method == "fuzzy"
        assert 0.0 < span.match_score < 1.0

    def test_low_overlap_returns_none(self):
        source = "Cortical gamma synchrony decreased after lesion in unrelated subjects."
        finding = "Hippocampal theta power increased markedly during spatial navigation tasks in mice."
        span = locate_evidence_span(source, finding, min_score=0.6)
        assert span is None


class TestEdgeCases:
    def test_empty_source_returns_none(self):
        assert locate_evidence_span("", "some finding text") is None

    def test_empty_finding_returns_none(self):
        assert locate_evidence_span("some source text", "") is None

    def test_whitespace_only_finding_returns_none(self):
        assert locate_evidence_span("some source text", "   ") is None
