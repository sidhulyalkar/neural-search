"""Tests for the explanation generation module."""

from __future__ import annotations

from neural_search.search.explanation import (
    ExplanationContext,
    ExplanationResult,
    MatchGroup,
    generate_explanation,
)


class TestMatchGroup:
    """Tests for MatchGroup dataclass."""

    def test_basic_match_group(self):
        """Test basic match group creation."""
        group = MatchGroup(
            category="task",
            query_terms=["reversal_learning", "decision_making"],
            matched_terms=["reversal_learning"],
            match_quality="full",
        )
        assert group.category == "task"
        assert len(group.query_terms) == 2
        assert len(group.matched_terms) == 1
        assert group.match_quality == "full"

    def test_partial_match_group(self):
        """Test match group with partial matches."""
        group = MatchGroup(
            category="modality",
            query_terms=["neuropixels"],
            matched_terms=[],
            partial_matches=["extracellular_ephys"],
            match_quality="partial",
        )
        assert len(group.matched_terms) == 0
        assert len(group.partial_matches) == 1
        assert group.match_quality == "partial"


class TestExplanationContext:
    """Tests for ExplanationContext."""

    def test_basic_context(self):
        """Test basic context creation."""
        context = ExplanationContext(
            query_text="reversal learning ephys",
            dataset_title="Test Dataset",
            dataset_id="test_001",
            match_groups=[
                MatchGroup(
                    category="task",
                    query_terms=["reversal_learning"],
                    matched_terms=["reversal_learning"],
                    match_quality="full",
                ),
            ],
            score=75.0,
            score_breakdown={"task_score": 0.8, "modality_score": 0.5},
            warnings=[],
            missing_metadata=[],
        )
        assert context.query_text == "reversal learning ephys"
        assert context.score == 75.0
        assert len(context.match_groups) == 1


class TestGenerateExplanation:
    """Tests for explanation generation."""

    def test_generate_excellent_match(self):
        """Test explanation for excellent match."""
        context = ExplanationContext(
            query_text="reversal learning ephys",
            dataset_title="Reversal Learning Ephys Dataset",
            dataset_id="test_001",
            match_groups=[
                MatchGroup(
                    category="task",
                    query_terms=["reversal_learning"],
                    matched_terms=["reversal_learning"],
                    match_quality="full",
                ),
                MatchGroup(
                    category="modality",
                    query_terms=["ephys"],
                    matched_terms=["extracellular_ephys"],
                    match_quality="full",
                ),
                MatchGroup(
                    category="species",
                    query_terms=[],
                    matched_terms=["mouse"],
                    match_quality="full",
                ),
            ],
            score=85.0,
            score_breakdown={
                "task_score": 1.0,
                "modality_score": 0.9,
                "species_score": 0.5,
            },
            warnings=[],
            missing_metadata=[],
        )

        result = generate_explanation(context)

        assert isinstance(result, ExplanationResult)
        assert result.quality_grade == "excellent"
        assert "reversal learning ephys" in result.detailed
        assert "task" in result.brief.lower()

    def test_generate_weak_match(self):
        """Test explanation for weak match."""
        context = ExplanationContext(
            query_text="calcium imaging behavioral task",
            dataset_title="Random Dataset",
            dataset_id="test_002",
            match_groups=[
                MatchGroup(
                    category="modality",
                    query_terms=["calcium_imaging"],
                    matched_terms=[],
                    match_quality="none",
                ),
            ],
            score=15.0,
            score_breakdown={"semantic_score": 0.15},
            warnings=["Modality mismatch"],
            missing_metadata=["description", "species"],
        )

        result = generate_explanation(context)

        assert result.quality_grade == "weak"
        assert "limited overlap" in result.brief.lower() or "limited" in result.detailed.lower()

    def test_generate_moderate_match(self):
        """Test explanation for moderate match."""
        context = ExplanationContext(
            query_text="neuropixels mouse",
            dataset_title="Mouse V1 Dataset",
            dataset_id="test_003",
            match_groups=[
                MatchGroup(
                    category="modality",
                    query_terms=["neuropixels"],
                    matched_terms=["neuropixels"],
                    match_quality="full",
                ),
                MatchGroup(
                    category="species",
                    query_terms=["mouse"],
                    matched_terms=[],
                    match_quality="none",
                ),
            ],
            score=45.0,
            score_breakdown={"modality_score": 0.9, "species_score": 0.0},
            warnings=[],
            missing_metadata=[],
        )

        result = generate_explanation(context)

        assert result.quality_grade in {"moderate", "good"}
        assert "modality" in result.brief.lower() or "neuropixels" in result.brief.lower()

    def test_technical_explanation(self):
        """Test technical explanation format."""
        context = ExplanationContext(
            query_text="test query",
            dataset_title="Test Dataset",
            dataset_id="test_004",
            match_groups=[
                MatchGroup(
                    category="task",
                    query_terms=["task1"],
                    matched_terms=["task1"],
                    match_quality="full",
                ),
            ],
            score=60.0,
            score_breakdown={
                "task_score": 0.8,
                "modality_score": 0.4,
                "final_score": 0.6,
            },
            warnings=[],
            missing_metadata=["description"],
        )

        result = generate_explanation(context)

        assert "test_004" in result.technical
        assert "Score Components:" in result.technical
        assert "task_score" in result.technical or "modality_score" in result.technical

    def test_match_summary(self):
        """Test match summary structure."""
        context = ExplanationContext(
            query_text="test",
            dataset_title="Test",
            dataset_id="test",
            match_groups=[
                MatchGroup(
                    category="task",
                    query_terms=["t1"],
                    matched_terms=["t1"],
                    match_quality="full",
                ),
                MatchGroup(
                    category="modality",
                    query_terms=["m1"],
                    matched_terms=["m1"],
                    match_quality="full",
                ),
            ],
            score=70.0,
            score_breakdown={},
            warnings=[],
            missing_metadata=[],
        )

        result = generate_explanation(context)

        assert "total_matches" in result.match_summary
        assert result.match_summary["total_matches"] == 2
        assert "categories_matched" in result.match_summary
        assert len(result.match_summary["categories_matched"]) == 2


class TestExplanationEdgeCases:
    """Tests for edge cases in explanation generation."""

    def test_empty_match_groups(self):
        """Test with no match groups."""
        context = ExplanationContext(
            query_text="test query",
            dataset_title="Test Dataset",
            dataset_id="test",
            match_groups=[],
            score=10.0,
            score_breakdown={},
            warnings=[],
            missing_metadata=[],
        )

        result = generate_explanation(context)

        assert result.quality_grade == "weak"
        assert result.brief  # Should still have some explanation

    def test_with_warnings(self):
        """Test explanation includes warnings."""
        context = ExplanationContext(
            query_text="calcium imaging",
            dataset_title="Test",
            dataset_id="test",
            match_groups=[],
            score=20.0,
            score_breakdown={},
            warnings=["Modality mismatch: expected calcium imaging"],
            missing_metadata=[],
        )

        result = generate_explanation(context)

        assert "Modality mismatch" in result.detailed or "⚠" in result.detailed

    def test_with_linked_papers(self):
        """Test explanation mentions linked papers."""
        context = ExplanationContext(
            query_text="test",
            dataset_title="Test",
            dataset_id="test",
            match_groups=[
                MatchGroup(
                    category="task",
                    query_terms=["t1"],
                    matched_terms=["t1"],
                    match_quality="full",
                )
            ],
            score=50.0,
            score_breakdown={},
            warnings=[],
            missing_metadata=[],
            graph_context={"enabled": True},
            linked_papers=[
                {"title": "Paper 1", "doi": "10.1234"},
                {"title": "Paper 2", "doi": "10.5678"},
            ],
        )

        result = generate_explanation(context)

        assert "publication" in result.detailed.lower() or "linked" in result.detailed.lower()
