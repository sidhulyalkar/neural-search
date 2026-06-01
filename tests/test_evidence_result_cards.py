"""Tests for evidence-backed result cards."""

import pytest

from neural_search.search.evidence_cards import (
    EvidenceResultCard,
    ReusabilityStatus,
    build_evidence_card,
    format_evidence_card_text,
    format_evidence_cards_report,
)


class TestEvidenceResultCard:
    """Tests for EvidenceResultCard model."""

    def test_create_basic_card(self):
        """Test creating a basic evidence card."""
        card = EvidenceResultCard(
            dataset_id="dandi:000026",
            dataset_title="Test Dataset",
            source="dandi",
            rank=1,
            score=0.85,
        )

        assert card.dataset_id == "dandi:000026"
        assert card.rank == 1
        assert card.score == 0.85
        assert card.reusability_status == ReusabilityStatus.UNKNOWN

    def test_card_with_affordances(self):
        """Test card with matched affordances."""
        card = EvidenceResultCard(
            dataset_id="test:001",
            dataset_title="Test",
            source="test",
            rank=1,
            score=0.9,
            reusability_status=ReusabilityStatus.SUPPORTED,
            matched_affordances=["choice_decoding", "q_learning"],
            matched_requirements=["choice", "reward", "trial_structure"],
        )

        assert len(card.matched_affordances) == 2
        assert "choice_decoding" in card.matched_affordances
        assert card.reusability_status == ReusabilityStatus.SUPPORTED

    def test_card_with_missing_requirements(self):
        """Test card with missing requirements."""
        card = EvidenceResultCard(
            dataset_id="test:001",
            dataset_title="Test",
            source="test",
            rank=1,
            score=0.6,
            reusability_status=ReusabilityStatus.PARTIAL,
            matched_requirements=["choice", "reward"],
            missing_requirements=["delay_duration"],
        )

        assert len(card.missing_requirements) == 1
        assert "delay_duration" in card.missing_requirements
        assert card.reusability_status == ReusabilityStatus.PARTIAL

    def test_card_with_evidence(self):
        """Test card with evidence claims."""
        card = EvidenceResultCard(
            dataset_id="test:001",
            dataset_title="Test",
            source="test",
            rank=1,
            score=0.9,
            evidence=[
                {
                    "claim_id": "claim:001",
                    "source_type": "paper_methods",
                    "predicate": "has_task",
                    "object_label": "delay discounting",
                    "summary": "Methods describe delay discounting task",
                    "confidence": 0.85,
                }
            ],
            evidence_claim_ids=["claim:001"],
        )

        assert len(card.evidence) == 1
        assert card.evidence[0]["source_type"] == "paper_methods"
        assert "claim:001" in card.evidence_claim_ids


class TestBuildEvidenceCard:
    """Tests for build_evidence_card function."""

    def test_build_basic_card(self):
        """Test building a basic evidence card."""
        dataset = {
            "dataset_id": "dandi:000026",
            "title": "Test Dataset",
            "source": "dandi",
            "description": "A test dataset",
        }

        card = build_evidence_card(
            dataset=dataset,
            query="test query",
            rank=1,
            score=0.8,
        )

        assert card.dataset_id == "dandi:000026"
        assert card.dataset_title == "Test Dataset"
        assert card.rank == 1
        assert card.score == 0.8

    def test_build_card_with_requirements(self):
        """Test building card with requirement checking."""
        dataset = {
            "dataset_id": "test:001",
            "title": "Choice Task Dataset",
            "source": "test",
            "behavioral_events": ["choice", "reward", "outcome"],
            "usability": {"has_trials": True, "has_behavior": True},
        }

        card = build_evidence_card(
            dataset=dataset,
            query="delay discounting",
            rank=1,
            score=0.7,
            must_have=["choice", "reward", "reaction_time"],
            should_have=["neural_data"],
        )

        assert "choice" in card.matched_requirements
        assert "reward" in card.matched_requirements
        assert "reaction_time" in card.missing_requirements

    def test_build_card_with_affordances(self):
        """Test building card with affordance results."""
        dataset = {
            "dataset_id": "test:001",
            "title": "Test",
            "source": "test",
        }

        affordance_results = [
            {
                "affordance_id": "choice_decoding",
                "supported": True,
                "support_level": "high",
                "confidence": 0.9,
                "found_required_features": ["neural_data", "choice_labels"],
                "missing_required_features": [],
            }
        ]

        card = build_evidence_card(
            dataset=dataset,
            query="choice decoding",
            rank=1,
            score=0.9,
            affordance_results=affordance_results,
        )

        assert "choice_decoding" in card.matched_affordances
        assert card.reusability_status in [ReusabilityStatus.SUPPORTED, ReusabilityStatus.PARTIAL]

    def test_build_card_with_claims(self):
        """Test building card with evidence claims."""
        dataset = {
            "dataset_id": "test:001",
            "title": "Test",
            "source": "test",
        }

        claims = [
            {
                "claim_id": "claim:test:has_task:abc123",
                "source_type": "paper_methods",
                "predicate": "has_task",
                "object_label": "delay discounting",
                "evidence_text": "Subjects chose between immediate and delayed rewards",
                "confidence": 0.85,
            }
        ]

        card = build_evidence_card(
            dataset=dataset,
            query="delay discounting",
            rank=1,
            score=0.9,
            claims=claims,
        )

        assert len(card.evidence) == 1
        assert "claim:test:has_task:abc123" in card.evidence_claim_ids
        assert card.evidence[0]["source_type"] == "paper_methods"

    def test_supported_status_when_all_requirements_met(self):
        """Test that supported status is assigned when all requirements met."""
        dataset = {
            "dataset_id": "test:001",
            "title": "Test",
            "source": "test",
            "behavioral_events": ["choice", "reward", "delay_duration", "outcome"],
        }

        affordance_results = [
            {
                "affordance_id": "delay_discounting_modeling",
                "supported": True,
                "support_level": "high",
                "confidence": 0.9,
                "found_required_features": ["all"],
                "missing_required_features": [],
            }
        ]

        card = build_evidence_card(
            dataset=dataset,
            query="delay discounting",
            rank=1,
            score=0.95,
            must_have=["choice", "reward"],
            affordance_results=affordance_results,
        )

        assert card.reusability_status == ReusabilityStatus.SUPPORTED


class TestFormatEvidenceCard:
    """Tests for evidence card formatting."""

    def test_format_basic_card(self):
        """Test formatting a basic card."""
        card = EvidenceResultCard(
            dataset_id="test:001",
            dataset_title="Test Dataset",
            source="test",
            rank=1,
            score=0.85,
            reusability_status=ReusabilityStatus.SUPPORTED,
            explanation="Dataset supports the analysis.",
        )

        text = format_evidence_card_text(card)

        assert "[1] Test Dataset" in text
        assert "test:001" in text
        assert "0.85" in text
        assert "supported" in text.lower()

    def test_format_card_with_requirements(self):
        """Test formatting card with requirements."""
        card = EvidenceResultCard(
            dataset_id="test:001",
            dataset_title="Test Dataset",
            source="test",
            rank=1,
            score=0.7,
            reusability_status=ReusabilityStatus.PARTIAL,
            matched_requirements=["choice", "reward"],
            missing_requirements=["delay_duration"],
            explanation="Partial support.",
        )

        text = format_evidence_card_text(card)

        assert "✓ Has:" in text
        assert "choice" in text
        assert "✗ Missing:" in text
        assert "delay_duration" in text

    def test_format_card_with_warnings(self):
        """Test formatting card with warnings."""
        card = EvidenceResultCard(
            dataset_id="test:001",
            dataset_title="Test Dataset",
            source="test",
            rank=1,
            score=0.5,
            reusability_status=ReusabilityStatus.UNSUPPORTED,
            warnings=["Limited evidence available", "May match wrong sense"],
            explanation="Not recommended.",
        )

        text = format_evidence_card_text(card)

        assert "⚠" in text
        assert "Limited evidence" in text


class TestFormatCardsReport:
    """Tests for multi-card report formatting."""

    def test_format_empty_report(self):
        """Test formatting empty results."""
        text = format_evidence_cards_report([])
        assert "No results found" in text

    def test_format_multiple_cards(self):
        """Test formatting multiple cards."""
        cards = [
            EvidenceResultCard(
                dataset_id="test:001",
                dataset_title="Dataset 1",
                source="test",
                rank=1,
                score=0.9,
                reusability_status=ReusabilityStatus.SUPPORTED,
                explanation="Good match.",
            ),
            EvidenceResultCard(
                dataset_id="test:002",
                dataset_title="Dataset 2",
                source="test",
                rank=2,
                score=0.7,
                reusability_status=ReusabilityStatus.PARTIAL,
                explanation="Partial match.",
            ),
        ]

        text = format_evidence_cards_report(cards)

        assert "Found 2 results" in text
        assert "Dataset 1" in text
        assert "Dataset 2" in text
        assert "Summary:" in text
        assert "1 supported" in text
        assert "1 partial" in text


class TestReusabilityStatus:
    """Tests for reusability status determination."""

    def test_status_values(self):
        """Test all status values exist."""
        assert ReusabilityStatus.SUPPORTED.value == "supported"
        assert ReusabilityStatus.PARTIAL.value == "partial"
        assert ReusabilityStatus.UNSUPPORTED.value == "unsupported"
        assert ReusabilityStatus.UNKNOWN.value == "unknown"

    def test_status_comparison(self):
        """Test status can be compared."""
        card1 = EvidenceResultCard(
            dataset_id="test:001",
            dataset_title="Test",
            source="test",
            rank=1,
            score=0.9,
            reusability_status=ReusabilityStatus.SUPPORTED,
        )
        card2 = EvidenceResultCard(
            dataset_id="test:002",
            dataset_title="Test 2",
            source="test",
            rank=2,
            score=0.7,
            reusability_status=ReusabilityStatus.PARTIAL,
        )

        assert card1.reusability_status != card2.reusability_status
