"""Tests for the ReusabilityClaim schema and ClaimStore."""

import tempfile
from pathlib import Path

import pytest

from neural_search.core.claims import (
    ClaimPredicate,
    ClaimStore,
    EvidenceSourceType,
    ReusabilityClaim,
    ReviewStatus,
    SOURCE_CONFIDENCE_DEFAULTS,
    claim_has_modality,
    claim_has_task,
    claim_has_variable,
    claim_linked_to_paper,
    claim_supports_affordance,
    create_claim,
    make_claim_id,
)


class TestReusabilityClaim:
    """Tests for the ReusabilityClaim model."""

    def test_create_basic_claim(self):
        """Test creating a basic claim."""
        claim = ReusabilityClaim(
            claim_id="claim:test:has_task:abc123",
            subject_id="dandi:000026",
            predicate="has_task",
            object_id="task:delay_discounting",
            object_label="Delay discounting",
            source_type=EvidenceSourceType.PAPER_METHODS,
            confidence=0.85,
        )

        assert claim.claim_id == "claim:test:has_task:abc123"
        assert claim.subject_id == "dandi:000026"
        assert claim.predicate == "has_task"
        assert claim.object_id == "task:delay_discounting"
        assert claim.object_label == "Delay discounting"
        assert claim.source_type == EvidenceSourceType.PAPER_METHODS
        assert claim.confidence == 0.85
        assert claim.review_status == ReviewStatus.UNREVIEWED

    def test_claim_validates_required_fields(self):
        """Test that required fields must be non-empty."""
        with pytest.raises(ValueError, match="must not be empty"):
            ReusabilityClaim(
                claim_id="",  # Empty
                subject_id="dandi:000026",
                predicate="has_task",
                object_id="task:test",
                source_type=EvidenceSourceType.ARCHIVE_METADATA,
                confidence=0.9,
            )

    def test_claim_validates_confidence_range(self):
        """Test confidence must be in 0-1 range."""
        with pytest.raises(ValueError):
            ReusabilityClaim(
                claim_id="claim:test",
                subject_id="dandi:000026",
                predicate="has_task",
                object_id="task:test",
                source_type=EvidenceSourceType.ARCHIVE_METADATA,
                confidence=1.5,  # Out of range
            )

    def test_claim_jsonl_serialization(self):
        """Test JSONL serialization round-trip."""
        claim = ReusabilityClaim(
            claim_id="claim:test:has_task:abc123",
            subject_id="dandi:000026",
            predicate="has_task",
            object_id="task:delay_discounting",
            source_type=EvidenceSourceType.PAPER_METHODS,
            evidence_text="subjects chose between immediate and delayed rewards",
            confidence=0.85,
        )

        jsonl = claim.to_jsonl()
        restored = ReusabilityClaim.from_jsonl(jsonl)

        assert restored.claim_id == claim.claim_id
        assert restored.subject_id == claim.subject_id
        assert restored.evidence_text == claim.evidence_text
        assert restored.confidence == claim.confidence

    def test_claim_with_review(self):
        """Test updating claim review status."""
        claim = ReusabilityClaim(
            claim_id="claim:test:has_task:abc123",
            subject_id="dandi:000026",
            predicate="has_task",
            object_id="task:delay_discounting",
            source_type=EvidenceSourceType.PAPER_METHODS,
            confidence=0.85,
        )

        reviewed = claim.with_review(ReviewStatus.TRUSTED, "expert@example.com")

        # Original unchanged
        assert claim.review_status == ReviewStatus.UNREVIEWED

        # New claim has review
        assert reviewed.review_status == ReviewStatus.TRUSTED
        assert reviewed.reviewed_by == "expert@example.com"
        assert reviewed.reviewed_at is not None


class TestMakeClaimId:
    """Tests for claim ID generation."""

    def test_deterministic_ids(self):
        """Test that same inputs produce same ID."""
        id1 = make_claim_id("dandi:000026", "has_task", "task:delay_discounting")
        id2 = make_claim_id("dandi:000026", "has_task", "task:delay_discounting")
        assert id1 == id2

    def test_different_inputs_produce_different_ids(self):
        """Test that different inputs produce different IDs."""
        id1 = make_claim_id("dandi:000026", "has_task", "task:delay_discounting")
        id2 = make_claim_id("dandi:000027", "has_task", "task:delay_discounting")
        assert id1 != id2

    def test_id_format(self):
        """Test ID follows expected format."""
        claim_id = make_claim_id("dandi:000026", "has_task", "task:delay_discounting")
        assert claim_id.startswith("claim:")
        assert "has_task" in claim_id


class TestCreateClaim:
    """Tests for the create_claim factory function."""

    def test_default_confidence_by_source_type(self):
        """Test that confidence defaults to source type default."""
        claim = create_claim(
            subject_id="dandi:000026",
            predicate="has_task",
            object_id="task:delay_discounting",
            source_type=EvidenceSourceType.FILE_INSPECTION,
            # No confidence specified
        )

        assert claim.confidence == SOURCE_CONFIDENCE_DEFAULTS[EvidenceSourceType.FILE_INSPECTION]

    def test_explicit_confidence_overrides_default(self):
        """Test that explicit confidence overrides default."""
        claim = create_claim(
            subject_id="dandi:000026",
            predicate="has_task",
            object_id="task:delay_discounting",
            source_type=EvidenceSourceType.FILE_INSPECTION,
            confidence=0.5,  # Override
        )

        assert claim.confidence == 0.5


class TestConvenienceFunctions:
    """Tests for convenience claim creation functions."""

    def test_claim_has_task(self):
        """Test has_task claim creation."""
        claim = claim_has_task(
            dataset_id="dandi:000026",
            task_id="delay_discounting",
            task_label="Delay discounting",
            source_type=EvidenceSourceType.PAPER_METHODS,
        )

        assert claim.predicate == ClaimPredicate.HAS_TASK
        assert claim.object_id == "task:delay_discounting"
        assert claim.object_label == "Delay discounting"

    def test_claim_has_modality(self):
        """Test has_modality claim creation."""
        claim = claim_has_modality(
            dataset_id="dandi:000026",
            modality_id="neuropixels",
            modality_label="Neuropixels recording",
            source_type=EvidenceSourceType.ARCHIVE_METADATA,
        )

        assert claim.predicate == ClaimPredicate.HAS_MODALITY
        assert claim.object_id == "modality:neuropixels"

    def test_claim_supports_affordance(self):
        """Test supports_affordance claim creation."""
        claim = claim_supports_affordance(
            dataset_id="dandi:000026",
            affordance_id="choice_decoding",
            affordance_label="Choice decoding",
            source_type=EvidenceSourceType.FILE_INSPECTION,
            evidence_text="Trial table contains choice labels",
        )

        assert claim.predicate == ClaimPredicate.SUPPORTS_AFFORDANCE
        assert claim.object_id == "affordance:choice_decoding"
        assert "choice labels" in claim.evidence_text

    def test_claim_has_variable(self):
        """Test has_variable claim creation."""
        claim = claim_has_variable(
            dataset_id="dandi:000026",
            variable_name="choice",
            source_type=EvidenceSourceType.FILE_INSPECTION,
        )

        assert claim.predicate == ClaimPredicate.HAS_VARIABLE
        assert claim.object_id == "variable:choice"

    def test_claim_linked_to_paper(self):
        """Test linked_to_paper claim creation."""
        claim = claim_linked_to_paper(
            dataset_id="dandi:000026",
            paper_doi="10.1234/example",
            paper_title="Example Paper Title",
        )

        assert claim.predicate == ClaimPredicate.LINKED_TO_PAPER
        assert claim.object_id == "paper:10.1234/example"
        assert claim.object_label == "Example Paper Title"


class TestClaimStore:
    """Tests for ClaimStore persistence and querying."""

    def test_add_and_get_claim(self):
        """Test adding and retrieving a claim."""
        store = ClaimStore()
        claim = create_claim(
            subject_id="dandi:000026",
            predicate="has_task",
            object_id="task:delay_discounting",
            source_type=EvidenceSourceType.PAPER_METHODS,
        )

        store.add(claim)
        retrieved = store.get(claim.claim_id)

        assert retrieved is not None
        assert retrieved.claim_id == claim.claim_id

    def test_query_by_subject(self):
        """Test querying claims by subject."""
        store = ClaimStore()
        store.add(create_claim(
            subject_id="dandi:000026",
            predicate="has_task",
            object_id="task:test1",
            source_type=EvidenceSourceType.ARCHIVE_METADATA,
        ))
        store.add(create_claim(
            subject_id="dandi:000026",
            predicate="has_modality",
            object_id="modality:ephys",
            source_type=EvidenceSourceType.ARCHIVE_METADATA,
        ))
        store.add(create_claim(
            subject_id="dandi:000027",
            predicate="has_task",
            object_id="task:test2",
            source_type=EvidenceSourceType.ARCHIVE_METADATA,
        ))

        results = store.query_by_subject("dandi:000026")
        assert len(results) == 2

    def test_query_with_filters(self):
        """Test querying with multiple filters."""
        store = ClaimStore()
        store.add(create_claim(
            subject_id="dandi:000026",
            predicate="has_task",
            object_id="task:test1",
            source_type=EvidenceSourceType.FILE_INSPECTION,
            confidence=0.95,
        ))
        store.add(create_claim(
            subject_id="dandi:000026",
            predicate="has_task",
            object_id="task:test2",
            source_type=EvidenceSourceType.INFERRED_ONTOLOGY,
            confidence=0.55,
        ))

        # Filter by min confidence
        high_conf = store.query(min_confidence=0.8)
        assert len(high_conf) == 1
        assert high_conf[0].confidence == 0.95

        # Filter by source type
        file_inspection = store.query(source_type=EvidenceSourceType.FILE_INSPECTION)
        assert len(file_inspection) == 1

    def test_jsonl_persistence(self):
        """Test saving and loading from JSONL."""
        store = ClaimStore()
        store.add(create_claim(
            subject_id="dandi:000026",
            predicate="has_task",
            object_id="task:delay_discounting",
            source_type=EvidenceSourceType.PAPER_METHODS,
        ))
        store.add(create_claim(
            subject_id="dandi:000027",
            predicate="has_modality",
            object_id="modality:ephys",
            source_type=EvidenceSourceType.ARCHIVE_METADATA,
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "claims.jsonl"

            # Save
            count = store.save_jsonl(path)
            assert count == 2
            assert path.exists()

            # Load into new store
            new_store = ClaimStore()
            loaded = new_store.load_jsonl(path)
            assert loaded == 2
            assert len(new_store) == 2

    def test_remove_claim(self):
        """Test removing a claim."""
        store = ClaimStore()
        claim = create_claim(
            subject_id="dandi:000026",
            predicate="has_task",
            object_id="task:test",
            source_type=EvidenceSourceType.ARCHIVE_METADATA,
        )
        store.add(claim)

        assert claim.claim_id in store
        removed = store.remove(claim.claim_id)
        assert removed is True
        assert claim.claim_id not in store


class TestSourceConfidenceDefaults:
    """Tests for source type confidence defaults."""

    def test_file_inspection_highest_confidence(self):
        """Test file inspection has highest confidence."""
        assert SOURCE_CONFIDENCE_DEFAULTS[EvidenceSourceType.FILE_INSPECTION] == 0.95

    def test_broad_taxonomy_lowest_confidence(self):
        """Test broad taxonomy has lowest confidence."""
        assert SOURCE_CONFIDENCE_DEFAULTS[EvidenceSourceType.BROAD_TAXONOMY] == 0.35

    def test_confidence_ordering(self):
        """Test that confidence values reflect reliability ordering."""
        assert (
            SOURCE_CONFIDENCE_DEFAULTS[EvidenceSourceType.FILE_INSPECTION]
            > SOURCE_CONFIDENCE_DEFAULTS[EvidenceSourceType.ARCHIVE_METADATA]
            > SOURCE_CONFIDENCE_DEFAULTS[EvidenceSourceType.PAPER_METHODS]
            > SOURCE_CONFIDENCE_DEFAULTS[EvidenceSourceType.PAPER_ABSTRACT]
            > SOURCE_CONFIDENCE_DEFAULTS[EvidenceSourceType.INFERRED_ONTOLOGY]
            > SOURCE_CONFIDENCE_DEFAULTS[EvidenceSourceType.BROAD_TAXONOMY]
        )
