"""Tests for DatasetCardV1 and CorpusSnapshot schemas."""

import json

import pytest

from neural_search.core.dataset_card import (
    AffordanceRequirement,
    AffordanceValidationResult,
    CorpusSnapshot,
    DatasetCardV1,
    ProvenanceEdge,
    ProvenanceEvidence,
    SourceSnapshot,
    create_corpus_snapshot,
)


class TestDatasetCardV1:
    """Tests for DatasetCardV1 schema."""

    def test_minimal_card_creation(self):
        """Test creating a card with minimal required fields."""
        card = DatasetCardV1(
            dataset_id="dandi:000001",
            source="dandi",
            source_id="000001",
            title="Test Dataset",
        )
        assert card.dataset_id == "dandi:000001"
        assert card.source == "dandi"
        assert card.card_version == "v1"

    def test_full_card_creation(self):
        """Test creating a card with all fields populated."""
        card = DatasetCardV1(
            dataset_id="dandi:000026",
            source="dandi",
            source_id="000026",
            source_url="https://dandiarchive.org/dandiset/000026",
            version="0.241028.1531",
            title="Neuropixels recordings from mouse visual cortex",
            description="Visual decision-making task with Neuropixels",
            species=["Mus musculus"],
            modality=["neuropixels", "electrophysiology"],
            brain_region=["visual cortex", "V1", "V2"],
            task=["visual_discrimination", "decision_making"],
            behavioral_events=["stimulus_onset", "choice", "reward"],
            data_standards=["nwb"],
            n_subjects=10,
            n_sessions=50,
            n_trials=5000,
            has_trials=True,
            has_behavior=True,
            has_neural_data=True,
            analysis_affordances=["choice_decoding", "event_aligned_psth"],
        )
        assert len(card.species) == 1
        assert len(card.modality) == 2
        assert len(card.brain_region) == 3
        assert card.n_trials == 5000

    def test_generate_text_card(self):
        """Test text card generation for embedding."""
        card = DatasetCardV1(
            dataset_id="test:001",
            source="test",
            source_id="001",
            title="Test Dataset",
            description="A test dataset for validation",
            species=["mouse"],
            modality=["calcium_imaging"],
            task=["reversal_learning"],
        )
        card.update_text_card()

        assert "Test Dataset" in card.text_card
        assert "A test dataset" in card.text_card
        assert "mouse" in card.text_card
        assert "calcium_imaging" in card.text_card
        assert "reversal_learning" in card.text_card

    def test_compute_hash_is_deterministic(self):
        """Test that card hash is deterministic."""
        card1 = DatasetCardV1(
            dataset_id="test:001",
            source="test",
            source_id="001",
            title="Test Dataset",
            species=["mouse"],
            modality=["calcium_imaging"],
        )
        card2 = DatasetCardV1(
            dataset_id="test:001",
            source="test",
            source_id="001",
            title="Test Dataset",
            species=["mouse"],
            modality=["calcium_imaging"],
        )
        assert card1.compute_hash() == card2.compute_hash()

    def test_hash_changes_with_content(self):
        """Test that hash changes when content changes."""
        card1 = DatasetCardV1(
            dataset_id="test:001",
            source="test",
            source_id="001",
            title="Test Dataset",
            species=["mouse"],
        )
        card2 = DatasetCardV1(
            dataset_id="test:001",
            source="test",
            source_id="001",
            title="Test Dataset",
            species=["rat"],  # Different species
        )
        assert card1.compute_hash() != card2.compute_hash()

    def test_required_fields_validation(self):
        """Test that required fields are validated."""
        with pytest.raises(ValueError):
            DatasetCardV1(
                dataset_id="",  # Empty
                source="test",
                source_id="001",
                title="Test",
            )

        with pytest.raises(ValueError):
            DatasetCardV1(
                dataset_id="test:001",
                source="test",
                source_id="001",
                title="",  # Empty
            )

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        card = DatasetCardV1(
            dataset_id="test:001",
            source="test",
            source_id="001",
            title="Test Dataset",
            species=["mouse"],
            modality=["calcium_imaging"],
        )

        json_str = card.model_dump_json()
        loaded = DatasetCardV1.model_validate_json(json_str)

        assert loaded.dataset_id == card.dataset_id
        assert loaded.species == card.species
        assert loaded.compute_hash() == card.compute_hash()


class TestCorpusSnapshot:
    """Tests for CorpusSnapshot schema."""

    def test_generate_snapshot_id(self):
        """Test snapshot ID generation."""
        id1 = CorpusSnapshot.generate_snapshot_id()
        id2 = CorpusSnapshot.generate_snapshot_id()

        assert id1.startswith("corpus_")
        assert len(id1) > 20
        # IDs should be different (different timestamps)
        # Note: Could be same if called in same millisecond, but unlikely

    def test_create_empty_snapshot(self):
        """Test creating an empty snapshot."""
        snapshot = CorpusSnapshot(
            snapshot_id=CorpusSnapshot.generate_snapshot_id(),
        )
        assert snapshot.total_records == 0
        assert snapshot.source_counts == {}

    def test_create_snapshot_from_cards(self):
        """Test creating a snapshot from cards."""
        cards = [
            DatasetCardV1(
                dataset_id="dandi:001",
                source="dandi",
                source_id="001",
                title="DANDI Dataset 1",
            ),
            DatasetCardV1(
                dataset_id="dandi:002",
                source="dandi",
                source_id="002",
                title="DANDI Dataset 2",
            ),
            DatasetCardV1(
                dataset_id="openneuro:001",
                source="openneuro",
                source_id="001",
                title="OpenNeuro Dataset 1",
            ),
        ]

        snapshot = create_corpus_snapshot(
            cards,
            repo_commit="abc123",
            notes="Test snapshot",
        )

        assert snapshot.total_records == 3
        assert snapshot.source_counts["dandi"] == 2
        assert snapshot.source_counts["openneuro"] == 1
        assert snapshot.repo_commit == "abc123"
        assert len(snapshot.records_hash) == 16

    def test_snapshot_hash_changes_with_cards(self):
        """Test that snapshot hash changes when cards change."""
        cards1 = [
            DatasetCardV1(
                dataset_id="test:001",
                source="test",
                source_id="001",
                title="Dataset 1",
            ),
        ]
        cards2 = [
            DatasetCardV1(
                dataset_id="test:001",
                source="test",
                source_id="001",
                title="Dataset 1 Modified",  # Different title
            ),
        ]

        snapshot1 = create_corpus_snapshot(cards1)
        snapshot2 = create_corpus_snapshot(cards2)

        assert snapshot1.records_hash != snapshot2.records_hash


class TestSourceSnapshot:
    """Tests for SourceSnapshot schema."""

    def test_create_source_snapshot(self):
        """Test creating a source snapshot."""
        source = SourceSnapshot(
            source_name="dandi",
            adapter_name="DandiAdapter",
            adapter_version="v0.7.3",
            retrieval_date="2026-05-27T10:00:00Z",
            dataset_count=163,
            byte_count=1000000000,
            api_version="v1.0",
        )
        assert source.source_name == "dandi"
        assert source.dataset_count == 163


class TestProvenanceEvidence:
    """Tests for ProvenanceEvidence schema."""

    def test_create_evidence(self):
        """Test creating provenance evidence."""
        evidence = ProvenanceEvidence(
            evidence_type="structured_metadata",
            source="dandi_api",
            field_path="$.measurementTechnique[0].name",
            confidence=0.95,
            extractor="dandi_adapter",
            extractor_version="v0.7.3",
        )
        assert evidence.evidence_type == "structured_metadata"
        assert evidence.confidence == 0.95

    def test_text_evidence(self):
        """Test creating text-based evidence."""
        evidence = ProvenanceEvidence(
            evidence_type="text_span",
            source="description",
            text="Neuropixels recordings from mouse visual cortex",
            confidence=0.8,
        )
        assert evidence.evidence_type == "text_span"
        assert "Neuropixels" in evidence.text


class TestProvenanceEdge:
    """Tests for ProvenanceEdge schema."""

    def test_create_edge_requires_evidence(self):
        """Test that edges require at least one evidence."""
        with pytest.raises(ValueError):
            ProvenanceEdge(
                edge_id="test_edge",
                source_id="dataset:dandi:000026",
                target_id="task:decision_making",
                edge_type="has_task",
                evidence=[],  # Empty - should fail
            )

    def test_create_valid_edge(self):
        """Test creating a valid edge with evidence."""
        evidence = ProvenanceEvidence(
            evidence_type="structured_metadata",
            source="metadata",
            confidence=0.9,
        )
        edge = ProvenanceEdge(
            edge_id="test_edge",
            source_id="dataset:dandi:000026",
            target_id="task:decision_making",
            edge_type="has_task",
            evidence=[evidence],
        )
        assert edge.edge_type == "has_task"
        assert len(edge.evidence) == 1

    def test_generate_edge_id(self):
        """Test edge ID generation is deterministic."""
        id1 = ProvenanceEdge.generate_edge_id(
            "dataset:dandi:000026",
            "task:decision_making",
            "has_task",
        )
        id2 = ProvenanceEdge.generate_edge_id(
            "dataset:dandi:000026",
            "task:decision_making",
            "has_task",
        )
        assert id1 == id2
        assert len(id1) == 16

    def test_aggregate_confidence(self):
        """Test confidence aggregation from multiple evidence."""
        evidence1 = ProvenanceEvidence(
            evidence_type="structured_metadata",
            source="api",
            confidence=0.9,
        )
        evidence2 = ProvenanceEvidence(
            evidence_type="text_span",
            source="description",
            confidence=0.7,
        )
        edge = ProvenanceEdge(
            edge_id="test",
            source_id="a",
            target_id="b",
            edge_type="has",
            evidence=[evidence1, evidence2],
        )

        agg_conf = edge.aggregate_confidence()
        # Should be > 0.9 (boost from second evidence) but < 1.0
        assert 0.9 < agg_conf <= 1.0


class TestAffordanceRequirement:
    """Tests for AffordanceRequirement schema."""

    def test_create_q_learning_requirement(self):
        """Test creating a Q-learning affordance requirement."""
        req = AffordanceRequirement(
            affordance_id="q_learning",
            label="Q-learning model fitting",
            required_features=[
                "trial_table",
                "ordered_trials",
                "choice_sequence",
                "reward_signal",
            ],
            optional_features=[
                "reaction_time",
                "stimulus_identity",
                "block_label",
            ],
            negative_conditions=[
                "only_summary_statistics",
                "no_trialwise_behavior",
            ],
            validation_methods=[
                "nwb_trials_column_check",
                "bids_events_column_check",
            ],
            min_trials=100,
        )
        assert req.affordance_id == "q_learning"
        assert len(req.required_features) == 4
        assert req.min_trials == 100

    def test_confidence_rules_default(self):
        """Test default confidence rules."""
        req = AffordanceRequirement(
            affordance_id="test",
            label="Test",
        )
        assert "high" in req.confidence_rules
        assert "medium" in req.confidence_rules
        assert "low" in req.confidence_rules


class TestAffordanceValidationResult:
    """Tests for AffordanceValidationResult schema."""

    def test_create_positive_result(self):
        """Test creating a positive validation result."""
        result = AffordanceValidationResult(
            dataset_id="dandi:000026",
            affordance_id="choice_decoding",
            supported=True,
            support_level="high",
            confidence=0.95,
            found_required_features=["trials", "choices", "neural_data"],
            missing_required_features=[],
            found_optional_features=["reaction_time"],
        )
        assert result.supported is True
        assert result.support_level == "high"
        assert len(result.missing_required_features) == 0

    def test_create_negative_result(self):
        """Test creating a negative validation result."""
        result = AffordanceValidationResult(
            dataset_id="dandi:000001",
            affordance_id="q_learning",
            supported=False,
            support_level="unsupported",
            confidence=0.2,
            found_required_features=["neural_data"],
            missing_required_features=["trial_table", "choice_sequence", "reward_signal"],
            validation_notes="Missing behavioral trial structure",
        )
        assert result.supported is False
        assert len(result.missing_required_features) == 3
