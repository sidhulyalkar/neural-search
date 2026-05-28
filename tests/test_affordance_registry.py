"""Tests for the affordance registry and validation."""

import pytest

from neural_search.affordances import (
    AFFORDANCE_REGISTRY,
    DatasetFeatures,
    SupportLevel,
    detect_features_from_metadata,
    get_affordance,
    get_all_affordances,
    list_affordances,
    validate_affordance,
    validate_all_affordances,
)


class TestAffordanceRegistry:
    """Tests for the affordance registry."""

    def test_registry_has_expected_affordances(self):
        """Test that registry contains expected affordances."""
        affordances = list_affordances()

        assert "event_aligned_psth" in affordances
        assert "choice_decoding" in affordances
        assert "q_learning" in affordances
        assert "dimensionality_reduction" in affordances
        assert len(affordances) >= 10

    def test_get_affordance_returns_requirement(self):
        """Test getting an affordance requirement."""
        req = get_affordance("choice_decoding")

        assert req is not None
        assert req.affordance_id == "choice_decoding"
        assert req.label == "Choice decoding"
        assert len(req.required_features) > 0

    def test_get_unknown_affordance_returns_none(self):
        """Test that unknown affordance returns None."""
        req = get_affordance("nonexistent")
        assert req is None

    def test_all_affordances_have_required_fields(self):
        """Test that all affordances have required fields."""
        for aff in get_all_affordances():
            assert aff.affordance_id
            assert aff.label
            assert len(aff.required_features) > 0
            assert len(aff.validation_methods) > 0


class TestFeatureDetection:
    """Tests for feature detection from metadata."""

    def test_detect_neural_modalities(self):
        """Test detection of neural recording modalities."""
        dataset = {
            "dataset_id": "test:001",
            "modality": ["electrophysiology", "neuropixels"],
        }
        features = detect_features_from_metadata(dataset)

        assert features.has_neural_data is True
        assert features.has_spike_times is True

    def test_detect_calcium_imaging(self):
        """Test detection of calcium imaging."""
        dataset = {
            "dataset_id": "test:002",
            "modality": ["calcium_imaging", "two_photon"],
        }
        features = detect_features_from_metadata(dataset)

        assert features.has_neural_data is True
        assert features.has_calcium_imaging is True

    def test_detect_behavioral_events(self):
        """Test detection of behavioral events."""
        dataset = {
            "dataset_id": "test:003",
            "behavioral_events": ["choice", "reward", "stimulus_onset"],
        }
        features = detect_features_from_metadata(dataset)

        assert features.has_choice_labels is True
        assert features.has_reward_signal is True
        assert features.has_event_timestamps is True

    def test_detect_trial_structure(self):
        """Test detection of trial structure."""
        dataset = {
            "dataset_id": "test:004",
            "n_trials": 100,
            "usability": {"has_trials": True, "has_behavior": True},
        }
        features = detect_features_from_metadata(dataset)

        assert features.has_trial_structure is True
        assert features.trial_count == 100
        assert features.has_behavior is True

    def test_detect_multiple_brain_regions(self):
        """Test detection of multiple brain regions."""
        dataset = {
            "dataset_id": "test:005",
            "brain_region": ["V1", "V2", "motor_cortex"],
        }
        features = detect_features_from_metadata(dataset)

        assert features.has_multiple_regions is True
        assert len(features.brain_regions) == 3

    def test_detect_data_standards(self):
        """Test detection of data standards."""
        dataset = {
            "dataset_id": "test:006",
            "data_standards": ["nwb"],
        }
        features = detect_features_from_metadata(dataset)

        assert "nwb" in features.data_standards


class TestAffordanceValidation:
    """Tests for affordance validation."""

    def test_choice_decoding_supported(self):
        """Test that choice decoding is supported with required features."""
        features = DatasetFeatures(
            dataset_id="test:001",
            has_neural_data=True,
            has_spike_times=True,
            has_choice_labels=True,
            has_trial_structure=True,
            trial_count=100,
        )

        result = validate_affordance("choice_decoding", features)

        assert result.supported is True
        assert result.support_level in ["high", "medium"]
        assert len(result.missing_required_features) == 0

    def test_choice_decoding_unsupported_without_choices(self):
        """Test that choice decoding is unsupported without choice labels."""
        features = DatasetFeatures(
            dataset_id="test:002",
            has_neural_data=True,
            has_spike_times=True,
            has_choice_labels=False,  # Missing!
            has_trial_structure=True,
        )

        result = validate_affordance("choice_decoding", features)

        # Should be unsupported or low
        assert "choice_labels" in result.missing_required_features

    def test_q_learning_requires_reward(self):
        """Test that Q-learning requires reward signal."""
        features = DatasetFeatures(
            dataset_id="test:003",
            has_trial_structure=True,
            has_choice_labels=True,
            has_reward_signal=False,  # Missing!
            has_outcome_labels=False,
        )

        result = validate_affordance("q_learning", features)

        assert "reward_signal" in result.missing_required_features

    def test_q_learning_fully_supported(self):
        """Test Q-learning with all features."""
        features = DatasetFeatures(
            dataset_id="test:004",
            has_trial_structure=True,
            has_choice_labels=True,
            has_reward_signal=True,
            has_outcome_labels=True,
            has_behavior=True,  # Required for negative condition check
            trial_count=200,
        )

        result = validate_affordance("q_learning", features)

        assert result.supported is True
        assert len(result.missing_required_features) == 0

    def test_event_aligned_psth_supported(self):
        """Test event-aligned PSTH support."""
        features = DatasetFeatures(
            dataset_id="test:005",
            has_neural_data=True,
            has_spike_times=True,
            has_event_timestamps=True,
            has_trial_structure=True,  # Needed to avoid aggregate_only negative
        )

        result = validate_affordance("event_aligned_psth", features)

        assert result.supported is True

    def test_cross_area_requires_multiple_regions(self):
        """Test that cross-area analysis requires multiple regions."""
        features = DatasetFeatures(
            dataset_id="test:006",
            has_neural_data=True,
            has_multiple_regions=False,  # Only one region
            brain_regions=["V1"],
        )

        result = validate_affordance("cross_area_interaction", features)

        assert "multiple_brain_regions" in result.missing_required_features

    def test_dimensionality_reduction_requires_population(self):
        """Test that dimensionality reduction requires population data."""
        features = DatasetFeatures(
            dataset_id="test:007",
            has_neural_data=True,
            unit_count=1,  # Single unit only
        )

        result = validate_affordance("dimensionality_reduction", features)

        # Should identify single neuron issue
        assert result.support_level in ["low", "unsupported"]

    def test_validate_all_affordances(self):
        """Test validating all affordances for a dataset."""
        features = DatasetFeatures(
            dataset_id="test:008",
            has_neural_data=True,
            has_spike_times=True,
            has_trial_structure=True,
            has_choice_labels=True,
            has_event_timestamps=True,
            unit_count=50,
        )

        results = validate_all_affordances(features)

        assert len(results) == len(list_affordances())
        # At least some should be supported
        supported = [r for r in results if r.supported]
        assert len(supported) > 0

    def test_validation_includes_evidence(self):
        """Test that validation results include evidence."""
        features = DatasetFeatures(
            dataset_id="test:009",
            has_neural_data=True,
            has_spike_times=True,
            has_event_timestamps=True,
            has_trial_structure=True,  # Needed to avoid aggregate_only negative
        )

        result = validate_affordance("event_aligned_psth", features)

        assert result.supported is True
        # Should have evidence for found features
        assert len(result.found_required_features) > 0


class TestFalsePositivePrevention:
    """Tests for preventing false positives.

    These tests verify that the system correctly identifies datasets
    that SEEM to support an analysis but actually don't.
    """

    def test_mentions_decision_making_but_no_choices(self):
        """Test: Dataset mentions decision-making but has no trial-level choices."""
        # This simulates a dataset with "decision-making" in description
        # but no actual choice data
        features = DatasetFeatures(
            dataset_id="decoy:001",
            has_neural_data=True,
            has_spike_times=True,
            has_trial_structure=False,
            has_choice_labels=False,
            has_behavior=False,
        )

        result = validate_affordance("choice_decoding", features)

        # Should NOT be supported
        assert result.supported is False or result.support_level in ["low", "unsupported"]

    def test_has_rewards_but_no_trial_structure(self):
        """Test: Dataset has rewards mentioned but no trial structure."""
        features = DatasetFeatures(
            dataset_id="decoy:002",
            has_neural_data=True,
            has_reward_signal=True,
            has_trial_structure=False,
            has_choice_labels=False,
        )

        result = validate_affordance("q_learning", features)

        # Q-learning requires trial structure
        assert "trial_structure" in result.missing_required_features or not result.supported

    def test_fmri_task_data_no_condition_labels(self):
        """Test: fMRI task data but no condition labels."""
        features = DatasetFeatures(
            dataset_id="decoy:003",
            has_neural_data=True,
            has_fmri=True,
            has_trial_structure=False,
            has_stimulus_info=False,
        )

        result = validate_affordance("stimulus_response_modeling", features)

        # Should identify missing stimulus info
        assert not result.supported or "stimulus" in str(result.missing_required_features)

    def test_calcium_imaging_no_roi_traces(self):
        """Test: Calcium imaging mentioned but no ROI traces."""
        features = DatasetFeatures(
            dataset_id="decoy:004",
            has_neural_data=True,
            has_calcium_imaging=False,
            has_roi_traces=False,
            has_trial_structure=True,
        )

        result = validate_affordance("trial_aligned_calcium_analysis", features)

        # Should require calcium data
        assert "calcium" in str(result.missing_required_features).lower() or not result.supported
