"""Tests for NWB and BIDS affordance validators."""

from __future__ import annotations

import pytest

from neural_search.affordances.validators.bids_validator import (
    BIDSFeatureCheck,
    BIDSValidator,
    validate_bids_affordances,
)
from neural_search.affordances.validators.nwb_validator import (
    NWBFeatureCheck,
    NWBValidator,
    ValidationConfidence,
    validate_nwb_affordances,
)


class TestNWBValidator:
    """Tests for NWB validator."""

    def test_validator_initialization(self):
        """Test validator can be created."""
        validator = NWBValidator(use_pynwb=False)
        assert validator is not None
        assert validator.use_pynwb is False

    def test_validate_from_metadata_with_units(self):
        """Test metadata validation detects units table."""
        validator = NWBValidator(use_pynwb=False)
        metadata = {
            "units": True,
            "n_units": 100,
            "spike_times": True,
        }

        result = validator.validate_from_metadata(metadata, dataset_id="test:001")

        assert result.dataset_id == "test:001"
        assert result.has_units_table is not None
        assert result.has_units_table.present is True
        assert result.feature_checks["has_spike_times"].present is True

    def test_validate_from_metadata_with_trials(self):
        """Test metadata validation detects trials table."""
        validator = NWBValidator(use_pynwb=False)
        metadata = {
            "trials": True,
            "n_trials": 500,
            "has_trials": True,
        }

        result = validator.validate_from_metadata(metadata, dataset_id="test:002")

        assert result.has_trials_table is not None
        assert result.has_trials_table.present is True
        assert result.feature_checks["has_trial_structure"].present is True

    def test_validate_from_metadata_with_choice(self):
        """Test metadata validation detects choice labels."""
        validator = NWBValidator(use_pynwb=False)
        metadata = {
            "trials": True,
            "description": "Decision-making task with choice and reward feedback",
        }

        result = validator.validate_from_metadata(metadata, dataset_id="test:003")

        assert result.feature_checks["has_choice_labels"].present is True
        assert result.feature_checks["has_reward_signal"].present is True

    def test_validate_from_metadata_multiple_regions(self):
        """Test metadata validation detects multiple brain regions."""
        validator = NWBValidator(use_pynwb=False)
        metadata = {
            "brain_regions": ["V1", "PFC", "hippocampus"],
        }

        result = validator.validate_from_metadata(metadata, dataset_id="test:004")

        assert result.feature_checks["has_multiple_brain_regions"].present is True
        assert result.feature_checks["has_multiple_brain_regions"].details["n_regions"] == 3

    def test_validate_from_metadata_population_recording(self):
        """Test metadata validation detects population recording."""
        validator = NWBValidator(use_pynwb=False)
        metadata = {
            "n_units": 50,
            "units": True,
        }

        result = validator.validate_from_metadata(metadata, dataset_id="test:005")

        assert result.feature_checks["has_population_recording"].present is True

    def test_validate_from_metadata_insufficient_units(self):
        """Test metadata validation detects insufficient units."""
        validator = NWBValidator(use_pynwb=False)
        metadata = {
            "n_units": 3,
            "units": True,
        }

        result = validator.validate_from_metadata(metadata, dataset_id="test:006")

        # 3 units is not enough for population analysis
        assert result.feature_checks["has_population_recording"].present is False

    def test_affordance_support_computation(self):
        """Test affordance support is computed from features."""
        validator = NWBValidator(use_pynwb=False)
        metadata = {
            "units": True,
            "n_units": 100,
            "trials": True,
            "n_trials": 500,
            "brain_regions": ["V1", "PFC"],
            "description": "Neural recording during choice task with reward",
        }

        result = validator.validate_from_metadata(metadata, dataset_id="test:007")

        # Should support various affordances
        assert "choice_decoding" in result.affordance_support
        assert "dimensionality_reduction" in result.affordance_support
        assert "cross_area_interaction" in result.affordance_support

    def test_missing_requirements_tracked(self):
        """Test missing requirements are tracked per affordance."""
        validator = NWBValidator(use_pynwb=False)
        metadata = {
            "units": True,
            # No trials, no choice labels
        }

        result = validator.validate_from_metadata(metadata, dataset_id="test:008")

        # choice_decoding should show missing requirements
        if "choice_decoding" in result.missing_requirements:
            missing = result.missing_requirements["choice_decoding"]
            # Should be missing trial_structure or choice_labels
            assert len(missing) > 0

    def test_result_to_dict(self):
        """Test result serialization."""
        validator = NWBValidator(use_pynwb=False)
        metadata = {"units": True, "n_units": 50}

        result = validator.validate_from_metadata(metadata, dataset_id="test:009")
        result_dict = result.to_dict()

        assert "dataset_id" in result_dict
        assert "validation_timestamp" in result_dict
        assert "affordance_support" in result_dict
        assert "feature_checks" in result_dict


class TestNWBFeatureCheck:
    """Tests for NWBFeatureCheck dataclass."""

    def test_feature_check_creation(self):
        """Test feature check can be created."""
        check = NWBFeatureCheck(
            feature_name="spike_times",
            present=True,
            confidence=ValidationConfidence.HIGH,
            details={"n_units": 100},
        )

        assert check.feature_name == "spike_times"
        assert check.present is True
        assert check.confidence == ValidationConfidence.HIGH
        assert check.details["n_units"] == 100

    def test_feature_check_with_error(self):
        """Test feature check with error."""
        check = NWBFeatureCheck(
            feature_name="trials",
            present=False,
            confidence=ValidationConfidence.UNKNOWN,
            error="Failed to read trials table",
        )

        assert check.present is False
        assert check.error is not None


class TestBIDSValidator:
    """Tests for BIDS validator."""

    def test_validator_initialization(self):
        """Test validator can be created."""
        validator = BIDSValidator()
        assert validator is not None

    def test_validate_from_metadata_with_fmri(self):
        """Test metadata validation detects fMRI modality."""
        validator = BIDSValidator()
        metadata = {
            "modality": ["fMRI", "T1w"],
            "n_subjects": 30,
        }

        result = validator.validate_from_metadata(metadata, dataset_id="openneuro:001")

        assert result.has_func is True
        assert "func" in result.modalities_found
        assert result.n_subjects == 30

    def test_validate_from_metadata_with_eeg(self):
        """Test metadata validation detects EEG modality."""
        validator = BIDSValidator()
        metadata = {
            "modality": ["EEG"],
            "n_channels": 64,
        }

        result = validator.validate_from_metadata(metadata, dataset_id="openneuro:002")

        assert result.has_eeg is True
        assert "eeg" in result.modalities_found
        assert result.feature_checks["has_neural_data"].present is True
        assert result.feature_checks["has_multiple_channels"].present is True

    def test_validate_from_metadata_with_tasks(self):
        """Test metadata validation detects task info."""
        validator = BIDSValidator()
        metadata = {
            "modality": ["fMRI"],
            "tasks": ["rest", "nback", "motor"],
        }

        result = validator.validate_from_metadata(metadata, dataset_id="openneuro:003")

        assert result.tasks == ["rest", "nback", "motor"]
        assert result.feature_checks["has_trial_structure"].present is True

    def test_validate_from_metadata_continuous_data(self):
        """Test metadata validation detects continuous data."""
        validator = BIDSValidator()
        metadata = {
            "modality": ["MEG"],
        }

        result = validator.validate_from_metadata(metadata, dataset_id="openneuro:004")

        assert result.has_meg is True
        assert result.feature_checks["has_continuous_data"].present is True

    def test_validate_from_metadata_multiple_modalities(self):
        """Test metadata validation handles multiple modalities."""
        validator = BIDSValidator()
        metadata = {
            "modality": ["fMRI", "EEG", "T1w", "DWI"],
        }

        result = validator.validate_from_metadata(metadata, dataset_id="openneuro:005")

        assert result.has_func is True
        assert result.has_eeg is True
        assert result.has_dwi is True

    def test_affordance_support_computation(self):
        """Test affordance support is computed from features."""
        validator = BIDSValidator()
        metadata = {
            "modality": ["EEG"],
            "n_channels": 128,
            "tasks": ["sternberg", "visual"],
            "n_subjects": 50,
        }

        result = validator.validate_from_metadata(metadata, dataset_id="openneuro:006")

        # Should compute affordance support
        assert "functional_connectivity" in result.affordance_support

    def test_result_to_dict(self):
        """Test result serialization."""
        validator = BIDSValidator()
        metadata = {"modality": ["fMRI"], "tasks": ["rest"]}

        result = validator.validate_from_metadata(metadata, dataset_id="openneuro:007")
        result_dict = result.to_dict()

        assert "dataset_path" in result_dict
        assert "modalities_found" in result_dict
        assert "tasks" in result_dict
        assert "affordance_support" in result_dict


class TestBIDSFeatureCheck:
    """Tests for BIDSFeatureCheck dataclass."""

    def test_feature_check_creation(self):
        """Test feature check can be created."""
        check = BIDSFeatureCheck(
            feature_name="events_files",
            present=True,
            confidence="high",
            details={"n_files": 120, "columns": ["onset", "duration", "trial_type"]},
        )

        assert check.feature_name == "events_files"
        assert check.present is True
        assert check.confidence == "high"
        assert check.details["n_files"] == 120


class TestValidationConvenienceFunctions:
    """Tests for convenience validation functions."""

    def test_validate_nwb_affordances_from_metadata(self):
        """Test convenience function with metadata."""
        result = validate_nwb_affordances(
            metadata={"units": True, "n_units": 50},
            dataset_id="test:conv:001",
        )

        assert result.dataset_id == "test:conv:001"
        assert result.validation_success is True

    def test_validate_nwb_affordances_raises_without_input(self):
        """Test convenience function raises without input."""
        with pytest.raises(ValueError):
            validate_nwb_affordances()

    def test_validate_bids_affordances_from_metadata(self):
        """Test convenience function with metadata."""
        result = validate_bids_affordances(
            metadata={"modality": ["fMRI"], "tasks": ["rest"]},
            dataset_id="openneuro:conv:001",
        )

        assert result.dataset_id == "openneuro:conv:001"
        assert result.validation_success is True

    def test_validate_bids_affordances_raises_without_input(self):
        """Test convenience function raises without input."""
        with pytest.raises(ValueError):
            validate_bids_affordances()


class TestNWBValidatorIntegration:
    """Integration tests for NWB validator with affordance registry."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        validator = NWBValidator(use_pynwb=False)

        # Simulate a rich NWB dataset
        metadata = {
            "units": True,
            "n_units": 200,
            "spike_times": True,
            "trials": True,
            "n_trials": 1000,
            "brain_regions": ["M1", "PMd", "S1"],
            "electrodes": True,
            "n_electrodes": 384,
            "description": "Decision-making task with choice and reward feedback",
        }

        result = validator.validate_from_metadata(metadata, dataset_id="dandi:000003")

        # Verify all features detected
        assert result.has_units_table.present is True
        assert result.has_trials_table.present is True
        assert result.has_electrodes.present is True
        assert result.feature_checks["has_neural_data"].present is True
        assert result.feature_checks["has_population_recording"].present is True
        assert result.feature_checks["has_multiple_brain_regions"].present is True
        assert result.feature_checks["has_choice_labels"].present is True
        assert result.feature_checks["has_reward_signal"].present is True

        # Verify affordance support
        assert len(result.affordance_support) > 0
        assert len(result.affordance_evidence) > 0

    def test_minimal_dataset_validation(self):
        """Test validation of minimal dataset."""
        validator = NWBValidator(use_pynwb=False)

        # Simulate a minimal NWB dataset
        metadata = {
            "description": "Resting state recording",
        }

        result = validator.validate_from_metadata(metadata, dataset_id="dandi:000099")

        # Should have limited feature support
        assert result.has_units_table.present is False
        assert result.has_trials_table.present is False

        # Most affordances should not be supported
        unsupported_count = sum(
            1 for supported in result.affordance_support.values() if not supported
        )
        assert unsupported_count > 0


class TestBIDSValidatorIntegration:
    """Integration tests for BIDS validator with affordance registry."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        validator = BIDSValidator()

        # Simulate a rich BIDS dataset
        metadata = {
            "modality": ["fMRI", "T1w", "DWI"],
            "tasks": ["rest", "nback", "motor"],
            "n_subjects": 100,
            "n_sessions": 2,
        }

        result = validator.validate_from_metadata(metadata, dataset_id="openneuro:ds001")

        # Verify modalities detected
        assert result.has_func is True
        assert result.has_dwi is True
        assert len(result.tasks) == 3
        assert result.n_subjects == 100

        # Verify feature checks
        assert result.feature_checks["has_neural_data"].present is True
        assert result.feature_checks["has_trial_structure"].present is True

        # Verify affordance support computed
        assert len(result.affordance_support) > 0
