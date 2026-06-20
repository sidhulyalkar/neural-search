"""Tests for the 6 new affordance types added in the expansion sprint."""

from neural_search.affordances.registry import (
    DataFormat,
    DatasetFeatures,
    detect_features_from_metadata,
    list_affordances,
    validate_affordance,
)

# ── Registry completeness ────────────────────────────────────────────────────

def test_registry_has_21_affordances():
    affordances = list_affordances()
    assert len(affordances) == 21


def test_new_affordances_present():
    affordances = list_affordances()
    for aff_id in [
        "speech_decoding",
        "seizure_detection",
        "sleep_stage_classification",
        "bci_decoding",
        "latent_dynamics_modeling",
        "representational_similarity_analysis",
    ]:
        assert aff_id in affordances, f"Missing: {aff_id}"


# ── Feature detection ────────────────────────────────────────────────────────

class TestNewFeatureDetection:
    def test_detects_speech_events(self):
        dataset = {
            "dataset_id": "test:speech",
            "behavioral_events": ["speech", "phoneme", "word"],
        }
        features = detect_features_from_metadata(dataset)
        assert features.has_speech_events is True

    def test_no_speech_events_when_absent(self):
        dataset = {"dataset_id": "test:nospch", "behavioral_events": ["choice"]}
        features = detect_features_from_metadata(dataset)
        assert features.has_speech_events is False

    def test_detects_seizure_annotations(self):
        dataset = {
            "dataset_id": "test:sz",
            "behavioral_events": ["seizure", "ictal"],
        }
        features = detect_features_from_metadata(dataset)
        assert features.has_seizure_annotations is True

    def test_detects_seizure_from_interictal(self):
        dataset = {"dataset_id": "test:sz2", "behavioral_events": ["interictal"]}
        features = detect_features_from_metadata(dataset)
        assert features.has_seizure_annotations is True

    def test_no_seizure_when_absent(self):
        dataset = {"dataset_id": "test:nosz", "behavioral_events": ["reward"]}
        features = detect_features_from_metadata(dataset)
        assert features.has_seizure_annotations is False

    def test_detects_sleep_stage_labels(self):
        dataset = {
            "dataset_id": "test:sleep",
            "behavioral_events": ["nrem", "rem", "wake"],
        }
        features = detect_features_from_metadata(dataset)
        assert features.has_sleep_stage_labels is True

    def test_detects_sleep_from_slow_wave(self):
        dataset = {"dataset_id": "test:sw", "behavioral_events": ["slow_wave"]}
        features = detect_features_from_metadata(dataset)
        assert features.has_sleep_stage_labels is True

    def test_no_sleep_when_absent(self):
        dataset = {"dataset_id": "test:nosl", "behavioral_events": ["choice"]}
        features = detect_features_from_metadata(dataset)
        assert features.has_sleep_stage_labels is False

    def test_detects_bci_context_from_events(self):
        dataset = {
            "dataset_id": "test:bci",
            "behavioral_events": ["motor_imagery", "imagined_movement"],
        }
        features = detect_features_from_metadata(dataset)
        assert features.has_bci_context is True

    def test_detects_bci_context_from_task_labels(self):
        dataset = {
            "dataset_id": "test:bci2",
            "task_labels": ["bci_spelling", "p300_bci"],
            "behavioral_events": [],
        }
        features = detect_features_from_metadata(dataset)
        assert features.has_bci_context is True

    def test_no_bci_when_absent(self):
        dataset = {"dataset_id": "test:nobci", "behavioral_events": ["choice"]}
        features = detect_features_from_metadata(dataset)
        assert features.has_bci_context is False


# ── Affordance validation ─────────────────────────────────────────────────────

class TestNewAffordanceValidation:
    def _ecog_speech_features(self) -> DatasetFeatures:
        return DatasetFeatures(
            dataset_id="ecog_speech",
            has_neural_data=True,
            has_ecog=True,
            has_speech_events=True,
            has_event_timestamps=True,
            data_format=DataFormat.NWB,
        )

    def _eeg_seizure_features(self) -> DatasetFeatures:
        return DatasetFeatures(
            dataset_id="eeg_seizure",
            has_neural_data=True,
            has_eeg=True,
            has_seizure_annotations=True,
            has_event_timestamps=True,
        )

    def _eeg_sleep_features(self) -> DatasetFeatures:
        return DatasetFeatures(
            dataset_id="eeg_sleep",
            has_neural_data=True,
            has_eeg=True,
            has_sleep_stage_labels=True,
            has_event_timestamps=True,
        )

    def _ephys_bci_features(self) -> DatasetFeatures:
        return DatasetFeatures(
            dataset_id="ephys_bci",
            has_neural_data=True,
            has_spike_times=True,
            has_bci_context=True,
            has_event_timestamps=True,
        )

    def _population_features(self) -> DatasetFeatures:
        return DatasetFeatures(
            dataset_id="population",
            has_neural_data=True,
            has_spike_times=True,
            unit_count=50,
            has_trial_structure=True,
            has_event_timestamps=True,
            data_format=DataFormat.NWB,
        )

    def test_speech_decoding_supported_on_ecog_speech(self):
        result = validate_affordance("speech_decoding", self._ecog_speech_features())
        assert result.supported is True
        assert result.support_level in ("medium", "high")

    def test_speech_decoding_unsupported_without_speech(self):
        features = DatasetFeatures(
            dataset_id="no_speech",
            has_neural_data=True,
            has_ecog=True,
            has_speech_events=False,
        )
        result = validate_affordance("speech_decoding", features)
        assert result.supported is False

    def test_seizure_detection_supported_with_labels(self):
        result = validate_affordance("seizure_detection", self._eeg_seizure_features())
        assert result.supported is True

    def test_seizure_detection_unsupported_without_labels(self):
        features = DatasetFeatures(dataset_id="no_sz", has_neural_data=True, has_eeg=True)
        result = validate_affordance("seizure_detection", features)
        assert result.supported is False

    def test_sleep_classification_supported(self):
        result = validate_affordance("sleep_stage_classification", self._eeg_sleep_features())
        assert result.supported is True

    def test_bci_decoding_supported(self):
        result = validate_affordance("bci_decoding", self._ephys_bci_features())
        assert result.supported is True

    def test_latent_dynamics_supported_on_population(self):
        result = validate_affordance("latent_dynamics_modeling", self._population_features())
        assert result.supported is True

    def test_rsa_supported_on_population_with_conditions(self):
        features = DatasetFeatures(
            dataset_id="pop_cond",
            has_neural_data=True,
            unit_count=30,
            has_trial_structure=True,
            event_types=["stim_a", "stim_b"],
        )
        result = validate_affordance("representational_similarity_analysis", features)
        assert result.supported is True
