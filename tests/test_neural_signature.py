"""Tests for NeuralSignatureV1 schema."""

from __future__ import annotations

import pytest
from neural_search.core.neural_signature import (
    CalciumStats,
    FiringRateStats,
    ISIStats,
    NeuralSignatureV1,
    RecordingModality,
    SignatureQuality,
    TrialStats,
    extract_signature_from_metadata,
)


class TestNeuralSignatureV1:
    """Tests for NeuralSignatureV1 schema."""

    def test_basic_creation(self):
        """Test basic signature creation."""
        sig = NeuralSignatureV1(
            dataset_id="dandi:000003",
            modality=RecordingModality.EPHYS,
            n_units=150,
        )

        assert sig.dataset_id == "dandi:000003"
        assert sig.modality == RecordingModality.EPHYS
        assert sig.n_units == 150
        assert sig.signature_id is not None

    def test_signature_id_generation(self):
        """Test automatic signature ID generation."""
        sig1 = NeuralSignatureV1(dataset_id="dandi:000003")
        sig2 = NeuralSignatureV1(dataset_id="dandi:000003")

        # Same dataset should get same signature ID
        assert sig1.signature_id == sig2.signature_id

        sig3 = NeuralSignatureV1(
            dataset_id="dandi:000003",
            asset_id="asset123",
        )
        # Different asset should get different ID
        assert sig1.signature_id != sig3.signature_id

    def test_brain_region_count(self):
        """Test automatic brain region count."""
        sig = NeuralSignatureV1(
            dataset_id="dandi:000003",
            brain_regions=["M1", "PMd", "S1"],
        )

        assert sig.n_brain_regions == 3

    def test_firing_rate_stats(self):
        """Test firing rate statistics."""
        stats = FiringRateStats(
            mean_hz=5.2,
            std_hz=3.1,
            n_units_sampled=100,
        )

        sig = NeuralSignatureV1(
            dataset_id="dandi:000003",
            modality=RecordingModality.EPHYS,
            firing_rate_stats=stats,
        )

        assert sig.firing_rate_stats.mean_hz == 5.2

    def test_trial_stats(self):
        """Test trial statistics."""
        trials = TrialStats(
            n_trials=500,
            trial_duration_mean_s=2.5,
            n_event_types=4,
            event_types=["stimulus", "response", "reward", "iti"],
        )

        sig = NeuralSignatureV1(
            dataset_id="dandi:000003",
            trial_stats=trials,
        )

        assert sig.trial_stats.n_trials == 500
        assert len(sig.trial_stats.event_types) == 4

    def test_to_feature_dict(self):
        """Test conversion to feature dictionary."""
        sig = NeuralSignatureV1(
            dataset_id="dandi:000003",
            duration_seconds=3600.0,
            n_units=150,
            brain_regions=["M1", "PMd"],
            firing_rate_stats=FiringRateStats(mean_hz=5.0, std_hz=2.0),
        )

        features = sig.to_feature_dict()

        assert features["duration_seconds"] == 3600.0
        assert features["n_units"] == 150.0
        assert features["n_brain_regions"] == 2.0
        assert features["firing_rate_mean"] == 5.0

    def test_compute_similarity(self):
        """Test similarity computation."""
        sig1 = NeuralSignatureV1(
            dataset_id="dandi:000003",
            feature_vector=[1.0, 0.0, 0.0],
        )
        sig2 = NeuralSignatureV1(
            dataset_id="dandi:000005",
            feature_vector=[1.0, 0.0, 0.0],
        )
        sig3 = NeuralSignatureV1(
            dataset_id="dandi:000007",
            feature_vector=[0.0, 1.0, 0.0],
        )

        # Identical vectors should have similarity 1.0
        assert sig1.compute_similarity(sig2) == pytest.approx(1.0)

        # Orthogonal vectors should have similarity 0.0
        assert sig1.compute_similarity(sig3) == pytest.approx(0.0)

    def test_modality_enum(self):
        """Test modality enum values."""
        assert RecordingModality.EPHYS == "ephys"
        assert RecordingModality.CALCIUM_IMAGING == "calcium_imaging"
        assert RecordingModality.FMRI == "fmri"

    def test_quality_enum(self):
        """Test quality enum values."""
        assert SignatureQuality.HIGH == "high"
        assert SignatureQuality.LOW == "low"


class TestExtractSignatureFromMetadata:
    """Tests for metadata-based signature extraction."""

    def test_basic_extraction(self):
        """Test basic metadata extraction."""
        metadata = {
            "n_units": 100,
            "n_trials": 500,
            "modality": "ephys",
        }

        sig = extract_signature_from_metadata("dandi:000003", metadata)

        assert sig.dataset_id == "dandi:000003"
        assert sig.n_units == 100
        assert sig.modality == RecordingModality.EPHYS
        assert sig.quality == SignatureQuality.LOW

    def test_modality_inference(self):
        """Test modality inference from metadata."""
        # Ephys
        sig = extract_signature_from_metadata("test:001", {"modality": "spike data"})
        assert sig.modality == RecordingModality.EPHYS

        # Calcium imaging
        sig = extract_signature_from_metadata("test:002", {"modality": "calcium imaging"})
        assert sig.modality == RecordingModality.CALCIUM_IMAGING

        # fMRI
        sig = extract_signature_from_metadata("test:003", {"modality": "BOLD fMRI"})
        assert sig.modality == RecordingModality.FMRI

    def test_trial_stats_extraction(self):
        """Test trial stats extraction from metadata."""
        metadata = {
            "n_trials": 200,
            "event_types": ["stimulus", "response"],
        }

        sig = extract_signature_from_metadata("test:001", metadata)

        assert sig.trial_stats is not None
        assert sig.trial_stats.n_trials == 200
        assert sig.trial_stats.event_types == ["stimulus", "response"]

    def test_extractor_notes(self):
        """Test extractor notes are added."""
        sig = extract_signature_from_metadata("test:001", {})

        assert len(sig.extractor_notes) > 0
        assert "metadata only" in sig.extractor_notes[0].lower()


class TestCalciumStats:
    """Tests for CalciumStats model."""

    def test_creation(self):
        """Test calcium stats creation."""
        stats = CalciumStats(
            mean_snr=5.5,
            active_roi_fraction=0.75,
            mean_event_rate_hz=0.3,
        )

        assert stats.mean_snr == 5.5
        assert stats.active_roi_fraction == 0.75


class TestISIStats:
    """Tests for ISIStats model."""

    def test_creation(self):
        """Test ISI stats creation."""
        stats = ISIStats(
            mean_ms=50.0,
            cv=0.8,
            burst_fraction=0.15,
        )

        assert stats.mean_ms == 50.0
        assert stats.cv == 0.8
