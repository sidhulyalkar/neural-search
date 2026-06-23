import pytest
from pydantic import ValidationError

from neural_search.spectral.schemas import (
    AperiodicEligibility,
    PeriodicPeak,
    SpectralEstimate,
    SpectralFeatureBundle,
    SpectralQCAssessment,
    SpectralRunConfig,
)


def _run_config(**overrides) -> SpectralRunConfig:
    payload = {"run_id": "run:test:1", "backend": "mock"}
    payload.update(overrides)
    return SpectralRunConfig(**payload)


def test_aperiodic_eligibility_requires_known_support_level():
    eligibility = AperiodicEligibility(dataset_id="dataset:dandi:1", support_level="high", confidence=0.9)
    assert eligibility.support_level == "high"
    with pytest.raises(ValidationError):
        AperiodicEligibility(dataset_id="dataset:dandi:1", support_level="maybe", confidence=0.9)


def test_run_config_rejects_invalid_freq_range():
    with pytest.raises(ValidationError):
        _run_config(freq_range_hz=(10.0, 5.0))
    with pytest.raises(ValidationError):
        _run_config(freq_range_hz=(0.0, 5.0))


def test_periodic_peak_requires_positive_values():
    peak = PeriodicPeak(center_frequency_hz=10.0, power=0.5, bandwidth_hz=2.0)
    assert peak.center_frequency_hz == 10.0
    with pytest.raises(ValidationError):
        PeriodicPeak(center_frequency_hz=-1.0, power=0.5, bandwidth_hz=2.0)


def test_qc_assessment_rejects_unknown_flags():
    SpectralQCAssessment(qc_id="qc:1", status="warn", flags=["low_fit_r_squared"])
    with pytest.raises(ValidationError):
        SpectralQCAssessment(qc_id="qc:1", status="warn", flags=["not_a_real_flag"])


def test_spectral_estimate_knee_requires_knee_mode():
    fixed_config = _run_config(aperiodic_mode="fixed")
    with pytest.raises(ValidationError):
        SpectralEstimate(
            estimate_id="estimate:1",
            dataset_id="dataset:dandi:1",
            run_config=fixed_config,
            aperiodic_offset=-2.0,
            aperiodic_exponent=2.0,
            aperiodic_knee_hz=5.0,
            fit_r_squared=0.9,
            fit_error=0.1,
            n_frequency_bins=50,
        )

    knee_config = _run_config(aperiodic_mode="knee")
    estimate = SpectralEstimate(
        estimate_id="estimate:1",
        dataset_id="dataset:dandi:1",
        run_config=knee_config,
        aperiodic_offset=-2.0,
        aperiodic_exponent=2.0,
        aperiodic_knee_hz=5.0,
        fit_r_squared=0.9,
        fit_error=0.1,
        n_frequency_bins=50,
    )
    assert estimate.aperiodic_knee_hz == 5.0


def test_feature_bundle_defaults_to_interpretation_cautions():
    eligibility = AperiodicEligibility(dataset_id="dataset:dandi:1", support_level="high", confidence=0.9)
    bundle = SpectralFeatureBundle(
        bundle_id="bundle:1",
        dataset_id="dataset:dandi:1",
        eligibility=eligibility,
        run_config=_run_config(),
    )
    assert bundle.interpretation_cautions
    assert any("excitation" in caution.lower() for caution in bundle.interpretation_cautions)
