import numpy as np
import pytest

from neural_search.spectral.schemas import SpectralRunConfig
from neural_search.spectral.specparam_backend import (
    BackendUnavailableError,
    MockSpectralParamBackend,
    SpecparamBackend,
    get_backend,
)
from neural_search.spectral.synthetic import (
    SyntheticPeak,
    SyntheticSpectrumSpec,
    synthetic_power_spectrum,
)


def _run_config(**overrides) -> SpectralRunConfig:
    payload = {"run_id": "run:test:1", "backend": "mock", "freq_range_hz": (2.0, 40.0)}
    payload.update(overrides)
    return SpectralRunConfig(**payload)


def test_mock_backend_recovers_exact_exponent_on_clean_spectrum():
    freqs = np.linspace(2.0, 40.0, 200)
    spec = SyntheticSpectrumSpec(exponent=2.0, offset=-2.0)
    power = synthetic_power_spectrum(freqs, spec)

    fit = MockSpectralParamBackend().fit(freqs, power, _run_config())

    assert fit.aperiodic_exponent == pytest.approx(2.0, abs=1e-3)
    assert fit.fit_r_squared > 0.99
    assert fit.n_frequency_bins == len(freqs)


def test_mock_backend_detects_injected_peak():
    freqs = np.linspace(2.0, 40.0, 400)
    spec = SyntheticSpectrumSpec(
        exponent=2.0,
        offset=-2.0,
        peaks=(SyntheticPeak(center_frequency_hz=10.0, power=0.002, bandwidth_hz=1.5),),
    )
    power = synthetic_power_spectrum(freqs, spec)

    fit = MockSpectralParamBackend().fit(freqs, power, _run_config())

    assert fit.aperiodic_exponent == pytest.approx(2.0, abs=0.3)
    assert len(fit.peaks) >= 1
    closest_peak = min(fit.peaks, key=lambda peak: abs(peak.center_frequency_hz - 10.0))
    assert abs(closest_peak.center_frequency_hz - 10.0) < 2.0


def test_mock_backend_requires_minimum_frequency_bins():
    freqs = np.array([2.0, 5.0, 10.0])
    power = np.array([1.0, 0.5, 0.1])
    with pytest.raises(ValueError):
        MockSpectralParamBackend().fit(freqs, power, _run_config())


def test_mock_backend_knee_mode_returns_a_knee():
    freqs = np.linspace(0.5, 40.0, 300)
    spec = SyntheticSpectrumSpec(exponent=2.0, offset=-1.0, knee=5.0)
    power = synthetic_power_spectrum(freqs, spec)

    fit = MockSpectralParamBackend().fit(freqs, power, _run_config(aperiodic_mode="knee"))

    assert fit.aperiodic_knee_hz is not None
    assert fit.aperiodic_knee_hz > 0


def test_get_backend_falls_back_to_mock_when_specparam_unavailable():
    backend = get_backend("specparam")
    assert backend.name in ("mock", "specparam")


def test_get_backend_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_backend("not_a_real_backend")


def test_specparam_backend_raises_or_succeeds_depending_on_install():
    try:
        backend = SpecparamBackend()
    except BackendUnavailableError:
        pytest.skip("specparam/fooof not installed in this environment")
    else:
        assert backend.name == "specparam"
