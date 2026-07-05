import numpy as np
import pytest

from neural_search.spectral.psd import restrict_freq_range, welch_psd
from neural_search.spectral.synthetic import (
    SyntheticPeak,
    SyntheticSpectrumSpec,
    synthetic_aperiodic_signal,
    synthetic_power_spectrum,
)

SAMPLE_RATE_HZ = 250.0
N_SAMPLES = 20_000
FREQ_RANGE_HZ = (2.0, 40.0)


def _loglog_slope(freqs: np.ndarray, power: np.ndarray) -> float:
    slope, _ = np.polyfit(np.log10(freqs), np.log10(power), 1)
    return -slope


def test_synthetic_power_spectrum_matches_known_exponent():
    freqs = np.linspace(2.0, 40.0, 200)
    spec = SyntheticSpectrumSpec(exponent=2.0, offset=-2.0)

    power = synthetic_power_spectrum(freqs, spec)

    assert _loglog_slope(freqs, power) == pytest.approx(2.0, abs=1e-6)


def test_synthetic_power_spectrum_adds_peak_bump():
    freqs = np.linspace(2.0, 40.0, 400)
    spec = SyntheticSpectrumSpec(exponent=2.0, offset=-2.0)
    peaked_spec = SyntheticSpectrumSpec(
        exponent=2.0, offset=-2.0, peaks=(SyntheticPeak(center_frequency_hz=10.0, power=0.01, bandwidth_hz=1.0),)
    )

    baseline = synthetic_power_spectrum(freqs, spec)
    peaked = synthetic_power_spectrum(freqs, peaked_spec)
    idx = np.argmin(np.abs(freqs - 10.0))

    assert peaked[idx] > baseline[idx]


def test_synthetic_power_spectrum_rejects_nonpositive_frequencies():
    spec = SyntheticSpectrumSpec(exponent=2.0, offset=-2.0)
    with pytest.raises(ValueError):
        synthetic_power_spectrum(np.array([0.0, 1.0, 2.0]), spec)


@pytest.mark.parametrize("true_exponent", [1.0, 1.5, 2.0, 2.5])
def test_welch_psd_of_synthetic_signal_recovers_known_slope(true_exponent: float):
    spec = SyntheticSpectrumSpec(exponent=true_exponent, offset=-2.0)
    signal = synthetic_aperiodic_signal(N_SAMPLES, SAMPLE_RATE_HZ, spec, seed=0)

    freqs, power = welch_psd(signal, SAMPLE_RATE_HZ, nperseg=500)
    freqs, power = restrict_freq_range(freqs, power, FREQ_RANGE_HZ)
    recovered = _loglog_slope(freqs, power)

    assert recovered == pytest.approx(true_exponent, abs=0.3)


def test_welch_psd_rejects_flat_signal():
    flat_signal = np.zeros(2000)
    with pytest.raises(ValueError):
        welch_psd(flat_signal, SAMPLE_RATE_HZ)


def test_welch_psd_rejects_nan_signal():
    signal = np.random.default_rng(0).standard_normal(2000)
    signal[10] = np.nan
    with pytest.raises(ValueError):
        welch_psd(signal, SAMPLE_RATE_HZ)


def test_restrict_freq_range_excludes_dc_and_out_of_range():
    freqs = np.array([0.0, 1.0, 5.0, 10.0, 50.0])
    power = np.array([100.0, 50.0, 10.0, 5.0, 1.0])

    restricted_freqs, restricted_power = restrict_freq_range(freqs, power, (2.0, 40.0))

    assert 0.0 not in restricted_freqs
    assert 50.0 not in restricted_freqs
    assert list(restricted_freqs) == [5.0, 10.0]
    assert list(restricted_power) == [10.0, 5.0]
