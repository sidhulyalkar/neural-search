"""Synthetic 1/f-like spectra and signals with known ground-truth parameters.

Used to validate that the PSD pipeline and spectral-parameterization backends
recover known aperiodic exponents (and injected oscillatory peaks) within a
documented tolerance — see ``scripts/eval/evaluate_aperiodic_synthetic.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SyntheticPeak:
    """Ground-truth oscillatory peak to inject into a synthetic spectrum."""

    center_frequency_hz: float
    power: float
    bandwidth_hz: float = 2.0


@dataclass(frozen=True)
class SyntheticSpectrumSpec:
    """Ground-truth parameters used to build a synthetic spectrum/signal."""

    exponent: float
    offset: float = 0.0
    knee: float | None = None
    peaks: tuple[SyntheticPeak, ...] = field(default_factory=tuple)


def synthetic_aperiodic_power(freqs_hz: np.ndarray, spec: SyntheticSpectrumSpec) -> np.ndarray:
    """Return linear-power aperiodic-only spectrum for the given frequencies."""

    freqs_hz = np.asarray(freqs_hz, dtype=float)
    if np.any(freqs_hz <= 0):
        raise ValueError("frequencies must be strictly positive")
    denominator = (
        spec.knee + np.power(freqs_hz, spec.exponent)
        if spec.knee is not None
        else np.power(freqs_hz, spec.exponent)
    )
    return (10.0**spec.offset) / denominator


def _add_gaussian_peak(
    freqs_hz: np.ndarray,
    power: np.ndarray,
    peak: SyntheticPeak,
) -> np.ndarray:
    bump = peak.power * np.exp(
        -((freqs_hz - peak.center_frequency_hz) ** 2) / (2.0 * peak.bandwidth_hz**2)
    )
    return power + bump


def synthetic_power_spectrum(freqs_hz: np.ndarray, spec: SyntheticSpectrumSpec) -> np.ndarray:
    """Build a full synthetic power spectrum: aperiodic background + peaks."""

    power = synthetic_aperiodic_power(freqs_hz, spec)
    for peak in spec.peaks:
        power = _add_gaussian_peak(np.asarray(freqs_hz, dtype=float), power, peak)
    return power


def synthetic_aperiodic_signal(
    n_samples: int,
    sample_rate_hz: float,
    spec: SyntheticSpectrumSpec,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a real-valued time series whose power spectrum follows
    ``spec`` (1/f-like aperiodic background plus optional oscillatory peaks),
    via amplitude-shaped white noise in the frequency domain."""

    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples)
    spectrum = np.fft.rfft(white)
    freqs_hz = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)

    safe_freqs = freqs_hz.copy()
    safe_freqs[0] = freqs_hz[1] if len(freqs_hz) > 1 else 1.0
    target_power = synthetic_power_spectrum(safe_freqs, spec)
    amplitude_scale = np.sqrt(target_power)
    amplitude_scale[0] = 0.0  # remove DC offset

    shaped_spectrum = spectrum * amplitude_scale
    signal = np.fft.irfft(shaped_spectrum, n=n_samples)
    return signal / (np.std(signal) + 1e-12)
