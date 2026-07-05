"""Power spectral density estimation (Welch's method).

Prefers ``scipy.signal.welch`` when available and falls back to a small
numpy-only Welch implementation otherwise, so this module never hard-fails
just because scipy (or MNE, which is never required here) is missing.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import welch as _scipy_welch
except ImportError:  # pragma: no cover - exercised only without scipy installed
    _scipy_welch = None


def _numpy_welch(
    signal: np.ndarray,
    sample_rate_hz: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("noverlap must be smaller than nperseg")
    window = np.hanning(nperseg)
    window_norm = np.sum(window**2)

    segments = []
    start = 0
    while start + nperseg <= len(signal):
        segment = signal[start : start + nperseg] * window
        spectrum = np.fft.rfft(segment)
        segments.append((np.abs(spectrum) ** 2) / (sample_rate_hz * window_norm))
        start += step

    if not segments:
        raise ValueError("signal is shorter than nperseg; cannot estimate PSD")

    power = np.mean(np.stack(segments, axis=0), axis=0)
    power[1:-1] *= 2.0  # one-sided spectrum correction (excludes DC/Nyquist)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sample_rate_hz)
    return freqs, power


def welch_psd(
    signal: np.ndarray,
    sample_rate_hz: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a one-sided power spectral density via Welch's method.

    Returns ``(freqs_hz, power)``. ``power`` units are arbitrary (signal
    units squared per Hz) — spectral-parameterization backends only need
    relative shape, not absolute calibration.
    """

    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains NaN or infinite values")
    if np.allclose(signal, signal[0]):
        raise ValueError("signal is flat (zero variance); cannot estimate PSD")

    nperseg = nperseg or min(len(signal), max(int(sample_rate_hz * 2), 64))
    nperseg = min(nperseg, len(signal))
    noverlap = nperseg // 2 if noverlap is None else noverlap

    if _scipy_welch is not None:
        freqs, power = _scipy_welch(
            signal,
            fs=sample_rate_hz,
            nperseg=nperseg,
            noverlap=noverlap,
        )
    else:
        freqs, power = _numpy_welch(signal, sample_rate_hz, nperseg, noverlap)
    return freqs, power


def restrict_freq_range(
    freqs_hz: np.ndarray,
    power: np.ndarray,
    freq_range_hz: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict a PSD to an inclusive frequency range, excluding 0 Hz (DC)."""

    low, high = freq_range_hz
    mask = (freqs_hz >= low) & (freqs_hz <= high) & (freqs_hz > 0)
    return np.asarray(freqs_hz)[mask], np.asarray(power)[mask]
