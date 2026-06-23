"""A compact IRASA (Irregular Resampling Auto-Spectral Analysis) backend.

IRASA separates periodic from aperiodic spectral components by resampling
the *raw signal* at several non-integer up/down-sampling factors, computing
Welch PSDs of each pair, and taking the median across factors — oscillatory
peaks shift with resampling while the aperiodic background does not, so the
median suppresses peaks and isolates the aperiodic component.

This is a simplified, self-contained implementation for research use; it is
not a byte-for-byte port of the published algorithm or the ``yasa`` package.
For publication-grade IRASA, prefer ``yasa.irasa`` and treat this backend as
a dependency-free approximation. See docs/aperiodic_reanalysis.md.
"""

from __future__ import annotations

import numpy as np

from neural_search.spectral.psd import welch_psd
from neural_search.spectral.schemas import SpectralRunConfig
from neural_search.spectral.specparam_backend import (
    BackendFitResult,
    _loglog_aperiodic_fit,
    _residual_peaks,
)

try:
    from scipy.signal import resample_poly as _resample_poly
except ImportError:  # pragma: no cover - exercised only without scipy installed
    _resample_poly = None


def _resample(signal: np.ndarray, factor: float) -> np.ndarray:
    if _resample_poly is not None:
        up = int(round(factor * 100))
        down = 100
        return _resample_poly(signal, up, down)
    n_new = max(int(round(len(signal) * factor)), 8)
    original_x = np.linspace(0.0, 1.0, num=len(signal))
    new_x = np.linspace(0.0, 1.0, num=n_new)
    return np.interp(new_x, original_x, signal)


def irasa_aperiodic_psd(
    signal: np.ndarray,
    sample_rate_hz: float,
    *,
    hset: tuple[float, ...] = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9),
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(freqs_hz, aperiodic_power, periodic_power)`` for ``signal``."""

    signal = np.asarray(signal, dtype=float)
    base_freqs, base_power = welch_psd(signal, sample_rate_hz, nperseg=nperseg)

    resampled_psds = []
    for factor in hset:
        up_signal = _resample(signal, factor)
        down_signal = _resample(signal, 1.0 / factor)

        up_freqs, up_power = welch_psd(up_signal, sample_rate_hz * factor, nperseg=nperseg)
        down_freqs, down_power = welch_psd(down_signal, sample_rate_hz / factor, nperseg=nperseg)

        up_interp = np.interp(base_freqs, up_freqs, up_power, left=np.nan, right=np.nan)
        down_interp = np.interp(base_freqs, down_freqs, down_power, left=np.nan, right=np.nan)
        geometric_mean = np.sqrt(np.clip(up_interp, 1e-300, None) * np.clip(down_interp, 1e-300, None))
        resampled_psds.append(geometric_mean)

    stacked = np.vstack(resampled_psds)
    aperiodic_power = np.nanmedian(stacked, axis=0)
    valid = np.isfinite(aperiodic_power) & (base_freqs > 0)
    freqs_hz = base_freqs[valid]
    aperiodic_power = aperiodic_power[valid]
    periodic_power = np.clip(base_power[valid] - aperiodic_power, 0.0, None)
    return freqs_hz, aperiodic_power, periodic_power


class IrasaBackend:
    """Fits aperiodic offset/exponent from the IRASA-derived aperiodic
    spectrum, computed directly from the raw signal (not a precomputed PSD)."""

    name = "irasa"

    def fit_from_signal(
        self,
        signal: np.ndarray,
        sample_rate_hz: float,
        run_config: SpectralRunConfig,
    ) -> BackendFitResult:
        freqs_hz, aperiodic_power, periodic_power = irasa_aperiodic_psd(
            signal,
            sample_rate_hz,
            nperseg=run_config.welch_nperseg,
        )
        low, high = run_config.freq_range_hz
        mask = (freqs_hz >= low) & (freqs_hz <= high)
        freqs_hz, aperiodic_power, periodic_power = freqs_hz[mask], aperiodic_power[mask], periodic_power[mask]
        if len(freqs_hz) < 4:
            raise ValueError("at least 4 frequency bins are required to fit a spectrum")

        knee = run_config.aperiodic_mode == "knee"
        offset, exponent, knee_hz, fitted_log_power = _loglog_aperiodic_fit(
            freqs_hz, aperiodic_power, knee=knee
        )
        log_aperiodic_power = np.log10(np.clip(aperiodic_power, 1e-300, None))
        ss_res = float(np.sum((log_aperiodic_power - fitted_log_power) ** 2))
        ss_tot = float(np.sum((log_aperiodic_power - np.mean(log_aperiodic_power)) ** 2)) or 1e-12
        r_squared = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
        fit_error = float(np.sqrt(ss_res / len(log_aperiodic_power)))

        log_periodic = np.log10(np.clip(periodic_power, 1e-300, None) + 1.0)
        peaks = _residual_peaks(
            freqs_hz,
            log_periodic,
            np.zeros_like(log_periodic),
            max_n_peaks=run_config.max_n_peaks,
            min_peak_height=run_config.min_peak_height,
            peak_threshold=run_config.peak_threshold,
            peak_width_limits_hz=run_config.peak_width_limits_hz,
        )

        return BackendFitResult(
            aperiodic_offset=offset,
            aperiodic_exponent=max(0.0, exponent),
            aperiodic_knee_hz=knee_hz,
            fit_r_squared=r_squared,
            fit_error=fit_error,
            n_frequency_bins=len(freqs_hz),
            peaks=peaks,
            backend_version="irasa-simplified-v0.1.0",
        )
