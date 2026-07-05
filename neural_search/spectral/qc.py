"""Quality control for spectral-parameterization fits.

Produces a conservative ``pass`` / ``warn`` / ``fail`` status plus the
specific flags that were triggered, so downstream consumers (KG export,
search ranking, dataset cards) can discount or exclude low-quality fits
instead of silently trusting every numeric output.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neural_search.spectral.schemas import PeriodicPeak, SpectralQCAssessment

# Flags that, when triggered, are severe enough to fail QC outright.
_FAIL_FLAGS = {
    "too_few_frequency_bins",
    "frequency_range_too_narrow",
    "flat_or_zero_signal",
    "nan_or_inf_values",
}

LINE_NOISE_BANDS_HZ: tuple[tuple[float, float], ...] = ((48.0, 52.0), (58.0, 62.0))

MIN_FREQUENCY_BINS_WARN = 10
MIN_FREQUENCY_BINS_FAIL = 5
MIN_FREQ_RANGE_WARN_HZ = 5.0
MIN_FREQ_RANGE_FAIL_HZ = 2.0
MIN_R_SQUARED_WARN = 0.9
MIN_R_SQUARED_FAIL = 0.7
MAX_FIT_ERROR_WARN = 0.3
MAX_FIT_ERROR_FAIL = 0.6
MAX_PEAKS_BEFORE_OVERFIT_WARNING = 6


def _range_overlaps_line_noise(freq_range_hz: tuple[float, float]) -> bool:
    low, high = freq_range_hz
    return any(low <= band_high and high >= band_low for band_low, band_high in LINE_NOISE_BANDS_HZ)


def assess_spectral_qc(
    *,
    qc_id: str,
    fit_r_squared: float,
    fit_error: float,
    n_frequency_bins: int,
    freq_range_hz: tuple[float, float],
    peaks: Sequence[PeriodicPeak] = (),
    sample_rate_hz: float | None = None,
    channel_metadata_present: bool | None = None,
    region_metadata_present: bool | None = None,
    task_state_present: bool | None = None,
    signal: np.ndarray | None = None,
) -> SpectralQCAssessment:
    """Run all QC checks and return a single pass/warn/fail assessment."""

    flags: list[str] = []
    notes: list[str] = []
    severe = False

    if fit_r_squared < MIN_R_SQUARED_FAIL:
        flags.append("low_fit_r_squared")
        severe = True
        notes.append(f"fit R^2={fit_r_squared:.3f} is below the fail threshold {MIN_R_SQUARED_FAIL}.")
    elif fit_r_squared < MIN_R_SQUARED_WARN:
        flags.append("low_fit_r_squared")
        notes.append(f"fit R^2={fit_r_squared:.3f} is below the warn threshold {MIN_R_SQUARED_WARN}.")

    if fit_error > MAX_FIT_ERROR_FAIL:
        flags.append("high_fit_error")
        severe = True
        notes.append(f"fit error={fit_error:.3f} exceeds the fail threshold {MAX_FIT_ERROR_FAIL}.")
    elif fit_error > MAX_FIT_ERROR_WARN:
        flags.append("high_fit_error")
        notes.append(f"fit error={fit_error:.3f} exceeds the warn threshold {MAX_FIT_ERROR_WARN}.")

    if n_frequency_bins < MIN_FREQUENCY_BINS_FAIL:
        flags.append("too_few_frequency_bins")
        notes.append(f"only {n_frequency_bins} frequency bins (< {MIN_FREQUENCY_BINS_FAIL}).")
    elif n_frequency_bins < MIN_FREQUENCY_BINS_WARN:
        flags.append("too_few_frequency_bins")
        notes.append(f"only {n_frequency_bins} frequency bins (< {MIN_FREQUENCY_BINS_WARN}).")

    freq_span = freq_range_hz[1] - freq_range_hz[0]
    if freq_span < MIN_FREQ_RANGE_FAIL_HZ:
        flags.append("frequency_range_too_narrow")
        notes.append(f"frequency range span {freq_span:.2f} Hz is below the fail threshold.")
    elif freq_span < MIN_FREQ_RANGE_WARN_HZ:
        flags.append("frequency_range_too_narrow")
        notes.append(f"frequency range span {freq_span:.2f} Hz is below the warn threshold.")

    if _range_overlaps_line_noise(freq_range_hz):
        flags.append("line_noise_overlap")
        notes.append("frequency range overlaps a 50/60 Hz mains line-noise band.")

    if len(peaks) > MAX_PEAKS_BEFORE_OVERFIT_WARNING:
        flags.append("many_peaks_possible_overfit")
        notes.append(f"{len(peaks)} peaks fitted (> {MAX_PEAKS_BEFORE_OVERFIT_WARNING}); possible overfitting.")

    if sample_rate_hz is None:
        flags.append("missing_sampling_rate")
        notes.append("sample rate is unknown.")

    if channel_metadata_present is False:
        flags.append("missing_channel_metadata")
        notes.append("channel/probe metadata is not available.")

    if region_metadata_present is False:
        flags.append("missing_region_metadata")
        notes.append("brain region metadata is not available.")

    if task_state_present is False:
        flags.append("missing_task_state")
        notes.append("task/behavioral state at time of recording is unknown.")

    if signal is not None:
        signal = np.asarray(signal, dtype=float)
        if not np.all(np.isfinite(signal)):
            flags.append("nan_or_inf_values")
            notes.append("signal contains NaN or infinite values.")
        elif np.allclose(signal, signal[0] if len(signal) else 0.0):
            flags.append("flat_or_zero_signal")
            notes.append("signal is flat or all-zero.")

    if severe or any(flag in _FAIL_FLAGS for flag in flags):
        status = "fail"
    elif flags:
        status = "warn"
    else:
        status = "pass"

    return SpectralQCAssessment(qc_id=qc_id, status=status, flags=sorted(set(flags)), notes=notes)
