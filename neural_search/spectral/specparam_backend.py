"""Spectral-parameterization backends that fit aperiodic + periodic
components to a power spectrum.

``MockSpectralParamBackend`` is dependency-free (numpy-only log-log linear
regression plus simple residual peak picking) and is used for tests and as
the default fallback. ``SpecparamBackend`` wraps the optional ``specparam``
package (the renamed successor to ``fooof``); ``FooofBackend`` is an alias
that also tries the legacy ``fooof`` package name. Both optional backends
degrade gracefully — importing this module never requires either package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from neural_search.spectral.schemas import PeriodicPeak, SpectralRunConfig


class BackendUnavailableError(RuntimeError):
    """Raised when an optional spectral-parameterization package is missing."""


@dataclass
class BackendFitResult:
    """Backend-agnostic fit output, later converted into a ``SpectralEstimate``."""

    aperiodic_offset: float
    aperiodic_exponent: float
    fit_r_squared: float
    fit_error: float
    n_frequency_bins: int
    aperiodic_knee_hz: float | None = None
    peaks: list[PeriodicPeak] = field(default_factory=list)
    backend_version: str = "unknown"


class SpectralParamBackend(Protocol):
    """Common interface for backends that fit a precomputed PSD."""

    name: str

    def fit(
        self,
        freqs_hz: np.ndarray,
        power: np.ndarray,
        run_config: SpectralRunConfig,
    ) -> BackendFitResult: ...


def _loglog_aperiodic_fit(
    freqs_hz: np.ndarray,
    power: np.ndarray,
    *,
    knee: bool,
) -> tuple[float, float, float | None, np.ndarray]:
    """Fit ``log10(power) ~ offset - exponent * log10(freq)`` (knee mode adds
    a knee term via a one-dimensional grid search). Returns
    ``(offset, exponent, knee_hz, fitted_log_power)``."""

    log_freqs = np.log10(freqs_hz)
    log_power = np.log10(np.clip(power, 1e-300, None))

    if not knee:
        exponent_neg, offset = np.polyfit(log_freqs, log_power, 1)
        fitted = offset + exponent_neg * log_freqs
        return float(offset), float(-exponent_neg), None, fitted

    best: tuple[float, float, float, float, np.ndarray] | None = None
    knee_candidates = np.logspace(np.log10(max(freqs_hz.min(), 1e-3)), np.log10(freqs_hz.max()), 12)
    for knee_value in knee_candidates:
        denom = np.log10(knee_value + np.power(freqs_hz, 1.0))
        design = np.vstack([np.ones_like(denom), -denom]).T
        coeffs, *_ = np.linalg.lstsq(design, log_power, rcond=None)
        offset, exponent = float(coeffs[0]), float(coeffs[1])
        fitted = offset - exponent * denom
        residual = float(np.sum((log_power - fitted) ** 2))
        if best is None or residual < best[0]:
            best = (residual, offset, exponent, float(knee_value), fitted)
    assert best is not None
    _, offset, exponent, knee_hz, fitted = best
    return offset, exponent, knee_hz, fitted


def _residual_peaks(
    freqs_hz: np.ndarray,
    log_power: np.ndarray,
    fitted_log_power: np.ndarray,
    *,
    max_n_peaks: int,
    min_peak_height: float,
    peak_threshold: float,
    peak_width_limits_hz: tuple[float, float],
) -> list[PeriodicPeak]:
    residual = log_power - fitted_log_power
    std = float(np.std(residual)) or 1e-12
    threshold = max(min_peak_height, peak_threshold * std)

    peaks: list[PeriodicPeak] = []
    for i in range(1, len(residual) - 1):
        if residual[i] <= threshold:
            continue
        if not (residual[i] > residual[i - 1] and residual[i] >= residual[i + 1]):
            continue
        half_max = residual[i] / 2.0
        left = i
        while left > 0 and residual[left] > half_max:
            left -= 1
        right = i
        while right < len(residual) - 1 and residual[right] > half_max:
            right += 1
        bandwidth = max(float(freqs_hz[right] - freqs_hz[left]), 1e-3)
        low, high = peak_width_limits_hz
        bandwidth = float(np.clip(bandwidth, low, high))
        peaks.append(
            PeriodicPeak(
                center_frequency_hz=float(freqs_hz[i]),
                power=float(10**residual[i] - 1) if residual[i] > 0 else float(residual[i]),
                bandwidth_hz=bandwidth,
            )
        )

    peaks.sort(key=lambda peak: peak.power, reverse=True)
    return peaks[:max_n_peaks]


class MockSpectralParamBackend:
    """Dependency-free aperiodic + periodic fit via log-log linear regression
    and simple residual peak detection. Used for tests and as the default
    backend when no optional package is installed."""

    name = "mock"

    def fit(
        self,
        freqs_hz: np.ndarray,
        power: np.ndarray,
        run_config: SpectralRunConfig,
    ) -> BackendFitResult:
        freqs_hz = np.asarray(freqs_hz, dtype=float)
        power = np.asarray(power, dtype=float)
        if len(freqs_hz) < 4:
            raise ValueError("at least 4 frequency bins are required to fit a spectrum")

        knee = run_config.aperiodic_mode == "knee"
        log_power = np.log10(np.clip(power, 1e-300, None))

        # Pass 1: fit on the full spectrum.
        offset, exponent, knee_hz, fitted_log_power = _loglog_aperiodic_fit(freqs_hz, power, knee=knee)

        # Pass 2: re-fit excluding bins whose pass-1 residual looks
        # peak-contaminated, so a handful of oscillatory peaks don't bias the
        # aperiodic slope (real FOOOF/specparam fit aperiodic and periodic
        # components jointly via iterative removal; this is a lightweight
        # one-shot approximation of the same idea).
        residual = log_power - fitted_log_power
        std = float(np.std(residual)) or 1e-12
        inliers = np.abs(residual) <= 2.0 * std
        if not inliers.any():
            # A near-perfect pass-1 fit can collapse `std` to a value so tiny
            # that ordinary floating-point noise excludes every point; treat
            # that degenerate case as "no peak contamination" rather than
            # dividing by zero below.
            inliers = np.ones_like(inliers)
        if inliers.sum() >= 4 and not inliers.all():
            offset, exponent, knee_hz, _ = _loglog_aperiodic_fit(
                freqs_hz[inliers], power[inliers], knee=knee
            )
            fitted_log_power = (
                offset - exponent * np.log10(knee_hz + np.power(freqs_hz, 1.0))
                if knee
                else offset - exponent * np.log10(freqs_hz)
            )
            residual = log_power - fitted_log_power

        ss_res = float(np.sum((residual[inliers]) ** 2))
        ss_tot = float(np.sum((log_power[inliers] - np.mean(log_power[inliers])) ** 2)) or 1e-12
        r_squared = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
        fit_error = float(np.sqrt(ss_res / int(inliers.sum())))

        peaks = _residual_peaks(
            freqs_hz,
            log_power,
            fitted_log_power,
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
            backend_version="mock-v0.1.0",
        )


class SpecparamBackend:
    """Wraps the optional ``specparam`` package (or legacy ``fooof`` name).
    Raises ``BackendUnavailableError`` if neither is installed."""

    name = "specparam"

    def __init__(self) -> None:
        self._model_cls, self._version = self._import_model()

    @staticmethod
    def _import_model() -> tuple[type, str]:
        try:
            import specparam as _specparam_module
            from specparam import SpectralModel  # type: ignore[import-not-found]

            return SpectralModel, getattr(_specparam_module, "__version__", "unknown")
        except ImportError:
            pass
        try:
            import fooof as _fooof_module
            from fooof import FOOOF  # type: ignore[import-not-found]

            return FOOOF, getattr(_fooof_module, "__version__", "unknown")
        except ImportError as exc:
            raise BackendUnavailableError(
                "Neither 'specparam' nor 'fooof' is installed. Install one "
                "(`pip install specparam`) or use MockSpectralParamBackend."
            ) from exc

    def fit(
        self,
        freqs_hz: np.ndarray,
        power: np.ndarray,
        run_config: SpectralRunConfig,
    ) -> BackendFitResult:
        model = self._model_cls(
            peak_width_limits=run_config.peak_width_limits_hz,
            max_n_peaks=run_config.max_n_peaks,
            min_peak_height=run_config.min_peak_height,
            peak_threshold=run_config.peak_threshold,
            aperiodic_mode=run_config.aperiodic_mode,
            verbose=False,
        )
        model.fit(np.asarray(freqs_hz, dtype=float), np.asarray(power, dtype=float), run_config.freq_range_hz)

        aperiodic_params = model.aperiodic_params_
        offset = float(aperiodic_params[0])
        if run_config.aperiodic_mode == "knee":
            knee_hz = float(aperiodic_params[1])
            exponent = float(aperiodic_params[2])
        else:
            knee_hz = None
            exponent = float(aperiodic_params[1])

        peaks = [
            PeriodicPeak(
                center_frequency_hz=float(row[0]),
                power=float(row[1]),
                bandwidth_hz=float(row[2]),
            )
            for row in model.peak_params_
        ]

        return BackendFitResult(
            aperiodic_offset=offset,
            aperiodic_exponent=exponent,
            aperiodic_knee_hz=knee_hz,
            fit_r_squared=float(model.r_squared_),
            fit_error=float(model.error_),
            n_frequency_bins=len(np.asarray(freqs_hz)),
            peaks=peaks,
            backend_version=self._version,
        )


FooofBackend = SpecparamBackend


def get_backend(name: str) -> SpectralParamBackend:
    """Resolve a PSD-based backend by name, falling back to the mock backend
    if an optional package is unavailable."""

    if name == "mock":
        return MockSpectralParamBackend()
    if name in ("specparam", "fooof"):
        try:
            return SpecparamBackend()
        except BackendUnavailableError:
            return MockSpectralParamBackend()
    raise ValueError(f"unknown spectral-parameterization backend: {name}")
