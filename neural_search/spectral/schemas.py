"""Pydantic schemas for the aperiodic spectral phenotype reanalysis layer.

All models are provenance-first: every computed quantity records the method,
settings, frequency range, backend, and QC status that produced it. None of
these schemas assert that an aperiodic exponent directly measures a single
biological mechanism (e.g. excitation/inhibition balance) — see
``interpretation_cautions`` on :class:`SpectralFeatureBundle` and
``docs/aperiodic_reanalysis.md``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SupportLevel = Literal["high", "medium", "low", "unsupported", "unknown"]
QCStatus = Literal["pass", "warn", "fail"]
AperiodicMode = Literal["fixed", "knee"]
SpectralBackendName = Literal["mock", "specparam", "fooof", "irasa"]

DEFAULT_DETECTOR_NAME = "rule_based_aperiodic_eligibility_detector"
DEFAULT_DETECTOR_VERSION = "v0.1.0"

QC_FLAGS = (
    "low_fit_r_squared",
    "high_fit_error",
    "too_few_frequency_bins",
    "line_noise_overlap",
    "many_peaks_possible_overfit",
    "frequency_range_too_narrow",
    "missing_sampling_rate",
    "missing_channel_metadata",
    "missing_region_metadata",
    "missing_task_state",
    "flat_or_zero_signal",
    "nan_or_inf_values",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class AperiodicEligibility(BaseModel):
    """Conservative estimate of whether a dataset can support aperiodic
    spectral parameterization, derived purely from existing metadata."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    support_level: SupportLevel
    confidence: float = Field(ge=0.0, le=1.0)
    compatible_modality: bool = False
    likely_continuous_signal: bool = False
    sampling_rate_likely_available: bool = False
    channel_or_probe_metadata_present: bool = False
    required_fields_present: list[str] = Field(default_factory=list)
    helpful_fields_present: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    detector_name: str = DEFAULT_DETECTOR_NAME
    detector_version: str = DEFAULT_DETECTOR_VERSION

    @field_validator("dataset_id", "detector_name", "detector_version")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned


class SpectralRunConfig(BaseModel):
    """Full provenance of a single spectral-parameterization run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    backend: SpectralBackendName
    backend_version: str = "unknown"
    freq_range_hz: tuple[float, float] = (1.0, 40.0)
    aperiodic_mode: AperiodicMode = "fixed"
    peak_width_limits_hz: tuple[float, float] = (0.5, 12.0)
    max_n_peaks: int = Field(default=6, ge=0)
    min_peak_height: float = Field(default=0.0, ge=0.0)
    peak_threshold: float = Field(default=2.0, ge=0.0)
    sample_rate_hz: float | None = Field(default=None, gt=0.0)
    welch_nperseg: int | None = Field(default=None, gt=0)
    welch_noverlap: int | None = Field(default=None, ge=0)
    random_seed: int | None = None
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utc_now)

    @field_validator("run_id", "backend_version")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @field_validator("freq_range_hz", "peak_width_limits_hz")
    @classmethod
    def _ordered_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if low <= 0 or high <= low:
            raise ValueError(f"invalid frequency range: {value}")
        return (float(low), float(high))


class PeriodicPeak(BaseModel):
    """A single oscillatory peak fit on top of the aperiodic component."""

    model_config = ConfigDict(extra="forbid")

    center_frequency_hz: float = Field(gt=0.0)
    power: float = Field(ge=0.0)
    bandwidth_hz: float = Field(gt=0.0)
    band_label: str | None = None


class SpectralQCAssessment(BaseModel):
    """Quality-control outcome for a single spectral estimate."""

    model_config = ConfigDict(extra="forbid")

    qc_id: str
    status: QCStatus
    flags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    qc_version: str = "v0.1.0"
    created_at: str = Field(default_factory=_utc_now)

    @field_validator("flags")
    @classmethod
    def _known_flags(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(QC_FLAGS))
        if unknown:
            raise ValueError(f"unknown QC flags: {unknown}")
        return value


class SpectralEstimate(BaseModel):
    """One aperiodic + periodic spectral parameterization result."""

    model_config = ConfigDict(extra="forbid")

    estimate_id: str
    dataset_id: str
    run_config: SpectralRunConfig
    channel_id: str | None = None
    region_id: str | None = None
    task_state_id: str | None = None
    aperiodic_offset: float
    aperiodic_exponent: float = Field(ge=0.0)
    aperiodic_knee_hz: float | None = Field(default=None, ge=0.0)
    fit_r_squared: float = Field(ge=0.0, le=1.0)
    fit_error: float = Field(ge=0.0)
    n_frequency_bins: int = Field(ge=0)
    peaks: list[PeriodicPeak] = Field(default_factory=list)
    qc: SpectralQCAssessment | None = None
    created_at: str = Field(default_factory=_utc_now)

    @field_validator("estimate_id", "dataset_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @model_validator(mode="after")
    def _knee_requires_knee_mode(self) -> SpectralEstimate:
        if self.aperiodic_knee_hz is not None and self.run_config.aperiodic_mode != "knee":
            raise ValueError("aperiodic_knee_hz requires run_config.aperiodic_mode == 'knee'")
        return self


class SpectralFeatureBundle(BaseModel):
    """All spectral estimates produced for one dataset by one run, plus the
    eligibility context and mandatory interpretation cautions."""

    model_config = ConfigDict(extra="forbid")

    bundle_id: str
    dataset_id: str
    eligibility: AperiodicEligibility
    run_config: SpectralRunConfig
    estimates: list[SpectralEstimate] = Field(default_factory=list)
    interpretation_cautions: list[str] = Field(
        default_factory=lambda: list(DEFAULT_INTERPRETATION_CAUTIONS)
    )
    overall_qc_status: QCStatus = "warn"
    created_at: str = Field(default_factory=_utc_now)

    @field_validator("bundle_id", "dataset_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned


DEFAULT_INTERPRETATION_CAUTIONS: tuple[str, ...] = (
    "The aperiodic exponent / spectral slope is a descriptive summary of the "
    "power spectrum, not a direct or validated measurement of "
    "excitation/inhibition (E/I) balance or any single circuit mechanism.",
    "Aperiodic features are sensitive to recording modality, reference "
    "scheme, electrode/probe placement, behavioral state, and preprocessing "
    "choices; cross-dataset or cross-species comparisons require matched "
    "methods and explicit caveats.",
    "Confidence in any interpretive claim should be downweighted when QC "
    "status is 'warn' or 'fail', when frequency range or channel/region "
    "metadata is missing, or when peak fits are unstable across runs.",
)
