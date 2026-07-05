"""Orchestrates PSD estimation, backend fitting, and QC into the final
``SpectralEstimate`` / ``SpectralFeatureBundle`` schemas."""

from __future__ import annotations

import numpy as np

from neural_search.spectral.irasa_backend import IrasaBackend
from neural_search.spectral.psd import restrict_freq_range, welch_psd
from neural_search.spectral.qc import assess_spectral_qc
from neural_search.spectral.schemas import (
    DEFAULT_INTERPRETATION_CAUTIONS,
    AperiodicEligibility,
    SpectralEstimate,
    SpectralFeatureBundle,
    SpectralRunConfig,
)
from neural_search.spectral.specparam_backend import get_backend

_QC_SEVERITY = {"pass": 0, "warn": 1, "fail": 2}


def compute_spectral_estimate(
    signal: np.ndarray,
    sample_rate_hz: float,
    *,
    dataset_id: str,
    estimate_id: str,
    run_config: SpectralRunConfig,
    channel_id: str | None = None,
    region_id: str | None = None,
    task_state_id: str | None = None,
    channel_metadata_present: bool | None = None,
    region_metadata_present: bool | None = None,
    task_state_present: bool | None = None,
) -> SpectralEstimate:
    """Run Welch PSD + a spectral-parameterization backend + QC for one
    channel/region/task-state and return a fully provenanced estimate."""

    if run_config.backend == "irasa":
        fit = IrasaBackend().fit_from_signal(signal, sample_rate_hz, run_config)
    else:
        freqs_hz, power = welch_psd(
            signal,
            sample_rate_hz,
            nperseg=run_config.welch_nperseg,
            noverlap=run_config.welch_noverlap,
        )
        freqs_hz, power = restrict_freq_range(freqs_hz, power, run_config.freq_range_hz)
        backend = get_backend(run_config.backend)
        fit = backend.fit(freqs_hz, power, run_config)

    qc = assess_spectral_qc(
        qc_id=f"qc:{estimate_id}",
        fit_r_squared=fit.fit_r_squared,
        fit_error=fit.fit_error,
        n_frequency_bins=fit.n_frequency_bins,
        freq_range_hz=run_config.freq_range_hz,
        peaks=fit.peaks,
        sample_rate_hz=sample_rate_hz,
        channel_metadata_present=channel_metadata_present,
        region_metadata_present=region_metadata_present,
        task_state_present=task_state_present,
        signal=signal,
    )

    return SpectralEstimate(
        estimate_id=estimate_id,
        dataset_id=dataset_id,
        run_config=run_config,
        channel_id=channel_id,
        region_id=region_id,
        task_state_id=task_state_id,
        aperiodic_offset=fit.aperiodic_offset,
        aperiodic_exponent=fit.aperiodic_exponent,
        aperiodic_knee_hz=fit.aperiodic_knee_hz,
        fit_r_squared=fit.fit_r_squared,
        fit_error=fit.fit_error,
        n_frequency_bins=fit.n_frequency_bins,
        peaks=fit.peaks,
        qc=qc,
    )


def build_feature_bundle(
    estimates: list[SpectralEstimate],
    *,
    bundle_id: str,
    dataset_id: str,
    eligibility: AperiodicEligibility,
    run_config: SpectralRunConfig,
) -> SpectralFeatureBundle:
    """Combine per-channel/region/state estimates into a single bundle with
    an overall QC status (the worst status across all estimates)."""

    overall_status = "pass"
    for estimate in estimates:
        if estimate.qc is None:
            continue
        if _QC_SEVERITY[estimate.qc.status] > _QC_SEVERITY[overall_status]:
            overall_status = estimate.qc.status

    return SpectralFeatureBundle(
        bundle_id=bundle_id,
        dataset_id=dataset_id,
        eligibility=eligibility,
        run_config=run_config,
        estimates=estimates,
        overall_qc_status=overall_status,
    )


def qc_summary(bundle: SpectralFeatureBundle) -> dict[str, int]:
    """Count estimates by QC status, e.g. ``{"pass": 3, "warn": 1, "fail": 0}``."""

    summary = {"pass": 0, "warn": 0, "fail": 0}
    for estimate in bundle.estimates:
        if estimate.qc is not None:
            summary[estimate.qc.status] += 1
    return summary


def summarize_for_card(
    eligibility: AperiodicEligibility,
    bundle: SpectralFeatureBundle | None = None,
) -> dict:
    """Build the dict consumed by ``DatasetCardRead.spectral_phenotype`` —
    aperiodic readiness, required/missing fields, estimate count, QC summary,
    and the mandatory interpretation cautions."""

    summary = {
        "aperiodic_readiness": eligibility.support_level,
        "eligibility_confidence": eligibility.confidence,
        "required_fields_present": eligibility.required_fields_present,
        "helpful_fields_present": eligibility.helpful_fields_present,
        "missing_fields": eligibility.missing_fields,
        "eligibility_reasons": eligibility.reasons,
        "existing_estimate_count": 0,
        "qc_summary": {"pass": 0, "warn": 0, "fail": 0},
        "overall_qc_status": None,
        "interpretation_cautions": list(DEFAULT_INTERPRETATION_CAUTIONS),
    }
    if bundle is not None:
        summary["existing_estimate_count"] = len(bundle.estimates)
        summary["qc_summary"] = qc_summary(bundle)
        summary["overall_qc_status"] = bundle.overall_qc_status
        summary["interpretation_cautions"] = bundle.interpretation_cautions
    return summary
