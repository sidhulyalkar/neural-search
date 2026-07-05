import numpy as np

from neural_search.spectral.qc import assess_spectral_qc
from neural_search.spectral.schemas import PeriodicPeak


def _peaks(n: int) -> list[PeriodicPeak]:
    return [
        PeriodicPeak(center_frequency_hz=5.0 + i, power=0.1, bandwidth_hz=1.0)
        for i in range(n)
    ]


def test_clean_fit_passes_qc():
    result = assess_spectral_qc(
        qc_id="qc:1",
        fit_r_squared=0.97,
        fit_error=0.05,
        n_frequency_bins=80,
        freq_range_hz=(2.0, 40.0),
        peaks=_peaks(2),
        sample_rate_hz=250.0,
        channel_metadata_present=True,
        region_metadata_present=True,
        task_state_present=True,
    )

    assert result.status == "pass"
    assert result.flags == []


def test_low_r_squared_and_high_error_trigger_warn_or_fail():
    warn_result = assess_spectral_qc(
        qc_id="qc:2",
        fit_r_squared=0.85,
        fit_error=0.4,
        n_frequency_bins=80,
        freq_range_hz=(2.0, 40.0),
    )
    assert warn_result.status in ("warn", "fail")
    assert "low_fit_r_squared" in warn_result.flags
    assert "high_fit_error" in warn_result.flags

    fail_result = assess_spectral_qc(
        qc_id="qc:3",
        fit_r_squared=0.5,
        fit_error=0.9,
        n_frequency_bins=80,
        freq_range_hz=(2.0, 40.0),
    )
    assert fail_result.status == "fail"


def test_too_few_frequency_bins_fails():
    result = assess_spectral_qc(
        qc_id="qc:4",
        fit_r_squared=0.95,
        fit_error=0.05,
        n_frequency_bins=3,
        freq_range_hz=(2.0, 40.0),
    )

    assert result.status == "fail"
    assert "too_few_frequency_bins" in result.flags


def test_narrow_frequency_range_fails():
    result = assess_spectral_qc(
        qc_id="qc:5",
        fit_r_squared=0.95,
        fit_error=0.05,
        n_frequency_bins=80,
        freq_range_hz=(10.0, 11.0),
    )

    assert result.status == "fail"
    assert "frequency_range_too_narrow" in result.flags


def test_line_noise_overlap_warns():
    result = assess_spectral_qc(
        qc_id="qc:6",
        fit_r_squared=0.97,
        fit_error=0.05,
        n_frequency_bins=80,
        freq_range_hz=(2.0, 55.0),
    )

    assert "line_noise_overlap" in result.flags
    assert result.status == "warn"


def test_many_peaks_warns_possible_overfit():
    result = assess_spectral_qc(
        qc_id="qc:7",
        fit_r_squared=0.97,
        fit_error=0.05,
        n_frequency_bins=80,
        freq_range_hz=(2.0, 40.0),
        peaks=_peaks(8),
    )

    assert "many_peaks_possible_overfit" in result.flags
    assert result.status == "warn"


def test_missing_metadata_flags_warn_without_failing():
    result = assess_spectral_qc(
        qc_id="qc:8",
        fit_r_squared=0.97,
        fit_error=0.05,
        n_frequency_bins=80,
        freq_range_hz=(2.0, 40.0),
        sample_rate_hz=None,
        channel_metadata_present=False,
        region_metadata_present=False,
        task_state_present=False,
    )

    assert result.status == "warn"
    assert {"missing_sampling_rate", "missing_channel_metadata", "missing_region_metadata", "missing_task_state"} <= set(
        result.flags
    )


def test_nan_signal_fails():
    signal = np.array([1.0, 2.0, np.nan, 4.0])
    result = assess_spectral_qc(
        qc_id="qc:9",
        fit_r_squared=0.97,
        fit_error=0.05,
        n_frequency_bins=80,
        freq_range_hz=(2.0, 40.0),
        signal=signal,
    )

    assert result.status == "fail"
    assert "nan_or_inf_values" in result.flags


def test_flat_signal_fails():
    signal = np.zeros(100)
    result = assess_spectral_qc(
        qc_id="qc:10",
        fit_r_squared=0.97,
        fit_error=0.05,
        n_frequency_bins=80,
        freq_range_hz=(2.0, 40.0),
        signal=signal,
    )

    assert result.status == "fail"
    assert "flat_or_zero_signal" in result.flags
