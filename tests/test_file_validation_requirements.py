"""Tests for the analysis_family -> file-validation requirement checks."""

from __future__ import annotations

from neural_search.graph.dandi_nwb_validator import DandiAssetValidation
from neural_search.graph.file_validation_requirements import (
    dandi_confirms_requirement,
    openneuro_confirms_requirement,
)
from neural_search.graph.openneuro_bids_validator import OpenNeuroValidation


def _dandi(**overrides):
    defaults = {"dandiset_id": "000003", "asset_path": "a.nwb", "size_bytes": 100}
    defaults.update(overrides)
    return DandiAssetValidation(**defaults)


def test_dandi_spike_train_analysis_requires_units():
    assert dandi_confirms_requirement("spike_train_analysis", _dandi(has_units=True, n_units=10))
    assert not dandi_confirms_requirement("spike_train_analysis", _dandi(has_units=False))
    assert not dandi_confirms_requirement("spike_train_analysis", _dandi(has_units=True, n_units=0))


def test_dandi_time_frequency_requires_electrodes():
    assert dandi_confirms_requirement("time_frequency", _dandi(has_electrodes=True, n_electrodes=32))
    assert not dandi_confirms_requirement("time_frequency", _dandi(has_electrodes=False))


def test_dandi_unknown_analysis_family_returns_false():
    assert not dandi_confirms_requirement("not_a_real_family", _dandi(has_units=True, n_units=5))


def test_dandi_error_result_never_confirms():
    assert not dandi_confirms_requirement(
        "spike_train_analysis", _dandi(has_units=True, n_units=10, error="boom")
    )


def _openneuro(**overrides):
    defaults = {"dataset_id": "ds000117"}
    defaults.update(overrides)
    return OpenNeuroValidation(**defaults)


def test_openneuro_time_frequency_requires_eeg_or_meg_modality():
    assert openneuro_confirms_requirement("time_frequency", _openneuro(modalities=["eeg"]))
    assert openneuro_confirms_requirement("time_frequency", _openneuro(modalities=["meg"]))
    assert not openneuro_confirms_requirement("time_frequency", _openneuro(modalities=["mri"]))


def test_openneuro_event_aligned_analysis_requires_tasks():
    assert openneuro_confirms_requirement(
        "event_aligned_analysis", _openneuro(tasks=["facerecognition"])
    )
    assert not openneuro_confirms_requirement("event_aligned_analysis", _openneuro(tasks=[]))


def test_openneuro_error_result_never_confirms():
    assert not openneuro_confirms_requirement(
        "time_frequency", _openneuro(modalities=["eeg"], error="boom")
    )
