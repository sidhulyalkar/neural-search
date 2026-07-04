from __future__ import annotations

from neural_search.experimentglancer.scene_builder import build_scene
from neural_search.experimentglancer.schemas import DatasetIntrospectionV1
from neural_search.experimentglancer.source_resolvers import (
    resolve_dandi_nwb,
    resolve_dataset_introspection,
    resolve_openneuro_bids_local,
)


def _openneuro_dataset(**overrides):
    dataset = {
        "dataset_id": "openneuro:ds003505",
        "source": "openneuro",
        "source_id": "ds003505",
        "title": "Motor imagery EEG BCI fixture",
        "data_standard": "BIDS",
    }
    dataset.update(overrides)
    return dataset


def _dandi_dataset(**overrides):
    dataset = {
        "dataset_id": "dandi:000100",
        "source": "dandi",
        "source_id": "000100",
        "title": "Example dandiset",
        "data_standard": "NWB",
    }
    dataset.update(overrides)
    return dataset


def test_openneuro_bids_local_resolver_reads_real_fixture():
    introspection = resolve_openneuro_bids_local(_openneuro_dataset(), {})

    assert introspection.resolver == "openneuro_bids_local"
    assert "timeline.events" in introspection.detected_layers
    assert "timeline.trials" in introspection.detected_layers
    assert "neural.lfp" in introspection.detected_layers
    assert {"onset", "duration", "trial_type", "value"} <= set(introspection.event_columns)
    assert "sub-01" in introspection.subjects
    assert introspection.clocks == ["bids_events_seconds"]


def test_openneuro_bids_local_resolver_falls_back_for_unknown_dataset():
    introspection = resolve_openneuro_bids_local(_openneuro_dataset(source_id="ds999999"), {})

    assert introspection.resolver == "metadata_only"
    assert any("No local BIDS fixture found" in warning for warning in introspection.source_warnings)


def test_dandi_nwb_resolver_falls_back_on_streaming_failure(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr("neural_search.data.dandi_streaming.list_dandiset_assets", _boom)

    introspection = resolve_dandi_nwb(_dandi_dataset(), {})

    assert introspection.resolver == "metadata_only"
    assert any(
        "DANDI streaming introspection unavailable" in warning
        for warning in introspection.source_warnings
    )


def test_dandi_nwb_resolver_reports_file_derived_layers_on_success(monkeypatch):
    class FakeAsset:
        asset_id = "asset-1"
        path = "sub-01/sub-01_ses-01_ecephys.nwb"
        size_bytes = 12345

    def _fake_list_assets(dandiset_id, max_assets=3):
        return [FakeAsset()]

    def _fake_extract_metadata(asset):
        return {
            "has_units": True,
            "has_spike_times": True,
            "has_trials": True,
            "n_trials": 42,
            "trial_columns": ["start_time", "stop_time", "trial_type"],
            "has_imaging": False,
        }

    monkeypatch.setattr("neural_search.data.dandi_streaming.list_dandiset_assets", _fake_list_assets)
    monkeypatch.setattr(
        "neural_search.data.dandi_streaming.extract_nwb_metadata_streaming", _fake_extract_metadata
    )

    introspection = resolve_dandi_nwb(_dandi_dataset(), {})

    assert introspection.resolver == "dandi_nwb_streaming"
    assert "neural.spikes" in introspection.detected_layers
    assert "timeline.trials" in introspection.detected_layers
    assert "trial_type" in introspection.trial_columns
    assert introspection.clocks == ["nwb_time_seconds"]


def test_dispatcher_uses_bids_local_for_openneuro_by_default():
    introspection = resolve_dataset_introspection(_openneuro_dataset(), {})
    assert introspection.resolver == "openneuro_bids_local"


def test_dispatcher_stays_metadata_only_for_dandi_unless_deep():
    shallow = resolve_dataset_introspection(_dandi_dataset(), {})
    assert shallow.resolver == "metadata_only"


def test_dispatcher_attempts_dandi_streaming_when_deep_requested(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("no network in test")

    monkeypatch.setattr("neural_search.data.dandi_streaming.list_dandiset_assets", _boom)

    # Even when explicitly requested, a streaming failure still degrades
    # gracefully to metadata-only rather than raising.
    introspection = resolve_dataset_introspection(_dandi_dataset(), {}, deep=True)
    assert introspection.resolver == "metadata_only"
    assert any(
        "DANDI streaming introspection unavailable" in warning
        for warning in introspection.source_warnings
    )


def test_scene_builder_marks_file_derived_layer_as_available():
    introspection = DatasetIntrospectionV1(
        dataset_id="openneuro:ds003505",
        source="openneuro",
        resolver="openneuro_bids_local",
        clocks=["bids_events_seconds"],
        event_columns=["onset", "duration", "trial_type"],
        trial_columns=["trial_type"],
        detected_layers=["timeline.events", "timeline.trials"],
    )

    scene = build_scene(
        dataset=_openneuro_dataset(),
        dataset_card={"experimental_structure": {"trial_event_structure": ["trial_type"]}},
        query="motor imagery",
        introspection=introspection,
    )

    events_layer = next(layer for layer in scene.layers if layer.kind == "timeline.events")
    assert events_layer.status == "available"
    assert events_layer.provenance.evidence_tier == "file_derived"
    assert events_layer.warnings == []
    assert scene.provenance.evidence_tier == "file_derived"
