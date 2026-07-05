import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from neural_search.experimentglancer.schemas import (
    CoordinateSpace,
    ExperimentGlancerSceneV1,
    LayerAlignment,
    LayerDataRef,
    LayerDisplay,
    LayerProvenance,
    SceneDatasetRef,
    SceneLayer,
    SceneProvenance,
    SceneSource,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "experimentglancer_scene_v1.json"


def _fixture_dict() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_fixture_scene_validates():
    scene = ExperimentGlancerSceneV1.model_validate(_fixture_dict())
    assert scene.schema_version == "experimentglancer.scene.v1"
    assert scene.dataset.dataset_id == "dandi:000000"
    assert len(scene.layers) == 4
    assert scene.anchors[0].kind == "dataset_overview"


def test_scene_round_trips_to_stable_json():
    scene = ExperimentGlancerSceneV1.model_validate(_fixture_dict())
    dumped = scene.model_dump(mode="json")
    reloaded = ExperimentGlancerSceneV1.model_validate(dumped)
    assert reloaded == scene


def _minimal_layer(**overrides) -> dict:
    layer = {
        "layer_id": "trials",
        "kind": "timeline.trials",
        "label": "Trials",
        "status": "probable",
        "data_ref": {"kind": "nwb_trials_table"},
        "alignment": {"clock": "nwb_time_seconds"},
        "display": {"track": "timeline"},
        "provenance": {"evidence_tier": "metadata_inferred", "detector": "dataset_card"},
    }
    layer.update(overrides)
    return layer


def test_layer_rejects_unsupported_kind():
    with pytest.raises(ValidationError):
        SceneLayer.model_validate(_minimal_layer(kind="video.holograms"))


def test_layer_rejects_unsupported_status():
    with pytest.raises(ValidationError):
        SceneLayer.model_validate(_minimal_layer(status="confirmed"))


def test_coordinate_space_rejects_unsupported_clock():
    with pytest.raises(ValidationError):
        CoordinateSpace.model_validate({"clock": "wall_clock_seconds"})


def test_scene_requires_dataset_and_provenance():
    with pytest.raises(ValidationError):
        ExperimentGlancerSceneV1.model_validate(
            {
                "scene_id": "eg_missing_fields",
                "created_at": "2026-07-03T00:00:00+00:00",
                "source": {"kind": "manual"},
            }
        )


def test_minimal_valid_scene_has_no_layers_or_anchors_by_default():
    scene = ExperimentGlancerSceneV1(
        scene_id="eg_minimal",
        created_at="2026-07-03T00:00:00+00:00",
        source=SceneSource(kind="manual"),
        dataset=SceneDatasetRef(dataset_id="dandi:000001"),
        coordinate_space=CoordinateSpace(clock="metadata_only"),
        provenance=SceneProvenance(
            generated_by="test",
            generator_version="v0.1.0",
            evidence_tier="unknown",
        ),
    )
    assert scene.layers == []
    assert scene.anchors == []
    assert scene.layout.tracks == ["timeline", "behavior", "neural", "model", "metadata"]


def test_layer_data_ref_alignment_display_provenance_construct():
    layer = SceneLayer(
        layer_id="spikes",
        kind="neural.spikes",
        label="Spike rasters",
        status="available",
        data_ref=LayerDataRef(kind="nwb_units_table", asset_id="asset-1"),
        alignment=LayerAlignment(clock="nwb_time_seconds", offset=0.5),
        display=LayerDisplay(track="neural", color="#f87171"),
        provenance=LayerProvenance(evidence_tier="file_derived", detector="nwb_resolver"),
    )
    assert layer.data_ref.asset_id == "asset-1"
    assert layer.alignment.offset == 0.5
