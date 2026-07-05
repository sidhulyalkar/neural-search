from neural_search.experimentglancer.scene_builder import build_scene


def _base_dataset(**overrides):
    dataset = {
        "dataset_id": "dandi:000001",
        "source": "dandi",
        "source_id": "000001",
        "title": "Example dataset",
        "data_standard": "NWB",
        "modalities": [],
    }
    dataset.update(overrides)
    return dataset


def _base_card(**overrides):
    card = {
        "modalities": [],
        "behaviors": [],
        "tasks": [],
        "experimental_structure": {},
        "neural_data": {"available_assets": []},
    }
    card.update(overrides)
    return card


def _layer_kinds(scene):
    return {layer.kind for layer in scene.layers}


def test_metadata_only_dataset_creates_metadata_and_provenance_layers():
    scene = build_scene(dataset=_base_dataset(), dataset_card=_base_card(), query="")

    kinds = _layer_kinds(scene)
    assert "metadata.labels" in kinds
    assert "provenance.evidence" in kinds
    # No modality/trial signal at all -> nothing else should be claimed.
    assert "neural.calcium" not in kinds
    assert "neural.spikes" not in kinds
    assert scene.anchors[0].kind == "dataset_overview"


def test_calcium_dataset_creates_calcium_layer():
    dataset = _base_dataset(modalities=["calcium_imaging"])
    card = _base_card(modalities=["calcium_imaging"])

    scene = build_scene(dataset=dataset, dataset_card=card, query="two-photon calcium imaging")

    kinds = _layer_kinds(scene)
    assert "neural.calcium" in kinds
    calcium_layer = next(layer for layer in scene.layers if layer.kind == "neural.calcium")
    assert calcium_layer.status == "probable"
    assert calcium_layer.provenance.evidence_tier == "metadata_inferred"


def test_ephys_neuropixels_dataset_creates_spike_layer():
    dataset = _base_dataset(modalities=["neuropixels"])
    card = _base_card(modalities=["neuropixels"])

    scene = build_scene(dataset=dataset, dataset_card=card, query="Neuropixels recordings")

    kinds = _layer_kinds(scene)
    assert "neural.spikes" in kinds
    # No spike-sorted-units asset was listed -> should surface as a warning.
    assert any("spike-sorted units" in warning for warning in scene.warnings)


def test_behavior_tracking_dataset_creates_pose_layer():
    dataset = _base_dataset(modalities=["behavior_tracking"])
    card = _base_card(modalities=["behavior_tracking"], behaviors=["deeplabcut_pose"])

    scene = build_scene(dataset=dataset, dataset_card=card, query="DeepLabCut pose tracking")

    kinds = _layer_kinds(scene)
    assert "behavior.pose" in kinds


def test_query_lick_onset_creates_lick_event_anchor():
    dataset = _base_dataset(modalities=["electrophysiology"])
    card = _base_card(
        modalities=["electrophysiology"],
        experimental_structure={"trial_event_structure": ["lick_times", "trial_id"]},
    )

    scene = build_scene(dataset=dataset, dataset_card=card, query="lick onset aligned spiking")

    assert scene.anchors[0].kind == "event"
    assert scene.anchors[0].event_type == "lick_onset"
    assert scene.anchors[0].time is None


def test_missing_video_becomes_warning_not_available_layer():
    dataset = _base_dataset(modalities=["electrophysiology"])
    card = _base_card(modalities=["electrophysiology"])

    scene = build_scene(dataset=dataset, dataset_card=card, query="show me the behavior video")

    kinds = _layer_kinds(scene)
    assert "video.frames" not in kinds
    assert "video layer requested by query but no video asset detected" in scene.warnings


def test_scene_is_deterministic_for_same_inputs():
    dataset = _base_dataset(modalities=["calcium_imaging"])
    card = _base_card(modalities=["calcium_imaging"])

    scene_a = build_scene(dataset=dataset, dataset_card=card, query="calcium imaging")
    scene_b = build_scene(dataset=dataset, dataset_card=card, query="calcium imaging")

    assert scene_a.scene_id == scene_b.scene_id
    assert scene_a.provenance.inputs == scene_b.provenance.inputs


def test_search_result_context_sets_source_kind_and_score():
    dataset = _base_dataset(modalities=["calcium_imaging"])
    card = _base_card(modalities=["calcium_imaging"])
    search_result = {"score": 0.87, "score_breakdown": {"ontology_score": 0.8}, "rank": 1}

    scene = build_scene(
        dataset=dataset,
        dataset_card=card,
        search_result=search_result,
        query="calcium imaging",
        rank=1,
        retrieval_method="hybrid_search",
    )

    assert scene.source.kind == "search_result"
    assert scene.source.score == 0.87
    assert scene.source.retrieval_method == "hybrid_search"


def test_affordance_without_query_event_still_produces_anchor():
    dataset = _base_dataset(modalities=["calcium_imaging"])
    card = _base_card(modalities=["calcium_imaging"])

    scene = build_scene(
        dataset=dataset,
        dataset_card=card,
        query="latent dynamics of population activity",
        affordance_ids=["latent_dynamics_modeling"],
    )

    assert scene.anchors[0].anchor_id == "latent_dynamics_modeling_anchor"
    assert scene.anchors[0].kind == "trial"
