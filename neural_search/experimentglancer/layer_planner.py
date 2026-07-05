"""Layer planning: decide which ExperimentGlancer layers a scene should offer.

``plan_layers_for_result`` never invents availability. A layer only becomes
``status="available"`` when the introspection resolver reports file-derived
evidence; everything inferred from dataset-card metadata is ``probable``;
anything explicitly requested (by query or caller) but unsupported by
evidence becomes a warning instead of a layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from neural_search.experimentglancer.schemas import (
    DatasetIntrospectionV1,
    LayerAlignment,
    LayerDataRef,
    LayerDisplay,
    LayerKind,
    LayerProvenance,
    SceneLayer,
)

_CALCIUM_KEYWORDS = ("calcium_imaging", "calcium", "two_photon", "two-photon", "ophys")
_EPHYS_KEYWORDS = ("electrophysiology", "ephys", "neuropixels", "extracellular")
_POSE_KEYWORDS = ("pose", "deeplabcut", "kinematics", "behavior_tracking", "video_tracking")
_VIDEO_KEYWORDS = ("video", "behavior_video")
_PUPIL_KEYWORDS = ("pupil", "pupillometry")
_LICK_KEYWORDS = ("lick",)
_WHEEL_KEYWORDS = ("wheel", "locomotion", "running_wheel")
_STIMULUS_KEYWORDS = ("stimulus", "cue", "target")
_REWARD_KEYWORDS = ("reward", "omission", "feedback")
_MODEL_KEYWORDS = ("model", "decoded", "latent", "embedding")

_LAYER_TRACK: dict[str, str] = {
    "timeline.events": "timeline",
    "timeline.trials": "timeline",
    "video.frames": "behavior",
    "behavior.pose": "behavior",
    "behavior.pupil": "behavior",
    "behavior.licks": "behavior",
    "behavior.wheel": "behavior",
    "stimulus.identity": "timeline",
    "reward.delivery": "timeline",
    "neural.spikes": "neural",
    "neural.calcium": "neural",
    "neural.lfp": "neural",
    "neural.population_heatmap": "neural",
    "model.predictions": "model",
    "model.latent_state": "model",
    "metadata.labels": "metadata",
    "provenance.evidence": "metadata",
}

_LAYER_LABEL: dict[str, str] = {
    "timeline.events": "Behavioral events",
    "timeline.trials": "Trials",
    "video.frames": "Video frames",
    "behavior.pose": "Pose keypoints",
    "behavior.pupil": "Pupil diameter",
    "behavior.licks": "Lick events",
    "behavior.wheel": "Wheel movement",
    "stimulus.identity": "Stimulus identity",
    "reward.delivery": "Reward delivery",
    "neural.spikes": "Spike rasters",
    "neural.calcium": "Calcium traces",
    "neural.lfp": "LFP",
    "neural.population_heatmap": "Population activity heatmap",
    "model.predictions": "Model predictions",
    "model.latent_state": "Model latent state",
    "metadata.labels": "Dataset metadata",
    "provenance.evidence": "Search evidence and provenance",
}


@dataclass
class LayerPlanResult:
    layers: list[SceneLayer] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _any_keyword(tokens: Sequence[str], keywords: Sequence[str]) -> bool:
    return any(keyword in token for token in tokens for keyword in keywords)


def _make_layer(
    kind: LayerKind,
    *,
    status: str,
    data_ref_kind: str,
    clock: str,
    detector: str,
    evidence_tier: str,
    color: str | None = None,
    warnings: list[str] | None = None,
) -> SceneLayer:
    return SceneLayer(
        layer_id=kind.split(".")[-1],
        kind=kind,
        label=_LAYER_LABEL[kind],
        status=status,
        data_ref=LayerDataRef(kind=data_ref_kind),
        alignment=LayerAlignment(clock=clock),
        display=LayerDisplay(track=_LAYER_TRACK[kind], color=color),
        provenance=LayerProvenance(evidence_tier=evidence_tier, detector=detector),
        warnings=warnings or [],
    )


def plan_layers_for_result(
    *,
    dataset_card: Mapping[str, Any] | None = None,
    search_result: Mapping[str, Any] | None = None,
    query: str = "",
    requested_layers: Sequence[str] = (),
    introspection: DatasetIntrospectionV1 | None = None,
) -> LayerPlanResult:
    """Plan which scene layers a dataset/search-result pair can support.

    ``introspection`` (from a source resolver) is the source of truth for
    what's file-derived vs. only metadata-inferred. ``dataset_card`` and
    ``query`` widen the *candidate* set of layers to consider; they never
    upgrade a layer's evidence tier on their own.
    """

    card = dataset_card or {}
    result = LayerPlanResult()

    modalities = [m.lower() for m in card.get("modalities", [])]
    behaviors = [b.lower() for b in card.get("behaviors", [])]
    experimental_structure = card.get("experimental_structure") or {}
    event_tokens = [str(v).lower() for v in experimental_structure.get("trial_event_structure", [])]
    if introspection is not None:
        modalities = list(dict.fromkeys([*modalities, *(m.lower() for m in introspection.available_modalities)]))
        event_tokens = list(dict.fromkeys([*event_tokens, *(c.lower() for c in introspection.event_columns)]))
    tokens = [*modalities, *behaviors, *event_tokens, query.lower()]

    detected_layers = set(introspection.detected_layers) if introspection else set()
    trial_columns = introspection.trial_columns if introspection else []
    has_trials = bool(experimental_structure) or bool(trial_columns)
    has_events = bool(event_tokens) or (introspection is not None and bool(introspection.event_columns))

    clock = introspection.clocks[0] if introspection and introspection.clocks else "metadata_only"

    # Always-present layers.
    result.layers.append(
        _make_layer(
            "metadata.labels",
            status="available",
            data_ref_kind="dataset_card",
            clock="metadata_only",
            detector="dataset_card",
            evidence_tier="metadata_inferred",
        )
    )
    result.layers.append(
        _make_layer(
            "provenance.evidence",
            status="available",
            data_ref_kind="search_result",
            clock="metadata_only",
            detector="search_result" if search_result else "dataset_card",
            evidence_tier="metadata_inferred",
        )
    )

    def add_metadata_inferred(kind: LayerKind, *, color: str | None = None) -> None:
        # Only a resolver that actually read file bytes (resolver != "metadata_only")
        # can justify `file_derived` evidence; the metadata-only resolver's own
        # detected_layers (from asset modality hints) still count as `probable`.
        is_file_derived = (
            introspection is not None
            and introspection.resolver != "metadata_only"
            and kind in detected_layers
        )
        status = "available" if is_file_derived else "probable"
        evidence_tier = "file_derived" if is_file_derived else "metadata_inferred"
        result.layers.append(
            _make_layer(
                kind,
                status=status,
                data_ref_kind=f"{kind.replace('.', '_')}_reference",
                clock=clock,
                detector="dataset_card" if not introspection else introspection.resolver,
                evidence_tier=evidence_tier,
                color=color,
                warnings=(
                    []
                    if is_file_derived
                    else [f"{_LAYER_LABEL[kind]} inferred from metadata; file validation pending."]
                ),
            )
        )

    if has_trials:
        add_metadata_inferred("timeline.trials", color="#7dd3fc")
    if has_events:
        add_metadata_inferred("timeline.events", color="#a5b4fc")

    if _any_keyword(tokens, _CALCIUM_KEYWORDS):
        add_metadata_inferred("neural.calcium", color="#34d399")
    if _any_keyword(tokens, _EPHYS_KEYWORDS):
        add_metadata_inferred("neural.spikes", color="#f87171")
    if _any_keyword(tokens, _POSE_KEYWORDS):
        add_metadata_inferred("behavior.pose", color="#fbbf24")
    if _any_keyword(tokens, _PUPIL_KEYWORDS):
        add_metadata_inferred("behavior.pupil")
    if _any_keyword(tokens, _LICK_KEYWORDS):
        add_metadata_inferred("behavior.licks")
    if _any_keyword(tokens, _WHEEL_KEYWORDS):
        add_metadata_inferred("behavior.wheel")
    if _any_keyword(tokens, _STIMULUS_KEYWORDS):
        add_metadata_inferred("stimulus.identity")
    if _any_keyword(tokens, _REWARD_KEYWORDS):
        add_metadata_inferred("reward.delivery")

    video_evidence = "video.frames" in detected_layers or _any_keyword(modalities, _VIDEO_KEYWORDS)
    video_requested = "video.frames" in requested_layers or _any_keyword([query.lower()], _VIDEO_KEYWORDS)
    if video_evidence:
        add_metadata_inferred("video.frames")
    elif video_requested:
        result.warnings.append("video layer requested by query but no video asset detected")

    model_evidence = "model.predictions" in detected_layers or "model.latent_state" in detected_layers
    model_requested = _any_keyword(requested_layers, _MODEL_KEYWORDS) or _any_keyword([query.lower()], _MODEL_KEYWORDS)
    if model_evidence:
        add_metadata_inferred("model.predictions")
        add_metadata_inferred("model.latent_state")
    elif model_requested:
        result.warnings.append("model latent state layer requires derived artifact generation")

    if introspection is not None:
        for requirement in introspection.missing_layer_requirements:
            result.warnings.append(requirement)
        result.warnings.extend(introspection.source_warnings)

    return result
