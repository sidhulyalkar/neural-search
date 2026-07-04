"""Versioned scene schema for the ExperimentGlancer bridge.

``ExperimentGlancerSceneV1`` is the portable contract between Neural Search
(intelligence/retrieval) and ExperimentGlancer (multimodal timeline viewer).
The schema is conservative by construction: a layer's ``status`` must never
imply availability beyond what evidence supports (see ``LayerStatus`` and
``EvidenceTier``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SceneSourceKind = Literal[
    "search_result",
    "dataset_card",
    "manual",
    "reanalysis_candidate",
]

ClockKind = Literal[
    "experiment_time_seconds",
    "nwb_time_seconds",
    "bids_events_seconds",
    "metadata_only",
]

AnchorKind = Literal[
    "event",
    "trial",
    "dataset_overview",
]

LayerKind = Literal[
    "timeline.events",
    "timeline.trials",
    "video.frames",
    "behavior.pose",
    "behavior.pupil",
    "behavior.licks",
    "behavior.wheel",
    "stimulus.identity",
    "reward.delivery",
    "neural.spikes",
    "neural.calcium",
    "neural.lfp",
    "neural.population_heatmap",
    "model.predictions",
    "model.latent_state",
    "metadata.labels",
    "provenance.evidence",
]

LayerStatus = Literal["available", "probable", "placeholder", "unsupported"]

EvidenceTier = Literal["file_derived", "metadata_inferred", "query_requested", "unknown"]


class QueryContext(BaseModel):
    query: str
    parsed_query: dict[str, Any] = Field(default_factory=dict)
    requested_layers: list[str] = Field(default_factory=list)
    affordance_ids: list[str] = Field(default_factory=list)


class SceneSource(BaseModel):
    kind: SceneSourceKind
    query: str | None = None
    rank: int | None = None
    retrieval_method: str | None = None
    score: float | None = None
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class SceneDatasetRef(BaseModel):
    dataset_id: str
    source: str | None = None
    source_id: str | None = None
    title: str | None = None
    url: str | None = None
    doi: str | None = None
    data_standard: str | None = None
    species: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    brain_regions: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)


class DefaultWindow(BaseModel):
    center_time: float | None = None
    pre: float = 2.0
    post: float = 5.0


class CoordinateSpace(BaseModel):
    clock: ClockKind
    time_unit: str = "s"
    default_window: DefaultWindow = Field(default_factory=DefaultWindow)
    session_id: str | None = None
    subject_id: str | None = None
    trial_id: str | None = None


class SceneAnchor(BaseModel):
    anchor_id: str
    kind: AnchorKind
    label: str
    time: float | None = None
    trial_id: str | None = None
    event_type: str | None = None
    reason: str
    evidence: list[str] = Field(default_factory=list)


class LayerDataRef(BaseModel):
    kind: str
    asset_id: str | None = None
    path: str | None = None
    column: str | None = None
    table: str | None = None


class LayerAlignment(BaseModel):
    clock: ClockKind
    offset: float = 0.0


class LayerDisplay(BaseModel):
    track: str
    color: str | None = None


class LayerProvenance(BaseModel):
    evidence_tier: EvidenceTier
    detector: str


class SceneLayer(BaseModel):
    layer_id: str
    kind: LayerKind
    label: str
    status: LayerStatus
    data_ref: LayerDataRef
    alignment: LayerAlignment
    display: LayerDisplay
    provenance: LayerProvenance
    warnings: list[str] = Field(default_factory=list)


class SceneLayout(BaseModel):
    primary_view: str = "timeline"
    tracks: list[str] = Field(
        default_factory=lambda: ["timeline", "behavior", "neural", "model", "metadata"]
    )


class SceneProvenance(BaseModel):
    generated_by: str
    generator_version: str
    inputs: dict[str, str] = Field(default_factory=dict)
    evidence_tier: EvidenceTier
    missing_requirements: list[str] = Field(default_factory=list)


class ExperimentGlancerSceneV1(BaseModel):
    """The full, versioned scene contract handed off to ExperimentGlancer."""

    schema_version: Literal["experimentglancer.scene.v1"] = "experimentglancer.scene.v1"
    scene_id: str
    created_at: str
    source: SceneSource
    query_context: QueryContext | None = None
    dataset: SceneDatasetRef
    coordinate_space: CoordinateSpace
    anchors: list[SceneAnchor] = Field(default_factory=list)
    layers: list[SceneLayer] = Field(default_factory=list)
    layout: SceneLayout = Field(default_factory=SceneLayout)
    provenance: SceneProvenance
    warnings: list[str] = Field(default_factory=list)


class DatasetIntrospectionV1(BaseModel):
    """Normalized output of a source resolver.

    Resolvers never download full datasets; they translate whatever
    metadata (and, in later phases, lightweight streamed file headers) is
    available into this shape so the layer planner never has to know which
    resolver produced it.
    """

    dataset_id: str
    source: str | None = None
    resolver: str
    assets: list[dict[str, Any]] = Field(default_factory=list)
    clocks: list[ClockKind] = Field(default_factory=list)
    sessions: list[str] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    trial_columns: list[str] = Field(default_factory=list)
    event_columns: list[str] = Field(default_factory=list)
    available_modalities: list[str] = Field(default_factory=list)
    detected_layers: list[LayerKind] = Field(default_factory=list)
    missing_layer_requirements: list[str] = Field(default_factory=list)
    source_warnings: list[str] = Field(default_factory=list)
