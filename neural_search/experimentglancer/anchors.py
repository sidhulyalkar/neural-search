"""Anchor selection: choose the timeline moment ExperimentGlancer opens on.

Anchors are picked in priority order (query-specified event/trial, then
affordance-implied anchor, then dataset-card structure, then a bare
overview fallback) and never invent a concrete timestamp that wasn't
file-validated -- ``time`` stays ``None`` until a later phase resolver can
confirm it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from neural_search.experimentglancer.schemas import DatasetIntrospectionV1, SceneAnchor

_EVENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "lick_onset": ("lick onset", "lick"),
    "reward_omission": ("reward omission", "omission"),
    "stimulus_onset": ("stimulus onset",),
    "trial_outcome": ("trial outcome", "failed trial", "confident but", "successful trial"),
    "movement_onset": ("movement onset", "wheel movement onset"),
    "replay_event": ("replay event", "hippocampal replay", "replay"),
    "strategy_switch": ("strategy switch", "switched strategy"),
}

# Registry affordance ids (see neural_search/affordances/registry.py) mapped to
# the anchor kind they imply when no concrete event/trial has been requested.
# `pose_neural_correlation` stands in for "brain <-> behavior alignment" since
# that is the closest affordance actually registered.
_AFFORDANCE_ANCHOR_KIND: dict[str, str] = {
    "event_aligned_psth": "event",
    "choice_decoding": "trial",
    "q_learning": "trial",
    "pose_neural_correlation": "event",
    "latent_dynamics_modeling": "trial",
}


def _event_anchor(event_type: str, matched_keyword: str) -> SceneAnchor:
    return SceneAnchor(
        anchor_id=event_type,
        kind="event",
        label=event_type.replace("_", " ").title(),
        time=None,
        trial_id=None,
        event_type=event_type,
        reason=f"Query specified the event of interest: '{matched_keyword}'.",
        evidence=[matched_keyword],
    )


def _affordance_anchor(affordance_id: str, kind: str) -> SceneAnchor:
    return SceneAnchor(
        anchor_id=f"{affordance_id}_anchor",
        kind=kind,
        label=f"Anchor for {affordance_id.replace('_', ' ')}",
        time=None,
        trial_id=None,
        event_type=None,
        reason=(
            f"Affordance '{affordance_id}' implies this anchor when no concrete "
            "event/trial has been file-validated."
        ),
        evidence=[affordance_id],
    )


def _hinted_anchor(anchor_hint: Mapping[str, Any]) -> SceneAnchor:
    event_type = str(anchor_hint["event_type"])
    kind = anchor_hint.get("kind") or "event"
    relative_time = anchor_hint.get("relative_time")
    reason = "Anchor hint provided by caller."
    if relative_time is not None:
        reason += f" Requested offset: {relative_time}s relative to {event_type}."
    return SceneAnchor(
        anchor_id=event_type,
        kind=kind,
        label=event_type.replace("_", " ").title(),
        time=None,
        trial_id=None,
        event_type=event_type,
        reason=reason,
        evidence=["caller-provided anchor_hint"],
    )


def select_scene_anchors(
    *,
    query: str = "",
    affordance_ids: Sequence[str] = (),
    dataset_card: Mapping[str, Any] | None = None,
    introspection: DatasetIntrospectionV1 | None = None,
    anchor_hint: Mapping[str, Any] | None = None,
) -> list[SceneAnchor]:
    """Return every jump target the query/affordances/dataset structure
    imply, ranked with the highest-confidence anchor first.

    ``anchors[0]`` remains the anchor the scene opens on (unchanged
    contract). The rest of the list is what an "anchor tray" UI can offer as
    additional jump targets, so a dataset that both matches an event keyword
    and has trial structure surfaces both instead of only the first match.
    An explicit ``anchor_hint`` is a caller override and always wins alone.
    """

    if anchor_hint and anchor_hint.get("event_type"):
        return [_hinted_anchor(anchor_hint)]

    query_lower = query.lower()
    anchors: list[SceneAnchor] = []
    seen_ids: set[str] = set()

    def add(anchor: SceneAnchor) -> None:
        if anchor.anchor_id not in seen_ids:
            seen_ids.add(anchor.anchor_id)
            anchors.append(anchor)

    for event_type, keywords in _EVENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                add(_event_anchor(event_type, keyword))
                break

    for affordance_id in affordance_ids:
        kind = _AFFORDANCE_ANCHOR_KIND.get(affordance_id)
        if kind is not None:
            add(_affordance_anchor(affordance_id, kind))

    has_trial_structure = bool((dataset_card or {}).get("experimental_structure")) or bool(
        introspection and introspection.trial_columns
    )
    if has_trial_structure:
        add(
            SceneAnchor(
                anchor_id="first_trial",
                kind="trial",
                label="First trial (from dataset structure)",
                time=None,
                trial_id=None,
                event_type=None,
                reason="Dataset card describes trial structure; using a representative trial as the anchor.",
                evidence=["experimental_structure"],
            )
        )

    if not anchors:
        add(
            SceneAnchor(
                anchor_id="overview",
                kind="dataset_overview",
                label="Dataset overview",
                time=None,
                trial_id=None,
                event_type=None,
                reason="No concrete event timestamp was file-validated yet.",
                evidence=[],
            )
        )

    return anchors
