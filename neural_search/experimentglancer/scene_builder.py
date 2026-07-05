"""Scene builder: assembles ``ExperimentGlancerSceneV1`` from a dataset,
optional dataset card, and optional search result.

This is the single entry point the API layer (Phase 2) and, eventually, the
frontend action (Phase 3) call. It owns no rendering logic and does not
import FastAPI or React types -- it only produces the scene contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from neural_search.experimentglancer.anchors import select_scene_anchors
from neural_search.experimentglancer.layer_planner import plan_layers_for_result
from neural_search.experimentglancer.schemas import (
    CoordinateSpace,
    DatasetIntrospectionV1,
    ExperimentGlancerSceneV1,
    QueryContext,
    SceneDatasetRef,
    SceneProvenance,
    SceneSource,
)
from neural_search.experimentglancer.serialization import make_scene_id, stable_hash
from neural_search.experimentglancer.source_resolvers import (
    resolve_dataset_introspection,
)

GENERATOR_VERSION = "v0.1.0"

_EVIDENCE_TIER_RANK = {
    "file_derived": 3,
    "metadata_inferred": 2,
    "query_requested": 1,
    "unknown": 0,
}


def _dataset_ref(dataset: Mapping[str, Any], dataset_card: Mapping[str, Any]) -> SceneDatasetRef:
    def pick(*keys: str) -> Any:
        for source in (dataset, dataset_card):
            for key in keys:
                value = source.get(key)
                if value:
                    return value
        return None

    dataset_id = str(pick("dataset_id", "id") or "unknown")
    return SceneDatasetRef(
        dataset_id=dataset_id,
        source=pick("source"),
        source_id=pick("source_id"),
        title=pick("title"),
        url=pick("url"),
        doi=pick("doi"),
        data_standard=pick("data_standard"),
        species=list(pick("species") or []),
        modalities=list(dict.fromkeys([*dataset.get("modalities", []), *dataset_card.get("modalities", [])])),
        brain_regions=list(pick("brain_regions") or []),
        tasks=list(pick("tasks") or []),
    )


def _highest_evidence_tier(tiers: Sequence[str]) -> str:
    if not tiers:
        return "unknown"
    return max(tiers, key=lambda tier: _EVIDENCE_TIER_RANK.get(tier, 0))


def build_scene(
    *,
    dataset: Mapping[str, Any],
    dataset_card: Mapping[str, Any] | None = None,
    search_result: Mapping[str, Any] | None = None,
    query: str = "",
    rank: int | None = None,
    retrieval_method: str | None = None,
    affordance_ids: Sequence[str] = (),
    requested_layers: Sequence[str] = (),
    introspection: DatasetIntrospectionV1 | None = None,
    anchor_hint: Mapping[str, Any] | None = None,
    deep_introspection: bool = False,
) -> ExperimentGlancerSceneV1:
    """Build a scene for one dataset, optionally in the context of a search result.

    ``introspection`` may be supplied directly by a caller that already ran a
    source-specific resolver. Otherwise ``resolve_dataset_introspection`` picks
    one: OpenNeuro/BIDS local fixtures are always attempted (fast, no
    network); DANDI/NWB streaming (real network I/O) only runs when
    ``deep_introspection=True`` is explicitly requested.
    """

    card = dict(dataset_card or {})
    result_dict = dict(search_result or {})

    resolved_introspection = introspection or resolve_dataset_introspection(
        dataset, card, deep=deep_introspection
    )

    source_kind = "search_result" if search_result else ("dataset_card" if dataset_card else "manual")

    dataset_ref = _dataset_ref(dataset, card)

    layer_plan = plan_layers_for_result(
        dataset_card=card,
        search_result=result_dict,
        query=query,
        requested_layers=requested_layers,
        introspection=resolved_introspection,
    )
    anchors = select_scene_anchors(
        query=query,
        affordance_ids=affordance_ids,
        dataset_card=card,
        introspection=resolved_introspection,
        anchor_hint=anchor_hint,
    )

    clock = resolved_introspection.clocks[0] if resolved_introspection.clocks else "metadata_only"
    coordinate_space = CoordinateSpace(
        clock=clock,
        session_id=resolved_introspection.sessions[0] if resolved_introspection.sessions else None,
        subject_id=resolved_introspection.subjects[0] if resolved_introspection.subjects else None,
    )

    evidence_tier = _highest_evidence_tier([layer.provenance.evidence_tier for layer in layer_plan.layers])
    missing_requirements = list(resolved_introspection.missing_layer_requirements)
    if evidence_tier != "file_derived":
        missing_requirements.append("file validated event timestamps")

    scene_id = make_scene_id(
        dataset_id=dataset_ref.dataset_id,
        query=query,
        anchor_hint=dict(anchor_hint) if anchor_hint else "",
        requested_layers=requested_layers,
        affordance_ids=affordance_ids,
        deep_introspection=deep_introspection,
    )

    return ExperimentGlancerSceneV1(
        scene_id=scene_id,
        created_at=datetime.now(UTC).isoformat(),
        source=SceneSource(
            kind=source_kind,
            query=query or None,
            rank=rank if rank is not None else result_dict.get("rank"),
            retrieval_method=retrieval_method or result_dict.get("retrieval_method"),
            score=result_dict.get("score"),
            score_breakdown=result_dict.get("score_breakdown") or {},
        ),
        query_context=(
            QueryContext(
                query=query,
                requested_layers=list(requested_layers),
                affordance_ids=list(affordance_ids),
            )
            if (query or requested_layers or affordance_ids)
            else None
        ),
        dataset=dataset_ref,
        coordinate_space=coordinate_space,
        anchors=anchors,
        layers=layer_plan.layers,
        provenance=SceneProvenance(
            generated_by="neural_search.experimentglancer.scene_builder",
            generator_version=GENERATOR_VERSION,
            inputs={
                "dataset_id": dataset_ref.dataset_id,
                "query": stable_hash(query),
            },
            evidence_tier=evidence_tier,
            missing_requirements=missing_requirements,
        ),
        warnings=layer_plan.warnings,
    )
