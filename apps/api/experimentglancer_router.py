"""FastAPI endpoints for the ExperimentGlancer bridge.

Mirrors the self-contained-router convention used by ``spectral_router.py``
/ ``claims_router.py``: this module loads the demo/combined corpus directly
from ``neural_search.ingestion.demo_seed`` instead of importing from
``apps.api.main``, avoiding a circular import.

Neural Search remains the intelligence layer; these endpoints only compile
search results / dataset records into the versioned
``ExperimentGlancerSceneV1`` contract. Rendering is ExperimentGlancer's job.
"""

from __future__ import annotations

import os
import time as _time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from neural_search.cards import generate_dataset_card_json
from neural_search.experimentglancer.persistence import load_scene, save_scene
from neural_search.experimentglancer.schemas import ExperimentGlancerSceneV1
from neural_search.experimentglancer.scene_builder import build_scene
from neural_search.experimentglancer.source_resolvers import resolve_dataset_introspection
from neural_search.extraction import extract_dataset_labels

router = APIRouter()

_DEMO_MODE = os.getenv("NEURAL_SEARCH_DEMO_MODE", "").lower() in ("1", "true", "yes")

_demo_data: list[dict[str, Any]] | None = None

# The in-memory cache is a hot-path optimization only; every scene is also
# written to artifacts/experimentglancer/scenes/ (see persistence.py) so a
# shareable URL keeps resolving after the TTL expires or the process
# restarts, without needing to re-run retrieval.
_SCENE_CACHE: dict[str, tuple[ExperimentGlancerSceneV1, float]] = {}
_SCENE_CACHE_TTL = 3600  # 1 hour


def _ensure_demo_data() -> list[dict[str, Any]]:
    global _demo_data
    if _demo_data is None:
        from neural_search.ingestion.demo_seed import (
            build_combined_corpus,
            build_demo_seed,
        )

        _demo_data = build_demo_seed() if _DEMO_MODE else build_combined_corpus()
    return _demo_data


def _find_dataset_record(dataset_id: str) -> dict[str, Any] | None:
    for record in _ensure_demo_data():
        ds = record["dataset"]
        if ds.get("id") == dataset_id or ds.get("source_id") == dataset_id:
            return record
    return None


def _card_for_record(record: dict[str, Any]) -> dict[str, Any]:
    ds = record["dataset"]
    extraction = record.get("extraction")
    if extraction is None:
        extraction = extract_dataset_labels(
            title=ds.get("title", ""),
            description=ds.get("description", ""),
            file_paths=[],
            source_metadata=ds,
            linked_paper_abstracts=[],
        )
    card = record.get("card")
    if card is None:
        card = generate_dataset_card_json(ds, extraction, record.get("papers", []))
    return card.model_dump(mode="json") if hasattr(card, "model_dump") else dict(card)


def _cache_scene(scene: ExperimentGlancerSceneV1) -> None:
    if len(_SCENE_CACHE) > 500:
        now = _time.time()
        expired = [key for key, (_, expires_at) in _SCENE_CACHE.items() if now >= expires_at]
        for key in expired[:100]:
            _SCENE_CACHE.pop(key, None)
    _SCENE_CACHE[scene.scene_id] = (scene, _time.time() + _SCENE_CACHE_TTL)
    save_scene(scene)


def _get_cached_scene(scene_id: str) -> ExperimentGlancerSceneV1 | None:
    entry = _SCENE_CACHE.get(scene_id)
    if entry and _time.time() < entry[1]:
        return entry[0]
    _SCENE_CACHE.pop(scene_id, None)

    persisted = load_scene(scene_id)
    if persisted is not None:
        _SCENE_CACHE[scene_id] = (persisted, _time.time() + _SCENE_CACHE_TTL)
    return persisted


def _scene_url(scene_id: str) -> str:
    return f"/experimentglancer?scene_id={scene_id}"


class AnchorHint(BaseModel):
    kind: str | None = None
    event_type: str | None = None
    relative_time: float | None = None


class SceneFromSearchResultRequest(BaseModel):
    query: str = ""
    dataset_id: str
    rank: int | None = None
    retrieval_method: str | None = None
    score: float | None = None
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    include_probable_layers: bool = True
    requested_layers: list[str] = Field(default_factory=list)
    affordance_ids: list[str] = Field(default_factory=list)
    anchor_hint: AnchorHint | None = None
    deep_introspection: bool = False


class SceneResponse(BaseModel):
    scene: dict[str, Any]
    scene_url: str
    external_url: str | None = None
    warnings: list[str] = Field(default_factory=list)


def _scene_response(scene: ExperimentGlancerSceneV1) -> SceneResponse:
    _cache_scene(scene)
    return SceneResponse(
        scene=scene.model_dump(mode="json"),
        scene_url=_scene_url(scene.scene_id),
        external_url=None,
        warnings=scene.warnings,
    )


@router.post("/api/experimentglancer/scenes/from-search-result", response_model=SceneResponse)
async def create_scene_from_search_result(request: SceneFromSearchResultRequest) -> SceneResponse:
    """Compile a search result (query + matched dataset) into a scene."""

    record = _find_dataset_record(request.dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")

    card = _card_for_record(record)
    search_result = {
        "rank": request.rank,
        "retrieval_method": request.retrieval_method,
        "score": request.score,
        "score_breakdown": request.score_breakdown,
    }

    scene = build_scene(
        dataset=record["dataset"],
        dataset_card=card,
        search_result=search_result,
        query=request.query,
        rank=request.rank,
        retrieval_method=request.retrieval_method,
        affordance_ids=request.affordance_ids,
        requested_layers=request.requested_layers,
        anchor_hint=request.anchor_hint.model_dump() if request.anchor_hint else None,
        deep_introspection=request.deep_introspection,
    )

    if not request.include_probable_layers:
        scene = scene.model_copy(
            update={"layers": [layer for layer in scene.layers if layer.status != "probable"]}
        )

    return _scene_response(scene)


@router.get("/api/experimentglancer/scenes/{scene_id}", response_model=SceneResponse)
async def get_scene(scene_id: str) -> SceneResponse:
    """Fetch a previously generated scene by id (shareable-URL target)."""

    scene = _get_cached_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")
    return SceneResponse(
        scene=scene.model_dump(mode="json"),
        scene_url=_scene_url(scene.scene_id),
        external_url=None,
        warnings=scene.warnings,
    )


@router.get("/api/experimentglancer/datasets/{dataset_id}/scene", response_model=SceneResponse)
async def get_dataset_scene(dataset_id: str, deep_introspection: bool = False) -> SceneResponse:
    """Compile a scene directly from a dataset's card, with no search context."""

    record = _find_dataset_record(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    card = _card_for_record(record)
    scene = build_scene(
        dataset=record["dataset"], dataset_card=card, query="", deep_introspection=deep_introspection
    )
    return _scene_response(scene)


@router.get("/api/experimentglancer/datasets/{dataset_id}/introspection")
async def get_dataset_introspection(dataset_id: str, deep: bool = False) -> dict[str, Any]:
    """Dataset introspection: OpenNeuro/BIDS local fixtures are always used when
    available (fast, no network). Pass ``deep=true`` to additionally attempt
    DANDI/NWB streaming introspection, which does real network I/O."""

    record = _find_dataset_record(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    card = _card_for_record(record)
    introspection = resolve_dataset_introspection(record["dataset"], card, deep=deep)
    return introspection.model_dump(mode="json")
