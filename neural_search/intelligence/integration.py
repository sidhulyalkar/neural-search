"""Bridge utilities for applying search intelligence without changing core search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from neural_search.awareness.search import search_datasets_with_awareness
from neural_search.intelligence.planner import (
    SearchIntelligencePlan,
    plan_search_intelligence,
)
from neural_search.schemas import SearchResponse


@dataclass(frozen=True)
class IntelligenceConfigApplication:
    """Result of applying a planner profile to a retrieval config."""

    plan: SearchIntelligencePlan
    retrieval_config: dict[str, Any]
    blended_weights: dict[str, float]
    enabled: bool

    def model_dump(self) -> dict[str, Any]:
        return {
            "plan": self.plan.model_dump(),
            "retrieval_config": self.retrieval_config,
            "blended_weights": self.blended_weights,
            "enabled": self.enabled,
        }


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def blend_planner_weights(
    base_weights: Mapping[str, Any],
    planner_weights: Mapping[str, float],
    *,
    strength: float = 0.35,
) -> dict[str, float]:
    """Blend existing retrieval weights with planner recommendations."""

    bounded = max(0.0, min(float(strength), 1.0))
    keys = sorted(set(base_weights) | set(planner_weights))
    blended: dict[str, float] = {}
    for key in keys:
        current = float(base_weights.get(key, 0.0) or 0.0)
        planned = float(planner_weights.get(key, current) or 0.0)
        blended[key] = round((1.0 - bounded) * current + bounded * planned, 4)
    return blended


def apply_search_intelligence_config(
    query: str,
    retrieval_config: Mapping[str, Any] | None = None,
    *,
    corpus_profile: Mapping[str, Any] | None = None,
) -> IntelligenceConfigApplication:
    """Return an effective retrieval config with optional planner blending."""

    base_config = deepcopy(dict(retrieval_config or {}))
    intelligence_config = base_config.get("intelligence", {})
    if not isinstance(intelligence_config, Mapping):
        intelligence_config = {}
    enabled = bool(intelligence_config.get("enabled", True))
    plan = plan_search_intelligence(query, corpus_profile=corpus_profile)
    if not enabled:
        return IntelligenceConfigApplication(
            plan=plan,
            retrieval_config=base_config,
            blended_weights=dict(base_config.get("weights", {})),
            enabled=False,
        )

    strength = float(intelligence_config.get("weight_blend_strength", 0.35))
    blended_weights = blend_planner_weights(
        base_config.get("weights", {}),
        plan.retrieval_weights,
        strength=strength,
    )
    effective = _deep_merge(base_config, {"weights": blended_weights})
    effective.setdefault("hard_negative_filters", {})
    if plan.mode == "constraint_filter_first":
        effective["hard_negative_filters"]["enabled"] = True
    effective["intelligence"] = {
        **dict(intelligence_config),
        "enabled": True,
        "active_intent": plan.intent,
        "active_mode": plan.mode,
        "quality_checks": list(plan.quality_checks),
    }
    return IntelligenceConfigApplication(
        plan=plan,
        retrieval_config=effective,
        blended_weights=blended_weights,
        enabled=True,
    )


def search_datasets_with_intelligence(
    query: str,
    filters: Mapping[str, Any] | None = None,
    structured_query: Mapping[str, Any] | None = None,
    datasets: Sequence[Mapping[str, Any]] | None = None,
    limit: int = 10,
    retrieval_config: Mapping[str, Any] | None = None,
    *,
    corpus_profile: Mapping[str, Any] | None = None,
    rerank: bool | None = None,
) -> SearchResponse:
    """Search with planner-derived config and awareness annotations."""

    application = apply_search_intelligence_config(
        query,
        retrieval_config,
        corpus_profile=corpus_profile,
    )
    intelligence_config = application.retrieval_config.get("intelligence", {})
    should_rerank = (
        bool(intelligence_config.get("rerank", False)) if rerank is None else rerank
    )
    awareness_weight = float(
        intelligence_config.get(
            "awareness_weight",
            application.blended_weights.get("awareness", 0.12),
        )
    )
    response = search_datasets_with_awareness(
        query=query,
        filters=filters,
        structured_query=structured_query,
        datasets=datasets,
        limit=limit,
        retrieval_config=application.retrieval_config,
        awareness_weight=awareness_weight,
        rerank=should_rerank,
    )
    response.parsed_query["search_intelligence_plan"] = application.plan.model_dump()
    response.parsed_query["search_intelligence_enabled"] = application.enabled
    response.parsed_query["search_intelligence_weights"] = application.blended_weights
    return response
