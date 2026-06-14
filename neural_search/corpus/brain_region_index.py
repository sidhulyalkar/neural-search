"""Build indexable brain-region atlas categories from ontology files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from neural_search.ontology import get_brain_regions

REGIONAL_TARGETS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "config" / "regional_map_targets.yaml"
)


class BrainRegionIndexEntry(BaseModel):
    """Frontend/search friendly expansion of one brain-region ontology entry."""

    id: str
    label: str
    system: str
    aliases: list[str] = Field(default_factory=list)
    parents: list[str] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)
    species_scope: list[str] = Field(default_factory=list)
    species_aliases: dict[str, list[str]] = Field(default_factory=dict)
    atlas_refs: dict[str, str] = Field(default_factory=dict)
    index_categories: list[str] = Field(default_factory=list)


def _load_regional_targets(path: Path = REGIONAL_TARGETS_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    targets = payload.get("regional_targets", [])
    if not isinstance(targets, list):
        return {}
    return {
        str(item.get("id")): item
        for item in targets
        if isinstance(item, dict) and item.get("id")
    }


def _children_by_parent(regions: dict[str, Any]) -> dict[str, list[str]]:
    children: dict[str, set[str]] = {region_id: set(region.children) for region_id, region in regions.items()}
    for region in regions.values():
        for parent in region.parents:
            children.setdefault(parent, set()).add(region.id)
    return {parent: sorted(values) for parent, values in children.items() if values}


def _infer_system(
    region_id: str,
    regions: dict[str, Any],
    targets: dict[str, dict[str, Any]],
    seen: set[str] | None = None,
) -> str:
    region = regions[region_id]
    if region.system:
        return region.system
    target_system = targets.get(region_id, {}).get("system")
    if target_system:
        return str(target_system)
    seen = seen or set()
    if region_id in seen:
        return "unmapped"
    seen.add(region_id)
    for parent in region.parents:
        if parent in regions:
            parent_system = _infer_system(parent, regions, targets, seen)
            if parent_system != "unmapped":
                return parent_system
    return "unmapped"


def _atlas_refs(region_id: str, region: Any, targets: dict[str, dict[str, Any]]) -> dict[str, str]:
    refs = dict(region.atlas_refs)
    refs.setdefault("neural_search_region", region_id)
    if region_id in targets:
        refs.setdefault("regional_map_target", region_id)
    return refs


def _index_categories(
    *,
    region_id: str,
    system: str,
    parents: list[str],
    children: list[str],
    species_scope: list[str],
    atlas_refs: dict[str, str],
) -> list[str]:
    categories = [
        f"brain_region:{region_id}",
        f"brain_system:{system}",
    ]
    categories.extend(f"parent_region:{parent}" for parent in parents)
    categories.extend(f"child_region:{child}" for child in children)
    categories.extend(f"species_scope:{species}" for species in species_scope)
    categories.extend(f"atlas:{atlas}:{value}" for atlas, value in sorted(atlas_refs.items()))
    return sorted(dict.fromkeys(categories))


@lru_cache(maxsize=1)
def build_brain_region_index() -> dict[str, BrainRegionIndexEntry]:
    """Return one indexable atlas entry per brain-region ontology ID."""

    regions = {region.id: region for region in get_brain_regions()}
    targets = _load_regional_targets()
    children_map = _children_by_parent(regions)
    entries: dict[str, BrainRegionIndexEntry] = {}
    for region_id, region in regions.items():
        system = _infer_system(region_id, regions, targets)
        children = children_map.get(region_id, [])
        refs = _atlas_refs(region_id, region, targets)
        entries[region_id] = BrainRegionIndexEntry(
            id=region_id,
            label=region.label,
            system=system,
            aliases=sorted(dict.fromkeys([*region.aliases, region.label, region_id])),
            parents=list(region.parents),
            children=children,
            species_scope=list(region.species_scope),
            species_aliases=dict(region.species_aliases),
            atlas_refs=refs,
            index_categories=_index_categories(
                region_id=region_id,
                system=system,
                parents=list(region.parents),
                children=children,
                species_scope=list(region.species_scope),
                atlas_refs=refs,
            ),
        )
    return entries
