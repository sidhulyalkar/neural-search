"""Allen Brain Atlas API endpoints.

Serves pre-built Allen structure artifacts for the Knowledge Graph Explorer.
Follows the same lazy-load module-level cache pattern as graph_router.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

REPO_ROOT = Path(__file__).parent.parent.parent
ATLAS_DIR = REPO_ROOT / "artifacts" / "atlas"
MOUSE_STRUCTURES_PATH = ATLAS_DIR / "allen_ccf_mouse_structures.json"
HUMAN_STRUCTURES_PATH = ATLAS_DIR / "allen_human_structures.json"
ATLAS_GRAPH_PATH = ATLAS_DIR / "atlas_graph.json"

router = APIRouter(prefix="/api/atlas", tags=["atlas"])

# ── Module-level artifact cache (None = not yet loaded) ──────────────────────

_mouse_structures: list[dict[str, Any]] | None = None
_human_structures: list[dict[str, Any]] | None = None
_atlas_graph: dict[str, Any] | None = None
_ontology_mapping: dict[str, int] | None = None


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _get_mouse_structures() -> list[dict[str, Any]]:
    global _mouse_structures
    if _mouse_structures is None:
        _mouse_structures = _load_json_list(MOUSE_STRUCTURES_PATH)
    return _mouse_structures


def _get_human_structures() -> list[dict[str, Any]]:
    global _human_structures
    if _human_structures is None:
        _human_structures = _load_json_list(HUMAN_STRUCTURES_PATH)
    return _human_structures


def _get_atlas_graph() -> dict[str, Any]:
    global _atlas_graph
    if _atlas_graph is None:
        if not ATLAS_GRAPH_PATH.exists():
            _atlas_graph = {"nodes": [], "edges": [], "meta": {}}
        else:
            _atlas_graph = json.loads(ATLAS_GRAPH_PATH.read_text(encoding="utf-8"))
    return _atlas_graph


def _get_ontology_mapping() -> dict[str, int]:
    """Build ontology → allen_id mapping from brain_regions.yaml on first call."""
    global _ontology_mapping
    if _ontology_mapping is not None:
        return _ontology_mapping

    try:
        import yaml  # type: ignore[import]
        ontology_path = REPO_ROOT / "data" / "ontology" / "brain_regions.yaml"
        if not ontology_path.exists():
            _ontology_mapping = {}
            return _ontology_mapping

        raw = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
        regions = raw.get("brain_regions", [])
    except Exception:
        _ontology_mapping = {}
        return _ontology_mapping

    from neural_search.ingestion.allen_structures import AllenStructure, load_structures

    mouse_structures = load_structures(MOUSE_STRUCTURES_PATH)
    from neural_search.graph.atlas_builder import map_ontology_to_allen

    _ontology_mapping = map_ontology_to_allen(regions, mouse_structures)
    return _ontology_mapping


def _structures_for_species(species: str) -> list[dict[str, Any]]:
    if species == "human":
        return _get_human_structures()
    return _get_mouse_structures()


def _collect_descendants(
    allen_id: int,
    by_parent: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """BFS over the children map to collect all descendants."""
    result: list[dict[str, Any]] = []
    queue = list(by_parent.get(allen_id, []))
    while queue:
        item = queue.pop(0)
        result.append(item)
        queue.extend(by_parent.get(item["allen_id"], []))
    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/structures")
def get_structures(
    species: str = Query("mouse", description="'mouse' or 'human'"),
    level: int | None = Query(None, description="Filter by st_level"),
    limit: int = Query(200),
) -> dict[str, Any]:
    """Return Allen structures, optionally filtered by species or st_level."""
    structs = _structures_for_species(species)
    if level is not None:
        structs = [s for s in structs if s.get("st_level") == level]
    return {
        "species": species,
        "total": len(structs),
        "structures": structs[:limit],
    }


@router.get("/structures/{allen_id}")
def get_structure(allen_id: int) -> dict[str, Any]:
    """Get a single Allen structure by ID, including its children_ids."""
    for species in ("mouse", "human"):
        for s in _structures_for_species(species):
            if s.get("allen_id") == allen_id:
                return s
    raise HTTPException(status_code=404, detail=f"Allen structure {allen_id} not found")


@router.get("/structures/{allen_id}/children")
def get_children(allen_id: int, recursive: bool = False) -> list[dict[str, Any]]:
    """Return direct children (or all descendants when recursive=True)."""
    # Determine which species index to search
    target: dict[str, Any] | None = None
    all_structs: list[dict[str, Any]] = []
    for species in ("mouse", "human"):
        structs = _structures_for_species(species)
        for s in structs:
            if s.get("allen_id") == allen_id:
                target = s
                all_structs = structs
                break
        if target:
            break

    if target is None:
        raise HTTPException(status_code=404, detail=f"Allen structure {allen_id} not found")

    children_ids = set(target.get("children_ids") or [])
    if not recursive:
        return [s for s in all_structs if s.get("allen_id") in children_ids]

    # Build parent → children map for BFS
    from collections import defaultdict
    by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for s in all_structs:
        pid = s.get("parent_id")
        if pid is not None:
            by_parent[pid].append(s)

    return _collect_descendants(allen_id, by_parent)


@router.get("/regions/mapping")
def get_ontology_mapping() -> dict[str, Any]:
    """Return the mapping: ontology_region_id → allen_structure_id."""
    mapping = _get_ontology_mapping()
    return {
        "total_mapped": len(mapping),
        "mapping": mapping,
    }


@router.get("/coverage")
def get_atlas_coverage() -> dict[str, Any]:
    """Stats: how many Allen structures are mapped to our ontology, by level."""
    mapping = _get_ontology_mapping()
    mapped_allen_ids = set(mapping.values())

    mouse_structs = _get_mouse_structures()
    by_level: dict[int, dict[str, int]] = {}
    for s in mouse_structs:
        level = s.get("st_level", -1)
        if level not in by_level:
            by_level[level] = {"total": 0, "mapped": 0}
        by_level[level]["total"] += 1
        if s.get("allen_id") in mapped_allen_ids:
            by_level[level]["mapped"] += 1

    return {
        "total_mouse_structures": len(mouse_structs),
        "total_human_structures": len(_get_human_structures()),
        "total_ontology_mapped": len(mapping),
        "by_level": by_level,
    }
