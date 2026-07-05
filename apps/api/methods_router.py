"""FastAPI router exposing methods taxonomy, species homology, oscillations, paradigms, and modalities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query

log = logging.getLogger(__name__)

router = APIRouter(prefix="/methods", tags=["methods"])

DATA_ROOT = Path(__file__).parent.parent.parent / "data"


# ── Module-level caches ────────────────────────────────────────────────────

_methods_cache: dict[str, Any] | None = None
_homology_cache: dict[str, Any] | None = None
_oscillations_cache: dict[str, Any] | None = None
_paradigms_cache: dict[str, Any] | None = None
_hcp_cache: dict[str, Any] | None = None
_modalities_cache: dict[str, Any] | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _merge_categories(base: dict[str, Any], ext: dict[str, Any]) -> dict[str, Any]:
    """Merge extension categories into base by appending methods within matching categories."""
    base_cats = {c["id"]: c for c in base.get("categories", [])}
    for ext_cat in ext.get("categories", []):
        cat_id = ext_cat["id"]
        if cat_id in base_cats:
            key = "methods" if "methods" in base_cats[cat_id] else "concepts"
            base_cats[cat_id].setdefault(key, []).extend(ext_cat.get("methods", ext_cat.get("concepts", [])))
        else:
            base.setdefault("categories", []).append(ext_cat)
    return base


def _get_methods() -> dict[str, Any]:
    global _methods_cache
    if _methods_cache is None:
        base = _load_yaml(DATA_ROOT / "methods" / "methods_taxonomy.yaml")
        ext_path = DATA_ROOT / "methods" / "methods_dl_multimodal.yaml"
        if ext_path.exists():
            _merge_categories(base, _load_yaml(ext_path))
        _methods_cache = base
    return _methods_cache


def _get_homology() -> dict[str, Any]:
    global _homology_cache
    if _homology_cache is None:
        _homology_cache = _load_yaml(DATA_ROOT / "species" / "species_homology.yaml")
    return _homology_cache


def _get_oscillations() -> dict[str, Any]:
    global _oscillations_cache
    if _oscillations_cache is None:
        base = _load_yaml(DATA_ROOT / "oscillations" / "oscillation_signatures.yaml")
        ext_path = DATA_ROOT / "oscillations" / "oscillation_signatures_ext.yaml"
        if ext_path.exists():
            ext = _load_yaml(ext_path)
            base.setdefault("oscillation_signatures", []).extend(ext.get("oscillation_signatures", []))
        _oscillations_cache = base
    return _oscillations_cache


def _get_paradigms() -> dict[str, Any]:
    global _paradigms_cache
    if _paradigms_cache is None:
        base = _load_yaml(DATA_ROOT / "paradigms" / "paradigm_registry.yaml")
        ext_path = DATA_ROOT / "paradigms" / "paradigm_registry_ext.yaml"
        if ext_path.exists():
            ext = _load_yaml(ext_path)
            base.setdefault("paradigms", []).extend(ext.get("paradigms", []))
        _paradigms_cache = base
    return _paradigms_cache


def _get_hcp() -> dict[str, Any]:
    global _hcp_cache
    if _hcp_cache is None:
        _hcp_cache = _load_yaml(DATA_ROOT / "hcp" / "structural_connectivity_priors.yaml")
    return _hcp_cache


def _get_modalities() -> dict[str, Any]:
    global _modalities_cache
    if _modalities_cache is None:
        _modalities_cache = _load_yaml(DATA_ROOT / "modalities" / "modality_registry.yaml")
    return _modalities_cache


# ── Methods endpoints ──────────────────────────────────────────────────────

@router.get("/categories")
def get_method_categories() -> list[dict[str, Any]]:
    """All method categories with counts."""
    data = _get_methods()
    return [
        {
            "id": cat["id"],
            "label": cat["label"],
            "description": cat.get("description", ""),
            "count": len(cat.get("methods", cat.get("concepts", []))),
        }
        for cat in data.get("categories", [])
    ]


@router.get("/list")
def list_methods(
    category: str | None = Query(default=None),
    topic: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    """List all methods, optionally filtered by category or topic."""
    data = _get_methods()
    results = []
    for cat in data.get("categories", []):
        if category and cat["id"] != category:
            continue
        for item in cat.get("methods", cat.get("concepts", [])):
            if topic and topic not in item.get("topics", []):
                continue
            results.append(
                {
                    "id": item["id"],
                    "label": item.get("label", item["id"]),
                    "category": cat["id"],
                    "category_label": cat["label"],
                    "formula": item.get("formula", ""),
                    "computes": item.get("computes", []),
                    "topics": item.get("topics", []),
                    "aliases": item.get("aliases", []),
                }
            )
    return results


@router.get("/detail/{method_id}")
def get_method_detail(method_id: str) -> dict[str, Any]:
    """Full details for a single method including assumptions, math, limitations."""
    data = _get_methods()
    for cat in data.get("categories", []):
        for item in cat.get("methods", cat.get("concepts", [])):
            if item["id"] == method_id:
                return {**item, "category": cat["id"], "category_label": cat["label"]}
    raise HTTPException(status_code=404, detail=f"Method '{method_id}' not found")


# ── Species homology endpoints ─────────────────────────────────────────────

@router.get("/homology/groups")
def get_homology_groups(
    confidence: str | None = Query(default=None, description="high|medium|low"),
    species: str | None = Query(default=None, description="mouse|rat|macaque|human"),
) -> list[dict[str, Any]]:
    """All homology groups, optionally filtered by confidence or species."""
    data = _get_homology()
    results = []
    for group in data.get("homologs", []):
        if confidence and group.get("confidence") != confidence:
            continue
        if species:
            member_species = [m["species"] for m in group.get("members", [])]
            if species not in member_species:
                continue
        results.append(
            {
                "group_id": group["group_id"],
                "confidence": group.get("confidence"),
                "basis": group.get("basis", []),
                "notes": group.get("notes", ""),
                "divergence": group.get("divergence", ""),
                "members": group.get("members", []),
            }
        )
    return results


@router.get("/homology/region/{region_id}")
def get_region_homologs(region_id: str) -> dict[str, Any]:
    """Get all cross-species homologs for a given region ID."""
    data = _get_homology()
    matching_groups = []
    for group in data.get("homologs", []):
        for member in group.get("members", []):
            if member["region_id"] == region_id:
                matching_groups.append(group)
                break
    if not matching_groups:
        raise HTTPException(status_code=404, detail=f"No homologs found for region '{region_id}'")
    return {"region_id": region_id, "homolog_groups": matching_groups}


@router.get("/homology/human-specific")
def get_human_specific_regions() -> list[dict[str, Any]]:
    """Return regions that are human-specific or massively expanded in humans."""
    data = _get_homology()
    return data.get("human_specific_or_expanded", [])


# ── Oscillation signature endpoints ────────────────────────────────────────

@router.get("/oscillations")
def get_oscillations(
    region: str | None = Query(default=None),
    band: str | None = Query(default=None),
    species: str | None = Query(default=None),
    topic: str | None = Query(default=None),
) -> dict[str, Any]:
    """Oscillation signatures with optional filters."""
    data = _get_oscillations()
    signatures = data.get("oscillation_signatures", [])

    filtered = []
    for sig in signatures:
        if region and sig["region_id"] != region:
            continue
        if band and sig["frequency_band"] != band:
            continue
        if species:
            if species not in sig.get("species", []):
                continue
        if topic:
            if topic not in sig.get("topics", []):
                continue
        filtered.append(sig)

    return {
        "frequency_bands": data.get("frequency_bands", []),
        "signatures": filtered,
        "total": len(filtered),
    }


@router.get("/oscillations/region/{region_id}")
def get_region_oscillations(region_id: str) -> list[dict[str, Any]]:
    """All oscillations recorded from a given region."""
    data = _get_oscillations()
    return [
        sig for sig in data.get("oscillation_signatures", [])
        if sig["region_id"] == region_id
    ]


# ── Paradigm endpoints ──────────────────────────────────────────────────────

@router.get("/paradigms")
def list_paradigms(
    species: str | None = Query(default=None),
    topic: str | None = Query(default=None),
    validity: str | None = Query(default=None, description="validated|adapted|analogous"),
) -> list[dict[str, Any]]:
    """All cross-species behavioral paradigms."""
    data = _get_paradigms()
    results = []
    for p in data.get("paradigms", []):
        impls = p.get("species_implementations", [])

        if species:
            species_list = [i["species"] for i in impls]
            if species not in species_list:
                continue
        if topic:
            if topic not in p.get("topics", []):
                continue
        if validity:
            has_validity = any(
                i.get("cross_species_validity") == validity for i in impls
            )
            if not has_validity:
                continue

        results.append(
            {
                "id": p["id"],
                "label": p.get("label", p["id"]),
                "cognitive_construct": p.get("cognitive_construct", ""),
                "description": p.get("description", ""),
                "topics": p.get("topics", []),
                "circuits_engaged": p.get("circuits_engaged", []),
                "species_available": [i["species"] for i in impls],
                "key_neural_signal": p.get("key_neural_signal", ""),
                "key_finding": p.get("key_finding", ""),
            }
        )
    return results


@router.get("/paradigms/{paradigm_id}")
def get_paradigm(paradigm_id: str) -> dict[str, Any]:
    """Full paradigm details with all species implementations."""
    data = _get_paradigms()
    for p in data.get("paradigms", []):
        if p["id"] == paradigm_id:
            return p
    raise HTTPException(status_code=404, detail=f"Paradigm '{paradigm_id}' not found")


# ── HCP structural connectivity endpoints ──────────────────────────────────

@router.get("/structural")
def get_structural_connections(
    region: str | None = Query(default=None),
    min_fa: float = Query(default=0.0, ge=0.0, le=1.0),
    human_specific: bool | None = Query(default=None),
) -> list[dict[str, Any]]:
    """HCP structural connectivity connections with optional filters."""
    data = _get_hcp()
    results = []
    for conn in data.get("structural_connections", []):
        if region and conn["source"] != region and conn["target"] != region:
            continue
        if conn.get("fa_estimate", 0) < min_fa:
            continue
        if human_specific is not None and conn.get("human_specific", False) != human_specific:
            continue
        results.append(conn)
    return results


@router.get("/structural/region/{region_id}")
def get_structural_neighbors(region_id: str) -> dict[str, Any]:
    """Get all structural connections for a given region."""
    data = _get_hcp()
    connections = [
        conn for conn in data.get("structural_connections", [])
        if conn["source"] == region_id or conn["target"] == region_id
    ]
    if not connections:
        raise HTTPException(
            status_code=404,
            detail=f"No structural connections found for region '{region_id}'"
        )
    return {
        "region_id": region_id,
        "connections": connections,
        "n_connections": len(connections),
    }


# ── Modality endpoints ─────────────────────────────────────────────────────

@router.get("/modalities")
def list_modalities(
    modality_class: str | None = Query(default=None, description="electrophysiology|hemodynamic|optical|structural|molecular|behavioral"),
    invasiveness: str | None = Query(default=None, description="invasive|semi_invasive|non_invasive|ex_vivo"),
    species: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    """All recording modalities with properties."""
    data = _get_modalities()
    results = []
    for mod in data.get("modalities", []):
        if modality_class and mod.get("modality_class") != modality_class:
            continue
        if invasiveness and mod.get("invasiveness") != invasiveness:
            continue
        if species and species not in mod.get("species", []):
            continue
        results.append(
            {
                "id": mod["id"],
                "label": mod.get("label", mod["id"]),
                "aliases": mod.get("aliases", []),
                "modality_class": mod.get("modality_class", ""),
                "signal_origin": mod.get("signal_origin", ""),
                "temporal_resolution_ms": mod.get("temporal_resolution_ms"),
                "spatial_resolution_mm": mod.get("spatial_resolution_mm"),
                "species": mod.get("species", []),
                "invasiveness": mod.get("invasiveness", ""),
                "cross_modal_value": mod.get("cross_modal_value", ""),
                "analysis_methods": mod.get("analysis_methods", []),
            }
        )
    return results


@router.get("/modalities/{modality_id}")
def get_modality(modality_id: str) -> dict[str, Any]:
    """Full modality details including frequency bands, limitations, and cross-modal value."""
    data = _get_modalities()
    for mod in data.get("modalities", []):
        if mod["id"] == modality_id:
            return mod
    raise HTTPException(status_code=404, detail=f"Modality '{modality_id}' not found")


@router.get("/modalities/cross-modal/pairs")
def get_cross_modal_pairs(
    modality: str | None = Query(default=None, description="Filter pairs involving this modality"),
    compatibility: str | None = Query(default=None, description="high|medium|low"),
) -> list[dict[str, Any]]:
    """Cross-modal compatibility pairs with integration methods."""
    data = _get_modalities()
    pairs = data.get("cross_modal_pairs", [])
    if modality:
        pairs = [p for p in pairs if p["modality_a"] == modality or p["modality_b"] == modality]
    if compatibility:
        pairs = [p for p in pairs if p.get("compatibility") == compatibility]
    return pairs


@router.get("/modalities/groups/summary")
def get_modality_groups() -> list[dict[str, Any]]:
    """Modality groups organized by temporal/spatial resolution tier."""
    data = _get_modalities()
    return data.get("modality_groups", [])
