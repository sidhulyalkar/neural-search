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
_ontology_regions: list[dict[str, Any]] | None = None
_topic_taxonomy: list[dict[str, Any]] | None = None


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


def _get_ontology_regions() -> list[dict[str, Any]]:
    global _ontology_regions
    if _ontology_regions is not None:
        return _ontology_regions
    try:
        import yaml  # type: ignore[import]
        path = REPO_ROOT / "data" / "ontology" / "brain_regions.yaml"
        if not path.exists():
            _ontology_regions = []
            return _ontology_regions
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        _ontology_regions = raw.get("brain_regions", [])
    except Exception:
        _ontology_regions = []
    return _ontology_regions


def _get_topic_taxonomy() -> list[dict[str, Any]]:
    global _topic_taxonomy
    if _topic_taxonomy is not None:
        return _topic_taxonomy
    try:
        import yaml  # type: ignore[import]
        path = REPO_ROOT / "data" / "ontology" / "topic_taxonomy.yaml"
        if not path.exists():
            _topic_taxonomy = []
            return _topic_taxonomy
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        _topic_taxonomy = raw.get("topics", [])
    except Exception:
        _topic_taxonomy = []
    return _topic_taxonomy


def _get_ontology_mapping() -> dict[str, int]:
    """Build ontology -> allen_id mapping from brain_regions.yaml on first call."""
    global _ontology_mapping
    if _ontology_mapping is not None:
        return _ontology_mapping

    regions = _get_ontology_regions()
    if not regions:
        _ontology_mapping = {}
        return _ontology_mapping

    from neural_search.ingestion.allen_structures import load_structures
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


@router.get("/regions/{region_id}/detail")
def get_region_detail(region_id: str) -> dict[str, Any]:
    """Rich detail for one ontology region: hierarchy, cross-refs, connected topics."""
    regions = _get_ontology_regions()
    region_by_id = {r["id"]: r for r in regions}

    region = region_by_id.get(region_id)
    if region is None:
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found in ontology")

    # Parent chain (breadcrumb)
    parents_chain: list[dict[str, str]] = []
    visited: set[str] = {region_id}
    current_parents = region.get("parents") or []
    for pid in current_parents:
        if pid in region_by_id and pid not in visited:
            p = region_by_id[pid]
            parents_chain.append({"id": pid, "label": p.get("label", pid)})
            visited.add(pid)
            # One more level up
            for gid in (p.get("parents") or []):
                if gid in region_by_id and gid not in visited:
                    g = region_by_id[gid]
                    parents_chain.insert(0, {"id": gid, "label": g.get("label", gid)})
                    visited.add(gid)

    # Children (regions whose parents list includes region_id)
    children = [
        {"id": r["id"], "label": r.get("label", r["id"])}
        for r in regions
        if region_id in (r.get("parents") or [])
    ]

    # Siblings (share at least one parent with this region)
    siblings: list[dict[str, str]] = []
    for pid in current_parents:
        for r in regions:
            if r["id"] != region_id and pid in (r.get("parents") or []):
                if not any(s["id"] == r["id"] for s in siblings):
                    siblings.append({"id": r["id"], "label": r.get("label", r["id"])})

    # Allen CCF cross-references
    atlas_refs = region.get("atlas_refs") or {}
    allen_ccf_id = atlas_refs.get("allen_ccf_mouse")
    allen_human_id = atlas_refs.get("allen_human")
    uberon_id = atlas_refs.get("uberon")
    waxholm_id = atlas_refs.get("waxholm_rat")

    allen_structure: dict[str, Any] | None = None
    if allen_ccf_id:
        try:
            allen_int = int(allen_ccf_id)
            for s in _get_mouse_structures():
                if s.get("allen_id") == allen_int:
                    allen_structure = {
                        "allen_id": allen_int,
                        "acronym": s.get("acronym", ""),
                        "color_hex": s.get("color_hex", ""),
                        "st_level": s.get("st_level"),
                    }
                    break
        except (ValueError, TypeError):
            pass

    # Connected topics (topics whose region list includes this region_id)
    topics = _get_topic_taxonomy()
    connected_topics = [
        {
            "id": t["id"],
            "label": t.get("label", t["id"]),
            "description": t.get("description", ""),
            "color": t.get("color", "#22d3ee"),
            "companion_topics": t.get("companion_topics", []),
        }
        for t in topics
        if region_id in (t.get("regions") or [])
    ]

    # Functional role — derive from topic memberships
    functional_systems = list({t["label"] for t in connected_topics})

    return {
        "id": region_id,
        "label": region.get("label", region_id),
        "aliases": region.get("aliases") or [],
        "is_strict": region.get("strict", False),
        "parents": parents_chain,
        "children": children,
        "siblings": siblings[:8],  # cap for UI
        "atlas_refs": {
            "allen_ccf_mouse": allen_ccf_id,
            "allen_human": allen_human_id,
            "uberon": uberon_id,
            "waxholm_rat": waxholm_id,
        },
        "allen_structure": allen_structure,
        "connected_topics": connected_topics,
        "functional_systems": functional_systems,
    }


@router.get("/circuits")
def get_circuits() -> list[dict[str, Any]]:
    """Return major functional circuits with their region chains and topic links."""
    return [
        {
            "id": "visual_pathway",
            "label": "Visual Processing Pathway",
            "description": "Primary visual information flow from retina through thalamus to cortex",
            "color": "#6366f1",
            "regions": [
                {"id": "lateral_geniculate", "label": "Lateral Geniculate Nucleus", "role": "relay"},
                {"id": "v1", "label": "Primary Visual Cortex (V1)", "role": "primary"},
                {"id": "v2", "label": "V2", "role": "secondary"},
                {"id": "v4", "label": "V4 — Color & Shape", "role": "ventral"},
                {"id": "area_mt", "label": "MT — Motion", "role": "dorsal"},
                {"id": "inferior_temporal_cortex", "label": "Inferior Temporal (IT)", "role": "object recognition"},
            ],
            "topics": ["visual_processing", "attention_and_salience"],
        },
        {
            "id": "hippocampal_circuit",
            "label": "Hippocampal Memory Circuit",
            "description": "Papez circuit for episodic memory encoding and spatial navigation",
            "color": "#10b981",
            "regions": [
                {"id": "entorhinal_cortex", "label": "Entorhinal Cortex", "role": "gateway"},
                {"id": "dentate_gyrus", "label": "Dentate Gyrus", "role": "pattern separation"},
                {"id": "ca3", "label": "CA3", "role": "pattern completion"},
                {"id": "ca1", "label": "CA1", "role": "output"},
                {"id": "subiculum", "label": "Subiculum", "role": "output relay"},
                {"id": "hippocampus", "label": "Hippocampus", "role": "integration"},
            ],
            "topics": ["episodic_memory", "spatial_navigation"],
        },
        {
            "id": "basal_ganglia_loop",
            "label": "Basal Ganglia – Cortex Loop",
            "description": "Action selection and reward-based learning via cortico-striatal loop",
            "color": "#f59e0b",
            "regions": [
                {"id": "prefrontal_cortex", "label": "Prefrontal Cortex", "role": "goal"},
                {"id": "striatum", "label": "Striatum", "role": "action gating"},
                {"id": "nucleus_accumbens", "label": "Nucleus Accumbens", "role": "reward"},
                {"id": "globus_pallidus", "label": "Globus Pallidus", "role": "inhibition"},
                {"id": "thalamus", "label": "Thalamus", "role": "relay back"},
                {"id": "motor_cortex", "label": "Motor Cortex", "role": "execution"},
            ],
            "topics": ["reward_learning", "decision_making", "motor_control"],
        },
        {
            "id": "fear_circuit",
            "label": "Fear & Threat Circuit",
            "description": "Threat detection and conditioned fear via amygdala–PFC interactions",
            "color": "#ef4444",
            "regions": [
                {"id": "amygdala", "label": "Amygdala", "role": "threat detection"},
                {"id": "lateral_amygdala", "label": "Lateral Amygdala", "role": "CS-US association"},
                {"id": "central_amygdala", "label": "Central Amygdala", "role": "fear output"},
                {"id": "anterior_cingulate_cortex", "label": "Anterior Cingulate", "role": "appraisal"},
                {"id": "prefrontal_cortex", "label": "vmPFC", "role": "extinction"},
                {"id": "hypothalamus", "label": "Hypothalamus", "role": "autonomic response"},
            ],
            "topics": ["fear_and_anxiety", "emotional_processing"],
        },
        {
            "id": "default_mode",
            "label": "Default Mode & Working Memory",
            "description": "Prefrontal–parietal networks supporting working memory and executive control",
            "color": "#8b5cf6",
            "regions": [
                {"id": "dlpfc", "label": "DLPFC", "role": "maintenance"},
                {"id": "anterior_cingulate_cortex", "label": "ACC", "role": "monitoring"},
                {"id": "posterior_parietal_cortex", "label": "Posterior Parietal", "role": "attention"},
                {"id": "hippocampus", "label": "Hippocampus", "role": "memory binding"},
                {"id": "thalamus", "label": "MD Thalamus", "role": "gating"},
            ],
            "topics": ["working_memory", "cognitive_control", "attention_and_salience"],
        },
        {
            "id": "motor_circuit",
            "label": "Motor Control Circuit",
            "description": "Voluntary movement generation from motor cortex through cerebellum",
            "color": "#22d3ee",
            "regions": [
                {"id": "supplementary_motor_area", "label": "SMA", "role": "planning"},
                {"id": "premotor_cortex", "label": "Premotor Cortex", "role": "preparation"},
                {"id": "motor_cortex", "label": "Primary Motor (M1)", "role": "execution"},
                {"id": "cerebellum", "label": "Cerebellum", "role": "coordination & timing"},
                {"id": "substantia_nigra", "label": "Substantia Nigra", "role": "dopamine modulation"},
            ],
            "topics": ["motor_control", "motor_learning", "sensorimotor_integration"],
        },
        {
            "id": "neuromodulatory",
            "label": "Neuromodulatory Systems",
            "description": "Subcortical nuclei broadcasting arousal, attention, and mood signals",
            "color": "#f97316",
            "regions": [
                {"id": "locus_coeruleus", "label": "Locus Coeruleus", "role": "norepinephrine — arousal"},
                {"id": "raphe_nucleus", "label": "Raphe Nuclei", "role": "serotonin — mood"},
                {"id": "ventral_tegmental_area", "label": "VTA", "role": "dopamine — reward"},
                {"id": "nucleus_accumbens", "label": "Nucleus Accumbens", "role": "dopamine target"},
                {"id": "thalamus", "label": "Thalamus", "role": "arousal gating"},
            ],
            "topics": ["neuromodulation", "reward_learning", "sleep_and_oscillations"],
        },
        {
            "id": "auditory_speech",
            "label": "Auditory & Language Circuit",
            "description": "Sound processing from cochlea to cortical language areas",
            "color": "#06b6d4",
            "regions": [
                {"id": "medial_geniculate", "label": "Medial Geniculate (MGN)", "role": "thalamic relay"},
                {"id": "auditory_cortex", "label": "Primary Auditory (A1)", "role": "tonotopy"},
                {"id": "superior_temporal_gyrus", "label": "Superior Temporal Gyrus", "role": "speech"},
                {"id": "inferior_frontal_gyrus", "label": "Inferior Frontal (Broca)", "role": "production"},
                {"id": "inferior_colliculus", "label": "Inferior Colliculus", "role": "subcortical processing"},
            ],
            "topics": ["auditory_processing", "language_and_speech"],
        },
    ]
