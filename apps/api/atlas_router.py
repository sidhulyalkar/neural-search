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
        # ── Sensory & Interoceptive ───────────────────────────────────────────
        {
            "id": "pain_somatosensory",
            "label": "Pain & Somatosensory Circuit",
            "description": "Ascending nociceptive pathway from spinal cord to conscious pain perception",
            "color": "#dc2626",
            "regions": [
                {"id": "somatosensory_cortex", "label": "S1 — Primary Somatosensory", "role": "tactile localization"},
                {"id": "s2", "label": "S2 — Secondary Somatosensory", "role": "pain intensity"},
                {"id": "thalamus", "label": "Ventral Posterior Thalamus", "role": "spinothalamic relay"},
                {"id": "anterior_cingulate_cortex", "label": "Anterior Cingulate (ACC)", "role": "affective pain"},
                {"id": "insular_cortex", "label": "Insular Cortex", "role": "interoception & salience"},
                {"id": "periaqueductal_gray", "label": "Periaqueductal Gray (PAG)", "role": "descending analgesia"},
            ],
            "topics": ["somatosensory_processing"],
        },
        {
            "id": "interoception",
            "label": "Interoception & Body Awareness",
            "description": "Neural mapping of internal body states — the basis of hunger, thirst, and emotion",
            "color": "#be185d",
            "regions": [
                {"id": "insular_cortex", "label": "Insular Cortex", "role": "primary interoceptive cortex"},
                {"id": "anterior_cingulate_cortex", "label": "ACC", "role": "salience & conflict"},
                {"id": "medial_prefrontal_cortex", "label": "vmPFC", "role": "body-state integration"},
                {"id": "hypothalamus", "label": "Hypothalamus", "role": "homeostatic regulation"},
                {"id": "amygdala", "label": "Amygdala", "role": "visceral-emotional binding"},
                {"id": "periaqueductal_gray", "label": "PAG", "role": "autonomic output"},
            ],
            "topics": ["emotional_processing", "fear_and_anxiety"],
        },
        {
            "id": "olfactory_circuit",
            "label": "Olfactory & Chemosensory Circuit",
            "description": "The only sensory system with direct cortical access, bypassing thalamic relay",
            "color": "#65a30d",
            "regions": [
                {"id": "olfactory_bulb", "label": "Olfactory Bulb", "role": "initial odor processing"},
                {"id": "piriform_cortex", "label": "Piriform Cortex", "role": "odor identification"},
                {"id": "entorhinal_cortex", "label": "Entorhinal Cortex", "role": "odor-memory bridge"},
                {"id": "amygdala", "label": "Amygdala", "role": "hedonic valence"},
                {"id": "orbitofrontal_cortex", "label": "OFC", "role": "flavor & reward integration"},
                {"id": "hippocampus", "label": "Hippocampus", "role": "olfactory memory"},
            ],
            "topics": ["episodic_memory", "emotional_processing"],
        },
        # ── Cognitive & Higher-Order ──────────────────────────────────────────
        {
            "id": "attention_salience",
            "label": "Attention & Salience Network",
            "description": "Top-down and bottom-up attention control via frontoparietal and salience networks",
            "color": "#7c3aed",
            "regions": [
                {"id": "anterior_cingulate_cortex", "label": "ACC / dACC", "role": "conflict detection"},
                {"id": "insular_cortex", "label": "Anterior Insula", "role": "salience detection"},
                {"id": "posterior_parietal_cortex", "label": "Posterior Parietal (IPS)", "role": "spatial attention"},
                {"id": "dlpfc", "label": "DLPFC", "role": "top-down control"},
                {"id": "superior_colliculus", "label": "Superior Colliculus", "role": "orienting reflexes"},
                {"id": "thalamus", "label": "Pulvinar", "role": "attentional gating"},
            ],
            "topics": ["attention_and_salience", "cognitive_control"],
        },
        {
            "id": "social_cognition",
            "label": "Social Cognition Circuit",
            "description": "Theory of mind, empathy, and social reward — the neural basis of social behavior",
            "color": "#d946ef",
            "regions": [
                {"id": "medial_prefrontal_cortex", "label": "mPFC", "role": "self/other representation"},
                {"id": "temporal_cortex", "label": "Temporoparietal Junction (TPJ)", "role": "perspective taking"},
                {"id": "posterior_parietal_cortex", "label": "Posterior STS", "role": "biological motion"},
                {"id": "amygdala", "label": "Amygdala", "role": "social threat & trust"},
                {"id": "anterior_cingulate_cortex", "label": "ACC", "role": "social pain / empathy"},
                {"id": "nucleus_accumbens", "label": "Nucleus Accumbens", "role": "social reward"},
                {"id": "orbitofrontal_cortex", "label": "OFC", "role": "social valuation"},
            ],
            "topics": ["social_behavior", "emotional_processing"],
        },
        {
            "id": "executive_control",
            "label": "Executive Control & Cognitive Flexibility",
            "description": "Prefrontal-parietal circuits enabling rule learning, task switching, and inhibition",
            "color": "#0ea5e9",
            "regions": [
                {"id": "dlpfc", "label": "DLPFC", "role": "rule representation & maintenance"},
                {"id": "vlpfc", "label": "vlPFC / IFG", "role": "response inhibition"},
                {"id": "anterior_cingulate_cortex", "label": "ACC", "role": "error monitoring"},
                {"id": "posterior_parietal_cortex", "label": "Posterior Parietal", "role": "task set"},
                {"id": "striatum", "label": "Caudate", "role": "habit vs. goal-directed"},
                {"id": "thalamus", "label": "MD Thalamus", "role": "prefrontal gating"},
            ],
            "topics": ["executive_function", "cognitive_control", "working_memory"],
        },
        # ── Memory Systems ────────────────────────────────────────────────────
        {
            "id": "spatial_navigation",
            "label": "Spatial Navigation & Place Coding",
            "description": "Grid cells, place cells, and head-direction cells — the brain's GPS system",
            "color": "#0d9488",
            "regions": [
                {"id": "entorhinal_cortex", "label": "Entorhinal Cortex (Grid Cells)", "role": "metric mapping"},
                {"id": "hippocampus", "label": "Hippocampus (Place Cells)", "role": "location encoding"},
                {"id": "ca1", "label": "CA1", "role": "sequence & time"},
                {"id": "subiculum", "label": "Subiculum", "role": "boundary detection"},
                {"id": "posterior_parietal_cortex", "label": "Posterior Parietal", "role": "egocentric navigation"},
                {"id": "prefrontal_cortex", "label": "PFC", "role": "goal-directed navigation"},
                {"id": "retrosplenial_cortex", "label": "Retrosplenial Cortex", "role": "landmark integration"},
            ],
            "topics": ["spatial_navigation", "episodic_memory"],
        },
        # ── Oscillations & State ──────────────────────────────────────────────
        {
            "id": "sleep_oscillations",
            "label": "Sleep, Memory Consolidation & Oscillations",
            "description": "Slow-wave sleep replay and REM consolidation via thalamocortical and hippocampal loops",
            "color": "#4338ca",
            "regions": [
                {"id": "thalamus", "label": "Thalamus", "role": "sleep spindle generation"},
                {"id": "hippocampus", "label": "Hippocampus", "role": "sharp-wave ripples & replay"},
                {"id": "prefrontal_cortex", "label": "Prefrontal Cortex", "role": "slow oscillation origin"},
                {"id": "hypothalamus", "label": "Hypothalamus (SCN)", "role": "circadian timing"},
                {"id": "locus_coeruleus", "label": "Locus Coeruleus", "role": "REM-off noradrenergic"},
                {"id": "raphe_nucleus", "label": "Raphe Nuclei", "role": "REM-off serotonergic"},
                {"id": "cerebellum", "label": "Cerebellum", "role": "timing & coordination"},
            ],
            "topics": ["sleep_and_oscillations", "neural_synchrony", "episodic_memory"],
        },
        {
            "id": "spectral_dynamics",
            "label": "Neural Synchrony & Spectral Dynamics",
            "description": "Frequency-band oscillations (theta, gamma, beta) coordinating distributed computation",
            "color": "#0891b2",
            "regions": [
                {"id": "hippocampus", "label": "Hippocampus", "role": "theta (4–8 Hz) generation"},
                {"id": "prefrontal_cortex", "label": "PFC", "role": "theta-gamma coupling"},
                {"id": "visual_cortex", "label": "Visual Cortex", "role": "gamma (30–80 Hz)"},
                {"id": "motor_cortex", "label": "Motor Cortex", "role": "beta (13–30 Hz)"},
                {"id": "thalamus", "label": "Thalamus", "role": "spindles & alpha (8–12 Hz)"},
                {"id": "basal_ganglia", "label": "Basal Ganglia", "role": "beta desynchronization"},
            ],
            "topics": ["neural_synchrony", "spectral_dynamics", "sleep_and_oscillations"],
        },
        # ── Neuromodulatory & Stress ──────────────────────────────────────────
        {
            "id": "stress_hpa",
            "label": "Stress & HPA Axis",
            "description": "Hypothalamic-pituitary-adrenal axis — the brain's central stress response system",
            "color": "#b45309",
            "regions": [
                {"id": "amygdala", "label": "Amygdala (CeA)", "role": "threat appraisal → CRH"},
                {"id": "hypothalamus", "label": "Hypothalamus (PVN)", "role": "CRH release"},
                {"id": "anterior_cingulate_cortex", "label": "ACC", "role": "cognitive appraisal"},
                {"id": "hippocampus", "label": "Hippocampus", "role": "glucocorticoid feedback"},
                {"id": "prefrontal_cortex", "label": "PFC", "role": "top-down stress regulation"},
                {"id": "locus_coeruleus", "label": "Locus Coeruleus", "role": "sympathetic arousal"},
            ],
            "topics": ["fear_and_anxiety", "neuromodulation"],
        },
        {
            "id": "reward_addiction",
            "label": "Reward, Motivation & Addiction",
            "description": "Mesolimbic dopamine circuit underlying drug-seeking, craving, and compulsion",
            "color": "#ea580c",
            "regions": [
                {"id": "ventral_tegmental_area", "label": "VTA", "role": "dopamine origin"},
                {"id": "nucleus_accumbens", "label": "Nucleus Accumbens (NAcc)", "role": "reward prediction error"},
                {"id": "prefrontal_cortex", "label": "PFC", "role": "cue-induced craving"},
                {"id": "amygdala", "label": "Amygdala (BLA)", "role": "cue-reward association"},
                {"id": "orbitofrontal_cortex", "label": "OFC", "role": "outcome expectation"},
                {"id": "striatum", "label": "Dorsal Striatum", "role": "habit formation"},
                {"id": "hippocampus", "label": "Hippocampus", "role": "context-cued relapse"},
            ],
            "topics": ["reward_learning", "value_computation", "decision_making"],
        },
        # ── Development & Plasticity ──────────────────────────────────────────
        {
            "id": "plasticity_circuit",
            "label": "Synaptic Plasticity & Learning",
            "description": "Hebbian and non-Hebbian mechanisms of long-term potentiation and depression",
            "color": "#15803d",
            "regions": [
                {"id": "hippocampus", "label": "Hippocampus", "role": "LTP — NMDA-dependent"},
                {"id": "amygdala", "label": "Amygdala", "role": "fear conditioning LTP"},
                {"id": "cerebellum", "label": "Cerebellum", "role": "LTD — motor error"},
                {"id": "striatum", "label": "Striatum", "role": "dopamine-gated plasticity"},
                {"id": "motor_cortex", "label": "Motor Cortex", "role": "skill acquisition"},
                {"id": "prefrontal_cortex", "label": "PFC", "role": "cognitive flexibility"},
            ],
            "topics": ["development_and_plasticity", "motor_learning", "episodic_memory"],
        },
        # ── Hippocampal Subcircuits ───────────────────────────────────────────
        {
            "id": "trisynaptic_loop",
            "label": "Hippocampal Trisynaptic Loop",
            "description": "The canonical EC→DG→CA3→CA1 subcircuit — the engine of episodic memory encoding",
            "color": "#059669",
            "subcircuit_of": "hippocampal_circuit",
            "scale": "microcircuit",
            "regions": [
                {"id": "entorhinal_cortex", "label": "Entorhinal Cortex (Layer II)", "role": "perforant path input — pattern separation trigger"},
                {"id": "dentate_gyrus", "label": "Dentate Gyrus (DG)", "role": "pattern separation via sparse coding"},
                {"id": "ca3", "label": "CA3", "role": "mossy fiber input + autoassociation (pattern completion)"},
                {"id": "ca1", "label": "CA1", "role": "Schaffer collateral input + mismatch detection"},
                {"id": "subiculum", "label": "Subiculum", "role": "main hippocampal output → cortex"},
                {"id": "entorhinal_cortex", "label": "Entorhinal Cortex (Layer V)", "role": "output return — memory consolidation"},
            ],
            "topics": ["episodic_memory", "spatial_navigation", "neural_synchrony"],
        },
        {
            "id": "sharp_wave_ripple",
            "label": "Sharp-Wave Ripple Circuit",
            "description": "CA3 → CA1 replay events during rest/sleep — the mechanism of offline memory consolidation",
            "color": "#10b981",
            "subcircuit_of": "hippocampal_circuit",
            "scale": "microcircuit",
            "regions": [
                {"id": "ca3", "label": "CA3 (Excitatory burst)", "role": "initiates sharp-wave via recurrent excitation"},
                {"id": "ca1", "label": "CA1 (Ripple generator ~150–250 Hz)", "role": "fast oscillation via PV interneuron network"},
                {"id": "entorhinal_cortex", "label": "Entorhinal Cortex", "role": "receives replayed sequences"},
                {"id": "prefrontal_cortex", "label": "PFC", "role": "coordinated cortical replay"},
                {"id": "thalamus", "label": "Nucleus Reuniens", "role": "hippocampal-PFC synchronization gate"},
            ],
            "topics": ["sleep_and_oscillations", "episodic_memory", "neural_synchrony"],
        },
        # ── Basal Ganglia Subcircuits ─────────────────────────────────────────
        {
            "id": "bg_direct_pathway",
            "label": "Basal Ganglia: Direct Pathway (Go)",
            "description": "Striatal D1-MSNs inhibit GPi/SNr, releasing thalamus — the action facilitation pathway",
            "color": "#d97706",
            "subcircuit_of": "basal_ganglia_loop",
            "scale": "microcircuit",
            "regions": [
                {"id": "motor_cortex", "label": "Motor/Premotor Cortex", "role": "corticostriatal glutamate input"},
                {"id": "striatum", "label": "Striatum (D1-MSN)", "role": "GABA output — inhibits GPi/SNr (Go)"},
                {"id": "substantia_nigra", "label": "GPi / SNr", "role": "tonic GABA inhibition of thalamus, released by D1-MSN"},
                {"id": "thalamus", "label": "VA/VL Thalamus", "role": "disinhibited — activates cortex"},
                {"id": "motor_cortex", "label": "Motor Cortex Output", "role": "movement execution"},
            ],
            "topics": ["motor_learning", "cognitive_control", "spectral_dynamics"],
        },
        {
            "id": "bg_indirect_pathway",
            "label": "Basal Ganglia: Indirect Pathway (NoGo)",
            "description": "Striatal D2-MSNs disinhibit STN, driving GPi/SNr inhibition of thalamus — action suppression",
            "color": "#b45309",
            "subcircuit_of": "basal_ganglia_loop",
            "scale": "microcircuit",
            "regions": [
                {"id": "striatum", "label": "Striatum (D2-MSN)", "role": "GABA → GPe (NoGo signal)"},
                {"id": "globus_pallidus_ext", "label": "GPe (External)", "role": "inhibits STN — removed by D2-MSN"},
                {"id": "subthalamic_nucleus", "label": "STN", "role": "glutamate burst → GPi/SNr (stop signal)"},
                {"id": "substantia_nigra", "label": "GPi / SNr", "role": "strong GABA inhibition of thalamus"},
                {"id": "thalamus", "label": "VA/VL Thalamus", "role": "suppressed — action cancelled"},
            ],
            "topics": ["cognitive_control", "executive_function", "motor_learning"],
        },
        {
            "id": "bg_hyperdirect_pathway",
            "label": "Basal Ganglia: Hyperdirect Pathway (Fast Stop)",
            "description": "Cortex bypasses striatum to drive STN directly — the brain's emergency brake for impulsive action",
            "color": "#92400e",
            "subcircuit_of": "basal_ganglia_loop",
            "scale": "microcircuit",
            "regions": [
                {"id": "prefrontal_cortex", "label": "rIFG / Pre-SMA (Stop signal)", "role": "cortical stop command"},
                {"id": "subthalamic_nucleus", "label": "STN (rapid glutamate burst)", "role": "bypasses striatum — arrives ~10 ms faster"},
                {"id": "substantia_nigra", "label": "GPi / SNr", "role": "broad inhibition cancels competing actions"},
                {"id": "thalamus", "label": "Thalamus", "role": "broadly suppressed"},
            ],
            "topics": ["cognitive_control", "executive_function", "attention_and_salience"],
        },
        # ── Neuromodulatory Subcircuits ───────────────────────────────────────
        {
            "id": "dopamine_mesolimbic",
            "label": "Dopamine: Mesolimbic Pathway",
            "description": "VTA → NAcc dopamine projection encoding reward prediction error (RPE) — Schultz circuit",
            "color": "#f59e0b",
            "scale": "subcircuit",
            "regions": [
                {"id": "ventral_tegmental_area", "label": "VTA (A10 dopamine neurons)", "role": "burst: reward > expectation; pause: omission"},
                {"id": "nucleus_accumbens", "label": "NAcc Core", "role": "RPE signal gates action"},
                {"id": "nucleus_accumbens", "label": "NAcc Shell", "role": "novelty, general motivation"},
                {"id": "amygdala", "label": "Amygdala (BLA)", "role": "CS-US association with DA modulation"},
                {"id": "hippocampus", "label": "Hippocampus", "role": "context gating of DA release"},
                {"id": "ventral_pallidum", "label": "Ventral Pallidum", "role": "hedonic hotspot — mu-opioid"},
            ],
            "topics": ["reward_learning", "decision_making", "spectral_dynamics"],
        },
        {
            "id": "dopamine_nigrostriatal",
            "label": "Dopamine: Nigrostriatal Pathway",
            "description": "SNc → dorsal striatum dopamine — motor initiation and habit formation",
            "color": "#d97706",
            "scale": "subcircuit",
            "regions": [
                {"id": "substantia_nigra", "label": "SNc (A9 dopamine neurons)", "role": "tonic DA maintains motor readiness"},
                {"id": "striatum", "label": "Dorsal Striatum (caudate/putamen)", "role": "DA gates D1/D2 corticostriatal plasticity"},
                {"id": "motor_cortex", "label": "Motor Cortex", "role": "corticostriatal input, DA modulates LTP"},
            ],
            "topics": ["motor_learning", "development_and_plasticity", "reward_learning"],
        },
        {
            "id": "lc_norepinephrine",
            "label": "Locus Coeruleus — Norepinephrine System",
            "description": "LC-NE modulates gain and signal-to-noise ratio across the entire cortex — arousal, attention, surprise",
            "color": "#0ea5e9",
            "scale": "neuromodulatory",
            "regions": [
                {"id": "locus_coeruleus", "label": "Locus Coeruleus (LC)", "role": "sole NE source for forebrain; tonic = mode, phasic = signal"},
                {"id": "prefrontal_cortex", "label": "PFC", "role": "NE enhances task-relevant representations"},
                {"id": "anterior_cingulate_cortex", "label": "ACC", "role": "NE gates conflict signals"},
                {"id": "hippocampus", "label": "Hippocampus", "role": "NE gates synaptic tagging and memory consolidation"},
                {"id": "amygdala", "label": "Amygdala", "role": "NE amplifies emotional memory encoding"},
                {"id": "cerebellum", "label": "Cerebellum", "role": "NE modulates timing/error signals"},
            ],
            "topics": ["attention_and_salience", "neural_synchrony", "sleep_and_oscillations"],
        },
        {
            "id": "cholinergic_basal_forebrain",
            "label": "Cholinergic Basal Forebrain System",
            "description": "BF-ACh broadcasts attentional gating and modulates cortical excitability — critical for memory formation",
            "color": "#7c3aed",
            "scale": "neuromodulatory",
            "regions": [
                {"id": "nucleus_basalis", "label": "Nucleus Basalis of Meynert (NBM)", "role": "main ACh source for neocortex"},
                {"id": "medial_septum", "label": "Medial Septum / Diagonal Band", "role": "ACh + GABA to hippocampus — drives theta"},
                {"id": "hippocampus", "label": "Hippocampus", "role": "ACh enables theta rhythm and LTP induction"},
                {"id": "prefrontal_cortex", "label": "PFC", "role": "ACh enhances attention — degraded in aging/AD"},
                {"id": "sensory_cortex", "label": "Sensory Cortices", "role": "ACh sharpens sensory tuning"},
            ],
            "topics": ["attention_and_salience", "development_and_plasticity", "episodic_memory"],
        },
        # ── Unique Circuits ───────────────────────────────────────────────────
        {
            "id": "claustrum_binding",
            "label": "Claustrum: Global Binding Circuit",
            "description": "Thin sheet of neurons receiving from all cortical areas and projecting back — proposed as the integration hub for conscious awareness",
            "color": "#a78bfa",
            "scale": "network",
            "regions": [
                {"id": "claustrum", "label": "Claustrum", "role": "dense bidirectional connectivity with entire cortex"},
                {"id": "prefrontal_cortex", "label": "PFC", "role": "strongest driver — attention/executive gating"},
                {"id": "visual_cortex", "label": "Visual Cortex", "role": "salience detection, figure-ground"},
                {"id": "insular_cortex", "label": "Insular Cortex", "role": "interoceptive-affective integration"},
                {"id": "anterior_cingulate_cortex", "label": "ACC", "role": "conflict signals fed through claustrum"},
            ],
            "topics": ["attention_and_salience", "neural_synchrony"],
        },
        {
            "id": "habenulo_interpeduncular",
            "label": "Lateral Habenula: Anti-Reward Circuit",
            "description": "LHb encodes punishment and negative prediction error — the brain's 'anti-VTA' suppressing dopamine on failure",
            "color": "#ef4444",
            "scale": "subcircuit",
            "regions": [
                {"id": "lateral_habenula", "label": "Lateral Habenula (LHb)", "role": "burst on negative PE → inhibits VTA/DRN"},
                {"id": "ventral_tegmental_area", "label": "VTA (RMTg/tail)", "role": "RMTg GABAergic inhibition of DA neurons"},
                {"id": "raphe_nucleus", "label": "Dorsal Raphe (DRN)", "role": "5-HT suppression on punishment"},
                {"id": "lateral_hypothalamus", "label": "Lateral Hypothalamus → LHb", "role": "reward expectation input"},
                {"id": "orbitofrontal_cortex", "label": "OFC → LHb", "role": "negative outcome prediction"},
            ],
            "topics": ["reward_learning", "fear_and_anxiety", "decision_making"],
        },
        {
            "id": "papez_circuit",
            "label": "Papez Circuit — Limbic Memory Loop",
            "description": "The original limbic circuit (1937): hippocampus → mammillary bodies → anterior thalamus → cingulate → entorhinal — emotional memory",
            "color": "#8b5cf6",
            "scale": "network",
            "regions": [
                {"id": "hippocampus", "label": "Hippocampus (Subiculum)", "role": "episodic memory output"},
                {"id": "mammillary_bodies", "label": "Mammillary Bodies", "role": "spatial/directional memory relay"},
                {"id": "thalamus", "label": "Anterior Thalamic Nuclei", "role": "head-direction signal, spatial context"},
                {"id": "anterior_cingulate_cortex", "label": "Cingulate Cortex", "role": "emotional coloring of memories"},
                {"id": "entorhinal_cortex", "label": "Entorhinal Cortex", "role": "hippocampal re-entry — consolidation loop"},
            ],
            "topics": ["episodic_memory", "spatial_navigation", "emotional_processing"],
        },
        {
            "id": "gut_brain_axis",
            "label": "Gut–Brain Axis (Vagal Pathway)",
            "description": "80% afferent signaling from gut → brainstem → forebrain via vagus nerve — the enteric nervous system's influence on mood and cognition",
            "color": "#16a34a",
            "scale": "systems",
            "regions": [
                {"id": "enteric_nervous_system", "label": "Enteric Nervous System / Gut", "role": "produces 90% of body's serotonin; signals gut state"},
                {"id": "vagus_nerve", "label": "Vagus Nerve (CN X)", "role": "fast afferent highway — gut → brainstem in ~100 ms"},
                {"id": "nts", "label": "Nucleus Tractus Solitarius (NTS)", "role": "first brainstem relay — visceral integration"},
                {"id": "locus_coeruleus", "label": "Locus Coeruleus", "role": "NTS → LC activates arousal/stress"},
                {"id": "raphe_nucleus", "label": "Dorsal Raphe", "role": "NTS → DRN modulates mood via 5-HT"},
                {"id": "amygdala", "label": "Amygdala", "role": "gut signals modulate fear and emotional state"},
                {"id": "prefrontal_cortex", "label": "PFC / Insular Cortex", "role": "conscious interoceptive awareness"},
            ],
            "topics": ["interoception", "emotional_processing", "neuromodulation"],
        },
        {
            "id": "corticostriatal_beta",
            "label": "Cortico-Striatal Beta Oscillation Circuit",
            "description": "Synchronized beta (13–30 Hz) between PFC/motor cortex and striatum — indexes reward certainty, habit stability, and motor readiness (Hulyalkar et al., 2025)",
            "color": "#f59e0b",
            "scale": "oscillatory",
            "regions": [
                {"id": "prefrontal_cortex", "label": "OFC / PFC (Layer 5)", "role": "cortical beta generator — outcome expectation"},
                {"id": "striatum", "label": "Dorsal Striatum", "role": "beta phase-locks to cortex on high-certainty trials"},
                {"id": "nucleus_accumbens", "label": "Ventral Striatum / NAcc", "role": "beta amplitude correlates with reward magnitude"},
                {"id": "thalamus", "label": "Thalamo-Striatal Projection", "role": "synchronization conduit"},
                {"id": "substantia_nigra", "label": "SNc (DA modulation)", "role": "DA release modulates beta amplitude"},
            ],
            "topics": ["spectral_dynamics", "reward_learning", "cognitive_control"],
        },
        {
            "id": "cerebellar_purkinje",
            "label": "Cerebellar Purkinje Cell Microcircuit",
            "description": "GC → PF → PC inhibited by CF (climbing fiber) error signal — the canonical supervised learning circuit of the brain",
            "color": "#22d3ee",
            "subcircuit_of": "motor_circuit",
            "scale": "microcircuit",
            "regions": [
                {"id": "inferior_olive", "label": "Inferior Olive", "role": "climbing fiber — teacher signal encoding movement error"},
                {"id": "granule_cells", "label": "Granule Cells (GC)", "role": "sparse coding of context via parallel fibers (PF)"},
                {"id": "purkinje_cells", "label": "Purkinje Cells (PC)", "role": "sole output — LTD at PF→PC synapse when CF fires"},
                {"id": "deep_cerebellar_nuclei", "label": "Deep Cerebellar Nuclei (DCN)", "role": "final output — thalamus and brainstem"},
                {"id": "thalamus", "label": "VL Thalamus", "role": "DCN → thalamus → motor cortex (prediction update)"},
            ],
            "topics": ["motor_learning", "development_and_plasticity", "neural_synchrony"],
        },
        {
            "id": "thalamocortical_sensory",
            "label": "Thalamo-Cortical Sensory Relay & Gating",
            "description": "Specific thalamic nuclei relay and transform sensory input; thalamic reticular nucleus gates what reaches cortex",
            "color": "#6366f1",
            "scale": "network",
            "regions": [
                {"id": "thalamus", "label": "Primary Relay (VPM, MGN, LGN)", "role": "sensory-specific relay to L4 cortex"},
                {"id": "thalamic_reticular_nucleus", "label": "Thalamic Reticular Nucleus (TRN)", "role": "GABA gate — selectively inhibits relay nuclei"},
                {"id": "sensory_cortex", "label": "Primary Sensory Cortex (L4)", "role": "initial cortical processing"},
                {"id": "prefrontal_cortex", "label": "PFC → TRN", "role": "top-down attention controls gating"},
                {"id": "brainstem", "label": "Brainstem Modulatory Input", "role": "cholinergic/noradrenergic unlock TRN during arousal"},
            ],
            "topics": ["attention_and_salience", "sleep_and_oscillations", "spectral_dynamics"],
        },
        {
            "id": "zona_incerta",
            "label": "Zona Incerta: Action Competition Suppressor",
            "description": "Subthalamic zone that tonically suppresses competing sensorimotor programs — releases specific actions via cortical disinhibition",
            "color": "#64748b",
            "scale": "subcircuit",
            "regions": [
                {"id": "zona_incerta", "label": "Zona Incerta (ZI)", "role": "tonic GABA suppression of superior colliculus, thalamus"},
                {"id": "motor_cortex", "label": "Motor / Premotor Cortex", "role": "selected action removes ZI suppression"},
                {"id": "superior_colliculus", "label": "Superior Colliculus", "role": "ZI-disinhibited → orienting / gaze shift"},
                {"id": "thalamus", "label": "Parafascicular Thalamus", "role": "ZI-disinhibited → motor action gating"},
                {"id": "basal_ganglia", "label": "Striatum (indirect path input)", "role": "receives ZI disinhibition signal"},
            ],
            "topics": ["motor_learning", "attention_and_salience", "cognitive_control"],
        },
        {
            "id": "arcuate_language",
            "label": "Arcuate Fasciculus: Human Language Circuit",
            "description": "White-matter superhighway uniquely expanded in humans connecting Wernicke's area (comprehension) to Broca's area (production)",
            "color": "#db2777",
            "scale": "network",
            "human_specific": True,
            "regions": [
                {"id": "wernickes_area", "label": "Wernicke's Area (pSTG / SMG)", "role": "speech comprehension — phonological decoding"},
                {"id": "arcuate_fasciculus", "label": "Arcuate Fasciculus (AF)", "role": "direct dorsal route — phonological working memory"},
                {"id": "inferior_frontal_gyrus", "label": "Broca's Area (IFG, BA44/45)", "role": "syntactic processing + speech production"},
                {"id": "premotor_cortex", "label": "Premotor / SMA", "role": "speech motor program"},
                {"id": "angular_gyrus", "label": "Angular Gyrus", "role": "semantic integration — reading/writing junction"},
            ],
            "topics": ["language_and_speech", "working_memory", "executive_function"],
        },
        {
            "id": "default_mode_subsystems",
            "label": "Default Mode Network: Core vs. Medial Temporal Subsystems",
            "description": "DMN splits into two dissociable subsystems: core (mPFC↔PCC) for self-referential processing and MTL subsystem for memory",
            "color": "#9333ea",
            "subcircuit_of": "default_mode",
            "scale": "network",
            "regions": [
                {"id": "medial_prefrontal_cortex", "label": "mPFC (Core subsystem)", "role": "self-referential, social cognition, future simulation"},
                {"id": "posterior_cingulate_cortex", "label": "PCC / Precuneus (Core subsystem)", "role": "autobiographical memory integration, mind-wandering"},
                {"id": "hippocampus", "label": "Hippocampus (MTL subsystem)", "role": "scene construction — episodic past and future"},
                {"id": "entorhinal_cortex", "label": "Entorhinal / Parahippocampal (MTL)", "role": "contextual and spatial memory binding"},
                {"id": "lateral_temporal_cortex", "label": "Lateral Temporal Cortex (MTL)", "role": "semantic memory store"},
                {"id": "angular_gyrus", "label": "Angular Gyrus (Core)", "role": "narrative + semantic integration"},
            ],
            "topics": ["episodic_memory", "social_behavior", "emotional_processing"],
        },
        {
            "id": "orexin_arousal",
            "label": "Orexin/Hypocretin Arousal Circuit",
            "description": "Lateral hypothalamic orexin neurons stabilize wake and suppress REM — lost in narcolepsy; key target for insomnia treatment",
            "color": "#f97316",
            "scale": "neuromodulatory",
            "regions": [
                {"id": "lateral_hypothalamus", "label": "Lateral Hypothalamus (LH)", "role": "sole source of orexin — projects widely to arousal centers"},
                {"id": "locus_coeruleus", "label": "Locus Coeruleus", "role": "orexin strongly excites LC — NE arousal"},
                {"id": "raphe_nucleus", "label": "Dorsal Raphe", "role": "orexin excites DRN — 5-HT wake promotion"},
                {"id": "hypothalamus", "label": "Tuberomammillary Nucleus (TMN)", "role": "orexin excites histamine neurons — cortical arousal"},
                {"id": "basal_forebrain", "label": "Basal Forebrain", "role": "orexin excites ACh — attention/wakefulness"},
                {"id": "prefrontal_cortex", "label": "Cortex (indirect)", "role": "wake-promoting arousal tone"},
            ],
            "topics": ["sleep_and_oscillations", "neuromodulation", "attention_and_salience"],
        },
        {
            "id": "cortical_column",
            "label": "Cortical Column: Canonical Microcircuit",
            "description": "The repeating computational unit of neocortex — feedforward (L4→L2/3→L5) + feedback (L6→thalamus) + inhibitory interneurons",
            "color": "#94a3b8",
            "scale": "microcircuit",
            "regions": [
                {"id": "cortical_l4", "label": "Cortical Layer 4 (Stellate cells)", "role": "thalamic input reception — feedforward entry"},
                {"id": "cortical_l23", "label": "Cortical Layer 2/3 (Pyramidal)", "role": "horizontal integration — same-area communication"},
                {"id": "cortical_l5", "label": "Cortical Layer 5 (Large pyramidals)", "role": "output to subcortex + brainstem; apical dendrites integrate feedback"},
                {"id": "cortical_l6", "label": "Cortical Layer 6 (Corticothalamic)", "role": "feedback to thalamus — controls gain"},
                {"id": "pv_interneurons", "label": "PV Interneurons (all layers)", "role": "fast perisomatic inhibition — gamma oscillation generation"},
                {"id": "sst_interneurons", "label": "SST Interneurons", "role": "dendritic inhibition — top-down gating"},
                {"id": "vip_interneurons", "label": "VIP Interneurons", "role": "disinhibition circuit — releases pyramidals from SST"},
            ],
            "topics": ["neural_synchrony", "spectral_dynamics", "attention_and_salience"],
        },
    ]
