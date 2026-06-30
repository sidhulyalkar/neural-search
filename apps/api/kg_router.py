"""FastAPI router exposing disorder registry, concept authority, and NeuroSynth topics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query

log = logging.getLogger(__name__)

router = APIRouter(prefix="/kg", tags=["knowledge-graph"])

DATA_ROOT = Path(__file__).parent.parent.parent / "data"

_disorders_cache: list[dict[str, Any]] | None = None
_concepts_cache: list[dict[str, Any]] | None = None


def _load_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get_disorders() -> list[dict[str, Any]]:
    global _disorders_cache
    if _disorders_cache is None:
        data = _load_yaml(DATA_ROOT / "disorders" / "disorder_registry.yaml")
        _disorders_cache = data.get("disorders", [])
    return _disorders_cache


def _get_concepts() -> list[dict[str, Any]]:
    global _concepts_cache
    if _concepts_cache is None:
        data = _load_yaml(DATA_ROOT / "concepts" / "concept_seed.yaml")
        _concepts_cache = data.get("concepts", [])
    return _concepts_cache


# ── Disorders ─────────────────────────────────────────────────────────────────

# ── Cross-disorder circuit matrix (must come BEFORE /disorders/{id}) ──────────

@router.get("/disorders/matrix/circuits")
def circuit_disorder_matrix() -> dict[str, Any]:
    """Return a matrix of circuits x disorders for heatmap visualisation."""
    disorders = _get_disorders()

    all_circuits: set[str] = set()
    for d in disorders:
        all_circuits.update(d.get("disrupted_circuits", []))
    circuits_list = sorted(all_circuits)

    rows = []
    for circuit in circuits_list:
        affected = [
            {"id": d["id"], "label": d["label"], "type": d.get("type")}
            for d in disorders
            if circuit in d.get("disrupted_circuits", [])
        ]
        rows.append({"circuit": circuit, "disorders": affected, "n_disorders": len(affected)})

    rows.sort(key=lambda r: r["n_disorders"], reverse=True)

    return {
        "circuits": rows,
        "disorders": [{"id": d["id"], "label": d["label"], "type": d.get("type")} for d in disorders],
        "n_circuits": len(circuits_list),
        "n_disorders": len(disorders),
    }


@router.get("/disorders")
def list_disorders(
    disorder_type: str | None = Query(None, description="Filter by type (e.g. psychotic_disorder, mood_disorder, neurodegenerative)"),
    circuit: str | None = Query(None, description="Filter by disrupted circuit ID"),
) -> dict[str, Any]:
    disorders = _get_disorders()
    result = disorders

    if disorder_type:
        result = [d for d in result if d.get("type") == disorder_type]
    if circuit:
        result = [d for d in result if circuit in d.get("disrupted_circuits", [])]

    # Group by type for the response
    by_type: dict[str, list[dict]] = {}
    for d in result:
        t = d.get("type", "other")
        by_type.setdefault(t, []).append({
            "id": d["id"],
            "label": d["label"],
            "icd11": d.get("icd11"),
            "type": t,
            "disrupted_circuits": d.get("disrupted_circuits", []),
            "n_biomarkers": len(d.get("oscillation_biomarkers", [])),
            "n_species_models": len(d.get("species_models", [])),
            "topics": d.get("topics", []),
        })

    return {
        "total": len(result),
        "by_type": by_type,
        "disorders": [{
            "id": d["id"],
            "label": d["label"],
            "icd11": d.get("icd11"),
            "type": d.get("type"),
            "disrupted_circuits": d.get("disrupted_circuits", []),
            "n_biomarkers": len(d.get("oscillation_biomarkers", [])),
            "topics": d.get("topics", []),
        } for d in result],
    }


@router.get("/disorders/{disorder_id}")
def get_disorder(disorder_id: str) -> dict[str, Any]:
    disorders = _get_disorders()
    match = next((d for d in disorders if d["id"] == disorder_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Disorder '{disorder_id}' not found")
    return match


@router.get("/disorders/{disorder_id}/circuits")
def get_disorder_circuits(disorder_id: str) -> dict[str, Any]:
    disorders = _get_disorders()
    match = next((d for d in disorders if d["id"] == disorder_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Disorder '{disorder_id}' not found")
    return {
        "disorder_id": disorder_id,
        "label": match["label"],
        "disrupted_circuits": match.get("disrupted_circuits", []),
        "oscillation_biomarkers": match.get("oscillation_biomarkers", []),
        "species_models": match.get("species_models", []),
        "key_papers": match.get("key_papers", []),
    }


# ── Concepts ──────────────────────────────────────────────────────────────────
# NOTE: /concepts/hierarchy/tree must come before /concepts/{concept_id} to avoid
# FastAPI matching "hierarchy" as a concept_id.

@router.get("/concepts/hierarchy/tree")
def concept_hierarchy_tree() -> dict[str, Any]:
    """Return concept hierarchy as a tree (root concepts + their narrower concepts)."""
    concepts = _get_concepts()
    by_id = {c["id"]: c for c in concepts}

    roots = [
        c for c in concepts
        if not c.get("broader_concept") or c.get("broader_concept") not in by_id
    ]

    def _subtree(concept: dict) -> dict:
        narrower_ids = concept.get("narrower_concepts", [])
        children = [_subtree(by_id[nid]) for nid in narrower_ids if nid in by_id]
        return {
            "id": concept["id"],
            "label": concept["label"],
            "concept_type": concept.get("concept_type"),
            "formula": concept.get("formula"),
            "definition": concept.get("definition", "")[:200],
            "children": children,
        }

    return {
        "roots": [_subtree(r) for r in roots],
        "total": len(concepts),
    }


@router.get("/concepts")
def list_concepts(
    concept_type: str | None = Query(None, description="Filter by concept_type"),
    topic: str | None = Query(None, description="Filter by associated topic"),
) -> dict[str, Any]:
    concepts = _get_concepts()
    result = concepts

    if concept_type:
        result = [c for c in result if c.get("concept_type") == concept_type]
    if topic:
        result = [c for c in result if topic in c.get("topics", [])]

    return {
        "total": len(result),
        "concepts": [{
            "id": c["id"],
            "label": c["label"],
            "aliases": c.get("aliases", []),
            "concept_type": c.get("concept_type"),
            "broader_concept": c.get("broader_concept"),
            "narrower_concepts": c.get("narrower_concepts", []),
            "definition": c.get("definition", ""),
            "formula": c.get("formula"),
            "related_methods": c.get("related_methods", []),
            "related_regions": c.get("related_regions", []),
            "testable_predictions": c.get("testable_predictions", []),
            "topics": c.get("topics", []),
            "scholarpedia_url": c.get("scholarpedia_url"),
        } for c in result],
    }


@router.get("/concepts/{concept_id}")
def get_concept(concept_id: str) -> dict[str, Any]:
    concepts = _get_concepts()
    match = next((c for c in concepts if c["id"] == concept_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_id}' not found")

    # Enrich: find child concepts
    narrower = [
        {"id": c["id"], "label": c["label"], "concept_type": c.get("concept_type")}
        for c in concepts
        if c.get("broader_concept") == concept_id
    ]
    return {**match, "narrower_resolved": narrower}


# ── NeuroSynth topic summary ──────────────────────────────────────────────────

@router.get("/neurosynth/regions")
def neurosynth_regions(
    topic: str | None = Query(None, description="Filter edges by topic or term"),
    min_freq: float = Query(0.05, description="Minimum activation frequency"),
) -> dict[str, Any]:
    """Return topic->region edges from NeuroSynth (reads composed KG JSONL)."""
    import json
    from pathlib import Path

    kg_path = Path(__file__).parent.parent.parent / "artifacts" / "kg" / "composed_kg.jsonl"
    if not kg_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Composed KG not yet generated. Run: python -m neural_search.ingestion.compose_kg",
        )

    edges = []
    with open(kg_path, encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if obj.get("type") != "edge":
                continue
            if obj.get("edge_type") not in ("topic_activates_region", "region_implicated_in_topic"):
                continue
            freq = obj.get("properties", {}).get("activation_frequency", 0)
            if freq < min_freq:
                continue
            if topic and topic not in (obj.get("properties", {}).get("neurosynth_term", "")):
                continue
            edges.append({
                "source": obj["source_node_id"],
                "target": obj["target_node_id"],
                "type": obj["edge_type"],
                "term": obj.get("properties", {}).get("neurosynth_term"),
                "freq": freq,
                "n_studies": obj.get("properties", {}).get("n_studies"),
            })

    # Aggregate: top regions per term
    from collections import defaultdict
    term_regions: dict[str, list] = defaultdict(list)
    for e in edges:
        if e["type"] == "topic_activates_region":
            term_regions[e["term"] or e["source"]].append({
                "region": e["target"].split(":")[-1],
                "freq": e["freq"],
                "n_studies": e["n_studies"],
            })

    return {
        "total_edges": len(edges),
        "terms": {
            t: sorted(rs, key=lambda r: r["freq"], reverse=True)[:10]
            for t, rs in list(term_regions.items())[:50]
        },
    }
