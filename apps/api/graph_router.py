"""Graph API endpoints for the Knowledge Explorer."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

REPO_ROOT = Path(__file__).parent.parent.parent
CLUSTER_GRAPH_PATH = REPO_ROOT / "artifacts/graph/cluster_graph.json"
GALAXY_PATH = REPO_ROOT / "artifacts/graph/galaxy_points.json"
CONSENSUS_PATH = REPO_ROOT / "artifacts/literature/relationships/consensus_summaries.jsonl"
FINDINGS_PATH = REPO_ROOT / "artifacts/literature/findings_tier1_ollama.jsonl"
LINKS_PATH = REPO_ROOT / "artifacts/literature/paper_dataset_links.jsonl"

router = APIRouter()

# ── Cache loaded artifacts in module-level dicts ────────────────────────────

_cluster_graph: dict[str, Any] | None = None
_consensus_rows: list[dict[str, Any]] | None = None
_findings_rows: list[dict[str, Any]] | None = None
_links_rows: list[dict[str, Any]] | None = None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _get_cluster_graph() -> dict[str, Any]:
    global _cluster_graph
    if _cluster_graph is None:
        if not CLUSTER_GRAPH_PATH.exists():
            _cluster_graph = {"nodes": [], "links": []}
        else:
            _cluster_graph = json.loads(CLUSTER_GRAPH_PATH.read_text())
    return _cluster_graph


def _get_consensus() -> list[dict[str, Any]]:
    global _consensus_rows
    if _consensus_rows is None:
        _consensus_rows = _load_jsonl(CONSENSUS_PATH)
    return _consensus_rows


def _get_findings() -> list[dict[str, Any]]:
    global _findings_rows
    if _findings_rows is None:
        _findings_rows = _load_jsonl(FINDINGS_PATH)
    return _findings_rows


def _get_links() -> list[dict[str, Any]]:
    global _links_rows
    if _links_rows is None:
        _links_rows = _load_jsonl(LINKS_PATH)
    return _links_rows


SUGGESTED_VIEWS: dict[str, dict[str, Any]] = {
    "hippocampal": {
        "slug": "hippocampal",
        "label": "Hippocampal Circuits",
        "description": "Memory, navigation, and place coding in hippocampal subfields",
        "regions": ["hippocampus", "ca1", "ca3", "dentate gyrus", "entorhinal cortex"],
        "species": [], "tasks": [], "layer": "literature",
        "companions": ["memory", "cross_species"],
    },
    "decision": {
        "slug": "decision",
        "label": "Decision Making",
        "description": "Value-based choices, reward learning, and prefrontal-striatal circuits",
        "regions": ["prefrontal cortex", "striatum", "anterior cingulate cortex"],
        "species": [], "tasks": ["decision_making", "reversal_learning"], "layer": "literature",
        "companions": ["reward"],
    },
    "tbi": {
        "slug": "tbi",
        "label": "TBI Landscape",
        "description": "Traumatic brain injury — multi-region effects and controversy",
        "regions": ["cortex", "hippocampus", "brainstem"],
        "species": ["human", "mouse"], "tasks": [], "layer": "consensus",
        "companions": ["cross_species"],
    },
    "cross_species": {
        "slug": "cross_species",
        "label": "Cross-Species",
        "description": "Comparative findings across mouse, rat, macaque, and human",
        "regions": [], "species": ["mouse", "human"], "tasks": [], "layer": "bridge",
        "companions": ["hippocampal", "decision"],
    },
    "reward": {
        "slug": "reward",
        "label": "Reward & Dopamine",
        "description": "Mesolimbic reward circuitry, dopamine, and conditioning",
        "regions": ["striatum", "nucleus accumbens", "basal ganglia"],
        "species": [], "tasks": [], "layer": "literature",
        "companions": ["decision"],
    },
    "memory": {
        "slug": "memory",
        "label": "Memory Systems",
        "description": "Working memory, spatial navigation, and consolidation",
        "regions": ["hippocampus", "prefrontal cortex"],
        "species": [], "tasks": ["working_memory"], "layer": "literature",
        "companions": ["hippocampal"],
    },
    "motor": {
        "slug": "motor",
        "label": "Motor Circuits",
        "description": "Movement control across cortex, cerebellum, and spinal cord",
        "regions": ["motor cortex", "cerebellum"],
        "species": [], "tasks": [], "layer": "literature",
        "companions": [],
    },
}


def _filter_graph(
    nodes: list[dict], links: list[dict],
    regions: list[str], species: list[str], tasks: list[str],
    limit: int,
) -> tuple[list[dict], list[dict]]:
    if not regions and not species and not tasks:
        # Return top nodes by size
        sorted_nodes = sorted(nodes, key=lambda n: -n["size"])[:limit]
    else:
        region_set = {r.lower() for r in regions}
        # Include nodes whose id or meta.region matches
        included_ids: set[str] = set()
        for n in nodes:
            nid = n["id"]
            meta = n.get("meta", {})
            node_region = str(meta.get("region", "")).lower()
            if any(r in nid.lower() or r in node_region for r in region_set):
                included_ids.add(nid)
            # Always include system nodes
            if n["type"] == "system":
                included_ids.add(nid)

        # Add one hop of neighbors via links
        neighbor_ids: set[str] = set()
        for link in links:
            if link["source"] in included_ids or link["target"] in included_ids:
                neighbor_ids.add(link["source"])
                neighbor_ids.add(link["target"])
        included_ids |= neighbor_ids

        id_to_node = {n["id"]: n for n in nodes}
        candidates = [id_to_node[nid] for nid in included_ids if nid in id_to_node]
        sorted_nodes = sorted(candidates, key=lambda n: -n["size"])[:limit]

    node_id_set = {n["id"] for n in sorted_nodes}
    filtered_links = [
        lnk for lnk in links
        if lnk["source"] in node_id_set and lnk["target"] in node_id_set
    ]
    return sorted_nodes, filtered_links


@router.get("/api/graph/overview")
async def get_graph_overview(limit: int = Query(400, le=800)) -> dict[str, Any]:
    graph = _get_cluster_graph()
    nodes, links = _filter_graph(
        graph["nodes"], graph["links"],
        regions=[], species=[], tasks=[], limit=limit,
    )
    return {"nodes": nodes, "links": links, "meta": {"node_count": len(nodes), "edge_count": len(links)}}


@router.get("/api/graph/subgraph")
async def get_graph_subgraph(
    regions: str = Query(""),
    species: str = Query(""),
    tasks: str = Query(""),
    limit: int = Query(400, le=800),
) -> dict[str, Any]:
    region_list = [r.strip() for r in regions.split(",") if r.strip()]
    species_list = [s.strip() for s in species.split(",") if s.strip()]
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    graph = _get_cluster_graph()
    nodes, links = _filter_graph(
        graph["nodes"], graph["links"],
        regions=region_list, species=species_list, tasks=task_list, limit=limit,
    )
    return {
        "nodes": nodes, "links": links,
        "meta": {
            "node_count": len(nodes), "edge_count": len(links),
            "filtered_by": {"regions": region_list, "species": species_list, "tasks": task_list},
        },
    }


@router.get("/api/graph/topic/{slug}")
async def get_topic_graph(slug: str, limit: int = Query(400, le=800)) -> dict[str, Any]:
    view = SUGGESTED_VIEWS.get(slug)
    if not view:
        raise HTTPException(status_code=404, detail=f"Unknown topic slug: {slug}")
    graph = _get_cluster_graph()
    nodes, links = _filter_graph(
        graph["nodes"], graph["links"],
        regions=view["regions"], species=view["species"], tasks=view["tasks"], limit=limit,
    )
    return {
        "nodes": nodes, "links": links,
        "meta": {"node_count": len(nodes), "edge_count": len(links), "filtered_by": {}},
        "topic": {
            "slug": slug,
            "label": view["label"],
            "description": view["description"],
            "companion_slugs": view["companions"],
        },
    }


@router.get("/api/graph/suggested-views")
async def get_suggested_views() -> list[dict[str, Any]]:
    return [
        {"slug": v["slug"], "label": v["label"], "description": v["description"],
         "layer": v["layer"], "companions": v["companions"]}
        for v in SUGGESTED_VIEWS.values()
    ]


@router.get("/api/literature/consensus")
async def get_consensus(region: str = Query(""), limit: int = Query(200, le=500)) -> list[dict[str, Any]]:
    rows = _get_consensus()
    if region:
        rows = [r for r in rows if region.lower() in r["region"].lower()]
    return [
        {
            "region": r["region"], "direction": r["direction"],
            "task": r.get("task"), "n_findings": r["n_findings"],
            "n_papers": r["n_papers"], "consensus_strength": r["consensus_strength"],
        }
        for r in sorted(rows, key=lambda x: -x["n_findings"])[:limit]
    ]


@router.get("/api/literature/findings")
async def get_findings(
    region: str = Query(""),
    direction: str = Query(""),
    limit: int = Query(20, le=100),
) -> list[dict[str, Any]]:
    rows = _get_findings()
    results = []
    for row in rows:
        if region and region.lower() not in " ".join(row.get("regions", [])).lower():
            continue
        if direction and direction != row.get("result_direction", ""):
            continue
        results.append({
            "finding_id": row.get("finding_id", ""),
            "finding_text": row.get("finding_text", ""),
            "region": (row.get("regions") or [""])[0],
            "direction": row.get("result_direction", ""),
            "confidence": row.get("confidence", 0.0),
            "paper_id": row.get("paper_id", ""),
        })
        if len(results) >= limit:
            break
    return results


@router.get("/api/datasets/{dataset_id}/neighborhood")
async def get_dataset_neighborhood(dataset_id: str) -> dict[str, Any]:
    links_data = _get_links()
    consensus = _get_consensus()

    paper_links = [
        lnk for lnk in links_data
        if lnk.get("dataset_record_id") == dataset_id and lnk.get("paper_openalex_id")
    ]

    # consensus for this dataset's typical brain regions (via paper links or cluster graph)
    paper_ids = {lnk["paper_openalex_id"] for lnk in paper_links}

    # Try to get dataset from corpus to find its brain_regions
    from neural_search.ingestion.demo_seed import build_combined_corpus  # type: ignore
    try:
        corpus = build_combined_corpus()
        dataset_rec = next((r for r in corpus if r.get("id") == dataset_id), None)
    except Exception:
        dataset_rec = None

    dataset_regions: list[str] = []
    if dataset_rec:
        dataset_regions = [r.lower() for r in (dataset_rec.get("brain_regions") or [])]

    # Consensus rows matching dataset's regions
    consensus_by_region = [
        {"region": r["region"], "direction": r["direction"],
         "n_findings": r["n_findings"], "n_papers": r["n_papers"],
         "consensus_strength": r["consensus_strength"]}
        for r in consensus
        if r["region"].lower() in dataset_regions
    ][:10]

    # Finding clusters for those regions
    finding_clusters = [
        {"region": r["region"], "direction": r["direction"],
         "n_findings": r["n_findings"], "consensus_strength": r["consensus_strength"]}
        for r in consensus
        if r["region"].lower() in dataset_regions
    ][:6]

    # Related datasets sharing regions via paper links
    related: list[dict[str, Any]] = []
    region_dataset_map: dict[str, list[str]] = defaultdict(list)
    for lnk in links_data:
        did = lnk.get("dataset_record_id", "")
        if did and did != dataset_id:
            region_dataset_map[lnk.get("paper_openalex_id", "")].append(did)

    return {
        "dataset_id": dataset_id,
        "linked_papers": [
            {
                "paper_openalex_id": lnk["paper_openalex_id"],
                "paper_title": lnk.get("paper_title"),
                "paper_year": lnk.get("paper_year"),
                "paper_doi": lnk.get("paper_doi"),
                "confidence": lnk.get("confidence", 0.0),
            }
            for lnk in paper_links
        ],
        "finding_clusters": finding_clusters,
        "related_datasets": related,
        "consensus_by_region": consensus_by_region,
    }
