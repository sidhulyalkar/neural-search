"""Build a cluster-level knowledge graph from literature artifacts.

Output: artifacts/graph/cluster_graph.json
Schema: {"nodes": [...], "links": [...]}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent.parent
CONSENSUS_PATH = REPO_ROOT / "artifacts/literature/relationships/consensus_summaries.jsonl"
EDGES_PATH = REPO_ROOT / "artifacts/literature/relationships/finding_edges.jsonl"
LINKS_PATH = REPO_ROOT / "artifacts/literature/paper_dataset_links.jsonl"
OUTPUT_DIR = REPO_ROOT / "artifacts/graph"
OUTPUT_PATH = OUTPUT_DIR / "cluster_graph.json"

DIRECTION_COLORS: dict[str, str] = {
    "increase": "#10b981",
    "decrease": "#ef4444",
    "correlation": "#8b5cf6",
    "no_change": "#6b7280",
}

BRAIN_SYSTEMS: dict[str, dict[str, Any]] = {
    "hippocampal_formation": {
        "label": "Hippocampal Formation",
        "color": "#f59e0b",
        "regions": {"hippocampus", "ca1", "ca3", "dentate gyrus", "entorhinal cortex", "subiculum"},
    },
    "prefrontal": {
        "label": "Prefrontal Cortex",
        "color": "#f59e0b",
        "regions": {"prefrontal cortex", "anterior cingulate", "anterior cingulate cortex",
                    "orbitofrontal cortex", "medial prefrontal cortex", "prelimbic", "infralimbic"},
    },
    "basal_ganglia": {
        "label": "Basal Ganglia",
        "color": "#f59e0b",
        "regions": {"striatum", "basal ganglia", "nucleus accumbens", "caudate", "putamen",
                    "globus pallidus", "substantia nigra"},
    },
    "cerebellum": {
        "label": "Cerebellum",
        "color": "#f59e0b",
        "regions": {"cerebellum", "cerebellar cortex", "purkinje cell layer"},
    },
    "brainstem": {
        "label": "Brainstem",
        "color": "#f59e0b",
        "regions": {"brainstem", "brain stem", "midbrain", "pons", "medulla",
                    "locus coeruleus", "raphe nuclei"},
    },
    "sensory_cortex": {
        "label": "Sensory Cortex",
        "color": "#f59e0b",
        "regions": {"visual cortex", "auditory cortex", "somatosensory cortex",
                    "barrel cortex", "primary visual cortex"},
    },
    "motor_cortex": {
        "label": "Motor Cortex",
        "color": "#f59e0b",
        "regions": {"motor cortex", "primary motor cortex", "supplementary motor area"},
    },
    "limbic": {
        "label": "Limbic System",
        "color": "#f59e0b",
        "regions": {"amygdala", "thalamus", "hypothalamus", "insula",
                    "anterior insula", "cingulate cortex"},
    },
}

# Build reverse lookup: region_name → system_id
_REGION_TO_SYSTEM: dict[str, str] = {}
for sys_id, sys_data in BRAIN_SYSTEMS.items():
    for r in sys_data["regions"]:
        _REGION_TO_SYSTEM[r.lower()] = sys_id


def assign_system(region: str) -> str:
    return _REGION_TO_SYSTEM.get(region.lower(), "other")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_graph(max_edges: int | None = None) -> dict[str, Any]:
    consensus = _load_jsonl(CONSENSUS_PATH)
    raw_edges = _load_jsonl(EDGES_PATH)
    links_data = _load_jsonl(LINKS_PATH)

    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    node_ids: set[str] = set()

    # ── System nodes (L0) ────────────────────────────────────────────
    for sys_id, sys_data in BRAIN_SYSTEMS.items():
        nid = f"system:{sys_id}"
        node_ids.add(nid)
        nodes.append({
            "id": nid, "type": "system", "label": sys_data["label"],
            "scale_level": 4, "size": 30, "color": sys_data["color"], "meta": {},
        })
    # "other" catch-all system
    node_ids.add("system:other")
    nodes.append({
        "id": "system:other", "type": "system", "label": "Other",
        "scale_level": 4, "size": 15, "color": "#6b7280", "meta": {},
    })

    # ── Region nodes (L1) from consensus ─────────────────────────────
    region_finding_counts: dict[str, int] = defaultdict(int)
    for row in consensus:
        region_finding_counts[row["region"]] += row["n_findings"]

    added_regions: set[str] = set()
    for region, count in region_finding_counts.items():
        nid = f"region:{region}"
        if nid in node_ids:
            continue
        node_ids.add(nid)
        added_regions.add(region)
        nodes.append({
            "id": nid, "type": "region", "label": region,
            "scale_level": 3,
            "size": max(8, min(25, count // 20)),
            "color": "#fcd34d",
            "meta": {"n_findings": count, "system": assign_system(region)},
        })
        # Edge: region → system
        sys_id = assign_system(region)
        links.append({
            "source": f"system:{sys_id}", "target": nid,
            "type": "contains", "weight": 1, "color": "#f59e0b30",
        })

    # ── Finding cluster nodes (L2) from consensus ─────────────────────
    # Map finding_id → cluster_id for edge aggregation
    finding_to_cluster: dict[str, str] = {}
    for row in consensus:
        cid = f"cluster:{row['region']}:{row['direction']}"
        for fid in row.get("finding_ids", []):
            finding_to_cluster[fid] = cid

    for row in consensus:
        cid = f"cluster:{row['region']}:{row['direction']}"
        if cid in node_ids:
            continue
        node_ids.add(cid)
        direction = row["direction"]
        nodes.append({
            "id": cid,
            "type": "finding_cluster",
            "label": f"{row['region']} {direction}",
            "scale_level": 2,
            "size": max(5, min(20, row["n_findings"] // 15)),
            "color": DIRECTION_COLORS.get(direction, "#6b7280"),
            "meta": {
                "region": row["region"],
                "direction": direction,
                "n_findings": row["n_findings"],
                "n_papers": row["n_papers"],
                "consensus_strength": row["consensus_strength"],
                "task": row.get("task"),
            },
        })
        links.append({
            "source": f"region:{row['region']}", "target": cid,
            "type": "contains", "weight": 1, "color": "#fcd34d30",
        })

    # ── Cluster→cluster edges aggregated from finding edges ──────────
    cluster_edge_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    limit = max_edges if max_edges is not None else len(raw_edges)
    for edge in raw_edges[:limit]:
        ca = finding_to_cluster.get(edge["finding_id_a"])
        cb = finding_to_cluster.get(edge["finding_id_b"])
        if ca and cb and ca != cb:
            key = (ca, cb, edge["edge_type"])
            cluster_edge_counts[key] += 1

    seen_cluster_pairs: set[tuple[str, str]] = set()
    for (ca, cb, etype), count in sorted(cluster_edge_counts.items(), key=lambda x: -x[1]):
        pair = (min(ca, cb), max(ca, cb))
        if pair in seen_cluster_pairs:
            continue
        seen_cluster_pairs.add(pair)
        if ca not in node_ids or cb not in node_ids:
            continue
        color = "#10b981" if etype == "supports" else "#ef4444"
        links.append({
            "source": ca, "target": cb,
            "type": etype, "weight": min(5, 1 + count // 3),
            "color": color,
        })

    # ── Paper nodes and dataset-paper edges ──────────────────────────
    paper_dataset_map: dict[str, list[str]] = defaultdict(list)
    for link in links_data:
        pid = link.get("paper_openalex_id")
        did = link.get("dataset_record_id")
        if not pid or not did:
            continue
        paper_nid = f"paper:{pid}"
        dataset_nid = f"dataset:{did}"

        if paper_nid not in node_ids:
            node_ids.add(paper_nid)
            nodes.append({
                "id": paper_nid, "type": "paper",
                "label": (link.get("paper_title") or pid)[:60],
                "scale_level": 3, "size": 6, "color": "#8b5cf6",
                "meta": {
                    "doi": link.get("paper_doi"),
                    "year": link.get("paper_year"),
                    "title": link.get("paper_title"),
                },
            })
        paper_dataset_map[pid].append(did)

    # Dataset nodes (from corpus)
    try:
        import sys as _sys
        _sys.path.insert(0, str(REPO_ROOT))
        from neural_search.ingestion.demo_seed import build_combined_corpus
        corpus = build_combined_corpus()
    except Exception:
        corpus = []

    dataset_lookup = {r.get("id", ""): r for r in corpus}

    added_datasets: set[str] = set()
    for pid, dids in paper_dataset_map.items():
        paper_nid = f"paper:{pid}"
        for did in dids:
            dataset_nid = f"dataset:{did}"
            if dataset_nid not in node_ids:
                node_ids.add(dataset_nid)
                added_datasets.add(did)
                rec = dataset_lookup.get(did, {})
                nodes.append({
                    "id": dataset_nid, "type": "dataset",
                    "label": (rec.get("title") or did)[:60],
                    "scale_level": 3,
                    "size": max(5, min(15, int(rec.get("analysis_readiness_score", 50)) // 8)),
                    "color": "#22d3ee",
                    "meta": {
                        "source": rec.get("source", ""),
                        "readiness": rec.get("analysis_readiness_score", 0),
                        "brain_regions": rec.get("brain_regions", []),
                    },
                })
            links.append({
                "source": dataset_nid, "target": paper_nid,
                "type": "linked", "weight": 1, "color": "#ffffff20",
            })

    # Dataset → region edges (from corpus brain_regions)
    for did in added_datasets:
        rec = dataset_lookup.get(did, {})
        for region in (rec.get("brain_regions") or [])[:3]:
            region_nid = f"region:{region.lower()}"
            if region_nid in node_ids:
                links.append({
                    "source": f"dataset:{did}", "target": region_nid,
                    "type": "covers", "weight": 1, "color": "#22d3ee20",
                })

    return {"nodes": nodes, "links": links}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building cluster graph…")
    graph = build_graph()
    OUTPUT_PATH.write_text(json.dumps(graph, indent=2))
    n = len(graph["nodes"])
    e = len(graph["links"])
    print(f"Done — {n} nodes, {e} links → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
