# Knowledge Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `/graph` into a multi-layer, multi-mode 3D/2D Knowledge Explorer with galaxy point-cloud overview, force-directed subgraph exploration, and inline relationship highlights in search results.

**Architecture:** Pre-compute a cluster-level graph from literature artifacts (consensus summaries → cluster nodes, finding edges → cluster-cluster edges, paper-dataset links → dataset-paper edges) and bake 3D positions via networkx. The frontend loads the static layout for instant galaxy rendering, then fetches subgraphs on demand. All graph components live under `apps/web/src/components/graph/`.

**Tech Stack:** Python networkx (layout), FastAPI (6 new endpoints), react-force-graph (3D/2D explorer), three.js (galaxy point cloud), TanStack Query v5, Tailwind, TypeScript strict.

## Global Constraints

- Python ≥ 3.11; all scripts runnable as `python scripts/literature/<name>.py`
- Artifacts output to `artifacts/graph/` (create dir if missing)
- No new npm packages besides `react-force-graph` and `three` + `@types/three`
- `react-force-graph` expects edge array key `links` (not `edges`), and each link `source`/`target` as node `id` strings
- All Tailwind classes must be in the existing dark neural theme (`neural-*`, `accent-cyan`, `accent-violet`, `accent-emerald`)
- No `console.log` in production code
- Frontend fetches via `apps/web/src/api/graph.ts` only — no inline `fetch` calls in components
- Python tests: `pytest tests/` from repo root
- Direction colors: increase=`#10b981`, decrease=`#ef4444`, correlation=`#8b5cf6`, no_change=`#6b7280`
- Node colors: system=`#f59e0b`, region=`#fcd34d`, finding_cluster=direction-based, dataset=`#22d3ee`, paper=`#8b5cf6`

---

## File Map

**Create:**
```
scripts/literature/build_cluster_graph.py
scripts/literature/compute_layout.py
artifacts/graph/                          (directory)
apps/api/graph_router.py
apps/web/src/types/graph.ts
apps/web/src/api/graph.ts
apps/web/src/components/graph/GalaxyGraph.tsx
apps/web/src/components/graph/ExplorerGraph.tsx
apps/web/src/components/graph/GraphControls.tsx
apps/web/src/components/graph/FilterPanel.tsx
apps/web/src/components/graph/SuggestedViews.tsx
apps/web/src/components/graph/OntologyTreePanel.tsx
apps/web/src/components/graph/NodeDetailPanel.tsx
apps/web/src/components/graph/GraphLegend.tsx
apps/web/src/components/graph/RelatedFindingsPanel.tsx
apps/web/src/pages/KnowledgeExplorerPage.tsx
tests/test_cluster_graph.py
tests/test_graph_api.py
```

**Modify:**
```
apps/api/main.py                          (include graph_router)
apps/web/src/App.tsx                      (swap GraphPage → KnowledgeExplorerPage)
apps/web/src/components/DatasetCard.tsx   (add Related Findings toggle)
apps/web/package.json                     (add react-force-graph, three, @types/three)
```

---

## Task 1: Install Frontend Dependencies + TypeScript Graph Types

**Files:**
- Modify: `apps/web/package.json`
- Create: `apps/web/src/types/graph.ts`

**Interfaces:**
- Produces: `GraphNode`, `GraphEdge`, `GalaxyPoint`, `SubgraphResponse`, `ViewMode`, `LayerMode`, `SuggestedView`, `DatasetNeighborhood` — consumed by all Tasks 6–11

- [ ] **Step 1: Install dependencies**

```bash
cd apps/web
npm install react-force-graph three
npm install --save-dev @types/three
```

Expected output: `added N packages`

- [ ] **Step 2: Create `apps/web/src/types/graph.ts`**

```typescript
export type NodeType = 'system' | 'region' | 'finding_cluster' | 'dataset' | 'paper'
export type EdgeType = 'contains' | 'covers' | 'linked' | 'supports' | 'contradicts'
export type ViewMode = 'galaxy' | 'explorer' | '2d'
export type LayerMode = 'corpus' | 'consensus' | 'literature' | 'bridge' | 'morphology'
export type Direction = 'increase' | 'decrease' | 'correlation' | 'no_change'

export interface GraphNode {
  id: string
  type: NodeType
  label: string
  scale_level: number    // 0=molecule, 2=cluster, 3=region, 4=system/dataset
  size: number
  color: string
  meta: Record<string, unknown>
}

export interface GraphEdge {
  source: string
  target: string
  type: EdgeType
  weight: number
  color: string
}

// react-force-graph uses 'links', not 'edges'
export interface GraphData {
  nodes: GraphNode[]
  links: GraphEdge[]
}

export interface GalaxyPoint {
  id: string
  x: number
  y: number
  z: number
  type: NodeType
  color: string
  size: number
  label: string
}

export interface GalaxyLayout {
  nodes: GalaxyPoint[]
}

export interface SubgraphResponse {
  nodes: GraphNode[]
  links: GraphEdge[]
  meta: { node_count: number; edge_count: number; filtered_by: Record<string, unknown> }
}

export interface TopicGraphResponse extends SubgraphResponse {
  topic: { slug: string; label: string; description: string; companion_slugs: string[] }
}

export interface SuggestedView {
  slug: string
  label: string
  description: string
  layer: LayerMode
  companions: string[]
}

export interface ConsensusRow {
  region: string
  direction: Direction
  task: string | null
  n_findings: number
  n_papers: number
  consensus_strength: number
}

export interface FindingRow {
  finding_id: string
  finding_text: string
  region: string
  direction: Direction
  confidence: number
  paper_id: string
}

export interface DatasetNeighborhood {
  dataset_id: string
  linked_papers: Array<{
    paper_openalex_id: string
    paper_title: string | null
    paper_year: number | null
    paper_doi: string | null
    confidence: number
  }>
  finding_clusters: ConsensusRow[]
  related_datasets: Array<{
    dataset_id: string
    title: string
    shared_regions: string[]
  }>
  consensus_by_region: ConsensusRow[]
}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd apps/web && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add apps/web/package.json apps/web/package-lock.json apps/web/src/types/graph.ts
git commit -m "feat(graph): add react-force-graph deps and TypeScript graph types"
```

---

## Task 2: Build Cluster Graph Script (Python)

**Files:**
- Create: `scripts/literature/build_cluster_graph.py`
- Create: `tests/test_cluster_graph.py`

**Interfaces:**
- Consumes: `artifacts/literature/relationships/consensus_summaries.jsonl`, `artifacts/literature/relationships/finding_edges.jsonl`, `artifacts/literature/paper_dataset_links.jsonl`, corpus loaded via `build_combined_corpus()`
- Produces: `artifacts/graph/cluster_graph.json` with schema `{nodes: GraphNode[], links: GraphEdge[]}`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cluster_graph.py`:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.literature.build_cluster_graph import (
    BRAIN_SYSTEMS,
    DIRECTION_COLORS,
    build_graph,
    assign_system,
)


def test_direction_colors_complete():
    for direction in ("increase", "decrease", "correlation", "no_change"):
        assert direction in DIRECTION_COLORS


def test_assign_system_known_region():
    assert assign_system("hippocampus") == "hippocampal_formation"


def test_assign_system_unknown_region():
    assert assign_system("unknown_region_xyz") == "other"


def test_build_graph_returns_nodes_and_links():
    graph = build_graph(max_edges=100)
    assert "nodes" in graph
    assert "links" in graph
    assert len(graph["nodes"]) > 0
    assert len(graph["links"]) > 0


def test_build_graph_node_schema():
    graph = build_graph(max_edges=10)
    for node in graph["nodes"]:
        assert "id" in node
        assert "type" in node
        assert node["type"] in ("system", "region", "finding_cluster", "dataset", "paper")
        assert "label" in node
        assert "scale_level" in node
        assert "size" in node
        assert "color" in node
        assert "meta" in node


def test_build_graph_link_schema():
    graph = build_graph(max_edges=10)
    node_ids = {n["id"] for n in graph["nodes"]}
    for link in graph["links"]:
        assert "source" in link
        assert "target" in link
        assert "type" in link
        assert "weight" in link
        assert "color" in link
        assert link["source"] in node_ids, f"source {link['source']} not in nodes"
        assert link["target"] in node_ids, f"target {link['target']} not in nodes"


def test_build_graph_no_duplicate_node_ids():
    graph = build_graph(max_edges=10)
    ids = [n["id"] for n in graph["nodes"]]
    assert len(ids) == len(set(ids)), "Duplicate node IDs found"


def test_build_graph_system_nodes_present():
    graph = build_graph(max_edges=10)
    system_nodes = [n for n in graph["nodes"] if n["type"] == "system"]
    assert len(system_nodes) >= 5
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_cluster_graph.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — script doesn't exist yet

- [ ] **Step 3: Create `scripts/literature/build_cluster_graph.py`**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cluster_graph.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Run the script**

```bash
python scripts/literature/build_cluster_graph.py
```

Expected: `Done — NNN nodes, NNN links → artifacts/graph/cluster_graph.json`

- [ ] **Step 6: Commit**

```bash
git add scripts/literature/build_cluster_graph.py tests/test_cluster_graph.py artifacts/graph/cluster_graph.json
git commit -m "feat(graph): build cluster-level KG from literature artifacts"
```

---

## Task 3: Compute 3D Layout Script (Python)

**Files:**
- Create: `scripts/literature/compute_layout.py`
- Produces: `artifacts/graph/galaxy_points.json`

**Interfaces:**
- Consumes: `artifacts/graph/cluster_graph.json`
- Produces: `artifacts/graph/galaxy_points.json` with schema `{nodes: GalaxyPoint[]}`

- [ ] **Step 1: Check networkx is available**

```bash
python -c "import networkx; print(networkx.__version__)"
```

If missing: `pip install networkx`

- [ ] **Step 2: Create `scripts/literature/compute_layout.py`**

```python
"""Compute 3D spring-layout positions for all cluster graph nodes.

Reads:  artifacts/graph/cluster_graph.json
Writes: artifacts/graph/galaxy_points.json
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(__file__).parent.parent.parent
INPUT_PATH = REPO_ROOT / "artifacts/graph/cluster_graph.json"
OUTPUT_PATH = REPO_ROOT / "artifacts/graph/galaxy_points.json"

# Scale factor — spread nodes across a 2000-unit cube for three.js
SCALE = 400


def compute_layout() -> dict:
    data = json.loads(INPUT_PATH.read_text())
    nodes = data["nodes"]
    links = data["links"]

    G = nx.Graph()
    node_meta = {n["id"]: n for n in nodes}
    G.add_nodes_from(n["id"] for n in nodes)
    for link in links:
        G.add_edge(link["source"], link["target"])

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Computing 3D spring layout (may take 30–60 s for large graphs)…")

    # Use scale_level as a y-axis seed to enforce the biological-scale axis.
    # Nodes with higher scale_level (system/dataset) start higher in y.
    seed_pos: dict[str, tuple[float, float, float]] = {}
    rng = random.Random(42)
    for n in nodes:
        scale = node_meta[n["id"]]["scale_level"]
        seed_pos[n["id"]] = (
            rng.uniform(-1, 1),
            (scale - 2) * 0.3 + rng.uniform(-0.1, 0.1),
            rng.uniform(-1, 1),
        )

    pos = nx.spring_layout(G, dim=3, seed=42, pos=seed_pos, k=0.8, iterations=80)

    galaxy_nodes = []
    for n in nodes:
        nid = n["id"]
        if nid not in pos:
            continue
        x, y, z = pos[nid]
        galaxy_nodes.append({
            "id": nid,
            "x": round(float(x) * SCALE, 2),
            "y": round(float(y) * SCALE, 2),
            "z": round(float(z) * SCALE, 2),
            "type": n["type"],
            "color": n["color"],
            "size": n["size"],
            "label": n["label"],
        })

    return {"nodes": galaxy_nodes}


def main() -> None:
    layout = compute_layout()
    OUTPUT_PATH.write_text(json.dumps(layout))
    print(f"Done — {len(layout['nodes'])} positioned nodes → {OUTPUT_PATH}")
    sizes = [abs(n["x"]) for n in layout["nodes"]]
    print(f"X range: {min(sizes):.1f} – {max(sizes):.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the script**

```bash
python scripts/literature/compute_layout.py
```

Expected: `Done — NNN positioned nodes → artifacts/graph/galaxy_points.json`
(Takes 30–90 s depending on node count)

- [ ] **Step 4: Verify output**

```bash
python -c "
import json
data = json.load(open('artifacts/graph/galaxy_points.json'))
print('nodes:', len(data['nodes']))
print('sample:', data['nodes'][0])
"
```

Expected: node count matches cluster_graph.json, each has x/y/z floats

- [ ] **Step 5: Commit**

```bash
git add scripts/literature/compute_layout.py artifacts/graph/galaxy_points.json
git commit -m "feat(graph): compute 3D spring layout for galaxy point cloud"
```

---

## Task 4: Graph API Endpoints (Python)

**Files:**
- Create: `apps/api/graph_router.py`
- Modify: `apps/api/main.py` (add `include_router`)
- Create: `tests/test_graph_api.py`

**Interfaces:**
- Consumes: `artifacts/graph/cluster_graph.json`, `artifacts/graph/galaxy_points.json`, `artifacts/literature/relationships/consensus_summaries.jsonl`, `artifacts/literature/findings_tier1_ollama.jsonl`, `artifacts/literature/paper_dataset_links.jsonl`, `_demo_data` corpus from `main.py`
- Produces:
  - `GET /api/graph/overview` → `SubgraphResponse`
  - `GET /api/graph/subgraph` → `SubgraphResponse`
  - `GET /api/graph/topic/{slug}` → `TopicGraphResponse`
  - `GET /api/literature/consensus` → `list[ConsensusRow]`
  - `GET /api/literature/findings` → `list[FindingRow]`
  - `GET /api/datasets/{id}/neighborhood` → `DatasetNeighborhood`

- [ ] **Step 1: Write failing tests**

Create `tests/test_graph_api.py`:

```python
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.api.main import app

client = TestClient(app)


def test_graph_overview_returns_nodes_and_links():
    r = client.get("/api/graph/overview")
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data
    assert "links" in data
    assert "meta" in data
    assert data["meta"]["node_count"] == len(data["nodes"])


def test_graph_overview_node_schema():
    r = client.get("/api/graph/overview")
    data = r.json()
    for node in data["nodes"][:5]:
        assert "id" in node
        assert "type" in node
        assert node["type"] in ("system", "region", "finding_cluster", "dataset", "paper")
        assert "label" in node
        assert "color" in node


def test_graph_subgraph_with_region_filter():
    r = client.get("/api/graph/subgraph", params={"regions": "hippocampus"})
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data
    assert len(data["nodes"]) > 0
    # Should include a hippocampus region or cluster node
    ids = [n["id"] for n in data["nodes"]]
    assert any("hippocampus" in nid for nid in ids)


def test_graph_subgraph_respects_limit():
    r = client.get("/api/graph/subgraph", params={"limit": 10})
    assert r.status_code == 200
    data = r.json()
    assert len(data["nodes"]) <= 10


def test_graph_topic_hippocampal():
    r = client.get("/api/graph/topic/hippocampal")
    assert r.status_code == 200
    data = r.json()
    assert "topic" in data
    assert data["topic"]["slug"] == "hippocampal"
    assert "nodes" in data


def test_graph_topic_unknown_slug():
    r = client.get("/api/graph/topic/nonexistent_slug_xyz")
    assert r.status_code == 404


def test_literature_consensus():
    r = client.get("/api/literature/consensus")
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list)
    if rows:
        row = rows[0]
        assert "region" in row
        assert "direction" in row
        assert "n_findings" in row
        assert "consensus_strength" in row


def test_literature_findings_with_region():
    r = client.get("/api/literature/findings", params={"region": "hippocampus", "limit": 5})
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list)
    assert len(rows) <= 5


def test_dataset_neighborhood():
    r = client.get("/api/datasets/dandi:000003/neighborhood")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        data = r.json()
        assert "dataset_id" in data
        assert "linked_papers" in data
        assert "finding_clusters" in data
        assert "consensus_by_region" in data
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_graph_api.py -v
```

Expected: failures — router doesn't exist yet

- [ ] **Step 3: Create `apps/api/graph_router.py`**

```python
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
```

- [ ] **Step 4: Add router to `apps/api/main.py`**

Find the line `app = FastAPI(` in `main.py`. After the `app.add_middleware(CORSMiddleware, ...)` block (around line 115), add:

```python
from apps.api.graph_router import router as graph_router
app.include_router(graph_router)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_graph_api.py -v
```

Expected: all 9 tests PASS (some may be 404 if data is sparse — that is acceptable)

- [ ] **Step 6: Smoke test the running API**

```bash
make api &
curl -s http://localhost:8000/api/graph/overview | python3 -c "import sys,json; d=json.load(sys.stdin); print('nodes:', len(d['nodes']), 'links:', len(d['links']))"
```

Expected: `nodes: NNN links: NNN`

- [ ] **Step 7: Commit**

```bash
git add apps/api/graph_router.py apps/api/main.py tests/test_graph_api.py
git commit -m "feat(api): add graph + literature API endpoints for Knowledge Explorer"
```

---

## Task 5: Frontend API Client

**Files:**
- Create: `apps/web/src/api/graph.ts`

**Interfaces:**
- Consumes: all 6 endpoints from Task 4
- Produces: `fetchGalaxyLayout`, `fetchGraphOverview`, `fetchSubgraph`, `fetchTopicGraph`, `fetchSuggestedViews`, `fetchConsensus`, `fetchFindings`, `fetchDatasetNeighborhood` — used by Tasks 6–11

- [ ] **Step 1: Create `apps/web/src/api/graph.ts`**

```typescript
import type {
  ConsensusRow,
  DatasetNeighborhood,
  FindingRow,
  GalaxyLayout,
  SubgraphResponse,
  SuggestedView,
  TopicGraphResponse,
} from '../types/graph'

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function get<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(`${BASE}${path}`)
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, String(v)))
  }
  const res = await fetch(url.toString())
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${path}`)
  return res.json() as Promise<T>
}

export function fetchGalaxyLayout(): Promise<GalaxyLayout> {
  // Static file served by vite dev server or built assets
  return fetch('/graph/galaxy_points.json').then((r) => {
    if (!r.ok) throw new Error('galaxy_points.json not found — run compute_layout.py first')
    return r.json() as Promise<GalaxyLayout>
  })
}

export function fetchGraphOverview(limit = 400): Promise<SubgraphResponse> {
  return get('/api/graph/overview', { limit })
}

export function fetchSubgraph(params: {
  regions?: string
  species?: string
  tasks?: string
  limit?: number
}): Promise<SubgraphResponse> {
  return get('/api/graph/subgraph', {
    regions: params.regions ?? '',
    species: params.species ?? '',
    tasks: params.tasks ?? '',
    limit: params.limit ?? 400,
  })
}

export function fetchTopicGraph(slug: string): Promise<TopicGraphResponse> {
  return get(`/api/graph/topic/${slug}`)
}

export function fetchSuggestedViews(): Promise<SuggestedView[]> {
  return get('/api/graph/suggested-views')
}

export function fetchConsensus(region?: string): Promise<ConsensusRow[]> {
  return get('/api/literature/consensus', region ? { region } : {})
}

export function fetchFindings(params: {
  region?: string
  direction?: string
  limit?: number
}): Promise<FindingRow[]> {
  return get('/api/literature/findings', {
    region: params.region ?? '',
    direction: params.direction ?? '',
    limit: params.limit ?? 20,
  })
}

export function fetchDatasetNeighborhood(datasetId: string): Promise<DatasetNeighborhood> {
  return get(`/api/datasets/${encodeURIComponent(datasetId)}/neighborhood`)
}
```

- [ ] **Step 2: Copy `galaxy_points.json` to the vite public dir**

```bash
mkdir -p apps/web/public/graph
cp artifacts/graph/galaxy_points.json apps/web/public/graph/galaxy_points.json
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd apps/web && npx tsc --noEmit
```

Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/api/graph.ts apps/web/public/graph/galaxy_points.json
git commit -m "feat(web): graph API client and galaxy static asset"
```

---

## Task 6: GalaxyGraph Component (three.js Point Cloud)

**Files:**
- Create: `apps/web/src/components/graph/GalaxyGraph.tsx`

**Interfaces:**
- Consumes: `GalaxyPoint[]` from `GalaxyLayout.nodes`, `onPointClick: (id: string) => void`
- Produces: full-bleed WebGL canvas, auto-rotating, raycasted click events

- [ ] **Step 1: Create `apps/web/src/components/graph/GalaxyGraph.tsx`**

```tsx
import { useCallback, useEffect, useRef } from 'react'
import * as THREE from 'three'
import type { GalaxyPoint } from '../../types/graph'

// OrbitControls is in three/examples — import path depends on @types/three version
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

interface GalaxyGraphProps {
  points: GalaxyPoint[]
  onPointClick: (id: string, label: string) => void
}

export function GalaxyGraph({ points, onPointClick }: GalaxyGraphProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const clickRef = useRef(onPointClick)
  clickRef.current = onPointClick

  useEffect(() => {
    const mount = mountRef.current
    if (!mount || points.length === 0) return

    const width = mount.clientWidth
    const height = mount.clientHeight

    // Scene
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 20000)
    camera.position.set(0, 0, 800)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(width, height)
    renderer.setClearColor(0x020b14)
    mount.appendChild(renderer.domElement)

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.4
    controls.enableDamping = true
    controls.dampingFactor = 0.05

    // Point cloud
    const geometry = new THREE.BufferGeometry()
    const positions = new Float32Array(points.length * 3)
    const colors = new Float32Array(points.length * 3)
    const sizes = new Float32Array(points.length)

    points.forEach((p, i) => {
      positions[i * 3] = p.x
      positions[i * 3 + 1] = p.y
      positions[i * 3 + 2] = p.z
      const c = new THREE.Color(p.color)
      colors[i * 3] = c.r
      colors[i * 3 + 1] = c.g
      colors[i * 3 + 2] = c.b
      sizes[i] = Math.max(2, p.size * 0.6)
    })

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))

    const material = new THREE.PointsMaterial({
      size: 4,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      sizeAttenuation: true,
    })

    const cloud = new THREE.Points(geometry, material)
    scene.add(cloud)

    // Raycaster for click
    const raycaster = new THREE.Raycaster()
    raycaster.params.Points = { threshold: 8 }
    const mouse = new THREE.Vector2()

    const handleClick = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect()
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(mouse, camera)
      const hits = raycaster.intersectObject(cloud)
      if (hits.length > 0 && hits[0].index !== undefined) {
        const p = points[hits[0].index]
        clickRef.current(p.id, p.label)
      }
    }
    renderer.domElement.addEventListener('click', handleClick)

    // Resize
    const handleResize = () => {
      const w = mount.clientWidth
      const h = mount.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', handleResize)

    // Animate
    let rafId: number
    const animate = () => {
      rafId = requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    return () => {
      cancelAnimationFrame(rafId)
      window.removeEventListener('resize', handleResize)
      renderer.domElement.removeEventListener('click', handleClick)
      controls.dispose()
      renderer.dispose()
      geometry.dispose()
      material.dispose()
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
  }, [points])

  return <div ref={mountRef} className="w-full h-full" />
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd apps/web && npx tsc --noEmit
```

Expected: no errors (if OrbitControls import causes issues, try `three/addons/controls/OrbitControls.js`)

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/graph/GalaxyGraph.tsx
git commit -m "feat(graph): GalaxyGraph three.js point cloud component"
```

---

## Task 7: ExplorerGraph Component (react-force-graph 3D/2D)

**Files:**
- Create: `apps/web/src/components/graph/ExplorerGraph.tsx`

**Interfaces:**
- Consumes: `graphData: GraphData`, `mode: '3d' | '2d'`, `onNodeClick: (node: GraphNode) => void`
- Produces: interactive force-directed graph; emits clicked node to parent

- [ ] **Step 1: Create `apps/web/src/components/graph/ExplorerGraph.tsx`**

```tsx
import { useCallback, useRef } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import ForceGraph3D from 'react-force-graph-3d'
import type { GraphData, GraphEdge, GraphNode } from '../../types/graph'

interface ExplorerGraphProps {
  graphData: GraphData
  mode: '3d' | '2d'
  onNodeClick: (node: GraphNode) => void
  highlightIds?: Set<string>
}

const NODE_REL_SIZE = 4

export function ExplorerGraph({ graphData, mode, onNodeClick, highlightIds }: ExplorerGraphProps) {
  const fgRef = useRef<unknown>(null)

  const handleNodeClick = useCallback(
    (node: object) => {
      onNodeClick(node as GraphNode)
    },
    [onNodeClick],
  )

  const nodeColor = useCallback(
    (node: object) => {
      const n = node as GraphNode
      if (highlightIds && highlightIds.size > 0) {
        return highlightIds.has(n.id) ? n.color : `${n.color}44`
      }
      return n.color
    },
    [highlightIds],
  )

  const nodeVal = useCallback((node: object) => (node as GraphNode).size, [])
  const linkColor = useCallback((link: object) => (link as GraphEdge).color, [])
  const linkWidth = useCallback((link: object) => (link as GraphEdge).weight * 0.8, [])
  const nodeLabel = useCallback((node: object) => (node as GraphNode).label, [])

  const commonProps = {
    ref: fgRef,
    graphData,
    nodeId: 'id' as const,
    nodeLabel,
    nodeColor,
    nodeVal,
    linkColor,
    linkWidth,
    linkDirectionalParticles: 1,
    linkDirectionalParticleSpeed: 0.004,
    onNodeClick: handleNodeClick,
    backgroundColor: '#020b14',
    nodeRelSize: NODE_REL_SIZE,
  }

  if (mode === '2d') {
    return (
      <ForceGraph2D
        {...commonProps}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const n = node as GraphNode & { x: number; y: number }
          const r = Math.sqrt(n.size) * NODE_REL_SIZE
          ctx.beginPath()
          ctx.arc(n.x, n.y, r, 0, 2 * Math.PI)
          ctx.fillStyle = nodeColor(node)
          ctx.fill()
          if (globalScale > 2 && n.label) {
            ctx.font = `${10 / globalScale}px Inter, sans-serif`
            ctx.fillStyle = '#e2e8f0'
            ctx.textAlign = 'center'
            ctx.fillText(n.label.slice(0, 20), n.x, n.y + r + 4 / globalScale)
          }
        }}
      />
    )
  }

  return <ForceGraph3D {...commonProps} />
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd apps/web && npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/graph/ExplorerGraph.tsx
git commit -m "feat(graph): ExplorerGraph react-force-graph 3D/2D component"
```

---

## Task 8: Control Panel Components

**Files:**
- Create: `apps/web/src/components/graph/GraphControls.tsx`
- Create: `apps/web/src/components/graph/FilterPanel.tsx`
- Create: `apps/web/src/components/graph/SuggestedViews.tsx`
- Create: `apps/web/src/components/graph/OntologyTreePanel.tsx`
- Create: `apps/web/src/components/graph/GraphLegend.tsx`

**Interfaces:**
- All consume state props from `KnowledgeExplorerPage` and emit callbacks
- `GraphControls` produces: `onViewModeChange`, `onLayerModeChange`
- `FilterPanel` produces: `onFiltersChange: (regions, species, tasks) => void`
- `SuggestedViews` produces: `onViewSelect: (slug: string) => void`
- `OntologyTreePanel` produces: `onRegionSelect: (region: string) => void`

- [ ] **Step 1: Create `apps/web/src/components/graph/GraphControls.tsx`**

```tsx
import type { LayerMode, ViewMode } from '../../types/graph'

const VIEW_MODES: { value: ViewMode; label: string; icon: string }[] = [
  { value: 'galaxy', label: 'Galaxy', icon: '🌌' },
  { value: 'explorer', label: 'Explorer', icon: '🔭' },
  { value: '2d', label: '2D', icon: '📐' },
]

const LAYER_MODES: { value: LayerMode; label: string }[] = [
  { value: 'corpus', label: 'Corpus' },
  { value: 'consensus', label: 'Consensus' },
  { value: 'literature', label: 'Literature' },
  { value: 'bridge', label: 'Bridge' },
  { value: 'morphology', label: 'Morphology' },
]

interface GraphControlsProps {
  viewMode: ViewMode
  layerMode: LayerMode
  onViewModeChange: (mode: ViewMode) => void
  onLayerModeChange: (mode: LayerMode) => void
  onLegendToggle: () => void
}

export function GraphControls({
  viewMode, layerMode, onViewModeChange, onLayerModeChange, onLegendToggle,
}: GraphControlsProps) {
  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2 bg-neural-950/80 backdrop-blur border border-neural-800/50 rounded-lg px-3 py-2">
      {/* View mode */}
      <div className="flex gap-1 border-r border-neural-800 pr-2 mr-1">
        {VIEW_MODES.map(({ value, label, icon }) => (
          <button
            key={value}
            type="button"
            onClick={() => onViewModeChange(value)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              viewMode === value
                ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30'
                : 'text-neural-400 hover:text-neural-200 border border-transparent'
            }`}
          >
            <span>{icon}</span>
            {label}
          </button>
        ))}
      </div>

      {/* Layer mode */}
      <select
        value={layerMode}
        onChange={(e) => onLayerModeChange(e.target.value as LayerMode)}
        className="bg-neural-900 border border-neural-700 rounded px-2 py-1.5 text-xs text-neural-200 focus:outline-none focus:border-neural-500"
      >
        {LAYER_MODES.map(({ value, label }) => (
          <option key={value} value={value}>Layer: {label}</option>
        ))}
      </select>

      {/* Legend */}
      <button
        type="button"
        onClick={onLegendToggle}
        className="ml-1 text-xs text-neural-500 hover:text-neural-200 px-2 py-1.5 rounded border border-transparent hover:border-neural-700 transition-colors"
        title="Toggle legend"
      >
        ?
      </button>
    </div>
  )
}
```

- [ ] **Step 2: Create `apps/web/src/components/graph/FilterPanel.tsx`**

```tsx
import { useState } from 'react'

const SPECIES_OPTIONS = ['mouse', 'rat', 'human', 'macaque', 'zebrafish']

interface FilterPanelProps {
  regions: string[]
  species: string[]
  tasks: string[]
  onFiltersChange: (filters: { regions: string[]; species: string[]; tasks: string[] }) => void
}

export function FilterPanel({ regions, species, tasks, onFiltersChange }: FilterPanelProps) {
  const [regionInput, setRegionInput] = useState('')

  const addRegion = () => {
    const r = regionInput.trim().toLowerCase()
    if (r && !regions.includes(r)) {
      onFiltersChange({ regions: [...regions, r], species, tasks })
    }
    setRegionInput('')
  }

  const removeRegion = (r: string) =>
    onFiltersChange({ regions: regions.filter((x) => x !== r), species, tasks })

  const toggleSpecies = (s: string) => {
    const next = species.includes(s) ? species.filter((x) => x !== s) : [...species, s]
    onFiltersChange({ regions, species: next, tasks })
  }

  const clearAll = () => onFiltersChange({ regions: [], species: [], tasks: [] })

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-widest text-neural-600">Filters</span>
        {(regions.length > 0 || species.length > 0 || tasks.length > 0) && (
          <button type="button" onClick={clearAll} className="text-xs text-neural-600 hover:text-neural-300">
            Clear
          </button>
        )}
      </div>

      {/* Region input */}
      <div>
        <span className="block text-xs text-neural-500 mb-1.5">Region</span>
        <div className="flex gap-1">
          <input
            value={regionInput}
            onChange={(e) => setRegionInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addRegion() } }}
            placeholder="e.g. hippocampus"
            className="flex-1 bg-neural-900 border border-neural-700 rounded px-2 py-1.5 text-xs text-neural-200 placeholder-neural-600 focus:outline-none focus:border-neural-500"
          />
          <button
            type="button"
            onClick={addRegion}
            className="px-2 py-1.5 text-xs bg-neural-800 text-neural-200 rounded hover:bg-neural-700"
          >
            +
          </button>
        </div>
        {regions.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1.5">
            {regions.map((r) => (
              <span key={r} className="inline-flex items-center gap-1 text-xs bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/30 rounded px-2 py-0.5">
                {r}
                <button type="button" onClick={() => removeRegion(r)} className="hover:text-white">×</button>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Species */}
      <div>
        <span className="block text-xs text-neural-500 mb-1.5">Species</span>
        <div className="flex flex-wrap gap-1">
          {SPECIES_OPTIONS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => toggleSpecies(s)}
              className={`text-xs rounded px-2 py-0.5 border transition-colors ${
                species.includes(s)
                  ? 'bg-accent-emerald/10 text-accent-emerald border-accent-emerald/30'
                  : 'text-neural-600 border-neural-800 hover:text-neural-300'
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Create `apps/web/src/components/graph/SuggestedViews.tsx`**

```tsx
import { useQuery } from '@tanstack/react-query'
import { fetchSuggestedViews } from '../../api/graph'
import type { SuggestedView } from '../../types/graph'

interface SuggestedViewsProps {
  activeSlug: string | null
  companionSlugs: string[]
  onViewSelect: (slug: string) => void
}

export function SuggestedViews({ activeSlug, companionSlugs, onViewSelect }: SuggestedViewsProps) {
  const { data: views = [] } = useQuery<SuggestedView[]>({
    queryKey: ['suggested-views'],
    queryFn: fetchSuggestedViews,
    staleTime: Infinity,
  })

  return (
    <div className="space-y-3">
      <span className="block text-xs uppercase tracking-widest text-neural-600">Views</span>

      <div className="space-y-1">
        {views.map((view) => (
          <button
            key={view.slug}
            type="button"
            onClick={() => onViewSelect(view.slug)}
            className={`w-full text-left px-2 py-1.5 rounded text-xs transition-colors ${
              activeSlug === view.slug
                ? 'bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20'
                : 'text-neural-500 hover:text-neural-200 hover:bg-neural-900/50'
            }`}
            title={view.description}
          >
            {view.label}
          </button>
        ))}
      </div>

      {companionSlugs.length > 0 && (
        <div>
          <span className="block text-xs text-neural-700 mb-1">Also explore</span>
          <div className="flex flex-wrap gap-1">
            {companionSlugs.map((slug) => {
              const view = views.find((v) => v.slug === slug)
              if (!view) return null
              return (
                <button
                  key={slug}
                  type="button"
                  onClick={() => onViewSelect(slug)}
                  className="text-xs text-neural-600 hover:text-accent-cyan border border-neural-800 hover:border-accent-cyan/30 rounded px-2 py-0.5 transition-colors"
                >
                  {view.label}
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Create `apps/web/src/components/graph/OntologyTreePanel.tsx`**

```tsx
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getOntology } from '../../api/search'
import type { Ontology } from '../../types'

// Brain system hierarchy for top-level tree
const SYSTEMS: { id: string; label: string; regions: string[] }[] = [
  { id: 'hippocampal_formation', label: 'Hippocampal Formation',
    regions: ['hippocampus', 'ca1', 'ca3', 'dentate gyrus', 'entorhinal cortex'] },
  { id: 'prefrontal', label: 'Prefrontal Cortex',
    regions: ['prefrontal cortex', 'anterior cingulate cortex', 'orbitofrontal cortex'] },
  { id: 'basal_ganglia', label: 'Basal Ganglia',
    regions: ['striatum', 'nucleus accumbens', 'globus pallidus', 'substantia nigra'] },
  { id: 'cerebellum', label: 'Cerebellum', regions: ['cerebellum', 'purkinje cell layer'] },
  { id: 'brainstem', label: 'Brainstem',
    regions: ['brainstem', 'midbrain', 'pons', 'medulla', 'locus coeruleus'] },
  { id: 'sensory_cortex', label: 'Sensory Cortex',
    regions: ['visual cortex', 'auditory cortex', 'somatosensory cortex'] },
  { id: 'motor_cortex', label: 'Motor Cortex',
    regions: ['motor cortex', 'primary motor cortex'] },
  { id: 'limbic', label: 'Limbic System',
    regions: ['amygdala', 'thalamus', 'hypothalamus', 'insula'] },
]

interface OntologyTreePanelProps {
  onRegionSelect: (region: string) => void
}

export function OntologyTreePanel({ onRegionSelect }: OntologyTreePanelProps) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })

  return (
    <div className="space-y-3">
      <span className="block text-xs uppercase tracking-widest text-neural-600">Ontology</span>
      <div className="space-y-0.5">
        {SYSTEMS.map((sys) => (
          <div key={sys.id}>
            <button
              type="button"
              onClick={() => toggle(sys.id)}
              className="w-full flex items-center gap-1.5 px-1.5 py-1 text-xs text-neural-400 hover:text-neural-200 rounded hover:bg-neural-900/50 transition-colors text-left"
            >
              <span className="text-neural-700 w-3 text-center flex-shrink-0">
                {expanded.has(sys.id) ? '▾' : '▸'}
              </span>
              {sys.label}
            </button>
            {expanded.has(sys.id) && (
              <div className="ml-4 space-y-0.5 mb-1">
                {sys.regions.map((region) => (
                  <button
                    key={region}
                    type="button"
                    onClick={() => onRegionSelect(region)}
                    className="w-full text-left px-2 py-0.5 text-xs text-neural-600 hover:text-accent-cyan hover:bg-accent-cyan/5 rounded transition-colors"
                  >
                    {region}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Create `apps/web/src/components/graph/GraphLegend.tsx`**

```tsx
interface GraphLegendProps {
  onClose: () => void
}

const NODE_TYPES = [
  { color: '#f59e0b', label: 'Brain System' },
  { color: '#fcd34d', label: 'Region' },
  { color: '#10b981', label: 'Finding cluster (↑ increase)' },
  { color: '#ef4444', label: 'Finding cluster (↓ decrease)' },
  { color: '#8b5cf6', label: 'Finding cluster (correlation)' },
  { color: '#22d3ee', label: 'Dataset' },
  { color: '#8b5cf6', label: 'Paper' },
]

const EDGE_TYPES = [
  { color: '#10b981', label: 'Supports' },
  { color: '#ef4444', label: 'Contradicts' },
  { color: '#ffffff30', label: 'Dataset → Paper' },
  { color: '#fcd34d30', label: 'Region → System' },
]

export function GraphLegend({ onClose }: GraphLegendProps) {
  return (
    <div className="absolute top-16 right-4 z-30 bg-neural-950/95 backdrop-blur border border-neural-800/50 rounded-lg p-4 w-56">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs uppercase tracking-widest text-neural-600">Legend</span>
        <button type="button" onClick={onClose} className="text-neural-600 hover:text-neural-300 text-xs">✕</button>
      </div>

      <p className="text-xs text-neural-600 mb-2">Nodes</p>
      <div className="space-y-1.5 mb-4">
        {NODE_TYPES.map(({ color, label }) => (
          <div key={label} className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
            <span className="text-xs text-neural-500">{label}</span>
          </div>
        ))}
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full flex-shrink-0 bg-neural-500" />
          <span className="text-xs text-neural-500">Node size ∝ finding count</span>
        </div>
      </div>

      <p className="text-xs text-neural-600 mb-2">Edges</p>
      <div className="space-y-1.5">
        {EDGE_TYPES.map(({ color, label }) => (
          <div key={label} className="flex items-center gap-2">
            <div className="w-5 h-0.5 flex-shrink-0" style={{ backgroundColor: color }} />
            <span className="text-xs text-neural-500">{label}</span>
          </div>
        ))}
        <div className="flex items-center gap-2">
          <div className="w-5 h-0.5 flex-shrink-0 bg-neural-500" />
          <span className="text-xs text-neural-500">Edge width ∝ evidence count</span>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 6: Verify TypeScript compiles**

```bash
cd apps/web && npx tsc --noEmit
```

- [ ] **Step 7: Commit**

```bash
git add apps/web/src/components/graph/
git commit -m "feat(graph): GraphControls, FilterPanel, SuggestedViews, OntologyTree, Legend"
```

---

## Task 9: NodeDetailPanel Component

**Files:**
- Create: `apps/web/src/components/graph/NodeDetailPanel.tsx`

**Interfaces:**
- Consumes: `node: GraphNode | null`, `findings: FindingRow[]`, `consensus: ConsensusRow[]`
- Produces: bottom-anchored detail strip that adapts to node type; emits `onSearchDatasets(query)`

- [ ] **Step 1: Create `apps/web/src/components/graph/NodeDetailPanel.tsx`**

```tsx
import { useNavigate } from 'react-router-dom'
import type { ConsensusRow, FindingRow, GraphNode } from '../../types/graph'

interface NodeDetailPanelProps {
  node: GraphNode | null
  findings: FindingRow[]
  consensus: ConsensusRow[]
  onClose: () => void
}

const DIRECTION_LABELS: Record<string, string> = {
  increase: '↑ increase',
  decrease: '↓ decrease',
  correlation: '↔ correlation',
  no_change: '— no change',
}
const DIRECTION_COLORS: Record<string, string> = {
  increase: 'text-accent-emerald',
  decrease: 'text-red-400',
  correlation: 'text-accent-violet',
  no_change: 'text-neural-500',
}

function ConsensusMiniBar({ strength }: { strength: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1 bg-neural-800 rounded overflow-hidden">
        <div
          className="h-full bg-accent-emerald rounded transition-all"
          style={{ width: `${Math.round(strength * 100)}%` }}
        />
      </div>
      <span className="text-xs text-neural-500 tabular-nums w-8 text-right">
        {(strength * 100).toFixed(0)}%
      </span>
    </div>
  )
}

export function NodeDetailPanel({ node, findings, consensus, onClose }: NodeDetailPanelProps) {
  const navigate = useNavigate()

  if (!node) return null

  const searchQuery = node.type === 'finding_cluster'
    ? `${(node.meta.region as string) ?? ''} ${(node.meta.direction as string) ?? ''}`.trim()
    : node.type === 'region'
    ? (node.label ?? '')
    : node.type === 'dataset'
    ? node.label
    : ''

  return (
    <div className="absolute bottom-0 left-0 right-0 z-20 bg-neural-950/95 backdrop-blur border-t border-neural-800/50 p-4">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-start justify-between gap-6">
          {/* Left: node info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs uppercase tracking-widest text-neural-600">{node.type.replace('_', ' ')}</span>
              <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: node.color }} />
            </div>
            <h3 className="text-base font-medium text-neural-100 mb-2 truncate">{node.label}</h3>

            {/* Finding cluster details */}
            {node.type === 'finding_cluster' && (
              <div className="flex flex-wrap gap-4 text-xs text-neural-400">
                <span>{node.meta.n_findings as number} findings</span>
                <span>{node.meta.n_papers as number} papers</span>
                <span className={DIRECTION_COLORS[node.meta.direction as string] ?? ''}>
                  {DIRECTION_LABELS[node.meta.direction as string] ?? node.meta.direction as string}
                </span>
                <span>consensus strength</span>
                <div className="w-32">
                  <ConsensusMiniBar strength={node.meta.consensus_strength as number ?? 0} />
                </div>
              </div>
            )}

            {/* Region details */}
            {node.type === 'region' && consensus.length > 0 && (
              <div className="flex flex-wrap gap-3 mt-1">
                {consensus.slice(0, 4).map((c) => (
                  <div key={`${c.region}-${c.direction}`} className="flex items-center gap-1.5">
                    <span className={`text-xs ${DIRECTION_COLORS[c.direction] ?? 'text-neural-400'}`}>
                      {DIRECTION_LABELS[c.direction]}
                    </span>
                    <span className="text-xs text-neural-600">{c.n_findings} findings</span>
                  </div>
                ))}
              </div>
            )}

            {/* Dataset details */}
            {node.type === 'dataset' && (
              <div className="text-xs text-neural-500">
                Source: {(node.meta.source as string) ?? '—'} · Readiness: {(node.meta.readiness as number) ?? 0}
              </div>
            )}

            {/* Paper details */}
            {node.type === 'paper' && (
              <div className="text-xs text-neural-500">
                {(node.meta.year as number) ? `${node.meta.year as number} · ` : ''}
                {(node.meta.doi as string) ?? ''}
              </div>
            )}

            {/* Top findings */}
            {findings.length > 0 && (
              <div className="mt-2 space-y-1">
                {findings.slice(0, 2).map((f) => (
                  <p key={f.finding_id} className="text-xs text-neural-500 line-clamp-1 italic">
                    "{f.finding_text}"
                  </p>
                ))}
              </div>
            )}
          </div>

          {/* Right: actions */}
          <div className="flex flex-col gap-2 flex-shrink-0">
            {searchQuery && (
              <button
                type="button"
                onClick={() => navigate(`/search?q=${encodeURIComponent(searchQuery)}`)}
                className="text-xs text-accent-cyan hover:text-white border border-accent-cyan/30 hover:border-accent-cyan rounded px-3 py-1.5 transition-colors whitespace-nowrap"
              >
                Search datasets →
              </button>
            )}
            {node.type === 'dataset' && (
              <button
                type="button"
                onClick={() => navigate(`/datasets/${node.id.replace('dataset:', '')}`)}
                className="text-xs text-neural-400 hover:text-neural-200 border border-neural-700 rounded px-3 py-1.5 transition-colors"
              >
                View card →
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              className="text-xs text-neural-600 hover:text-neural-400"
            >
              Close ✕
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd apps/web && npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/graph/NodeDetailPanel.tsx
git commit -m "feat(graph): NodeDetailPanel with type-adaptive node details"
```

---

## Task 10: KnowledgeExplorerPage — Main Page Wiring

**Files:**
- Create: `apps/web/src/pages/KnowledgeExplorerPage.tsx`
- Modify: `apps/web/src/App.tsx`

**Interfaces:**
- Consumes: all graph components from Tasks 6–9, all API functions from Task 5
- Produces: the `/graph` route, complete interactive Knowledge Explorer

- [ ] **Step 1: Create `apps/web/src/pages/KnowledgeExplorerPage.tsx`**

```tsx
import { useCallback, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  fetchDatasetNeighborhood,
  fetchFindings,
  fetchGalaxyLayout,
  fetchGraphOverview,
  fetchSubgraph,
  fetchTopicGraph,
} from '../api/graph'
import { GalaxyGraph } from '../components/graph/GalaxyGraph'
import { ExplorerGraph } from '../components/graph/ExplorerGraph'
import { GraphControls } from '../components/graph/GraphControls'
import { FilterPanel } from '../components/graph/FilterPanel'
import { SuggestedViews } from '../components/graph/SuggestedViews'
import { OntologyTreePanel } from '../components/graph/OntologyTreePanel'
import { NodeDetailPanel } from '../components/graph/NodeDetailPanel'
import { GraphLegend } from '../components/graph/GraphLegend'
import type { GraphData, GraphNode, LayerMode, ViewMode } from '../types/graph'

interface ActiveFilters {
  regions: string[]
  species: string[]
  tasks: string[]
}

const EMPTY_FILTERS: ActiveFilters = { regions: [], species: [], tasks: [] }

function filtersToParams(f: ActiveFilters) {
  return {
    regions: f.regions.join(','),
    species: f.species.join(','),
    tasks: f.tasks.join(','),
  }
}

function hasFilters(f: ActiveFilters) {
  return f.regions.length > 0 || f.species.length > 0 || f.tasks.length > 0
}

export function KnowledgeExplorerPage() {
  const [viewMode, setViewMode] = useState<ViewMode>('galaxy')
  const [layerMode, setLayerMode] = useState<LayerMode>('corpus')
  const [filters, setFilters] = useState<ActiveFilters>(EMPTY_FILTERS)
  const [activeTopicSlug, setActiveTopicSlug] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [legendOpen, setLegendOpen] = useState(false)
  const [companionSlugs, setCompanionSlugs] = useState<string[]>([])

  // Galaxy layout (static file, loads once)
  const { data: galaxyLayout } = useQuery({
    queryKey: ['galaxy-layout'],
    queryFn: fetchGalaxyLayout,
    staleTime: Infinity,
  })

  // Overview graph (no filters)
  const { data: overviewData } = useQuery({
    queryKey: ['graph-overview'],
    queryFn: () => fetchGraphOverview(400),
    staleTime: 60_000,
  })

  // Subgraph (when filters active)
  const { data: subgraphData } = useQuery({
    queryKey: ['graph-subgraph', filters],
    queryFn: () => fetchSubgraph(filtersToParams(filters)),
    enabled: hasFilters(filters) && !activeTopicSlug,
    staleTime: 30_000,
  })

  // Topic graph
  const { data: topicData } = useQuery({
    queryKey: ['graph-topic', activeTopicSlug],
    queryFn: () => fetchTopicGraph(activeTopicSlug!),
    enabled: Boolean(activeTopicSlug),
    staleTime: 60_000,
  })

  // Findings for selected node
  const { data: nodefindings = [] } = useQuery({
    queryKey: ['node-findings', selectedNode?.id],
    queryFn: () => {
      const region = (selectedNode?.meta?.region as string) ?? selectedNode?.label ?? ''
      return fetchFindings({ region, limit: 5 })
    },
    enabled: Boolean(selectedNode && (selectedNode.type === 'finding_cluster' || selectedNode.type === 'region')),
    staleTime: 60_000,
  })

  const graphData = useMemo<GraphData>(() => {
    const raw = activeTopicSlug
      ? topicData
      : hasFilters(filters)
      ? subgraphData
      : overviewData
    if (!raw) return { nodes: [], links: [] }
    return { nodes: raw.nodes, links: raw.links }
  }, [activeTopicSlug, topicData, filters, subgraphData, overviewData])

  const handleTopicSelect = useCallback((slug: string) => {
    setActiveTopicSlug(slug)
    setFilters(EMPTY_FILTERS)
    setViewMode('explorer')
  }, [])

  const handleFiltersChange = useCallback((f: ActiveFilters) => {
    setFilters(f)
    setActiveTopicSlug(null)
    if (hasFilters(f)) setViewMode('explorer')
  }, [])

  const handleRegionSelect = useCallback((region: string) => {
    setFilters((prev) => ({
      ...prev,
      regions: prev.regions.includes(region) ? prev.regions : [...prev.regions, region],
    }))
    setActiveTopicSlug(null)
    setViewMode('explorer')
  }, [])

  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(node)
  }, [])

  const handleGalaxyPointClick = useCallback((id: string, label: string) => {
    // Extract region from id (e.g. "region:hippocampus" → "hippocampus")
    const parts = id.split(':')
    if (parts[0] === 'region' || parts[0] === 'cluster') {
      const region = parts[1]
      setFilters({ regions: [region], species: [], tasks: [] })
      setActiveTopicSlug(null)
      setViewMode('explorer')
    }
  }, [])

  // When topic data arrives, set companion slugs
  useMemo(() => {
    if (topicData?.topic?.companion_slugs) {
      setCompanionSlugs(topicData.topic.companion_slugs)
    } else {
      setCompanionSlugs([])
    }
  }, [topicData])

  return (
    <div className="relative w-full" style={{ height: 'calc(100vh - 56px)' }}>
      {/* Top controls */}
      <GraphControls
        viewMode={viewMode}
        layerMode={layerMode}
        onViewModeChange={setViewMode}
        onLayerModeChange={setLayerMode}
        onLegendToggle={() => setLegendOpen((v) => !v)}
      />

      {/* Legend overlay */}
      {legendOpen && <GraphLegend onClose={() => setLegendOpen(false)} />}

      {/* Left sidebar */}
      <div className="absolute top-0 left-0 bottom-0 z-10 w-52 bg-neural-950/80 backdrop-blur border-r border-neural-800/40 p-4 overflow-y-auto flex flex-col gap-6">
        <FilterPanel
          regions={filters.regions}
          species={filters.species}
          tasks={filters.tasks}
          onFiltersChange={handleFiltersChange}
        />
        <SuggestedViews
          activeSlug={activeTopicSlug}
          companionSlugs={companionSlugs}
          onViewSelect={handleTopicSelect}
        />
        <OntologyTreePanel onRegionSelect={handleRegionSelect} />
      </div>

      {/* Canvas area */}
      <div className="absolute top-0 left-52 right-0 bottom-0">
        {viewMode === 'galaxy' && (
          <GalaxyGraph
            points={galaxyLayout?.nodes ?? []}
            onPointClick={handleGalaxyPointClick}
          />
        )}
        {(viewMode === 'explorer' || viewMode === '2d') && (
          <ExplorerGraph
            graphData={graphData}
            mode={viewMode === '2d' ? '2d' : '3d'}
            onNodeClick={handleNodeClick}
          />
        )}
      </div>

      {/* Status bar */}
      {(activeTopicSlug || hasFilters(filters)) && (
        <div className="absolute top-14 right-4 z-10 flex items-center gap-2">
          <span className="text-xs text-neural-500 bg-neural-950/80 backdrop-blur border border-neural-800/40 rounded px-3 py-1.5">
            {graphData.nodes.length} nodes · {graphData.links.length} links
            {activeTopicSlug && topicData?.topic && ` · ${topicData.topic.label}`}
          </span>
          <button
            type="button"
            onClick={() => {
              setFilters(EMPTY_FILTERS)
              setActiveTopicSlug(null)
              setCompanionSlugs([])
              setSelectedNode(null)
              setViewMode('galaxy')
            }}
            className="text-xs text-neural-600 hover:text-neural-300 bg-neural-950/80 backdrop-blur border border-neural-800/40 rounded px-3 py-1.5 transition-colors"
          >
            Reset
          </button>
        </div>
      )}

      {/* Node detail panel */}
      <NodeDetailPanel
        node={selectedNode}
        findings={nodefindings}
        consensus={[]}
        onClose={() => setSelectedNode(null)}
      />

      {/* Loading states */}
      {viewMode === 'galaxy' && !galaxyLayout && (
        <div className="absolute inset-0 flex items-center justify-center bg-neural-950/50 z-30">
          <div className="flex items-center gap-3 text-neural-400 text-sm">
            <span className="w-4 h-4 border-2 border-neural-700 border-t-accent-cyan rounded-full animate-spin" />
            Loading galaxy…
          </div>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Update `apps/web/src/App.tsx`**

```tsx
import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { SearchPage } from './pages/SearchPage'
import { ResultsPage } from './pages/ResultsPage'
import { DatasetPage } from './pages/DatasetPage'
import { OntologyPage } from './pages/OntologyPage'
import { ReportsPage } from './pages/ReportsPage'
import { EvaluationPage } from './pages/EvaluationPage'
import { DemoPage } from './pages/DemoPage'
import { KnowledgeExplorerPage } from './pages/KnowledgeExplorerPage'
import { CoveragePage } from './pages/CoveragePage'
import { BrainAtlasPage } from './pages/BrainAtlasPage'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<SearchPage />} />
        <Route path="/search" element={<ResultsPage />} />
        <Route path="/datasets/:id" element={<DatasetPage />} />
        <Route path="/ontology" element={<OntologyPage />} />
        <Route path="/reports" element={<ReportsPage />} />
        <Route path="/evaluation" element={<EvaluationPage />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="/graph" element={<KnowledgeExplorerPage />} />
        <Route path="/coverage" element={<CoveragePage />} />
        <Route path="/atlas" element={<BrainAtlasPage />} />
      </Routes>
    </Layout>
  )
}

export default App
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd apps/web && npx tsc --noEmit
```

- [ ] **Step 4: Start dev server and verify `/graph` loads**

```bash
make api &    # terminal 1
cd apps/web && npm run dev   # terminal 2
# Open http://localhost:5173/graph
```

Expected:
- Galaxy view renders (rotating point cloud) once `galaxy_points.json` loads
- Left sidebar shows Filters + Views + Ontology sections
- Top bar shows Galaxy / Explorer / 2D toggles
- Clicking Galaxy → Explorer switches to force graph
- Clicking a suggested view applies filters and shows subgraph

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/pages/KnowledgeExplorerPage.tsx apps/web/src/App.tsx
git commit -m "feat(graph): KnowledgeExplorerPage — galaxy + explorer + controls wired"
```

---

## Task 11: Search Result Relationship Highlights

**Files:**
- Create: `apps/web/src/components/graph/RelatedFindingsPanel.tsx`
- Modify: `apps/web/src/components/DatasetCard.tsx`

**Interfaces:**
- Consumes: `datasetId: string`, `linkedPapers: LinkedPaper[]`, `brainRegions: string[]`, fetches from `fetchDatasetNeighborhood`
- Produces: expandable panel below "Open evidence" showing consensus badges, finding snippets, mini SVG, and "View in Knowledge Graph" link

- [ ] **Step 1: Create `apps/web/src/components/graph/RelatedFindingsPanel.tsx`**

```tsx
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { fetchDatasetNeighborhood } from '../../api/graph'
import type { ConsensusRow } from '../../types/graph'
import type { LinkedPaper } from '../../types'

interface RelatedFindingsPanelProps {
  datasetId: string
  brainRegions: string[]
  linkedPapers: LinkedPaper[]
}

const DIRECTION_LABELS: Record<string, string> = {
  increase: '↑',
  decrease: '↓',
  correlation: '↔',
  no_change: '—',
}
const DIRECTION_COLORS: Record<string, string> = {
  increase: 'text-accent-emerald border-accent-emerald/30 bg-accent-emerald/5',
  decrease: 'text-red-400 border-red-500/30 bg-red-500/5',
  correlation: 'text-accent-violet border-accent-violet/30 bg-accent-violet/5',
  no_change: 'text-neural-500 border-neural-700 bg-neural-900',
}

function ConsensusBadge({ row }: { row: ConsensusRow }) {
  const cls = DIRECTION_COLORS[row.direction] ?? 'text-neural-500 border-neural-700 bg-neural-900'
  return (
    <span className={`inline-flex items-center gap-1 text-xs border rounded px-2 py-0.5 ${cls}`}>
      {row.region} {DIRECTION_LABELS[row.direction]} {(row.consensus_strength * 100).toFixed(0)}%
      <span className="text-neural-600">({row.n_findings})</span>
    </span>
  )
}

function MiniGraph({ datasetId, paperCount, clusterCount }: { datasetId: string; paperCount: number; clusterCount: number }) {
  // Static SVG schematic — not force-simulated
  const paperNodes = Array.from({ length: Math.min(paperCount, 4) }, (_, i) => ({
    cx: 40 + i * 60,
    cy: 30,
  }))
  const clusterNodes = Array.from({ length: Math.min(clusterCount, 6) }, (_, i) => ({
    cx: 20 + i * 46,
    cy: 80,
    color: i % 2 === 0 ? '#10b981' : '#ef4444',
  }))

  return (
    <svg width="280" height="110" className="overflow-visible">
      {/* Paper nodes */}
      {paperNodes.map((p, i) => (
        <g key={i}>
          <circle cx={p.cx} cy={p.cy} r={10} fill="#8b5cf620" stroke="#8b5cf6" strokeWidth={1} />
          <text x={p.cx} y={p.cy + 1} textAnchor="middle" dominantBaseline="middle" fill="#8b5cf6" fontSize={6}>P</text>
        </g>
      ))}
      {/* Cluster nodes */}
      {clusterNodes.map((c, i) => (
        <g key={i}>
          {paperNodes.slice(0, 2).map((p, j) => (
            <line key={j} x1={p.cx} y1={p.cy + 10} x2={c.cx} y2={c.cy - 6} stroke="#ffffff15" strokeWidth={0.5} />
          ))}
          <circle cx={c.cx} cy={c.cy} r={7} fill={`${c.color}20`} stroke={c.color} strokeWidth={1} />
        </g>
      ))}
      <text x={140} y={105} textAnchor="middle" fill="#374151" fontSize={7}>
        papers → finding clusters
      </text>
    </svg>
  )
}

export function RelatedFindingsPanel({ datasetId, brainRegions, linkedPapers }: RelatedFindingsPanelProps) {
  const { data: neighborhood, isLoading } = useQuery({
    queryKey: ['dataset-neighborhood', datasetId],
    queryFn: () => fetchDatasetNeighborhood(datasetId),
    staleTime: 300_000,
    enabled: true,
  })

  if (isLoading) {
    return (
      <div className="mt-4 border border-neural-800/60 rounded-lg p-4 animate-pulse">
        <div className="h-3 w-32 bg-neural-800 rounded mb-2" />
        <div className="h-3 w-48 bg-neural-800 rounded" />
      </div>
    )
  }

  const consensusRows = neighborhood?.consensus_by_region ?? []
  const findings = neighborhood?.finding_clusters ?? []
  const papers = neighborhood?.linked_papers ?? []

  if (consensusRows.length === 0 && papers.length === 0) {
    return (
      <div className="mt-4 border border-neural-800/40 rounded-lg p-3">
        <p className="text-xs text-neural-600">No literature links found for this dataset yet.</p>
      </div>
    )
  }

  return (
    <div className="mt-4 border border-neural-800/60 rounded-lg p-4 bg-neural-950/50">
      <p className="text-xs uppercase tracking-wide text-neural-600 mb-3">Related Findings</p>

      {/* Consensus badges */}
      {consensusRows.length > 0 && (
        <div className="mb-3">
          <p className="text-xs text-neural-600 mb-1.5">Consensus by region</p>
          <div className="flex flex-wrap gap-1.5">
            {consensusRows.slice(0, 4).map((row) => (
              <ConsensusBadge key={`${row.region}-${row.direction}`} row={row} />
            ))}
          </div>
        </div>
      )}

      {/* Mini graph schematic */}
      {(papers.length > 0 || findings.length > 0) && (
        <div className="mb-3 overflow-hidden">
          <MiniGraph
            datasetId={datasetId}
            paperCount={papers.length}
            clusterCount={findings.length}
          />
        </div>
      )}

      {/* Finding snippets placeholder (top clusters only) */}
      {findings.slice(0, 2).map((f) => (
        <p key={`${f.region}-${f.direction}`} className="text-xs text-neural-500 italic mb-1">
          "{f.region}: {f.n_findings} findings toward {f.direction} (strength {(f.consensus_strength * 100).toFixed(0)}%)"
        </p>
      ))}

      {/* Deep-link */}
      <div className="mt-3">
        <Link
          to={`/graph?dataset=${encodeURIComponent(datasetId)}`}
          className="text-xs text-accent-cyan hover:text-white transition-colors"
        >
          View neighborhood in Knowledge Graph →
        </Link>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Add Related Findings toggle to `apps/web/src/components/DatasetCard.tsx`**

Find the block in `DatasetCard.tsx` that renders the action buttons (around line 563, the `<div className="flex items-center gap-4">` block). Add a state variable and toggle button:

At the top of the `DatasetCard` function, after the existing `useState` declarations, add:

```tsx
const [relatedFindingsOpen, setRelatedFindingsOpen] = useState(false)
```

In the actions `<div className="flex items-center gap-4">`, after the "Open evidence" button, add:

```tsx
<button
  onClick={(e) => { e.preventDefault(); setRelatedFindingsOpen((v) => !v) }}
  className="text-xs text-neural-400 hover:text-neural-200 transition-colors"
>
  {relatedFindingsOpen ? 'Close findings' : 'Related findings'}
</button>
```

After the `{detailsOpen && <EvidencePanel ... />}` block, add:

```tsx
{relatedFindingsOpen && (
  <RelatedFindingsPanel
    datasetId={dataset.id}
    brainRegions={dataset.brain_regions}
    linkedPapers={linked_papers ?? []}
  />
)}
```

Add the import at the top of `DatasetCard.tsx`:

```tsx
import { RelatedFindingsPanel } from './graph/RelatedFindingsPanel'
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd apps/web && npx tsc --noEmit
```

- [ ] **Step 4: Manual test**

```bash
# With API running, open http://localhost:5173/search?q=hippocampus
# 1. Find a result card that shows linked papers
# 2. Click "Related findings"
# 3. Verify: panel expands, consensus badges appear (or "No literature links" message)
# 4. Verify: "View neighborhood in Knowledge Graph →" link navigates to /graph
```

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/graph/RelatedFindingsPanel.tsx apps/web/src/components/DatasetCard.tsx
git commit -m "feat(search): RelatedFindingsPanel with consensus badges and KG deep-link"
```

---

## Task 12: Update Nav Link + Smoke Test Full Flow

**Files:**
- Modify: `apps/web/src/components/Layout.tsx`

**Interfaces:**
- Update nav link text from "Graph" to "Knowledge Graph" (or "KG Explorer")

- [ ] **Step 1: Update nav in `apps/web/src/components/Layout.tsx`**

Find the nav link that points to `/graph` and update its label:

```tsx
// Find the existing link (exact text varies) and change label to:
<Link to="/graph">Knowledge Graph</Link>
```

- [ ] **Step 2: Full smoke test — run both servers**

```bash
# Terminal 1
make api

# Terminal 2
cd apps/web && npm run dev
```

Walk through this checklist manually:

- [ ] `/` — search page loads, "Knowledge Graph" link visible in nav
- [ ] `/graph` — page loads without console errors
- [ ] Galaxy mode — point cloud appears and auto-rotates
- [ ] Clicking Galaxy → Explorer — switches to force-directed 3D graph
- [ ] Clicking a suggested view (e.g. "Hippocampal Circuits") — graph re-renders with filtered nodes
- [ ] Filter panel — type "hippocampus" + Enter, graph updates
- [ ] Ontology tree — expand "Hippocampal Formation", click "hippocampus", graph updates
- [ ] Node click — detail panel slides up from bottom
- [ ] "Search datasets →" button in detail panel — navigates to `/search?q=...`
- [ ] `/search?q=hippocampus` — results load, "Related findings" button visible
- [ ] "Related findings" toggle — panel opens, consensus badges render
- [ ] Legend `?` button — legend overlay opens
- [ ] Reset button — returns to galaxy view

- [ ] **Step 3: Run Python test suite**

```bash
pytest tests/test_cluster_graph.py tests/test_graph_api.py -v
```

Expected: all tests pass

- [ ] **Step 4: Final commit**

```bash
git add apps/web/src/components/Layout.tsx
git commit -m "feat(graph): update nav link; full Knowledge Explorer shipped"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Galaxy point cloud (three.js) — Task 6
- [x] Explorer 3D/2D (react-force-graph) — Task 7
- [x] Filter panel — Task 8
- [x] Suggested views + companion slugs — Task 8
- [x] Ontology tree sidebar (BrainKnow addition) — Task 8
- [x] Node detail panel — Task 9
- [x] Graph legend — Task 8
- [x] Layer switcher (corpus/consensus/literature/bridge/morphology) — Task 8 (UI) + Task 4 (data filtering)
- [x] KnowledgeExplorerPage wiring — Task 10
- [x] App.tsx route swap — Task 10
- [x] RelatedFindingsPanel in DatasetCard — Task 11
- [x] Consensus badges — Task 11
- [x] "View in Knowledge Graph" deep-link — Task 11
- [x] Pre-computation scripts — Tasks 2–3
- [x] API endpoints — Task 4
- [x] TypeScript types — Task 1
- [x] Frontend API client — Task 5
- [ ] **Insight Synthesis Panel** — not implemented (requires Anthropic API call; add as Sprint 2 Task 1)
- [ ] **Morphology layer data** — morphology node type defined but not populated; add as Sprint 2 Task 2

**Type consistency:** `GraphData.links` (not `.edges`) used consistently across all components. `GraphNode.id`, `GraphNode.color`, `GraphNode.size`, `GraphNode.label`, `GraphNode.meta` used consistently in ExplorerGraph, GalaxyGraph, NodeDetailPanel.

**Placeholder scan:** No TBDs or vague steps found. All code blocks are complete.

**Known limitations:**
- OrbitControls import path may need adjustment depending on `@types/three` version — fallback: `import { OrbitControls } from 'three/addons/controls/OrbitControls.js'`
- `galaxy_points.json` must be copied to `apps/web/public/graph/` after each rebuild; add to Makefile as `make graph-rebuild`
- Layer mode currently affects UI label only — layer-specific graph data differentiation is Sprint 2 work
