# Knowledge Explorer — Design Spec
**Date:** 2026-06-18  
**Status:** Approved  
**Branch:** claude/neuronpedia-foundation  

---

## Overview

Upgrade `GraphPage` into a multi-layer, multi-mode **Knowledge Explorer** that visualizes the full neural search knowledge graph — from a 80K+ point galaxy down to individual finding clusters and dataset neighborhoods. Simultaneously enrich search result cards with inline relationship highlights drawn from the literature layer.

This is a **starting point** designed for extensibility. New layers, views, and modes can be added without architectural changes. The system is built assuming the KG will continue to grow (more findings, more papers, more adapters).

---

## Goals

1. Showcase the depth and complexity of the KG visually — first impression should feel like exploring a scientific universe
2. Smooth, performant navigation at any scale (80K points in galaxy, 500 in explorer)
3. Multi-layer views: corpus map, finding consensus, cross-dataset bridges
4. Focused "lens" views driven by topic presets (TBI, hippocampus, reward, etc.)
5. Inline relationship highlighting in search results without leaving the search flow
6. Extensible: new views = new filter presets; new layers = new node/edge types

---

## Architecture

### Rendering Stack

Two renderers, one page:

| Mode | Renderer | Node count | Edges | Use |
|------|----------|-----------|-------|-----|
| Galaxy | `three.js Points` geometry | 80K+ | None | Full-scale overview, orientation |
| Explorer | `react-force-graph-3d` (WebGL) | 200–500 | 500–2000 | Interactive subgraph navigation |
| 2D | `react-force-graph-2d` (Canvas) | 200–500 | 500–2000 | Relationship density reading |

**Key insight:** Galaxy mode is a point cloud — no edge rendering — so it scales to the full KG without any cap. Explorer and 2D use the same `react-force-graph` API (one prop swap between 3D/2D), so switching between them is trivial.

### Data Model

**Node types:**

| Type | Color | Size basis | Source |
|------|-------|------------|--------|
| Brain system (L0) | amber `#f59e0b` | # sub-regions | Hardcoded (8 systems) |
| Region (L1) | amber lighter `#fcd34d` | # findings | `consensus_summaries.jsonl` |
| Finding cluster (L2) | emerald→red gradient by direction | # findings | `consensus_summaries.jsonl` |
| Dataset | cyan `#22d3ee` | readiness score | Corpus |
| Paper | violet `#8b5cf6` | # linked findings | `paper_dataset_links.jsonl` |

Finding cluster color: `#10b981` (increase) → `#f59e0b` (mixed/correlation) → `#ef4444` (decrease), based on consensus `direction` field.

**Edge types:**

| Edge | Color | Thickness | Layer |
|------|-------|-----------|-------|
| Region → System | amber dim | fixed | Corpus, Consensus |
| Dataset → Region | cyan dim | fixed | Corpus, Bridge |
| Paper → Region | violet dim | fixed | Literature, Bridge |
| Dataset → Paper | white 0.3 alpha | fixed | Bridge |
| Cluster supports cluster | `#10b981` | `n_supporting_papers` | Literature |
| Cluster contradicts cluster | `#ef4444` | `n_supporting_papers` | Literature |

### Cluster Hierarchy

Findings are never rendered as individual nodes. They're aggregated into semantic clusters that map to how neuroscientists think about the literature:

```
Level 0 — Brain systems (~8 nodes)
  cortex · hippocampal formation · basal ganglia · cerebellum · brainstem …

Level 1 — Specific regions (~100 nodes)
  hippocampus · CA1 · prefrontal cortex · striatum · VTA …

Level 2 — Finding clusters (~500–2K nodes)
  "hippocampus | increase | mouse | working_memory" → 1 node (n=47 findings)
  "hippocampus | decrease | human | epilepsy" → 1 node (n=23 findings)
  Source: consensus_summaries.jsonl, extended with species + task breakdown

Level 3 — Individual findings (side panel only, never graph nodes)
  Actual finding text, paper ID, confidence score
  Shown in NodeDetailPanel on cluster click, max 10 per panel
```

`consensus_summaries.jsonl` already gives Level 2 for free (`{region, direction, n_findings, consensus_strength}`). The build script extends this with species/task breakdown.

### Pre-computation (one-time backend jobs)

| Script | Input | Output | When to run |
|--------|-------|--------|-------------|
| `scripts/literature/build_cluster_graph.py` | findings JSONL + relationships JSONL | `artifacts/graph/cluster_graph.json` | After each extraction batch |
| `scripts/literature/compute_layout.py` | `cluster_graph.json` | `artifacts/graph/galaxy_points.json` | After cluster graph rebuild |

`galaxy_points.json` contains pre-computed `{id, x, y, z, type, color_value, size}` for every node. The frontend loads this as a static file — no API call, instant load. Explorer subgraphs are fetched dynamically on demand.

---

## New API Endpoints

All added to `apps/api/main.py` (or extracted to `apps/api/graph_router.py`):

```
GET /api/graph/overview
  → { nodes: GraphNode[], edges: GraphEdge[], meta: { node_count, edge_count } }
  Full cluster-level graph (no individual findings). Used by Explorer default state.

GET /api/graph/subgraph
  ?regions=hippocampus,prefrontal
  &species=mouse,human
  &tasks=working_memory
  &layers=corpus,literature
  &limit=400
  → { nodes: GraphNode[], edges: GraphEdge[] }
  Filtered subgraph. Limit enforced server-side; largest-connected nodes prioritized.

GET /api/graph/topic/{slug}
  slug: tbi | memory | reward | motor | decision_making | hippocampal | cross_species
  → { nodes: GraphNode[], edges: GraphEdge[], meta: { label, description, companion_slugs[] } }
  Pre-defined topic lens. companion_slugs drives "also explore" suggestions.

GET /api/literature/consensus
  → [{ region, direction, n_findings, n_papers, consensus_strength }]
  Powers the Consensus layer and RelatedFindingsPanel consensus badges.

GET /api/literature/findings
  ?region=hippocampus&species=mouse&direction=increase&limit=50
  → [{ finding_id, finding_text, region, direction, confidence, paper_id }]
  Powers NodeDetailPanel Level 3 drill-down.

GET /api/datasets/{id}/neighborhood
  → {
      dataset: DatasetRecord,
      linked_papers: LinkedPaper[],
      finding_clusters: FindingCluster[],
      related_datasets: { dataset_id, title, shared_regions[], shared_tasks[], shared_papers[] }[],
      consensus_by_region: { region, direction, n_findings, consensus_strength }[]
    }
  Powers RelatedFindingsPanel and "View in Knowledge Graph" deep-link.
```

**Shared types** (`apps/web/src/types/graph.ts`):
```typescript
type NodeType = 'system' | 'region' | 'finding_cluster' | 'dataset' | 'paper'
type EdgeType = 'covers' | 'studies' | 'linked' | 'supports' | 'contradicts'
type ViewMode = 'galaxy' | 'explorer' | '2d'
type LayerMode = 'corpus' | 'consensus' | 'literature' | 'bridge'

interface GraphNode {
  id: string
  type: NodeType
  label: string
  x?: number; y?: number; z?: number   // pre-computed for galaxy
  size: number
  color: string
  meta: Record<string, unknown>         // type-specific payload
}

interface GraphEdge {
  source: string
  target: string
  type: EdgeType
  weight: number
  color: string
}

interface FindingCluster {
  id: string
  region: string
  direction: 'increase' | 'decrease' | 'correlation' | 'no_change'
  n_findings: number
  n_papers: number
  consensus_strength: number
  species?: string[]
  tasks?: string[]
}
```

---

## UI Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│ [🌌 Galaxy] [🔭 Explorer] [📐 2D]   Layer: [Corpus ▾]       [?]   │ ← GraphControls
├──────────────┬──────────────────────────────────────────────────────┤
│              │                                                        │
│  FILTERS     │                                                        │
│  Region      │                                                        │
│  Species     │           3D Canvas (full bleed)                      │
│  Task        │                                                        │
│  Topic text  │   Galaxy: 80K points, auto-rotate, click to zoom      │
│              │   Explorer: force graph, drag/zoom/click               │
│  ──────────  │   2D: same graph flat, easier edge reading            │
│  VIEWS       │                                                        │
│  Hippocampal │                                                        │
│  Decision    │                                                        │
│  TBI         ├───────────────────────────────────────────────────────┤
│  Reward      │  NODE DETAIL PANEL  (slides up on node click)         │
│  Memory      │  Name · Type · Stats · Top findings · [→ Search]      │
│  Motor       │  Consensus chart for region nodes                     │
│  Cross-sp.   │  Related datasets for finding cluster nodes           │
└──────────────┴───────────────────────────────────────────────────────┘
```

**GraphControls** (top bar):
- View mode toggle: Galaxy | Explorer | 2D
- Layer switcher dropdown: Corpus | Consensus | Literature | Bridge
- Help icon (opens legend overlay)

**FilterPanel** (left sidebar):
- Region multi-select (typeahead, driven by ontology)
- Species checkboxes
- Task multi-select
- Free-text topic search (drives `/api/graph/subgraph` with text filter)
- Clear filters button

**SuggestedViews** (left sidebar, below filters):
- 7 preset chips; click auto-populates filters and fetches topic subgraph
- After topic loads: "Also explore: [Companion A] [Companion B]" appears

**NodeDetailPanel** (bottom strip, slides up):
- Region node: name, system, n_findings, n_papers, consensus chart (↑/↓/∼ bar)
- Finding cluster node: region, direction badge, top 5 finding texts, [→ search datasets]
- Dataset node: title, source, readiness score, linked papers, [→ view card]
- Paper node: title, year, linked finding count, [→ DOI]

**GraphLegend** (overlay, toggled by ?):
- Node type → color/shape mapping
- Edge type → color/thickness mapping
- Mode explanation (Galaxy vs Explorer vs 2D)

---

## Interaction Flow

```
1. Open /graph
   → galaxy_points.json loads (static, instant)
   → GalaxyGraph renders 80K+ points, auto-rotates

2. Click a region cluster (e.g. hippocampus blob)
   → Camera flies to cluster (800ms ease-out)
   → Mode switches: Galaxy → Explorer
   → GET /api/graph/subgraph?regions=hippocampus
   → ExplorerGraph renders hippocampus subgraph
   → NodeDetailPanel shows hippocampus stats

3. Apply filter: species = mouse
   → GET /api/graph/subgraph?regions=hippocampus&species=mouse
   → Nodes animate: non-matching fade out, new nodes fade in
   → Panel updates

4. Click "TBI" suggested view
   → Filters auto-set: regions=cortex,hippocampus,brainstem
   → GET /api/graph/topic/tbi
   → Explorer re-renders TBI subgraph
   → Panel shows: "TBI landscape · 847 findings · controversy index high"
   → Companion chips appear: "Neuroinflammation · Axonal injury"

5. Click a finding cluster node
   → Panel: cluster summary, top 5 finding texts, consensus bar
   → [→ Search datasets] button: navigates to /search?q=hippocampus+increase+mouse

6. Click a dataset node
   → Panel: dataset summary, readiness score, linked papers
   → [→ View card] navigates to /datasets/{id}

7. Switch to 2D
   → Same data, flat layout, edges more readable
   → Useful for relationship density comparison

8. Switch layer to Consensus
   → Region nodes recolor by dominant direction
   → Edge weights scale by consensus_strength
   → Finding cluster nodes suppress; region + system nodes emphasize
```

---

## Search Result Enhancements

### DatasetCard additions

**Knowledge badge** added to the existing source/id header row:
```
DANDI  000582  NWB  · 4 papers · 18 findings
```
Shown only if `linked_papers.length > 0`. Fetched as part of existing card data or the new `/api/datasets/{id}/neighborhood` endpoint.

**Related Findings panel** (new expandable section, toggle alongside "Open evidence"):

```
┌─ Related Findings ─────────────────────────────────────────────────┐
│                                                                     │
│  Consensus by region:                                               │
│  hippocampus  ↑ 0.48  (289 findings)                               │
│  striatum     ↑ 0.57  (74 findings)                                │
│                                                                     │
│  Top findings from linked papers:                                   │
│  "LTP in CA1 increases following contextual fear conditioning…"    │
│  "Striatal dopamine release is time-locked to reward delivery…"    │
│  "Place cell remapping occurs during spatial learning in CA1…"     │
│                                                                     │
│  [mini SVG: paper nodes → cluster nodes, 40–60 nodes]              │
│                                                                     │
│  [View neighborhood in Knowledge Graph →]                          │
└─────────────────────────────────────────────────────────────────────┘
```

Mini SVG graph: static (no force simulation), papers as violet circles, finding clusters as colored circles, edges as lines. Sized at ~300×150px. Not interactive — tap "View in KG" to go interactive.

"View in Knowledge Graph →" deep-links to `/graph?dataset={id}` which opens Explorer pre-focused on that dataset's neighborhood.

---

## Suggested Views

| Slug | Label | Filters | Layer | Companions |
|------|-------|---------|-------|------------|
| `hippocampal` | Hippocampal circuits | regions: hippocampus, CA1, CA3, entorhinal cortex | Literature | memory, cross_species |
| `decision` | Decision making | regions: PFC, striatum, ACC · tasks: decision_making, reversal_learning | Literature | reward |
| `tbi` | TBI landscape | regions: cortex, hippocampus, brainstem · species: human, mouse | Consensus | cross_species |
| `cross_species` | Cross-species | species: mouse + human · same region/task filter | Bridge | hippocampal, decision |
| `reward` | Reward & dopamine | regions: striatum, NAcc, VTA | Literature | decision |
| `memory` | Memory systems | regions: hippocampus, PFC, parietal · tasks: working_memory, spatial_navigation | Literature | hippocampal |
| `motor` | Motor circuits | regions: motor_cortex, cerebellum, spinal_cord · tasks: reaching, locomotion | Literature | — |

**Extensibility note:** New views are JSON config entries, not code changes. A `suggested_views.json` file or Python dict in `graph_router.py` drives the preset list. Adding a "Seizures" view = adding one entry.

---

## New Files

### Frontend

```
apps/web/src/pages/KnowledgeExplorerPage.tsx
  ↳ replaces GraphPage.tsx routing entry
  ↳ owns: viewMode, layerMode, filters, selectedNode state
  ↳ orchestrates all graph sub-components

apps/web/src/components/graph/
  GalaxyGraph.tsx           three.js Points cloud; props: points[], onNodeClick
  ExplorerGraph.tsx         react-force-graph-3d/2d; props: graphData, mode, onNodeClick
  GraphControls.tsx         mode/layer switcher bar
  FilterPanel.tsx           region/species/task/topic filter sidebar
  SuggestedViews.tsx        preset lens chips + companion suggestions
  NodeDetailPanel.tsx       bottom detail strip; adapts to node type
  GraphLegend.tsx           overlay legend
  RelatedFindingsPanel.tsx  card-level: consensus badges + mini SVG + findings

apps/web/src/pages/LiteraturePage.tsx
  /literature route — consensus heatmap + finding search table
  (stub in sprint 1, fully built in sprint 2)

apps/web/src/api/graph.ts
  fetchGraphOverview, fetchSubgraph, fetchTopicGraph,
  fetchConsensus, fetchFindings, fetchDatasetNeighborhood

apps/web/src/types/graph.ts
  GraphNode, GraphEdge, FindingCluster, SubgraphResponse,
  TopicGraphResponse, DatasetNeighborhood, ViewMode, LayerMode
```

### Backend

```
apps/api/graph_router.py (or inline in main.py)
  GET /api/graph/overview
  GET /api/graph/subgraph
  GET /api/graph/topic/{slug}
  GET /api/literature/consensus
  GET /api/literature/findings
  GET /api/datasets/{id}/neighborhood

scripts/literature/build_cluster_graph.py
  input:  artifacts/literature/findings_tier1_ollama.jsonl
          artifacts/literature/relationships/finding_edges.jsonl
          artifacts/literature/relationships/consensus_summaries.jsonl
  output: artifacts/graph/cluster_graph.json

scripts/literature/compute_layout.py
  input:  artifacts/graph/cluster_graph.json
  output: artifacts/graph/galaxy_points.json
  method: networkx spring_layout in 3D, serialize node positions

artifacts/graph/
  cluster_graph.json        cluster-level nodes + edges
  galaxy_points.json        pre-computed xyz positions for all nodes
```

### New dependency

```
react-force-graph   (peer: three.js already pulled by 3d variant)
```

---

## Performance Contract

| Mode | Nodes | Edges | Renderer | Target FPS |
|------|-------|-------|----------|------------|
| Galaxy | 80K+ | 0 | three.js Points | 60fps |
| Explorer 3D | 200–500 | 500–2000 | react-force-graph-3d | 60fps |
| Explorer 2D | 200–500 | 500–2000 | react-force-graph-2d | 60fps |
| Mini-graph (card) | 30–60 | 60–120 | SVG (static) | n/a |

Galaxy loads from `galaxy_points.json` (static file, no API call). Explorer subgraphs fetched on demand, capped at 400 nodes server-side (largest-connected-first prioritization).

---

## Extensibility Notes

These design choices are deliberate to keep the system open:

- **New views** = one entry in `suggested_views` config (slug, label, filters, layer, companions). No component changes.
- **New node types** = extend `NodeType` union + add color/size mapping in `ExplorerGraph`. No renderer changes.
- **New layers** = add to `LayerMode` union + add edge/node filter logic in subgraph endpoint. No UI changes.
- **New graph modes** = new component alongside `GalaxyGraph`/`ExplorerGraph`, plug into `KnowledgeExplorerPage` mode switcher.
- **Growing KG** = `galaxy_points.json` regenerated by CI after each extraction batch. Galaxy scales to 1M+ nodes without code changes (WebGL point budget).
- **Researcher-specific views** = future: user-saved lens configs stored in `/api/frontend/saved-views` (not in scope now, but the filter schema supports serialization).

---

## Out of Scope (v1)

- User-saved custom views (serialize filter state to URL params only)
- Real-time KG updates (static JSON, rebuilt on demand)
- Finding text search within the graph (side-panel only)
- 3D brain atlas overlay (future: align region nodes to anatomical coordinates)
- Collaborative annotation on graph nodes
- LiteraturePage full build (stub only in v1, full in sprint 2)
