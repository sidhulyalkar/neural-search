# Technical Architecture

Neural Search runs against a real, growing corpus and knowledge graph — not an in-memory demo fixture. It's organized as:

- FastAPI backend in `apps/api`.
- React/Vite frontend in `apps/web`.
- Core Python package in `neural_search`.
- YAML-backed ontology, seed data, benchmark queries, notebook templates, and reports in `data`.
- A small agent-orchestration scaffold in `artifacts/agents` + `neural_search/agents`.

`NEURAL_SEARCH_DEMO_MODE=1` still selects a small 26-record fixture corpus for CI and quick local demos; the default is the full combined corpus (7,171 records / 625 unique datasets).

## Data Flow

1. **Ingestion**
   Source adapters (DANDI, OpenNeuro, OpenAlex, and others) fetch raw payloads for provenance, then normalize into the internal dataset/paper schema. `neural_search.ingestion.services` provides deterministic functions the API and tests both use, so default CI never depends on live archive availability.

2. **Ontology and extraction**
   The behavioral ontology defines tasks, synonyms, modalities, brain regions, behaviors, and suggested analyses. Titles, descriptions, assets, and linked-paper abstracts are used to extract scientific labels with a confidence and an evidence string per label.

3. **Knowledge graph build**
   `scripts/build_real_corpus_graph.py` merges every KG-producing layer (dataset/paper/concept/method/affordance nodes; cross-dataset similarity, reanalysis-candidate, reanalysis-bridge, reinterpretation-candidate, citation, and paper-link edges) into one graph, resolves any dangling edge endpoints into stub nodes, and writes `data/graph/neural_search_graph.real_corpus.json` (~12,750 nodes / ~150,000 edges). Every change to this pipeline is checked against the NDCG@10 ablation ladder before being considered done — see [Evaluation](evaluation.md).

4. **Search**
   Query parsing combines free text with structured filters. Ranking fuses BM25, BGE-large dense field embeddings, and graph/concept-memory signals via reciprocal rank fusion, with readiness and provenance-oriented warnings surfaced rather than hidden.

5. **Dataset cards**
   Cards summarize experimental structure, neural data, readiness, missing fields, reuse instructions, linked literature (with retraction status and evidence tier per link), provenance, QA status, and suggested analyses.

6. **ExperimentGlancer scenes**
   A search result or dataset card can be compiled into a versioned scene (`neural_search/experimentglancer/`) — which layers (trials, events, spikes, calcium, pose, model output) are plausible, at what evidence tier, anchored at a query-relevant moment — and rendered by a dedicated synchronized-timeline viewer.

7. **Notebooks**
   Template selection generates starter notebooks for inspection and first analysis.

8. **Evaluation, agents, and reports**
   The canonical 317-query benchmark measures ranking quality via the ablation ladder. A small agent scaffold (registry + append-only ledger + playbooks) runs the connectivity-audit and benchmark-gate discipline on a schedule, writing results to the ledger and to linked Obsidian notes.

## Backend Modules

| Module | Responsibility |
| --- | --- |
| `neural_search/ontology` | Load and match task ontology terms |
| `neural_search/search` | Query parsing and hybrid ranking (`search/core.py` is the main pipeline) |
| `neural_search/extraction.py` | Scientific label extraction |
| `neural_search/cards` | Dataset-card generation |
| `neural_search/graph` | Knowledge graph schema, builders (paper nodes, reanalysis/reinterpretation candidates, citation edges, reprocessing candidates, evidence-tier upgrades), query/traversal, search-feature scoring |
| `neural_search/kg/schemas` | The 6-tier `EvidenceTier` enum and method-registry schema |
| `neural_search/literature` | Multi-source paper-dataset linking (OpenAlex, DataCite, Crossref, PubMed/bioRxiv, Semantic Scholar) and retraction/correction checking |
| `neural_search/experimentglancer` | Scene schema, layer planner, anchor selection, source resolvers (metadata-only, OpenNeuro/BIDS local, DANDI/NWB streaming) |
| `neural_search/agents` | The append-only run ledger for the agent-orchestration scaffold |
| `neural_search/affordances` | Analysis affordance requirement registry and validation |
| `neural_search/notebooks` | Notebook template matching and generation |
| `neural_search/evaluation` | Benchmark execution and reporting |
| `neural_search/qa` | Dataset-card QA state |
| `apps/api/main.py` + routers | API surface (search, graph, claims, coverage, methods, spectral, timeline, atlas, KG, ExperimentGlancer) |

## Frontend Pages

| Page | Purpose |
| --- | --- |
| `/` | Search entry point |
| `/search` | Results, structured query controls, match evidence, retraction/evidence-tier badges on linked papers |
| `/datasets/:id` | Full dataset card, QA, notebook generation, literature (with retraction status), provenance, ExperimentGlancer preview |
| `/experimentglancer?scene_id=...` | The synchronized multimodal timeline viewer |
| `/ontology`, `/atlas`, `/methods`, `/disorders` | Browsers for the ontology, Allen CCF atlas, method registry, and disorder maps |
| `/graph` | Knowledge graph explorer |
| `/evaluation` | Benchmark report display |
| `/reports` | Corpus compilation report |

## Retrieval Model

Hybrid, not single-signal:

- Free-text query terms and structured filters both constrain results.
- Ontology matching normalizes task, behavior, modality, and region synonyms.
- BGE-large dense field embeddings and BM25 sparse scoring are fused via reciprocal rank fusion.
- Graph and concept-memory signals add relational context (shared region/task/species, method-supports-analysis links).
- Readiness, QA, and evidence-tier signals help identify genuinely reusable, verified datasets.
- Provenance and missing metadata remain visible so ranking never hides uncertainty.

## Evidence Tiers, Everywhere

A single 6-tier enum (`neural_search/kg/schemas/evidence_tier.py`: `heuristic_candidate → evidence_backed_bridge → source_declared → file_validated → human_validated → computed`) is the trust model threaded through the graph, the reanalysis/reinterpretation candidate builders, the file validators, and — as of this cycle — the search API and dataset-card responses themselves, so a user sees the same tier the graph already tracked internally. ExperimentGlancer uses its own 4-tier scheme (`available`/`probable`/`placeholder`/`unsupported`) at the layer level, deliberately not yet unified with the KG's 6-tier scheme — see [Known Limitations](known_limitations.md).

## Agent Orchestration

A thin scaffold, not a heavyweight framework: `artifacts/agents/registry.yaml` describes task classes (trigger, cost, gate strictness), `neural_search/agents/ledger.py` is an append-only JSONL run log, and `artifacts/agents/playbooks/*.md` are step-by-step procedures written from already-proven manual audits. Two playbooks are live and running on a weekly schedule: `kg-connectivity-auditor` (is every KG-producing module reachable, side-channel, dead, or a stated orphan?) and `benchmark-gatekeeper` (does the NDCG@10 ablation ladder still pass?). Findings are written to the ledger and to linked notes under `obsidian_vault/11_Agent_Runs/`.

## Local Runtime

```bash
make demo   # or NEURAL_SEARCH_DEMO_MODE=1 for the 26-record fixture
make api
make web
```

Optional database and Docker targets exist for development. The KG rebuild and ablation ladder are run explicitly, not on every API start:

```bash
python scripts/build_real_corpus_graph.py
python scripts/eval/run_ablation_ladder.py --skip-rungs bm25 bm25_structured dense_bge
```

## Production Gaps

Durable indexing, scheduled ingestion, authenticated review workflows, persistent QA state at scale, larger embedding infrastructure, observability, source-specific rate-limit handling (Semantic Scholar's unauthenticated tier is currently blocked without an API key), and explicit data governance around generated cards are all still open. Gold/human-labeled qrels remain at 0 rows — the single largest gap between "measured" and "scientifically validated" in this system today.
