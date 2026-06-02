# Neural Search v2.0 — Expansion & Optimization Spec

**Date:** 2026-06-01
**Branch target:** `claude/neural-search-v2`
**Status:** Approved for implementation

---

## 1. Vision & Goals

Neural Search v2.0 transforms the current 738-dataset retrieval prototype into a **publication-quality, pitch-ready agentic scientific discovery tool** covering all major public neuroscience data repositories.

**Three measurable goals:**

| Goal | Current | Target |
|------|---------|--------|
| Usefulness correlation (Spearman r) | 0.5044 | ≥ 0.70 |
| Corpus size | 738 datasets | 4000–5000 datasets |
| Retrieval benchmark (NDCG@10) | 0.822 | ≥ 0.85 on 5000+ corpus |

**One qualitative goal:** A single showcase query — "Map the circuit mechanisms of flexible cognitive control across prefrontal, hippocampal, dopaminergic and motor systems" — returns a structured, multi-dataset synthesis that demonstrates the system's scientific value in a way that is immediately compelling to a neuroscience consortium or agentic scientist company.

---

## 2. Overall Architecture

The system is a four-layer stack. Each layer is independently improvable and independently testable.

```
┌─────────────────────────────────────────────────────────┐
│  QUERY LAYER                                            │
│  • Intent classification (UsefulnessIntent, 7 classes) │
│  • Query embedding: local bge-large-en-v1.5 (default)  │
│    OR OpenAI text-embedding-3-small (if API key set)   │
│  • Boolean constraint parsing                           │
├─────────────────────────────────────────────────────────┤
│  RETRIEVAL LAYER                                        │
│  • turbovec TurboQuantIndex (ANN, 1024-dim, 8-bit)     │
│  • Hybrid: dense ANN + BM25 keyword (RRF fusion)       │
│  • 10-dim intent-weighted usefulness scorer            │
│  • s9 graph proximity: real PathSim (no neutral prior) │
├─────────────────────────────────────────────────────────┤
│  KNOWLEDGE GRAPH                                        │
│  • 5000+ nodes: datasets × papers × methods ×          │
│    brain regions × species × affordances               │
│  • Typed edges: has_modality, has_task, cites,         │
│    replicates, extends, shares_method, shares_region   │
│  • Multi-hop traversal for complex query fan-out       │
├─────────────────────────────────────────────────────────┤
│  CORPUS                                                 │
│  • DANDI, OpenNeuro, CRCNS, NeuroVault, GIN, HCP,      │
│    ABIDE/ADHD-200, OSF, figshare, zenodo, EBRAINS      │
│  • Normalized to NWB/BIDS schema (NormalizedRecord)    │
│  • Deduplicated by SHA-256(title + author + year)      │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Execution Strategy: Parallel Tracks

Two tracks run simultaneously. Track 1 produces metric improvements on the current corpus within days. Track 2 builds ingestion adapters in parallel. Both converge at the Integration Point.

```
Track 1: Embedding + Retrieval Upgrade
  ├── DenseEmbeddingProvider (bge-large-en-v1.5, GPU)
  ├── turbovec TurboQuantIndex
  ├── Hybrid query path (local bge / OpenAI fallback)
  └── Wire s9 real graph proximity

Track 2: Corpus Expansion (8 new adapters)
  ├── CRCNS, NeuroVault, GIN, HCP
  ├── ABIDE/ADHD-200, OSF, figshare/zenodo, EBRAINS
  └── Deduplication pipeline

Integration Point:
  ├── Rebuild graph from 5000+ corpus
  ├── Re-embed all records (GPU batch)
  ├── Re-run benchmark + correlation evaluation
  └── Build killer demo on full system
```

---

## 4. Track 1 — Embedding & Retrieval Upgrade

### 4.1 DenseEmbeddingProvider

**File:** `neural_search/embeddings/dense_provider.py`

Replaces `HashingEmbeddingProvider` (64-dim token hashing, no semantics) with a proper sentence-transformer model.

```python
class DenseEmbeddingProvider:
    model_name: str = "BAAI/bge-large-en-v1.5"
    dim: int = 1024
    device: str  # "cuda" if available, else "cpu"
    batch_size: int = 64

    def embed_batch(self, texts: list[str]) -> np.ndarray  # shape (N, 1024)
    def embed_query(self, text: str) -> np.ndarray          # shape (1024,)
```

**Query-time hybrid path:**
- If `OPENAI_API_KEY` is set: use `text-embedding-3-small` (1536-dim) projected to 1024 via a fixed SVD-derived linear projection (computed once from any 500+ random corpus embeddings from both models; no training data required; stored at `data/embeddings/openai_projection_1536_to_1024.npy`)
- Otherwise: use local bge-large for both corpus and query

**Why bge-large-en-v1.5:**
- MTEB leaderboard top-5 for scientific retrieval tasks
- 1024-dim captures neuroscience-specific terminology ("neuropixels" ≠ "calcium imaging")
- Fits in ~4GB VRAM on 3070 Ti; batch-embed 5000 records in ~45 seconds

### 4.2 TurboVec Index

**File:** `neural_search/embeddings/turbovec_index.py`

Wraps `turbovec.TurboQuantIndex` with the project's existing corpus interface.

```python
class NeuralSearchTurboIndex:
    def build(self, records: list[NormalizedRecord], provider: DenseEmbeddingProvider) -> None
    def search(self, query_vec: np.ndarray, k: int, filter_ids: set[str] | None = None) -> list[SearchHit]
    def save(self, path: Path) -> None
    def load(self, path: Path) -> None
```

Index file: `data/embeddings/turbovec_index.bin` (~20MB for 5000 records at 8-bit quantization)

**Hybrid retrieval (RRF fusion):**
- Dense: top-50 from turbovec ANN
- Sparse: top-50 from existing BM25 keyword scorer
- RRF merge: `score(d) = 1/(k + rank_dense) + 1/(k + rank_bm25)`, k=60
- Final: top-20 passed to usefulness scorer

### 4.3 Wiring Real Graph Proximity (s9)

**Current problem:** `graph_proximity` always returns 0.3 neutral prior because dataset IDs don't resolve in the graph at score time.

**Fix:** `score_usefulness()` accepts an optional `graph: KnowledgeGraph` parameter. When provided, it calls `normalized_metapath_score(graph, query_id, candidate_id, "dataset_has_task")` directly. The search pipeline passes the loaded graph on every call.

**File changes:**
- `neural_search/retrieval/usefulness_scorer.py`: add `graph` param to `score_usefulness()`
- `neural_search/search/search.py`: load graph at startup, pass to scorer
- **Expected impact:** s9 goes from 0.3 flat → 0.0–1.0 discriminative signal → Spearman r +0.10–0.15

### 4.4 Evaluation Gate (Track 1 exit criterion)

Before Track 1 is considered complete:
```bash
python scripts/evaluate_usefulness_correlation.py --n-queries 30
```
Must produce: `spearman_r >= 0.65`

If below threshold: diagnose which dimensions are dragging (per-dimension contribution analysis in `scripts/analyze_scorer_dimensions.py`) and tune before proceeding to Integration Point.

---

## 5. Track 2 — Corpus Expansion

### 5.1 Target Repositories

| Source | API / Method | Normalized fields | Est. yield |
|--------|-------------|-------------------|------------|
| **CRCNS** | HTTP scrape + metadata JSON | title, species, brain_region, modality, tasks | ~300 |
| **NeuroVault** | REST API `neurovault.org/api/collections/` | title, modalities, cognitive_atlas_tasks, species | ~1500 |
| **G-Node GIN** | GIN REST API + BIDS `dataset_description.json` | title, modalities, data_standards | ~400 |
| **HCP** | S3 manifest + ConnectomeDB metadata | title, modalities, species | ~50 |
| **ABIDE / ADHD-200** | CSV metadata files from COINS/LORIS | title, modalities, species, tasks | ~200 |
| **OSF** | OSF REST API, tag filter: `neuroscience` | title, description, modalities | ~600 |
| **figshare + zenodo** | REST APIs, category filter: neuroscience | title, description, linked DOIs | ~800 |
| **EBRAINS** | KG API `search.kg.ebrains.eu` | title, species, brain_atlas_regions | ~300 |

**Total target:** 4150–4150 new records + 738 existing = **~5000 deduplicated datasets**

### 5.2 Ingestion Adapter Interface

All adapters implement the same interface:

```python
class CorpusAdapter(Protocol):
    source_name: str              # e.g. "neurovault"
    source_url: str

    def fetch(self, limit: int | None = None) -> list[dict]
    def normalize(self, raw: dict) -> NormalizedRecord | None  # None = skip
    def fetch_and_normalize(self, limit: int | None = None) -> list[NormalizedRecord]
```

**File layout:**
```
neural_search/ingestion/
  crcns.py          # new
  neurovault.py     # new
  gin.py            # new
  hcp.py            # new
  abide.py          # new
  osf.py            # new
  figshare.py       # new
  ebrains.py        # new
  registry.py       # maps source_name → adapter class (new)
```

### 5.3 Deduplication Pipeline

**File:** `scripts/dedup_corpus.py`

Two-stage dedup:
1. **Exact:** SHA-256(normalize(title) + first_author + year) — catches identical entries across sources
2. **Near-duplicate:** cosine similarity > 0.97 between bge-large embeddings — catches reposts with slightly different titles

Output: `data/corpus/normalized/real_all_deduped.jsonl` (single merged file)

### 5.4 Corpus Quality Gate (Track 2 exit criterion)

```bash
python scripts/validate_corpus.py
```
Must produce:
- Total records ≥ 4000
- Records with `modalities` populated ≥ 70%
- Records with `tasks` populated ≥ 40%
- 0 duplicate IDs

---

## 6. Integration Point

Both tracks must pass their exit criteria before integration begins.

### Step 1: Rebuild knowledge graph
```bash
python scripts/rebuild_corpus_graph.py
```
Expected: 5000+ nodes, 15000+ edges (3× growth from typed edge expansion)

### Step 2: Re-embed all records
```bash
python scripts/recompute_embeddings.py --provider dense --device cuda
```
Expected: ~45 seconds on 3070 Ti for 5000 records

### Step 3: Build turbovec index
```bash
python scripts/build_turbovec_index.py
```
Expected: ~10 seconds; writes `data/embeddings/turbovec_index.bin`

### Step 4: Re-run evaluation suite
```bash
python scripts/evaluate_usefulness_correlation.py --n-queries 30
python -m neural_search.evaluation.run_benchmark --suite real_corpus
```
**Target metrics:**
- Spearman r ≥ 0.70
- NDCG@10 ≥ 0.85
- P@5 ≥ 72%

### Step 5: Side-by-side comparison table
```bash
python scripts/compare_versions.py --v1 reports/v09_metrics.json --v2 reports/v20_metrics.json
```
Produces the table that goes in the whitepaper and pitch deck.

---

## 7. The Killer Demo Query

### 7.1 The Query

> **"Map the neural circuit mechanisms underlying flexible cognitive control — integrating datasets spanning prefrontal-hippocampal interactions, dopaminergic reward modulation, motor adaptation, and cross-species learning-dependent plasticity — to identify convergent computational mechanisms that could be tested in a single unified experiment."**

### 7.2 How the System Handles It

**Fan-out into 5 sub-queries (rule-based in v2.0 via concept-cluster keyword extraction; LLM-decomposition deferred to v3.0):**
1. `"prefrontal cortex hippocampus interaction working memory"` → intent: cross_dataset_comparison
2. `"dopamine reward prediction error striatum"` → intent: meta_analysis
3. `"motor cortex adaptation learning plasticity"` → intent: method_transfer
4. `"cross-species decision making flexible behavior"` → intent: cross_dataset_comparison
5. `"population dynamics prefrontal cortex latent space"` → intent: method_transfer

**Multi-hop graph traversal links them:**
- Datasets sharing `brain_region: prefrontal_cortex` AND `task: working_memory`
- Datasets citing the same anchor papers (cross-hop through paper nodes)
- Datasets with complementarity score > 0.6 (non-overlapping affordances)

**usefulness scorer ranks by contribution role:**
- **Anchor studies** (highly_useful): direct match to ≥3 sub-queries
- **Replication candidates** (useful): same task/species, different modality
- **Methodological complements** (useful): different brain region, same affordances
- **Cross-species bridges** (weakly_useful/useful): different species, same computational task

**Expected output structure:**
```
Query: "Map circuit mechanisms of flexible cognitive control..."
Intent: cross_dataset_comparison + meta_analysis (compound)
Datasets retrieved: 12-18
Organized by role:
  Anchor (4): [IBL brain-wide map, Steinmetz 2019, Miller PFC WM, ...]
  Replication (3): [...]
  Cross-species (3): [macaque PFC, human fMRI WM, ...]
  Methodological (3): [calcium imaging PFC, fiber photometry DA, ...]
Suggested experiment: "Record simultaneously from PFC, hippocampus, and
  striatum during a multi-step planning task with dopamine perturbation,
  across mouse and rat, using NWB format for cross-dataset analysis."
Missing data gaps: ["No large-scale human intracranial + fMRI paired dataset
  with both reward and working memory tasks"]
```

### 7.3 Success Criteria for Demo

- Returns ≥ 4 distinct modalities (neuropixels, calcium, fMRI, fiber photometry)
- Returns ≥ 3 species (mouse, macaque, human)
- At least 1 cross-species comparison possible within the result set
- 0 hard-negative violations (no dataset that contradicts the query concept)
- Synthesis paragraph is coherent (LLM-generated from structured result, reviewed manually)

---

## 8. turbovec Integration Details

**Install:** `pip install turbovec`

**Index creation:**
```python
from turbovec import TurboQuantIndex

index = TurboQuantIndex(dim=1024, bit_width=8)
index.add(embeddings_matrix)  # np.ndarray (N, 1024)
distances, indices = index.search(query_vec, k=50)
```

**Why turbovec over FAISS:**
- No C++ build dependency (pure Python wheel)
- 8× memory reduction vs float32 (5000 × 1024 × 1 byte = 5MB)
- `IdMapIndex` wrapper supports O(1) deletion when corpus records are updated
- `filter_ids` allowlist supports pre-filtered search (e.g., "only mouse datasets")

---

## 9. sourced Integration

**What sourced provides:** MCP-based access to PyPI/npm package source code — enables the system to index *which analysis tools/packages* are used with each dataset type.

**Planned use (post-v2.0):** Build a `pipeline_provenance` edge type in the knowledge graph: `dataset --uses_pipeline--> spikeinterface/kilosort/suite2p`. This enriches `s7: pipeline_transferability` with real dependency data rather than just data standard overlap.

**Not in v2.0 scope** — flagged for v2.1 after integration point lands.

---

## 10. File Map (new + modified)

### New files
```
neural_search/embeddings/dense_provider.py       Track 1
neural_search/embeddings/turbovec_index.py       Track 1
neural_search/ingestion/crcns.py                 Track 2
neural_search/ingestion/neurovault.py            Track 2
neural_search/ingestion/gin.py                   Track 2
neural_search/ingestion/hcp.py                   Track 2
neural_search/ingestion/abide.py                 Track 2
neural_search/ingestion/osf.py                   Track 2
neural_search/ingestion/figshare.py              Track 2
neural_search/ingestion/ebrains.py               Track 2
neural_search/ingestion/registry.py              Track 2
scripts/build_turbovec_index.py                  Integration
scripts/dedup_corpus.py                          Track 2
scripts/validate_corpus.py                       Track 2
scripts/analyze_scorer_dimensions.py             Track 1
scripts/compare_versions.py                      Integration
scripts/run_killer_demo.py                        Demo
tests/test_dense_provider.py                     Track 1
tests/test_turbovec_index.py                     Track 1
tests/test_ingestion_crcns.py                    Track 2
tests/test_ingestion_neurovault.py               Track 2
tests/test_ingestion_gin.py                      Track 2
tests/test_ingestion_osf.py                      Track 2
tests/test_ingestion_figshare.py                 Track 2
tests/test_ingestion_ebrains.py                  Track 2
tests/test_dedup_pipeline.py                     Track 2
tests/test_killer_demo.py                        Demo
```

### Modified files
```
neural_search/embeddings/__init__.py             export DenseEmbeddingProvider, NeuralSearchTurboIndex
neural_search/retrieval/usefulness_scorer.py     add graph param to score_usefulness()
neural_search/search/search.py                   load graph at startup, pass to scorer
neural_search/ingestion/__init__.py              export adapter registry
scripts/recompute_embeddings.py                  add --provider flag (dense vs hashing)
docs/whitepaper/neural_search_whitepaper.tex     v2.0 results section
```

---

## 11. Success Criteria Summary

| Criterion | Measurement | Pass |
|-----------|-------------|------|
| Spearman r (usefulness correlation) | `evaluate_usefulness_correlation.py --n-queries 30` | ≥ 0.70 |
| Corpus size | `validate_corpus.py` | ≥ 4000 deduplicated records |
| NDCG@10 | `run_benchmark --suite real_corpus` | ≥ 0.85 |
| P@5 | same | ≥ 72% |
| Demo: modality coverage | `run_killer_demo.py --check` | ≥ 4 modalities |
| Demo: species coverage | same | ≥ 3 species |
| Demo: HN violations | same | 0 |
| Test suite | `pytest tests/ -q` | 0 failures |

---

## 12. Out of Scope (v2.0)

- **Agentic pipeline / experiment formulation** — deferred to v3.0
- **sourced pipeline provenance edges** — deferred to v2.1
- **LLM-in-the-loop relevance scoring** — deferred to v2.1
- **Learned ranking model** (neural ranker trained on labeled pairs) — deferred to v2.1
- **Real-time corpus updates** (webhook/polling from repositories) — deferred to v3.0
- **UI / web interface** — deferred to v3.0
