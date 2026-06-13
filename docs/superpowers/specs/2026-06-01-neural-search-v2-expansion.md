# Neural Search v2.0 — Expansion & Optimization Spec

**Date:** 2026-06-01 (revised after technical review)
**Branch target:** `claude/neural-search-v2`
**Status:** Approved for implementation

---

## 1. Vision & Goals

Neural Search v2.0 transforms the current 738-dataset retrieval prototype into a **publication-quality, pitch-ready scientific discovery tool** covering all major public neuroscience data repositories.

**Core scientific claim (revised):**

> "We evaluate whether dense retrieval, compressed vector indexing, and real graph proximity improve latent usefulness ranking beyond lexical, ontology, and metadata-only baselines at 5,000-dataset scale."

This framing separates *implemented and validated* from *proposed and aspirational*. Every metric below is a measurement target, not a promise.

**Measurement targets:**

| Metric | v0.9 baseline | v2.0 target | How measured |
|--------|--------------|-------------|--------------|
| Spearman r (usefulness correlation) | 0.5044 | Evaluate; hypothesis ≥ 0.65 | `evaluate_usefulness_correlation.py` |
| Corpus size (deduplicated, usable) | 738 | ≥ 4000 | `validate_corpus.py` |
| NDCG@10 | 0.822 | ≥ 0.85 | `run_benchmark --suite real_corpus` |
| s9 graph proximity (neutral prior → real) | 0.3 flat | Ablation shows ranking change | graph proximity ablation script |
| Corpus field completeness | unknown | median ≥ 75% across modality/task/species | `validate_corpus.py` |

---

## 2. Overall Architecture

Four-layer stack, each independently improvable and testable:

```
┌─────────────────────────────────────────────────────────┐
│  QUERY LAYER                                            │
│  • Intent classification (7 UsefulnessIntent classes)  │
│  • Query embedding: BGE-large-en-v1.5 (local, default) │
│    OR OpenAI text-embedding-3-small (separate index,   │
│    rank-fused via RRF — NOT projected into BGE space)  │
│  • Boolean + typed constraint parsing                   │
├─────────────────────────────────────────────────────────┤
│  RETRIEVAL LAYER                                        │
│  • turbovec IdMapIndex (ANN, 1024-dim, 4-bit quant)    │
│  • Hybrid: dense ANN + BM25 keyword (RRF fusion)       │
│  • 10-dim intent-weighted usefulness scorer            │
│  • s9 graph proximity: real PathSim (not neutral prior)│
├─────────────────────────────────────────────────────────┤
│  KNOWLEDGE GRAPH                                        │
│  • 5000+ nodes: datasets × papers × methods ×          │
│    brain regions × species × affordances               │
│  • Typed edges: has_modality, has_task, cites,         │
│    replicates, extends, shares_method, shares_region   │
│  • Multi-hop traversal for complex query fan-out       │
├─────────────────────────────────────────────────────────┤
│  CORPUS                                                 │
│  • Tier 1 (high-confidence): DANDI, OpenNeuro, EBRAINS,│
│    G-Node GIN, NeuroVault, HCP                         │
│  • Tier 2 (high-yield, needs classifier): OSF,         │
│    figshare, zenodo                                     │
│  • Normalized to NWB/BIDS schema (NormalizedRecord)    │
│  • 5-layer deduplication pipeline                      │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Execution Strategy: Parallel Tracks

Two tracks run simultaneously. Converge at the Integration Point only after both pass their exit criteria.

```
Track 1: Embedding + Retrieval Upgrade (days → Spearman r measurable)
  ├── DenseEmbeddingProvider (bge-large-en-v1.5, GPU, BOTH corpus + query)
  ├── turbovec IdMapIndex (dim=1024, bit_width=4)
  ├── Exact-search recall check (verify ANN ≈ brute-force cosine)
  ├── Wire s9 real graph proximity (ablation proves ranking change)
  └── Optional: separate OpenAI ANN index, RRF-fused (not projected)

Track 2: Corpus Expansion
  ├── Tier 1: DANDI, OpenNeuro, EBRAINS, G-Node, NeuroVault, HCP
  ├── Tier 2: OSF, figshare, zenodo (with dataset inclusion classifier)
  ├── 5-layer deduplication pipeline
  └── Corpus quality dashboard (completeness, provenance, license)

Integration Point (both tracks must pass exit criteria):
  ├── Rebuild graph from 5000+ corpus
  ├── Re-embed all records (GPU batch, BGE-large)
  ├── Build/update turbovec IdMapIndex
  ├── Run 5-layer evaluation suite
  └── Build killer demo on full validated system
```

---

## 4. Track 1 — Embedding & Retrieval Upgrade

### 4.1 Embedding Strategy (IMPORTANT: single embedding space)

**BGE-large-en-v1.5 for both corpus and query.** This is the correct default.

**Why this is non-negotiable:** OpenAI `text-embedding-3-small` (1536-dim) and BGE-large (1024-dim) live in different semantic spaces. Projecting between them via SVD or learned linear map does not align the spaces — it only matches dimensions. Cosine similarity across spaces is semantically undefined and will silently corrupt rankings. This is the most dangerous error to avoid.

**If OpenAI embeddings are desired (optional, additive):**
- Maintain a **separate `IdMapIndex`** built from OpenAI embeddings
- Query against **both** indexes independently
- Fuse the two ranked lists via RRF: `score(d) = 1/(k + rank_BGE) + 1/(k + rank_OpenAI)`, k=60
- This is the only safe way to combine two embedding spaces

```python
class DenseEmbeddingProvider:
    model_name: str = "BAAI/bge-large-en-v1.5"
    dim: int = 1024
    device: str  # "cuda" if available, else "cpu"
    batch_size: int = 64

    def embed_batch(self, texts: list[str]) -> np.ndarray  # shape (N, 1024)
    def embed_query(self, text: str) -> np.ndarray          # shape (1024,)
```

**Why bge-large-en-v1.5:**
- MTEB top-5 for scientific retrieval; 512-token context (adequate for dataset titles + abstracts)
- 1024-dim captures domain-specific terminology ("neuropixels" ≠ "calcium imaging")
- Fits in ~4GB VRAM on 3070 Ti; batch-embeds 5000 records in ~45s
- Same model for corpus + query → cosine similarity is well-defined

### 4.2 TurboVec Index (corrected config)

**File:** `neural_search/embeddings/turbovec_index.py`

```python
# CORRECT: IdMapIndex preserves stable external dataset IDs (O(1) deletion)
# WRONG: TurboQuantIndex — slot IDs can shift after deletion
from turbovec import IdMapIndex

index = IdMapIndex(dim=1024, bit_width=4)  # bit_width ∈ {2, 4} only
index.add(ids=dataset_ids, vectors=embeddings_matrix)
distances, ids = index.search(query_vec, k=50)
```

**Memory reality at 5000 datasets:**
- float32 baseline: 5000 × 1024 × 4 bytes = ~20.5 MB
- 4-bit TurboQuant codes: ~2.6 MB (plus norms/metadata overhead)
- 2-bit: ~1.3 MB

At 5000 records, the value of turbovec is **not primarily memory**. It is:
1. **Filtered search** — `filter_ids` allowlist for "only mouse datasets" or "only NWB" queries
2. **Future scalability** — same code works at 500k records where memory does matter
3. **Online ingest** — `IdMapIndex.add()` supports incremental corpus growth without full rebuild

**Recall validation (mandatory at integration):**
```bash
python scripts/validate_turbovec_recall.py --k 50
```
Must show: ANN recall@50 ≥ 0.95 vs brute-force cosine search. If below, reduce bit_width to 2.

### 4.3 Wiring Real Graph Proximity (s9)

**Current problem:** `s9` always returns 0.3 neutral prior — dataset IDs don't resolve in the graph at score time. This means the graph is built but never actually used during retrieval.

**Fix:** `score_usefulness()` accepts an optional `graph: KnowledgeGraph` parameter. When provided, `normalized_metapath_score(graph, query_id, candidate_id, "dataset_has_task")` is called directly. The search pipeline loads the graph at startup and passes it on every call.

**Files:**
- `neural_search/retrieval/usefulness_scorer.py`: add `graph` param to `score_usefulness()`
- `neural_search/search/search.py`: load graph at startup, pass to scorer

**Mandatory ablation before Track 1 exits:**
```bash
python scripts/ablate_graph_proximity.py
```
Must show: with real graph, ≥10% of query-candidate pairs change rank vs neutral prior. If no ranking change is observed, investigate ID mapping before claiming s9 is fixed.

### 4.4 Track 1 Exit Criterion

Both conditions must hold:
1. `evaluate_usefulness_correlation.py --n-queries 30` runs without error and produces a report
2. `ablate_graph_proximity.py` shows ≥10% of pairs change rank when real graph is used

The Spearman r result is **recorded and reported**, not a gate. The hypothesis (r will improve) is tested, not assumed.

---

## 5. Track 2 — Corpus Expansion

### 5.1 Source Classification by Metadata Quality

Sources are ordered by metadata reliability. Do not treat all sources as equivalent.

**Tier 1 — High-confidence (target these first):**

| Source | API | Notes |
|--------|-----|-------|
| DANDI | REST API (documented) | Best metadata, NWB-native, existing adapter — just increase limits |
| OpenNeuro | GraphQL API | BIDS-native, existing adapter — deeper crawl |
| G-Node GIN | GIN REST API + BIDS sidecars | DOIs, neuroscience-specific, clean metadata |
| EBRAINS | KG Core API (`search.kg.ebrains.eu`) | openMINDS/JSON-LD metadata; **may need access token** via `fairgraph` client — verify before building adapter |
| NeuroVault | REST API `/api/collections/`, `/api/images/` | **Important caveat:** NeuroVault is mostly statistical maps, parcellations, and MRI outputs — not always raw reusable datasets. Normalize at collection level. Do not conflate every image with a full dataset. |
| HCP | ConnectomeDB metadata + S3 manifest | **Access friction:** requires ConnectomeDB registration and data-use agreement. Programmatic S3 access needs generated credentials. Build adapter but document the auth requirement. |

**Tier 2 — High-yield, needs dataset inclusion classifier:**

| Source | API | Risk |
|--------|-----|------|
| OSF | APIv2 (`osf.io/api/v2/`) | Mixed content (papers, slides, code, datasets). Must classify before ingesting. |
| zenodo | REST API + OAI-PMH bulk dumps | Large but noisy. Keyword filter alone is insufficient. |
| figshare | Search API (title, tag, category) | Similar to zenodo. Category filter helps but needs classifier. |

### 5.2 Dataset Inclusion Classifier (required for Tier 2)

Before any Tier 2 record is ingested, it must pass all four checks:

```python
def is_valid_dataset(record: dict) -> bool:
    return (
        has_raw_or_processed_data(record)        # not paper/slides/code-only
        and has_species_or_modality_signal(record) # at least one science field
        and has_reuse_license(record)              # CC-BY, CC0, or equivalent
        and has_doi_or_accession(record)           # persistent identifier
    )
```

Records that fail go to `data/corpus/rejected/tier2_rejected.jsonl` with failure reason — not silently dropped.

### 5.3 5-Layer Deduplication Pipeline

**File:** `scripts/dedup_corpus.py`

SHA-256(title + author + year) alone is too brittle: it misses cross-source duplicates and collapses non-duplicates with similar titles.

**Layer 1 — Exact identifiers (highest priority):**
- DOI match → same record
- Repository accession number match (DANDI:000123, ds003505) → same record
- Canonical URL match → same record

**Layer 2 — Canonical metadata:**
- Normalized title + first author family name + year → probable duplicate, flag for Layer 3

**Layer 3 — File-level hints:**
- Shared NWB filenames or BIDS `dataset_description.json` checksums → probable duplicate

**Layer 4 — Embedding similarity:**
- BGE-large cosine similarity > 0.97 on title + abstract → near-duplicate, flag for human review queue

**Layer 5 — Human review queue:**
- `data/corpus/dedup_review_queue.jsonl` — flagged pairs that scored 0.90–0.97 in Layer 4
- Resolved as: `duplicate`, `mirror`, `derived_from`, `companion`, or `distinct`

**Output enriches records with provenance fields:**
```json
{
  "duplicate_of": "dandi:000123",
  "same_record_as": null,
  "derived_from": null
}
```

These become edges in the knowledge graph (`derived_from`, `mirrors`, `companion_to`).

### 5.4 Corpus Quality Dashboard

**File:** `scripts/validate_corpus.py` — produces both pass/fail output and a Markdown table:

| Source | Raw | Normalized | Unique | Usable | Modality % | Task % | Species % | License issues |
|--------|-----|------------|--------|--------|------------|--------|-----------|----------------|
| DANDI | ... | ... | ... | ... | ... | ... | ... | 0 |
| OpenNeuro | ... | ... | ... | ... | ... | ... | ... | 0 |
| ... | | | | | | | | |
| **TOTAL** | | | ≥4000 | | ≥75% | ≥40% | ≥80% | — |

**Track 2 exit criteria:**
- Total usable records ≥ 4000
- Median modality field completeness ≥ 75%
- No records without a persistent identifier (DOI or accession number)
- Tier 2 classifier rejection log exists and is non-empty (proves classifier ran)

---

## 6. Five-Layer Evaluation Suite

At the Integration Point and for every subsequent measurement, run all five layers.

### Layer 1: Retrieval Quality

Metrics already in place:
- P@5, Recall@10, MRR, NDCG@10

Add:
- NDCG@10 broken down by query intent class (strict_lookup / replication / exploration etc.)
- Source-diversity-aware recall: did the result set draw from ≥2 repositories?
- Hard-negative violation rate (already computed)
- Modality / species / task match rates (already computed)

### Layer 2: Latent Usefulness Quality

New metrics:
- **Spearman r**: correlation between `usefulness_score.total_score` and expert usefulness judgments over query-dataset pairs
- **Pairwise preference accuracy**: for pairs (A, B) where a human said "A is more useful," does the scorer rank A above B?
- **Complementarity score for multi-dataset sets**: given the top-10 result set, what fraction of analysis affordances are covered by any one dataset vs the full set?
- **"Would support the proposed analysis?" binary judgment**: for 20 complex queries, does the result set contain the datasets a domain expert would include?

### Layer 3: Corpus Quality

The dashboard from Section 5.4, frozen at evaluation time. This goes directly into the paper as a methods table.

### Layer 4: Index Quality (TurboVec)

```bash
python scripts/validate_turbovec_recall.py
```

Reports:
- Recall@50 vs brute-force cosine (target ≥ 0.95)
- Query latency: p50, p95, p99 in milliseconds
- Memory footprint: index file size vs float32 baseline
- Filtered search latency: with a 100-item allowlist vs unfiltered
- Result stability after insert + delete cycle (IdMapIndex property)

### Layer 5: Graph Contribution

This is critical because the whitepaper currently claims graph features help, but s9 was returning a neutral prior. That claim must be reconciled.

```bash
python scripts/ablate_graph_proximity.py
```

Reports:
- % of query-candidate pairs where real graph score ≠ neutral prior (0.3)
- NDCG@10 with s9=0.3 vs NDCG@10 with real s9
- Metapath type ablation: which edge type (dataset_has_task, dataset_has_region, etc.) contributes most to s9 signal
- Graph overgeneralization error rate: % of pairs where graph proximity is high but expert relevance is low (graph false positives)

---

## 7. The Killer Demo Query — Structured 5-Stage Pipeline

### 7.1 The Query

> **"Map the neural circuit mechanisms underlying flexible cognitive control — integrating datasets spanning prefrontal-hippocampal interactions, dopaminergic reward modulation, motor adaptation, and cross-species learning-dependent plasticity — to identify convergent computational mechanisms that could be tested in a single unified experiment."**

This is the **product demo**. For evaluation, it is decomposed into a formal 5-stage pipeline.

### 7.2 Stage 1: Query Decomposition (rule-based, v2.0)

The query is explicitly broken into typed sub-queries. In v2.0 this is rule-based (concept-cluster keyword extraction). LLM-based decomposition is deferred to v3.0.

| Sub-query ID | Query text | Intent class |
|---|---|---|
| SQ1 | "prefrontal cortex hippocampus interaction working memory" | cross_dataset_comparison |
| SQ2 | "dopamine reward prediction error striatum" | meta_analysis |
| SQ3 | "motor cortex adaptation learning plasticity" | method_transfer |
| SQ4 | "cross-species decision making flexible behavior reversal learning" | cross_dataset_comparison |
| SQ5 | "population dynamics prefrontal cortex latent space manifold" | method_transfer |

### 7.3 Stage 2: Typed Constraint Extraction

For each sub-query, extract:
- `species`: mouse | macaque | human | rat | any
- `brain_regions`: prefrontal_cortex | hippocampus | striatum | motor_cortex | any
- `modalities`: neuropixels | calcium_imaging | fmri | fiber_photometry | ecog | any
- `task_family`: working_memory | reward_learning | motor_task | decision_making | any
- `analysis_affordances`: choice_decoding | population_dynamics | dimensionality_reduction | glm | any
- `hard_negatives`: modalities/species that would make the result irrelevant

### 7.4 Stage 3: Retrieval + Set-Coverage Scoring

Rank not just by individual usefulness score, but by contribution to the **set**:

```
final_set_score(D) =
    mean(usefulness_score(d) for d in D)              # individual quality
  + α * coverage_bonus(D, modality, species, region)  # set diversity
  + β * complementarity_bonus(D)                      # unique affordances across set
  + γ * provenance_bonus(D)                           # DOI + license + metadata completeness
  - δ * redundancy_penalty(D)                         # penalize near-duplicate datasets
  - ε * missing_metadata_penalty(D)                   # penalize incomplete records
  - ζ * hard_negative_penalty(D)                      # penalize constraint violations
```

Weights α–ζ are set to equal values in v2.0; tunable in v2.1.

### 7.5 Stage 4: Output Roles

Each returned dataset is assigned exactly one role. A dataset with no assignable role is excluded.

| Role | Criteria |
|------|----------|
| **Anchor** | Matches ≥3 sub-queries; highest individual usefulness score |
| **Replication** | Same task + species as an anchor, different modality |
| **Cross-species comparator** | Same task, different species from anchor |
| **Methodological complement** | Different brain region, shares ≥2 affordances with anchor |
| **Perturbation / causal evidence** | Has optogenetic, pharmacological, or lesion manipulation |
| **Behavior-rich** | Rich trial-by-trial behavioral events, minimal neural recording |
| **Population dynamics** | Large cell count (≥100 units), suitable for dimensionality reduction |
| **Imaging-ephys bridge** | Contains both fMRI/EEG and electrophysiology on same subjects |

### 7.6 Stage 5: Demo Success Metrics

**Hard criteria (must pass):**
- Every returned dataset has an assigned role with evidence
- Every role has ≥1 supporting dataset
- 0 hard-negative violations
- No dataset included only for diversity (diversity bonus cannot be the sole reason)
- Final experiment design cites which specific datasets support each design decision

**Coverage criteria (measured, not gates):**
- ≥4 distinct modalities represented
- ≥3 species represented
- ≥3 brain regions represented
- ≥1 cross-species comparison possible within the result set

---

## 8. Implementation Order (mandatory sequence)

The following order minimizes wasted work and surfaces problems early.

1. **Freeze and record v0.9/v1.0 baseline** — corpus size, query set, all 5-layer metrics, config, git commit hash. This is the comparison baseline for every claim in the paper.

2. **Fix s9 graph proximity** — wire real graph lookup, run ablation, confirm ranking change. This is low-effort, high-signal, and reconciles a current whitepaper inconsistency.

3. **Deploy BGE-large embedding provider** — both corpus re-embed and query path. Confirm no semantic space mixing. Run Spearman r evaluation. Record result.

4. **Deploy turbovec IdMapIndex (bit_width=4)** — run recall@50 validation. Confirm ≥0.95 vs brute-force before any retrieval evaluation.

5. **Expand Tier 1 corpus** — DANDI (deeper), OpenNeuro (deeper), G-Node GIN, EBRAINS (check auth), NeuroVault (collection-level). Verify access requirements for HCP before building adapter.

6. **Add Tier 2 corpus with classifier** — OSF, figshare, zenodo. Deploy dataset inclusion classifier first; ingest only passing records.

7. **Build corpus quality dashboard** — do not claim 5000 records until field completeness and provenance metrics are published.

8. **Expand usefulness labels** — add human-judged query-dataset pairs (or strong proxy labels) to make Spearman r statistically meaningful. 186 auto-labeled pairs is a start; 500+ with human review makes the correlation reportable.

9. **Integration Point** — rebuild graph, re-embed, rebuild index, run full 5-layer evaluation suite, produce side-by-side comparison table.

10. **Killer demo** — build last, on the fully validated system.

---

## 9. File Map

### New files
```
neural_search/embeddings/dense_provider.py           Track 1
neural_search/embeddings/turbovec_index.py           Track 1
neural_search/ingestion/crcns.py                     Track 2 Tier 1
neural_search/ingestion/neurovault.py                Track 2 Tier 1
neural_search/ingestion/gin.py                       Track 2 Tier 1
neural_search/ingestion/hcp.py                       Track 2 Tier 1 (auth-gated)
neural_search/ingestion/ebrains.py                   Track 2 Tier 1 (token-gated)
neural_search/ingestion/abide.py                     Track 2 Tier 1
neural_search/ingestion/osf.py                       Track 2 Tier 2
neural_search/ingestion/figshare.py                  Track 2 Tier 2
neural_search/ingestion/zenodo.py                    Track 2 Tier 2
neural_search/ingestion/dataset_classifier.py        Track 2 Tier 2 gate
neural_search/ingestion/registry.py                  Track 2
scripts/ablate_graph_proximity.py                    Track 1
scripts/build_turbovec_index.py                      Integration
scripts/dedup_corpus.py                              Track 2
scripts/validate_corpus.py                           Track 2
scripts/validate_turbovec_recall.py                  Integration
scripts/analyze_scorer_dimensions.py                 Track 1
scripts/compare_versions.py                          Integration
scripts/run_killer_demo.py                           Demo
tests/test_dense_provider.py                         Track 1
tests/test_turbovec_index.py                         Track 1
tests/test_turbovec_recall.py                        Track 1
tests/test_ingestion_neurovault.py                   Track 2
tests/test_ingestion_gin.py                          Track 2
tests/test_ingestion_osf.py                          Track 2
tests/test_ingestion_figshare.py                     Track 2
tests/test_dataset_classifier.py                     Track 2
tests/test_dedup_pipeline.py                         Track 2
tests/test_killer_demo.py                            Demo
```

### Modified files
```
neural_search/embeddings/__init__.py             export DenseEmbeddingProvider, NeuralSearchTurboIndex
neural_search/retrieval/usefulness_scorer.py     add graph param to score_usefulness()
neural_search/search/search.py                   load graph at startup, pass to scorer
neural_search/ingestion/__init__.py              export adapter registry
scripts/recompute_embeddings.py                  add --provider flag (dense vs hashing)
docs/whitepaper/neural_search_whitepaper.tex     v2.0 results section (all 5 eval layers)
```

---

## 10. Out of Scope (v2.0)

- **OpenAI single-index hybrid** — only safe as separate index with RRF fusion; not projected into BGE space
- **LLM-based query decomposition** — deferred to v3.0 (rule-based fan-out in v2.0)
- **Agentic pipeline / experiment formulation** — deferred to v3.0
- **sourced pipeline provenance edges** — deferred to v2.1
- **Learned ranking model** — deferred to v2.1 (need ≥500 human-judged pairs first)
- **Real-time corpus updates** — deferred to v3.0
- **UI / web interface** — deferred to v3.0

---

## 11. Key Technical Invariants (do not violate)

1. **Corpus and query embeddings must use the same model.** Cosine similarity across different embedding spaces is semantically undefined.
2. **turbovec bit_width must be 2 or 4.** Value 8 is not supported.
3. **Use IdMapIndex, not TurboQuantIndex.** Dataset IDs must be stable across insert/delete.
4. **Graph proximity fix must be proven by ablation before claiming it works.** A code change is not evidence.
5. **Spearman r is a measurement, not a target.** Report the number; do not gate progress on it.
6. **Tier 2 corpus requires a dataset classifier.** Raw keyword filtering produces garbage-in-glitter-out.
7. **Every "duplicate" needs a provenance classification.** Mirror ≠ derived ≠ companion ≠ distinct.
