# Corpus Expansion, Integration & Search Roadmap
_Generated: 2026-06-14 | Branch: claude/latent-usefulness-v08_

---

## Current State

| Dimension          | Value                      | Source               |
|--------------------|---------------------------|----------------------|
| Corpus size        | 7,176 datasets             | DuckDB ledger        |
| Brain region cov.  | 48.1% (via LLM + ontology) | coverage_entries     |
| Modality coverage  | 81.9%                      | coverage_entries     |
| Species coverage   | 72.8%                      | coverage_entries     |
| Task coverage      | 23.4%                      | coverage_entries     |
| Known dark pairs   | retina×fMRI, barrel_cortex×fMRI, septum×fMRI | gap queries |
| Silver qrels       | 0 adjudicated (eval blocked) | neuro_judge_silver  |
| Benchmark queries  | 100 (v2)                   | expand_query_set.py  |

---

## Phase 1: Corpus Expansion (Near-term — 1–2 weeks)

Priority from `data/reports/coverage/acquisition_plan.json`:

### P0: Dark-Pair Filling (highest ROI)

| Target                  | What to do                                        | Est. datasets |
|-------------------------|--------------------------------------------------|---------------|
| Brain Image Library     | Parse BIL metadata API → extract lightsheet/fMRI + region tags | 200+ |
| Retina fMRI studies     | OpenNeuro BIDS search: task=retinotopy, modality=bold | 50–80 |
| Barrel cortex imaging   | DANDI search: barrel_cortex + calcium_imaging    | 30–50 |
| Septum ephys datasets   | DANDI + OpenNeuro search: septum + LFP/ephys     | 20–40 |

**Implementation**: Extend `neural_search/corpus/adapters/` with a `brain_image_library.py` adapter following the DANDI adapter pattern.

### P1: Region Metadata Enrichment

| Source     | Gap                    | Fix                                          |
|------------|------------------------|----------------------------------------------|
| OpenNeuro  | 20.3% region coverage  | Add BIDS `electrodes.tsv` + `T1w.json` parser to extract region labels |
| GIN        | 27.5% region coverage  | CrossRef DOI → abstract → rule-based region extraction (already have `fetch_paper_abstracts.py`) |
| NeuroVault | Task coverage 23.4%    | Parse NeuroVault `cognitive_paradigm_cogatlas` field for task labels |

### P2: New Source Adapters

| Source          | Data type                     | Acquisition path            |
|-----------------|-------------------------------|-----------------------------|
| SPARC           | Peripheral nervous system     | SPARC public API            |
| figshare        | Mixed neuroscience datasets   | figshare search API, neuro tags |
| Neurodata.io    | NWB files (not just DANDI)    | AWS S3 bucket listing       |
| UK Biobank      | Human fMRI (population)       | Registered access API       |

---

## Phase 2: Integration Hardening (Sprint 4–5)

### 2a: Coverage Gap Boost → Production
- [x] `coverage_gap.enabled = true` in `data/config/retrieval.yaml` (done in Sprint 4)
- [ ] Measure NDCG delta: run `run_neural_search_baseline.py` before vs after enabling gap boost
- [ ] Tune `MAX_BOOST` (currently 0.10) based on measured NDCG impact

### 2b: Source Diversity → Production
- [x] `diversity.enabled = true`, `max_per_source = 3` in config (done in Sprint 4)
- [ ] A/B test: check whether diversity reranking increases result set entropy without hurting NDCG@10
- [ ] Consider per-query override: large corpus queries (fMRI, calcium) may want tighter limit (2)

### 2c: Memory Graph → Wider Enable
- Currently: `memory_graph.enabled = true` in YAML but bridge fires on 860 nodes only
- TODO: Re-index after corpus expansion to cover new datasets
- TODO: Evaluate `memory_graph.weight` (currently 0.06) vs retrieval quality

### 2d: Acquisition Plan → Auto-Crawl
- `data/reports/coverage/acquisition_plan.json` has 45 prioritised items
- Build `scripts/corpus/run_acquisition_plan.py` that:
  1. Reads acquisition plan JSON
  2. For P0 items: triggers the appropriate adapter's `fetch()` method
  3. Appends to corpus, re-runs `build_duckdb_ledger.py --fast`
  4. Generates updated acquisition plan to track progress

---

## Phase 3: Search Quality (Sprint 5–6)

### 3a: SPECTER2 Hybrid Retrieval
- `scripts/eval/run_specter2_comparison.py` is ready — needs `--build-embeddings` run (~2h first time)
- Planned approach: **score fusion** — `final_score = 0.7 * neural_search + 0.3 * specter2_cosine`
- Hypothesis: SPECTER2 will outperform BGE-large on abstract-heavy queries; BGE-large wins on metadata-heavy
- Evaluation: compute_bootstrap_ci.py with Wilcoxon test for pairwise significance

### 3b: Query Expansion via LLM
- Current: `expand_query_terms()` uses rule-based synonym expansion
- Target: LLM-in-the-loop expansion for rare terms ("multiunit activity", "UP states")
- Implementation: add `llm_expansion` config block; call Claude claude-haiku-4-5 with structured output
- Gate: only trigger if `len(region_terms) == 0` after rule-based parsing (fallback path)

### 3c: Human Qrels Bootstrap (CRITICAL)
- Currently 0 adjudicated qrels → NDCG/MRR are uninterpretable
- 100-query benchmark set exists; neuro_judge silver qrels can seed candidate list
- **Action**: annotate 30 query-dataset pairs per session using `scripts/eval/annotate_candidates.py`
- Target: 200 annotated pairs to compute meaningful NDCG@10 (needs ≥ 5 relevant per query)
- Unlocks: all eval comparisons (DANDI baseline, SPECTER2, gap boost A/B)

### 3d: Search Result Explanations v2
- Current: `why_matched` is a flat list of matched labels
- Target: structured explanation by dimension (region, modality, task, species)
- Frontend: `DatasetCard` already shows `why_matched`; add dimension badges
- Implementation: extend `_generate_rich_explanation()` to group by dimension

### 3e: Adaptive Diversity per Query Type
- fMRI queries → stricter diversity (NeuroVault dominates)
- Rodent ephys queries → looser (DANDI/OpenNeuro have complementary datasets)
- Implementation: extend planner to emit `diversity_override: {max_per_source: N}` per query class

---

## Evaluation Milestone Targets

| Milestone                        | Requirement                          | Status     |
|----------------------------------|--------------------------------------|------------|
| First real NDCG@10 number        | ≥200 human-annotated qrels           | BLOCKED     |
| Gap boost A/B result             | Run file before/after enabling       | Ready to run|
| SPECTER2 vs BGE comparison       | Build embeddings (~2h GPU)           | Script ready|
| DANDI baseline comparison        | Run `run_dandi_search_baseline.py`   | Script ready|
| Coverage 60%+ regions            | BIL ingest + OpenNeuro enrichment    | In plan     |
| Task coverage 40%+               | NeuroVault CogAtlas + SPARC          | In plan     |

---

## Recommended Execution Order

```
Week 1:
  1. Annotate 50 qrels (unblocks eval)
  2. Run DANDI baseline (1 script, ~5 min)
  3. Run neural_search baseline (1 script, ~2 min)
  → First real comparison numbers

Week 2:
  4. BIL ingest adapter (P0 dark pairs)
  5. OpenNeuro BIDS electrode extractor (P1 region coverage)
  → Coverage 55%+ brain regions

Week 3:
  6. SPECTER2 embeddings build (GPU, 2h)
  7. Score fusion A/B against NDCG
  → Evidence for hybrid retrieval value

Sprint 5:
  8. Query expansion via LLM (fallback path)
  9. Structured explanation groups in frontend
  10. Adaptive diversity per query class
```

---

## Files Ready to Use

| Script                                        | What it does                           |
|-----------------------------------------------|----------------------------------------|
| `scripts/eval/run_neural_search_baseline.py`  | BGE-large run on 100 queries (NEW)     |
| `scripts/eval/run_dandi_search_baseline.py`   | DANDI native search baseline           |
| `scripts/eval/run_specter2_comparison.py`     | SPECTER2 embeddings + retrieval        |
| `scripts/eval/compute_bootstrap_ci.py`        | Bootstrap CI + Wilcoxon significance   |
| `scripts/eval/annotate_candidates.py`         | Human qrel annotation CLI              |
| `scripts/coverage/generate_acquisition_plan.py` | Corpus gap → prioritised crawl list  |
| `data/reports/coverage/acquisition_plan.json` | 45 prioritised acquisition targets     |
