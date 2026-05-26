# Neural Search Development Roadmap

**Generated:** 2026-05-26
**Branch:** mvp-stabilization (merged from claude/full-paper-and-experiment-upgrade)

---

## Current State Summary

### Recently Committed Work (12 commits)

1. **Core Search & Ontology** - Intent classification, weight blending, confidence scoring
2. **Source Connectors** - Allen Brain and NeMO Archive ingestion
3. **Embedding Infrastructure** - Concept embeddings, semantic fingerprints
4. **Evaluation Framework** - Calibration, relevance labeling, ablation analysis
5. **Search Modules** - Fusion, sparse retrieval, semantic expansion, graph metapath
6. **Configuration** - Database registry, intent profiles, modalities ontology
7. **Real Corpus** - Normalized records from Allen, DANDI, NeMO, OpenNeuro
8. **Artifacts** - Field embeddings, concept embeddings, knowledge graphs
9. **Evaluation Data** - Benchmark queries v01, relevance labels, quality reports
10. **Web Frontend** - Enhanced search, evaluation, and reports views
11. **Documentation** - Architecture, graph schema, task planning docs
12. **Test Suite** - Comprehensive tests for new modules

### Implementation Status

| Component | Status |
|-----------|--------|
| v0.3 Scientific Labels | ✅ COMPLETE |
| v0.3 Analysis Affordances | ✅ COMPLETE |
| v0.4 Embedding Providers | ✅ COMPLETE |
| v0.4 Field Embeddings | ✅ COMPLETE |
| v0.4 Ablation Runner | ✅ COMPLETE |
| v0.5 Graph Schema | ✅ COMPLETE |
| v0.5 Graph Builder | ✅ COMPLETE |
| v0.5 Search Features | ✅ COMPLETE |
| v0.5 Semantic Edges | ✅ COMPLETE |
| v0.5 Metapath Traversal | ✅ COMPLETE |
| v0.7 Source Quality Gates | ✅ COMPLETE |
| v0.7 Graph QA | ✅ COMPLETE |
| Query Intent Router | 🟡 IN PROGRESS |
| Human Relevance Labeling | 🟡 IN PROGRESS |
| Real Corpus Benchmark Expansion | 🔴 NOT STARTED |
| Planner Default Promotion | 🔴 NOT STARTED |

---

## Phase 1: Immediate Priorities (1-2 weeks)

### 1.1 Validate Current Implementation

**Objective:** Ensure all recent commits pass quality gates

```bash
pytest -q
ruff check neural_search tests
python -m neural_search.evaluation.run_benchmark --suite demo_v02
python -m neural_search.evaluation.run_benchmark --suite adversarial
```

**Tasks:**
- [ ] Fix any lint warnings in test files (unused imports)
- [ ] Verify benchmark suites pass without regression
- [ ] Run real_v07 benchmark if corpus artifacts are complete

### 1.2 Wire Awareness Scoring into Main Retrieval

**Files:** `neural_search/search/core.py`, `neural_search/search/intent.py`

**Tasks:**
- [ ] Integrate `query_awareness` scoring from awareness wrapper into `search_datasets`
- [ ] Expose `awareness_score` and data-form evidence in standard responses
- [ ] Add config flag for awareness weight (default 0.0 for backward compatibility)
- [ ] Test that existing benchmarks pass with awareness disabled

### 1.3 Integrate Planner into Main Retrieval

**Files:** `neural_search/search/core.py`, `data/config/retrieval.yaml`

**Tasks:**
- [ ] Wire planner-selected weights through `search_datasets_with_intelligence`
- [ ] Validate demo, adversarial, and real_v07 before enabling by default
- [ ] Add planner metadata to search traces
- [ ] Define CI/local/exploratory config presets

---

## Phase 2: Real Corpus Stabilization (2-3 weeks)

### 2.1 Corpus Expansion Targets

| Source | Current | Target | Priority |
|--------|---------|--------|----------|
| DANDI | ~10 | 50-100 | HIGH |
| OpenNeuro | ~10 | 50-100 | HIGH |
| OpenAlex Papers | ~50 | 500-2000 | MEDIUM |
| Allen Brain | ~5 | 20-50 | MEDIUM |
| NeMO Archive | ~5 | 20-50 | MEDIUM |
| ModelDB | 0 | 10-30 | LOW |
| cellxgene | 0 | 10-20 | LOW |

### 2.2 Benchmark Expansion

**File:** `data/eval/benchmark_queries_real_corpus.yaml`

**Tasks:**
- [ ] Add 5 direct dataset/paper lookup queries
- [ ] Add 5 modality-region-species queries
- [ ] Add 5 task/behavior queries
- [ ] Add 5 analysis-affordance queries
- [ ] Add 5 paper-dataset linking queries
- [ ] Add real-corpus benchmarks for fMRI, MEG, connectomics, molecular, clinical

### 2.3 Human Relevance Labeling

**Files:** `data/eval/relevance_labels_*.jsonl`, `docs/HUMAN_RELEVANCE_LABELING_PROTOCOL.md`

**Tasks:**
- [ ] Complete labeling protocol documentation
- [ ] Generate review queues from coverage gaps
- [ ] Label 100+ query-result pairs for calibration
- [ ] Use labels to prioritize corpus and benchmark expansion

---

## Phase 3: Knowledge Graph Enrichment (2-3 weeks)

### 3.1 Graph Coverage Gates

**Tasks:**
- [ ] Add coverage gates for required node types (dataset, paper, task, modality, region)
- [ ] Add coverage gates for required edge types (dataset_has_task, paper_uses_dataset)
- [ ] Promote dangling references to hard blockers
- [ ] Add invalid confidence detection

### 3.2 Dataset-Paper Linking

**Tasks:**
- [ ] Expand linking beyond weak linked ID fields
- [ ] Add OpenAlex citation graph integration
- [ ] Add semantic similarity-based linking
- [ ] Report linking coverage and confidence distribution

### 3.3 Source-Specific Provenance

**Tasks:**
- [ ] Add graph provenance summaries for each source
- [ ] Track source quality alongside graph structure
- [ ] Add source-balance checks to reports

---

## Phase 4: Search Intelligence (3-4 weeks)

### 4.1 Query Intent Router v2

**File:** `neural_search/search/intent.py`

**Intents:**
- `dataset_lookup` - Direct ID or name lookup
- `paper_lookup` - Paper by title, DOI, or author
- `task_search` - Search by behavioral task
- `modality_region_species_search` - Search by recording modality, brain region, or species
- `analysis_affordance_search` - Search by analysis capability
- `paper_to_dataset_linking` - Find datasets used in a paper
- `dataset_to_paper_linking` - Find papers using a dataset
- `similar_dataset_search` - Find similar datasets
- `negative_constraint_search` - Search with explicit exclusions
- `ambiguous_exploratory_search` - Exploratory/discovery mode

**Tasks:**
- [ ] Implement intent-specific field weighting
- [ ] Add fallback behavior for ambiguous queries
- [ ] Parse and handle negative constraints
- [ ] Tune thresholds for intent detection

### 4.2 Score Calibration

**Tasks:**
- [ ] Validate score distributions against human judgments
- [ ] Add per-intent calibration curves
- [ ] Report calibration metrics in evaluation output
- [ ] Adjust weight profiles based on calibration results

### 4.3 Explanation Quality

**Tasks:**
- [ ] Make graph reranking explanations more explicit
- [ ] Add requirement-aware explanations
- [ ] Improve missing metadata warnings
- [ ] Add confidence indicators for each score component

---

## Phase 5: Release Readiness (1-2 weeks)

### 5.1 Quality Gates

```bash
# All implementations must pass
pytest -q
ruff check neural_search tests
make artifacts-build
python -m neural_search.evaluation.run_benchmark --suite demo_v02
python -m neural_search.evaluation.run_benchmark --suite adversarial

# Real corpus work
make real-artifacts-build
python -m neural_search.evaluation.run_benchmark --suite real_v07
make release-check
make awareness-report
```

### 5.2 Release Checks

**Tasks:**
- [ ] Combine graph QA, source quality, calibration, and benchmark deltas
- [ ] Create promotion-readiness dashboard
- [ ] Define blocking vs. warning thresholds
- [ ] Add per-query benchmark failure handling

### 5.3 Documentation

**Tasks:**
- [ ] Update README with new capabilities
- [ ] Add query examples for all intent types
- [ ] Document embedding model strategy
- [ ] Create release notes template

---

## Open Engineering Decisions

1. **Corpus Artifacts** - Whether real corpus raw payloads should be committed, sampled, or generated only locally
2. **File Inspection Claims** - Inside normalized records or as linked sidecar artifacts
3. **Graph Reports** - Whether to summarize by corpus tag and timestamp
4. **Benchmark Labeling** - How much should be automated vs. manually reviewed
5. **Search Improvement Timing** - Before, during, or after real-corpus ingestion
6. **Artifact Commit Policy** - Which real-corpus artifacts are small and stable enough to commit
7. **Release Blocking** - Per-query benchmark failures or only aggregate thresholds

---

## Critical Constraints

1. Do NOT build frontend features beyond existing scope
2. Do NOT tune only the demo benchmark
3. Do NOT remove provenance/evidence requirements
4. Do NOT make graph score dominate retrieval
5. Do NOT require external graph database
6. Existing benchmark suites must continue to pass
7. No conflict with existing embedding work

---

## Recommended Next Actions

### This Week
1. Run quality gates and fix any issues
2. Complete awareness scoring integration
3. Begin human relevance labeling protocol

### Next Week
1. Expand DANDI and OpenNeuro normalized records
2. Add 10+ new benchmark queries
3. Wire planner into retrieval core

### This Month
1. Complete corpus expansion to target levels
2. Label 100+ query-result pairs
3. Add graph coverage gates
4. Validate and promote planner defaults
