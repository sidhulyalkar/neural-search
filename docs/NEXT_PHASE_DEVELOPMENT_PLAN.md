# Neural Search: Next Phase Development Plan

**Generated:** 2026-05-27
**Branch:** claude/full-paper-and-experiment-upgrade
**Current Version:** v0.7.3

---

## Executive Summary

Neural Search has achieved a strong MVP foundation with:
- **371 normalized datasets** from DANDI (163), OpenNeuro (190), Allen (8), NeMO (10)
- **Knowledge graph** with 614 nodes and 1697 edges
- **Benchmark performance:** P@5 = 78%, MRR = 0.894, zero hard-negative violations
- **Comprehensive evaluation framework** with calibration, ablation, and baseline ladder

This plan outlines the next development phases to **optimize retrieval quality**, **broaden corpus coverage**, and **enhance production readiness**.

---

## Phase A: Retrieval Core Stabilization (Priority: CRITICAL)

### A.1 Promote Awareness Scoring to Primary Retrieval

**Current State:** Awareness scoring implemented as optional wrapper, ready for promotion
**Target:** Integrate into main `search_datasets()` path

**Tasks:**
1. Wire `awareness_score` into score fusion in [core.py](../neural_search/search/core.py)
2. Add config flag `enable_awareness_scoring: true` with bounded weight (0.0-0.3)
3. Validate demo_v02 and adversarial benchmarks maintain metrics
4. Add awareness score breakdown to search result explanations

**Acceptance Criteria:**
- [ ] Demo benchmark P@5 >= 78%
- [ ] Adversarial hard-negative violations = 0
- [ ] Awareness score appears in result explanations

### A.2 Promote Planner Defaults to Primary Retrieval

**Current State:** Planner selects intent-specific weights, exposed via `search_datasets_with_intelligence()`
**Target:** Make planner-selected profiles the default path

**Tasks:**
1. Validate planner intent detection accuracy (>90% on labeled queries)
2. Wire planner weights through main retrieval config
3. Add fallback to baseline weights for unrecognized intents
4. Update search traces with planner metadata

**Acceptance Criteria:**
- [ ] Intent classification accuracy > 90%
- [ ] No regression on any benchmark suite
- [ ] Planner metadata in all search traces

### A.3 Define CI/Local/Exploratory Config Presets

**Current State:** Retrieval presets exist but not fully differentiated
**Target:** Clear separation for different use cases

| Preset | Embedding | Graph | Awareness | Intent Router |
|--------|-----------|-------|-----------|---------------|
| `ci` | hashing | disabled | disabled | basic |
| `demo` | hashing | enabled | enabled | basic |
| `local` | sentence-transformers | enabled | enabled | full |
| `exploratory` | sentence-transformers | enabled | enabled | full |
| `benchmark` | sentence-transformers | enabled | enabled | full |
| `production` | sentence-transformers | enabled | enabled | full |

**Tasks:**
1. Update [retrieval_presets.yaml](../data/config/retrieval_presets.yaml) with clear preset definitions
2. Add preset selection to CLI and API
3. Document preset differences and use cases
4. Add preset validation to release checks

---

## Phase B: Corpus Expansion (Priority: HIGH)

### B.1 OpenAlex Paper Linking Expansion

**Current State:** ~50 papers linked
**Target:** 500-2000 papers with semantic linking to datasets

**Tasks:**
1. Expand OpenAlex ingestion with neuroscience domain filters
2. Add semantic paper-dataset linking using shared concepts
3. Add DOI-based explicit linking where available
4. Generate paper linking coverage report

**Data Targets:**
| Source | Current | Target | Priority |
|--------|---------|--------|----------|
| OpenAlex papers | ~50 | 500-2000 | HIGH |
| Concept-based links | ~20 | 200+ | HIGH |
| DOI explicit links | ~5 | 50+ | MEDIUM |

### B.2 Additional Data Source Integration

**Pending Sources:**
| Source | Priority | Estimated Datasets | Notes |
|--------|----------|-------------------|-------|
| ModelDB | MEDIUM | 10-30 | Computational models |
| cellxgene | LOW | 10-20 | Single-cell expression |
| MICrONS | LOW | 5-10 | Connectomics |
| Allen Cell Types | MEDIUM | 20-40 | Patch-seq, electrophysiology |

**Tasks:**
1. Add fixture-backed connectors for each source
2. Implement source-specific normalization
3. Add source balance checks to reports
4. Document ingestion protocols

### B.3 Benchmark Query Expansion

**Current State:** 30 demo + 35 adversarial + 30 real corpus = 95 queries
**Target:** 150+ queries with human labels

**New Query Categories Needed:**
- [ ] 10 fMRI task/resting-state queries
- [ ] 10 MEG/EEG queries
- [ ] 10 connectomics queries
- [ ] 10 molecular/transcriptomic queries
- [ ] 10 clinical/disease queries
- [ ] 10 computational modeling queries

---

## Phase C: Evaluation and Calibration (Priority: HIGH)

### C.1 Human Relevance Labeling Campaign

**Current State:** Protocol defined, active learning implemented
**Target:** 100+ query-result pairs with human labels

**Tasks:**
1. Generate labeling queues using `select_samples_for_labeling()`
2. Create labeling interface (CLI or simple web UI)
3. Collect judgments using uncertainty sampling strategy
4. Compute calibration metrics and publish report

**Labeling Targets:**
| Phase | Queries | Labels/Query | Total Labels |
|-------|---------|--------------|--------------|
| Initial | 20 | 5 | 100 |
| Core | 40 | 10 | 400 |
| Extended | 60 | 10 | 600 |

### C.2 Score Calibration Pipeline

**Current State:** Calibration metrics implemented
**Target:** Automated calibration reporting and adjustment

**Tasks:**
1. Compute baseline calibration (ECE, Brier score) from existing labels
2. Implement automatic temperature scaling based on ECE
3. Add calibration metrics to benchmark reports
4. Track calibration drift over time

**Metrics Targets:**
- ECE < 0.10 (good calibration)
- Brier score < 0.20
- Calibration slope 0.9-1.1

### C.3 Baseline Ladder Automation

**Current State:** Baseline ladder implemented, manual runs
**Target:** Automated ladder in CI with regression detection

**Tasks:**
1. Add `make baseline-ladder` target
2. Generate comparison reports (keyword → BM25 → dense → full)
3. Fail CI if full system drops below BM25+ontology
4. Track improvement attribution over time

---

## Phase D: Knowledge Graph Enrichment (Priority: MEDIUM)

### D.1 Graph Coverage Gates

**Current State:** Graph quality module exists
**Target:** Automated coverage validation

**Required Node Types:**
- [x] dataset
- [x] paper
- [x] task
- [x] modality
- [x] region
- [ ] author (expand)
- [ ] institution (add)
- [ ] cell_type (add)
- [ ] cognitive_domain (add)

**Required Edge Types:**
- [x] dataset_has_task
- [x] paper_uses_dataset
- [x] dataset_has_modality
- [ ] author_wrote_paper
- [ ] paper_cites_paper
- [ ] dataset_from_institution
- [ ] task_in_cognitive_domain

**Tasks:**
1. Define coverage thresholds in [graph_coverage.yaml](../data/config/graph_coverage.yaml)
2. Add validation CLI with JSON output
3. Add graph coverage to release checks
4. Generate gap reports for missing edges

### D.2 Paper-Dataset Semantic Linking

**Current State:** Basic concept matching
**Target:** Multi-signal semantic linking

**Linking Signals:**
1. Shared behavioral tasks (weight: 0.3)
2. Shared modalities (weight: 0.25)
3. Shared brain regions (weight: 0.2)
4. Shared species (weight: 0.15)
5. Citation/DOI explicit links (weight: 0.1)

**Tasks:**
1. Implement multi-signal similarity scoring
2. Add confidence thresholds (0.5 for automatic linking)
3. Generate linking coverage report
4. Add manual review queue for borderline cases

### D.3 Analysis Requirement Edges

**Current State:** Basic requirements defined
**Target:** Full analysis-to-requirement mapping

**Requirements to Track:**
- Data format requirements (NWB, BIDS, etc.)
- Minimum trial/sample counts
- Required metadata fields
- Preprocessing requirements
- Tool/package dependencies

**Tasks:**
1. Define requirement schema in graph
2. Link analysis types to requirements
3. Score dataset readiness for each analysis
4. Surface requirements in search explanations

---

## Phase E: Production Readiness (Priority: MEDIUM)

### E.1 API Hardening

**Current State:** FastAPI backend functional
**Target:** Production-grade API

**Tasks:**
1. Add request validation and error handling
2. Implement rate limiting
3. Add response caching for common queries
4. Document all API endpoints (OpenAPI spec)
5. Add health check and metrics endpoints

### E.2 Search Performance Optimization

**Current State:** Adequate for demo corpus
**Target:** Sub-200ms p95 latency for 1000+ datasets

**Tasks:**
1. Profile search path for bottlenecks
2. Optimize embedding similarity computation
3. Add result caching with TTL
4. Implement batch search API
5. Add performance regression tests

### E.3 Monitoring and Observability

**Tasks:**
1. Add structured logging with correlation IDs
2. Implement search quality metrics tracking
3. Add error rate and latency monitoring
4. Create operational dashboard
5. Set up alerting for quality regressions

---

## Phase F: Research Extensions (Priority: LOW)

### F.1 Latent Neural Signature Search

**Vision:** Search by neural activity patterns, not just metadata

**Capabilities:**
- Extract firing rate distributions from NWB
- Compute ISI statistics and correlations
- Build PSTH-based signatures
- Enable "find datasets with similar neural dynamics"

**Tasks:**
1. Implement NWB unit extraction
2. Compute neural signature features
3. Build signature embedding index
4. Add latent search API endpoint

### F.2 Causal Claim Graph

**Vision:** Map intervention-outcome-evidence relationships

**Structure:**
- Intervention nodes (drug, stimulation, lesion)
- Outcome nodes (behavior, neural activity)
- Evidence edges with confidence scores

**Tasks:**
1. Define causal claim schema
2. Extract claims from paper abstracts
3. Link claims to datasets
4. Enable causal reasoning queries

### F.3 Cross-Species Alignment

**Vision:** Find analogous experiments across species

**Tasks:**
1. Build species equivalence mappings
2. Add cross-species task alignment
3. Enable "find mouse equivalent of human fMRI study"
4. Track alignment confidence

---

## Implementation Priorities

### Immediate (This Week)
1. **Complete Phase A.1**: Wire awareness scoring into main retrieval
2. **Run baseline ladder**: Validate current performance
3. **Begin labeling**: Start human relevance labeling campaign

### Short-Term (Next 2 Weeks)
1. **Complete Phase A.2-A.3**: Planner promotion and presets
2. **Complete Phase B.1**: OpenAlex paper expansion
3. **Complete Phase C.1**: Initial 100 labels

### Medium-Term (Next Month)
1. **Complete Phase B.2-B.3**: Additional sources and benchmark expansion
2. **Complete Phase C.2-C.3**: Calibration pipeline and ladder automation
3. **Complete Phase D.1**: Graph coverage gates

### Long-Term (Next Quarter)
1. **Complete Phase D.2-D.3**: Full graph enrichment
2. **Complete Phase E**: Production readiness
3. **Begin Phase F**: Research extensions

---

## Success Metrics

### Retrieval Quality
| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| P@5 (demo) | 78% | 82% | 85% |
| MRR (demo) | 0.894 | 0.92 | 0.95 |
| Hard-negative violations | 0 | 0 | 0 |
| Adversarial pass rate | 85.7% | 90% | 95% |

### Corpus Coverage
| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Total datasets | 371 | 500 | 750 |
| Papers linked | ~50 | 500 | 2000 |
| Benchmark queries | 95 | 150 | 200 |
| Human labels | 0 | 100 | 500 |

### Calibration
| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| ECE | Unknown | <0.10 | <0.05 |
| Brier score | Unknown | <0.20 | <0.15 |
| Calibration slope | Unknown | 0.9-1.1 | 0.95-1.05 |

---

## Risk Mitigation

### Technical Risks
1. **Embedding model inconsistency**: Use hashing in CI, document model versions
2. **Graph scalability**: Monitor node/edge counts, add pagination
3. **Calibration drift**: Track over time, alert on degradation

### Process Risks
1. **Label quality**: Use multi-reviewer protocol, track inter-rater agreement
2. **Benchmark overfitting**: Maintain held-out test set
3. **Feature creep**: Prioritize core retrieval over extensions

---

## Open Questions

1. **Target venue for paper submission?** (ICLR, NeurIPS, Neuroinformatics)
2. **Acceptable corpus size for initial release?** (500 vs 1000 datasets)
3. **Include user study in initial submission?**
4. **Priority: broader coverage or deeper validation?**

---

## Commands Reference

```bash
# Quality gates
make lint
make test
make benchmark

# Artifact builds
make real-artifacts-build
make real-graph-build

# Reports
make awareness-report
make baseline-ladder
make release-check

# Development
make api                # Start backend
make web                # Start frontend
make demo-search QUERY="..." # Test search
```

---

*Last updated: 2026-05-27*
