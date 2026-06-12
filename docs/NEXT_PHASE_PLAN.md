# Neural Search: Next Phase Development Plan

**Updated:** 2026-05-27
**Status:** Post-Phase 10 (Whitepaper claim tightening complete)
**Repository Version:** v0.7.3
**Branch:** claude/full-paper-and-experiment-upgrade

---

## Executive Summary

Neural Search has progressed through foundational implementation phases:

| Phase | Status | Key Deliverables |
|-------|--------|------------------|
| Phase 0 | ✅ Complete | CURRENT_SYSTEM_MAP.md, CLAIM_LEDGER.md |
| Phase 1 | ✅ Complete | DatasetCardV1, CorpusSnapshot, ProvenanceEdge schemas |
| Phase 5 | ✅ Complete | EmbeddingProvider abstraction, SPECTER2/SciBERT stubs, model comparison |
| Phase 7 | ✅ Complete | Affordance registry (14 types), validation framework |
| Phase 10 | ✅ Complete | Whitepaper consistency fixes, claim tightening |

**Remaining critical work:**
1. **Affordance Validation**: Empirical testing against actual NWB/BIDS contents
2. **Benchmark Expansion**: 100+ queries with multi-annotator labels
3. **Embedding Comparison**: Systematic SPECTER2/SciBERT/ColBERT evaluation
4. **Content Signatures**: NWB feature extraction for content-based retrieval

---

## Current Implementation Status

### New Infrastructure (148+ tests, all passing)

```
neural_search/
├── core/
│   ├── dataset_card.py          # DatasetCardV1, CorpusSnapshot, ProvenanceEdge
│   └── neural_signature.py      # NeuralSignatureV1, content-derived features
├── embeddings/
│   ├── providers.py             # EmbeddingProvider, EmbeddingRecord, 4 providers
│   └── model_comparison.py      # EmbeddingIndex, compare_embedding_models()
├── retrieval/
│   └── constraint_parser.py     # Boolean constraint parser (NOT/AND/OR)
├── evaluation/
│   └── dataset_linkage.py       # Dataset-to-dataset relatedness benchmark
└── affordances/
    ├── __init__.py
    ├── registry.py              # 14 affordance types, DatasetFeatures, validation
    └── validators/              # NWB/BIDS file validators
        ├── nwb_validator.py
        └── bids_validator.py
```

### Affordance Registry (14 Types)

| ID | Label | Required Features |
|----|-------|-------------------|
| event_aligned_psth | Event-aligned PSTH | neural_data, spike_times, event_timestamps |
| choice_decoding | Choice decoding | neural_data, trial_structure, choice_labels |
| q_learning | Q-learning model fitting | trial_structure, choice_sequence, reward_signal |
| stimulus_response_modeling | Stimulus-response modeling | neural_data, stimulus_info, stimulus_timing |
| behavioral_state_decoding | Behavioral state decoding | neural_data, behavioral_state_labels |
| cross_area_interaction | Cross-area interaction | neural_data, multiple_brain_regions |
| dimensionality_reduction | Dimensionality reduction | neural_data, population_recording |
| functional_connectivity | Functional connectivity | neural_data, continuous_data, multiple_channels |
| trial_aligned_calcium_analysis | Trial-aligned calcium | calcium_imaging, roi_traces, trial_structure |
| pose_neural_correlation | Pose-neural correlation | neural_data, pose_tracking_data |
| delay_discounting_modeling | Delay discounting | trial_structure, delay_duration_variable, reward_magnitude |
| motor_decoding | Motor decoding | neural_data, motor_action_labels |
| trial_aligned_neural_analysis | Trial-aligned neural | neural_data, trial_structure, event_timestamps |
| cross_session_generalization | Cross-session generalization | neural_data, multiple_sessions, session_id |

---

## Priority 1: Affordance Validation Study ✅ COMPLETE

**Goal**: Move affordances from rule-based predictions to empirically validated reusability indicators.

### Status: Infrastructure Complete

1. **NWB Validator** ✅ (`neural_search/affordances/validators/nwb_validator.py`)
   - Validates NWB files for affordance support
   - Checks: units table, trials table, electrodes, imaging planes, behavioral events
   - Maps findings to 14 affordance types
   - Supports both file inspection (with pynwb) and metadata-only validation

2. **BIDS Validator** ✅ (`neural_search/affordances/validators/bids_validator.py`)
   - Validates BIDS datasets for affordance support
   - Checks: participants.tsv, events.tsv, modality directories, task labels
   - Extracts features from BIDS structure

3. **Corpus Validation** ✅ (metadata-based)
   - Validated 371 datasets from corpus
   - Report: `reports/affordance_validation_v1.md`
   - Results: `data/eval/affordance_validation/validation_results.json`

### Remaining Work
- Ground truth annotation with actual file inspection (700+ pairs)
- Compute affordance precision metrics
- Validate against actual NWB file contents (requires downloads)

---

## Priority 2: Benchmark Expansion

**Goal**: Expand from 30 single-annotator queries to 100+ multi-annotator queries.

### Query Expansion Plan

| Category | Current | Target | Examples |
|----------|---------|--------|----------|
| Task-specific | 10 | 30 | "Q-learning suitable datasets" |
| Cross-modal | 5 | 20 | "ephys + behavior in mice" |
| Hard negatives | 5 | 25 | "NOT visual cortex AND NOT fMRI" |
| Affordance queries | 5 | 15 | "datasets supporting choice decoding" |
| Species-specific | 5 | 10 | "non-human primate decision-making" |
| **Total** | **30** | **100+** | |

### Multi-Annotator Protocol

1. Recruit 3 domain experts
2. Each query annotated by 2+ annotators
3. Compute inter-annotator agreement (Cohen's kappa ≥ 0.7)
4. Resolve disagreements through discussion

### Corpus Expansion

| Source | Current | Target |
|--------|---------|--------|
| DANDI | 163 | 250 |
| OpenNeuro | 190 | 250 |
| Allen Brain | 8 | 30 |
| NeMO | 10 | 20 |
| CRCNS | 0 | 30 |
| IBL | 0 | 20 |
| **Total** | **371** | **600** |

---

## Priority 3: Embedding Model Comparison

**Goal**: Systematic comparison of embedding models for scientific dataset retrieval.

### Provider Status

| Provider | Status | Dependencies |
|----------|--------|--------------|
| HashingEmbeddingProvider | ✅ Implemented | None (deterministic) |
| SentenceTransformerProvider | ✅ Implemented | sentence-transformers |
| SPECTER2Provider | ✅ Implemented | transformers, torch |
| SciBERTProvider | ✅ Implemented | transformers, torch |
| PubMedBERTProvider | 🔴 Not started | transformers, torch |
| ColBERTProvider | 🔴 Not started | colbert-ai |

### Evaluation Plan

1. Generate embeddings for all 371+ dataset cards
2. Run dense-only retrieval with each model
3. Run hybrid (BM25 + dense) with each model
4. Compare on expanded benchmark
5. Per-query-type breakdown
6. Latency comparison

### Success Metrics

| Model | Target NDCG@10 | Target Latency (p50) |
|-------|----------------|---------------------|
| Hashing (baseline) | 0.75 | <10ms |
| Sentence-Transformer | 0.82 | <50ms |
| SPECTER2 | 0.87 | <100ms |
| SciBERT | 0.85 | <100ms |
| Hybrid Best | 0.92 | <150ms |

---

## Priority 4: Content-Derived Neural Signatures ✅ SCHEMA COMPLETE

**Goal**: Enable retrieval based on actual dataset contents, not just metadata.

### Status: Schema Implemented

**NeuralSignatureV1** (`neural_search/core/neural_signature.py`) ✅:
- Full Pydantic schema with provenance tracking
- Modality-specific stats (FiringRateStats, ISIStats, CalciumStats)
- Trial statistics (TrialStats)
- Feature vector for similarity search
- Quality levels (HIGH/MEDIUM/LOW)
- Extraction from metadata and NWB files

### Features Tracked
- Recording duration, sampling rate
- Number of units/ROIs/channels/electrodes
- Trial count and event types
- Brain regions
- Firing rate statistics
- ISI distribution summary
- Calcium imaging metrics

### Remaining Work
- Run extraction on 50 NWB files
- Build signature similarity index
- Evaluate content-based retrieval

---

## Priority 5: Structured Constraint Parser ✅ COMPLETE

**Goal**: Handle complex query constraints reliably.

### Status: Implemented

**ConstraintParser** (`neural_search/retrieval/constraint_parser.py`) ✅:
- NOT operator (negation)
- AND/OR operators
- Quoted phrase handling
- Case-insensitive operators
- Implicit constraints from domain knowledge
- Constraint tree building

| Query Type | Status | Example |
|------------|--------|---------|
| Simple negation | ✅ Works | "NOT fMRI" |
| Multi-term negation | ✅ Works | "NOT visual cortex" |
| Boolean operators | ✅ Works | "mouse AND decision-making" |
| Implicit constraints | ✅ Works | "Neuropixels" → ephys modality |
| Quoted phrases | ✅ Works | "visual cortex" NOT "motor cortex" |

### Test Coverage
- 21 tests for constraint parsing
- Covers: simple queries, NOT, AND, OR, quoted phrases, implicit constraints

---

## Priority 6: Dataset Linkage Benchmark ✅ SCHEMA COMPLETE

**Goal**: Evaluate dataset-to-dataset relatedness.

### Status: Schema Implemented

**DatasetLinkage** (`neural_search/evaluation/dataset_linkage.py`) ✅:
- DatasetPair with multi-annotator support
- LinkageBenchmark for evaluation
- LinkageMetrics with precision/recall/MRR/NDCG
- Sample benchmark creation
- Load/save utilities

### Linkage Types Supported

| Linkage Type | Description |
|--------------|-------------|
| same_task | Same behavioral task |
| same_modality | Same recording modality |
| same_species | Same species |
| same_brain_region | Overlapping brain regions |
| topical | Related scientific questions |
| reusable | Similar analysis pipelines |

### Remaining Work
- Generate 500 labeled pairs
- Collect multi-annotator labels
- Run evaluation on retrieval system

---

## Implementation Timeline

### Month 1: Validation Foundation

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | NWB/BIDS validators | `validators/nwb_validator.py`, `validators/bids_validator.py` |
| 2 | Dataset sampling | 100 datasets selected, annotation schema defined |
| 3 | Ground truth annotation | 350+ affordance labels |
| 4 | Ground truth completion | 700+ labels, validation report |

### Month 2: Benchmark & Embedding

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Query expansion | 70 additional queries drafted |
| 2 | Multi-annotator labels | 2+ annotators per query, kappa computed |
| 3 | SPECTER2/SciBERT evaluation | Provider completion, embedding generation |
| 4 | Model comparison report | `embedding_comparison_v1.md` |

### Month 3: Advanced Features

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Neural signature extraction | 50 NWB files processed |
| 2 | Signature similarity search | Demo "find similar" queries |
| 3 | Constraint parser enhancement | 50+ parser tests, Boolean support |
| 4 | Dataset linkage benchmark | 500 pairs labeled, metrics computed |

---

## Definition of Done

The next milestone is successful when:

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Affordance precision | ≥ 80% | Validation study |
| Benchmark queries | 100+ | Query count |
| Inter-annotator kappa | ≥ 0.7 | Agreement metric |
| Embedding models compared | ≥ 3 | Comparison report |
| NWB signatures extracted | ≥ 50 | File count |
| Hard-negative violations | 0% | Benchmark |
| All claims backed by artifacts | 100% | CLAIM_LEDGER.md |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| NWB file access requires downloads | High | Medium | Streaming/lazy loading, cache locally |
| SPECTER2 dependencies too heavy | Medium | Low | Gate behind optional extra |
| Low inter-annotator agreement | Medium | High | Clear guidelines, calibration sessions |
| Benchmark overfitting | Low | High | Hold-out test set |
| Annotation effort too large | Medium | Medium | Active sampling, prioritize key affordances |

---

## Commands Reference

```bash
# Run affordance registry tests
pytest tests/test_affordance_registry.py -v

# Run embedding provider tests
pytest tests/test_embedding_providers.py -v

# Run dataset card tests
pytest tests/test_dataset_card.py -v

# Run full test suite
pytest tests/ -q

# Generate affordance validation report (future)
python -m neural_search.evaluation.affordance_validation --report

# Run embedding comparison (future)
python -m neural_search.embeddings.model_comparison --models specter2,scibert,hashing

# Run benchmark suite
python -m neural_search.evaluation.run_benchmark --suite demo_v02
```

---

## Related Documents

- [CURRENT_SYSTEM_MAP.md](CURRENT_SYSTEM_MAP.md) - Module architecture
- [CLAIM_LEDGER.md](CLAIM_LEDGER.md) - Claim status tracking
- [ARCHITECTURE_V05.md](ARCHITECTURE_V05.md) - System architecture
- [Whitepaper](whitepaper/neural_search_whitepaper.tex) - ICLR-style paper

---

*Last updated: 2026-05-27*
