# Neural Search: Current System Map

**Generated:** 2026-05-27
**Repository Version:** v0.7.3
**Branch:** claude/full-paper-and-experiment-upgrade

---

## Executive Summary

Neural Search is a provenance-aware retrieval system for scientific datasets that combines structured experimental constraints, ontology-aware metadata, knowledge graph linkages, and learned embeddings to identify datasets that are experimentally reusable, not merely topically similar.

**Current Corpus:**
- 371 normalized datasets (DANDI: 163, OpenNeuro: 190, Allen: 8, NeMO: 10)
- 614 knowledge graph nodes, 1697 edges
- ~50 linked papers via OpenAlex

**Current Performance:**
- Precision@5: 78%
- MRR: 0.894
- Hard-negative violations: 0

---

## Module Architecture

```
neural_search/                    (~44,000 LOC across 166 files)
├── core/                         Core retrieval pipeline
│   ├── retrieval.py             Multi-stage retrieval orchestration
│   ├── query.py                 Query parsing and intent classification
│   └── records.py               ScientificRecord data model
├── ingestion/                    Source connectors
│   ├── dandi.py                 DANDI Archive adapter
│   ├── openneuro.py             OpenNeuro/BIDS adapter
│   ├── allen_brain.py           Allen Brain adapter
│   ├── nemo_archive.py          NeuroMorpho adapter
│   ├── openalex.py              Paper/citation adapter
│   └── demo_seed.py             Demo fixture loader
├── graph/                        Knowledge graph
│   ├── schema.py                Node/edge type definitions
│   ├── builder.py               Graph construction
│   ├── query.py                 Graph traversal/querying
│   └── provenance.py            Edge provenance tracking
├── embeddings/                   Vector representations
│   ├── base.py                  EmbeddingProvider protocol
│   ├── sentence_transformers.py Transformer embeddings
│   ├── hashing.py               Deterministic hash embeddings
│   └── index.py                 Vector similarity search
├── search/                       Retrieval components
│   ├── core.py                  Main search_datasets() function
│   ├── hybrid.py                Multi-signal fusion
│   ├── constraints.py           Hard constraint enforcement
│   ├── explanation.py           Result explanation generation
│   └── semantic_scoring.py      Semantic similarity scoring
├── ontology/                     Scientific vocabulary
│   ├── loader.py                Ontology loading
│   └── matcher.py               Term matching and expansion
├── evaluation/                   Benchmarking
│   ├── run_benchmark.py         Benchmark runner
│   ├── baseline_ladder.py       Ablation studies
│   ├── calibration.py           Confidence calibration
│   └── affordance_validation.py Affordance testing
├── analysis_affordances.py       Analysis capability detection
├── schemas.py                    Pydantic models
└── scientific_labels.py          Label vocabularies
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                            │
├─────────────────────────────────────────────────────────────────┤
│  DANDI API ──┐                                                   │
│  OpenNeuro ──┼──▶ Source Adapters ──▶ NormalizedDatasetRecord   │
│  Allen Brain ┤                              │                    │
│  OpenAlex ───┘                              ▼                    │
│                                     EvidenceLabel provenance     │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                       NORMALIZATION                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐       │
│  │ Task        │  │ Modality     │  │ Brain Region      │       │
│  │ Ontology    │  │ Ontology     │  │ Ontology          │       │
│  └──────┬──────┘  └──────┬───────┘  └─────────┬─────────┘       │
│         └────────────────┼──────────────────────┘                │
│                          ▼                                       │
│              Synonym expansion + canonical IDs                   │
│              Confidence scoring per field                        │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE GRAPH                              │
├─────────────────────────────────────────────────────────────────┤
│  Node Types (39): dataset, paper, task, modality, brain_region  │
│                   species, affordance, data_standard, ...       │
│                                                                  │
│  Edge Types (40+): has_task, has_modality, described_by_paper   │
│                    supports_affordance, related_dataset, ...    │
│                                                                  │
│  Storage: File-backed JSON (data/graph/)                        │
│  Evidence: ProvenanceEdge with source, extractor, confidence    │
└─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│  Query ──▶ QueryPlan (intent, constraints, stages)              │
│                │                                                 │
│                ▼                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              CANDIDATE GENERATION                        │    │
│  │  ┌────────────┐  ┌────────────┐  ┌───────────────────┐  │    │
│  │  │  Lexical   │  │  Ontology  │  │   Affordance      │  │    │
│  │  │ Generator  │  │ Generator  │  │   Generator       │  │    │
│  │  └─────┬──────┘  └──────┬─────┘  └─────────┬─────────┘  │    │
│  │        └────────────────┼──────────────────┘             │    │
│  └─────────────────────────┼───────────────────────────────┘    │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  SCORE FUSION                            │    │
│  │  Signals: ontology, modality, affordance, metadata,     │    │
│  │           semantic, graph, readiness                     │    │
│  │  Method: Weighted sum with intent-specific profiles      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  RERANKING                               │    │
│  │  - Provenance boost (linked papers, DOI)                 │    │
│  │  - Constraint satisfaction scoring                       │    │
│  │  - Hard-negative filtering                               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  EXPLANATION                             │    │
│  │  - Score component breakdown                             │    │
│  │  - why_matched strings                                   │    │
│  │  - matched/unmatched constraints                         │    │
│  │  - uncertainty_flags                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Current Schemas

### Core Data Model: ScientificRecord
Location: [core/records.py](../neural_search/core/records.py)

```python
@dataclass
class ScientificRecord:
    # Identity
    record_id: str
    record_type: str  # dataset, paper
    source: str
    source_id: str

    # Text fields
    title: str
    description: str | None
    abstract: str | None
    methods_summary: str | None
    scientific_summary: str | None

    # Scientific entities (with provenance)
    species: list[ScientificEntity]
    modalities: list[ScientificEntity]
    brain_regions: list[ScientificEntity]
    tasks: list[ScientificEntity]
    behavioral_events: list[ScientificEntity]
    cell_types: list[ScientificEntity]
    recording_technologies: list[ScientificEntity]
    analysis_methods: list[ScientificEntity]
    file_formats: list[ScientificEntity]
    software_tools: list[ScientificEntity]

    # Analysis affordances
    analysis_affordances: list[AnalysisAffordanceV2]

    # Usability signals
    has_trials: bool
    has_behavior: bool
    has_neural_data: bool
    has_continuous_behavior: bool
    has_event_timestamps: bool
    has_raw_data: bool
    has_processed_data: bool

    # Provenance
    raw_metadata: dict
    extraction_provenance: list[ExtractionProvenance]

    # Linkages
    linked_papers: list[str]
    linked_datasets: list[str]
```

### Query Model: QueryPlan
Location: [core/query.py](../neural_search/core/query.py)

```python
@dataclass
class QueryPlan:
    query_text: str
    normalized_query: str
    primary_intent: str  # 11 intent types
    intent_confidence: float
    secondary_intents: list[str]

    # Constraints
    required_modalities: list[str]
    excluded_modalities: list[str]
    required_species: list[str]
    excluded_species: list[str]
    required_tasks: list[str]
    excluded_tasks: list[str]
    required_regions: list[str]
    excluded_regions: list[str]
    required_analyses: list[str]

    # Stage configuration
    stages: list[StageConfig]
    weight_overrides: dict[str, float]

    # Expansion
    ontology_matches: dict[str, list[str]]
    expanded_terms: list[str]
```

### Legacy Schema: NormalizedDatasetRecord
Location: [schemas.py](../neural_search/schemas.py)

Used by ingestion adapters. Contains EvidenceLabel lists with provenance tracking.

---

## Current Retrievers

### 1. Lexical Generator
- BM25-style keyword matching
- Field-weighted scoring (title > description > abstract)
- Configurable field weights

### 2. Ontology Generator
- Task ontology matching with synonym expansion
- Modality/region/species ontology alignment
- Hierarchical concept matching

### 3. Affordance Generator
- 16 analysis affordance types
- Rule-based detection from dataset features
- Support levels: high, medium, low, unsupported

### 4. Semantic Embedding Scorer (Partial)
- Sentence Transformer embeddings (all-MiniLM-L6-v2)
- Hash-based deterministic fallback for CI
- Cosine similarity scoring

### 5. Graph Features (Partial)
- Metapath traversal
- Node similarity scoring
- Path-based relatedness

### Fusion: Reciprocal Rank Fusion + Weighted Sum
- Intent-specific weight profiles
- Configurable per-signal weights
- Hard constraint enforcement post-fusion

---

## Current Test Coverage

Location: [tests/](../tests/)

| Category | Files | Status |
|----------|-------|--------|
| Core retrieval | test_retrieval_core.py | Passing |
| Query parsing | test_query_planning.py | Passing |
| Ontology matching | test_ontology_matching.py | Passing |
| Affordance detection | test_affordance_validation.py | Passing |
| Graph construction | test_graph_fixtures.py | Modified |
| Benchmark harness | test_evaluation_benchmark.py | Passing |
| Calibration | test_calibration.py | Passing |
| Baseline ladder | test_baseline_ladder.py | Passing |

### Run Tests
```bash
pytest tests/ -q              # Quick test suite
make test-backend             # Full backend tests
bash scripts/quality_gate.sh  # Full quality gate
```

---

## Current Benchmarks

### Query Sets
| Set | Count | Purpose |
|-----|-------|---------|
| demo_v02 | 30 | Core capability queries |
| adversarial | 35 | Hard negatives, exclusions |
| real_corpus | 30 | Real dataset retrieval |

### Metrics Computed
- Precision@K (K=1,3,5,10)
- Recall@K
- NDCG@K
- MRR (Mean Reciprocal Rank)
- Hard-negative violation rate
- Constraint satisfaction rate

### Run Benchmark
```bash
make benchmark                                    # Full benchmark
python -m neural_search.evaluation.run_benchmark  # CLI
```

---

## Known Gaps

### Critical (Required for Paper)

| Gap | Impact | Phase |
|-----|--------|-------|
| No embedding model comparison | Cannot claim embedding contribution | Phase 5 |
| Affordances not validated against actual data | Overstated analysis support claims | Phase 7 |
| Small corpus (371 datasets) | Limited generalization evidence | Phase 1 |
| Single-annotator benchmark | Questionable label quality | Phase 8 |

### Important (Improves Credibility)

| Gap | Impact | Phase |
|-----|--------|-------|
| No DatasetCardV1 canonical schema | Inconsistent dataset representation | Phase 1 |
| No CorpusSnapshot versioning | Non-reproducible corpus state | Phase 1 |
| No SPECTER2/SciBERT/ColBERT comparison | Missing scientific embedding baselines | Phase 5 |
| No pairwise dataset linkage benchmark | Unvalidated relatedness claims | Phase 6 |
| No content-derived neural signatures | Missing key differentiator | Phase 9 |

### Future Work (Acknowledged Limitations)

| Gap | Impact | Phase |
|-----|--------|-------|
| No cross-species alignment benchmark | Speculative cross-species claims | Future |
| No causal claim extraction | Missing causal reasoning capability | Future |
| No user study | No task completion evidence | Future |

---

## Recommended Implementation Order

Based on the critique document and current state:

### Milestone 1: Corpus Foundation (Phase 0-1)
1. Create CURRENT_SYSTEM_MAP.md (this document)
2. Create CLAIM_LEDGER.md with status tracking
3. Define DatasetCardV1 and CorpusSnapshot schemas
4. Add deterministic corpus export and hashing
5. Update ingestion to produce canonical cards

### Milestone 2: Embedding Reality (Phase 5)
1. Add EmbeddingProvider abstraction
2. Add EmbeddingRecord with versioning
3. Implement true dense retrieval baseline
4. Add hybrid BM25+dense with RRF
5. Run embedding ablation study

### Milestone 3: Affordance Validation (Phase 7)
1. Define AffordanceRequirement schemas
2. Map affordances to required NWB/BIDS fields
3. Add validation against actual file structure
4. Add false-positive tracking
5. Update affordance confidence scoring

### Milestone 4: Provenance Graph (Phase 4, 6)
1. Define ProvenanceEdge schema with evidence
2. Ensure all edges have evidence
3. Add dataset linkage scoring
4. Add pairwise explanation generation
5. Create linkage benchmark

### Milestone 5: Evaluation Hardening (Phase 8)
1. Expand query sets (hard negatives, affordance queries)
2. Add per-query failure analysis
3. Add ablation automation
4. Add calibration metrics to reports
5. Update whitepaper with validated claims only

---

## Commands Reference

```bash
# Development
make api                           # Start FastAPI backend
make web                           # Start React frontend
make demo                          # Run full demo pipeline

# Testing
make test-backend                  # Run pytest
make lint                          # Run ruff
bash scripts/quality_gate.sh       # Full quality gate

# Benchmarking
make benchmark                     # Run benchmark suite
make baseline-ladder               # Run ablation studies

# Reports
make reports                       # Generate corpus reports
make awareness-report              # Search quality report

# Artifacts
make real-artifacts-build          # Build real corpus
make real-graph-build              # Build knowledge graph
```

---

## File Locations

| Artifact | Location |
|----------|----------|
| Normalized datasets | data/corpus/normalized/ |
| Knowledge graph | data/graph/ |
| Embedding indices | data/indexes/embeddings/ |
| Benchmark queries | data/eval/queries/ |
| Benchmark results | data/eval/results/ |
| Ontology definitions | data/ontology/ |
| Config presets | data/config/ |
| Generated reports | data/reports/ |

---

*Last updated: 2026-05-27*
