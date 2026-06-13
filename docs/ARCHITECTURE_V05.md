# Neural Search Architecture v0.5

This document describes the next-generation architecture of Neural Search: a provenance-aware, graph-enhanced, embedding-driven neuroscience retrieval system.

## Design Principles

1. **Provenance-aware**: Every label, relationship, and score tracks its source
2. **Confidence-scored**: Uncertainty is quantified throughout the system
3. **Explainable**: Results explain why they matched and what's uncertain
4. **Layered**: Raw, normalized, extracted, inferred, and learned metadata are distinct
5. **Modular**: Components can be enabled/disabled independently
6. **Testable**: Deterministic behavior with fixture data

## System Layers

### Layer 1: Canonical Scientific Record Model

**Location**: `neural_search/core/records.py`

All searchable objects (datasets, papers, methods) are normalized to a unified `ScientificRecord` model that maintains:

- **Stable ID**: `{type}:{source}:{source_id}` (e.g., `dataset:dandi:000026`)
- **Text fields**: title, description, abstract, methods_summary, scientific_summary
- **Scientific entities**: species, modalities, brain_regions, tasks, etc. (all as `ScientificEntity` with provenance)
- **Metadata layers**: Raw source metadata preserved separately from normalized/inferred
- **Quality tracking**: completeness, provenance strength, confidence averages
- **Status tracking**: embedding status, graph status, QA status

**Key types**:
- `ScientificRecord` - Unified record model
- `ScientificEntity` - Entity with type, confidence, provenance
- `ExtractionProvenance` - Source tracking for extracted labels
- `MetadataLayer` - Enum: raw, normalized, extracted, inferred, learned, graph, human

### Layer 2: Scientific Entity and Affordance Extraction

**Location**: `neural_search/scientific_labels.py`, `neural_search/extraction.py`

Entity extraction produces `ScientificEntity` objects with:
- Entity type (species, modality, task, region, etc.)
- Confidence score
- Source field and evidence text
- Extraction method (rule, embedding, LLM, human)

Supported entity types (from `EntityType` enum):
- SPECIES, STRAIN, MODALITY, SIGNAL_TYPE, BRAIN_REGION
- CELL_TYPE, BEHAVIORAL_TASK, STIMULUS_TYPE, DISEASE_MODEL
- PERTURBATION, RECORDING_TECHNOLOGY, ANALYSIS_METHOD
- FILE_FORMAT, MODEL_ARCHITECTURE, METRIC, CLAIM, etc.

Analysis affordances (`AnalysisAffordanceV2`):
- Support level: high, medium, low, unsupported, unknown
- Required/helpful/missing fields
- Compatible tools and preprocessing recommendations

### Layer 3: Query Understanding and Planner

**Location**: `neural_search/core/query.py`

The query planner transforms natural language queries into structured `QueryPlan` objects:

```python
plan = parse_and_plan_query("Find datasets for latent-state modeling")
# Returns QueryPlan with:
# - primary_intent: FIND_LATENT_MODELING_DATA
# - required_analyses: ["latent_state_modeling"]
# - stages: [ONTOLOGY_MATCH, LEXICAL, EMBEDDING_SEARCH, AFFORDANCE_MATCH, ...]
# - weight_overrides: {"affordance": 0.30, ...}
```

**Supported intents** (from `QueryIntent` enum):
- FIND_DATASETS, FIND_PAPERS
- LINK_PAPERS_TO_DATASETS
- FIND_ANALYSIS_DATASETS
- FIND_LATENT_MODELING_DATA
- FIND_MULTIMODAL_DATASETS
- FIND_SIMILAR_EXPERIMENTS
- FIND_EVIDENCE_FOR_CLAIM
- PROPOSE_EXPERIMENT

**Planner outputs**:
- Intent classification with confidence
- Extracted constraints (task, modality, species, region, analysis)
- Retrieval stages to execute with weights
- Graph expansion decision
- Weight overrides based on intent

### Layer 4: Multi-Stage Retrieval Pipeline

**Location**: `neural_search/core/retrieval.py`

Modular pipeline with configurable stages:

**Stage A: Candidate Generation**
- `LexicalGenerator`: Keyword matching
- `OntologyGenerator`: Task/modality/species/region matching
- `AffordanceGenerator`: Analysis support matching
- (Future: `EmbeddingGenerator`, `GraphGenerator`)

**Stage B: Fusion**
- `ScoreFuser`: Merges candidates by record_id
- Weighted combination of signals
- Intent-specific weight profiles

**Stage C: Reranking**
- `DeterministicReranker`: Provenance boost, constraint satisfaction
- (Future: Cross-encoder reranking, LLM reranking)

**Stage D: Calibration**
- Relevance score vs. confidence score separation
- Uncertainty flags
- Provenance strength tracking

**Stage E: Explanation**
- Why matched (per constraint)
- Score breakdown by source
- Uncertainty and limitation flags

### Layer 5: Knowledge Graph as Retrieval Engine

**Location**: `neural_search/graph/`

39 node types including:
- Dataset, Paper, Task, Modality, BrainRegion, Species
- AnalysisAffordance, BehavioralEvent, DataStandard
- Organism model, Taxon group, Recording context

39 edge types including:
- dataset_has_modality, dataset_has_task, dataset_supports_analysis
- paper_uses_dataset, paper_mentions_dataset
- analysis_requires_modality, analysis_requires_behavioral_event
- species_in_taxon_group, species_has_model_role

Every edge includes:
- Confidence (0.0-1.0)
- Evidence text and source field
- Extraction method

Graph features feed into scoring via:
- `graph_context_score()`: Weighted sum of linked papers, affordances, matched requirements
- Transitive expansion via BFS (configurable max_hops)

### Layer 6: Embedding Strategy

**Location**: `neural_search/embeddings/`

**Provider abstraction**:
- `HashingEmbeddingProvider`: Deterministic, test-friendly (SHA256 → 16-dim)
- `SentenceTransformersEmbeddingProvider`: Real embeddings (all-MiniLM-L6-v2)

**Field-aware indexing**:
- Pre-computed embeddings per field: title, description, scientific_summary, tasks, modalities
- Stored as JSONL: `v07_genomics.field_embeddings.jsonl`
- Query-time weighted combination via `field_semantic_score_for_result()`

**Future roadmap** (not yet implemented):
- NWB/Zarr/HDF5 feature extraction
- Signal-level summary statistics
- Neural-behavior alignment embeddings
- Learned cross-dataset representations

### Layer 7: Paper-Dataset Linking

**Location**: `neural_search/core/linking.py`, `neural_search/graph/paper_linking.py`

Multi-signal linker computing:

| Signal | Weight | Description |
|--------|--------|-------------|
| DOI/Accession explicit | 0.30 | Direct reference |
| Task overlap | 0.12 | Shared behavioral tasks |
| Modality overlap | 0.10 | Shared recording modalities |
| Species overlap | 0.08 | Shared species |
| Region overlap | 0.06 | Shared brain regions |
| Author overlap | 0.04 | Shared authors |
| Title similarity | 0.03 | Word overlap |
| Embedding similarity | 0.02 | Abstract-description similarity |

Output (`PaperDatasetLinkV2`):
- Link type: explicit, inferred, weak, speculative
- Evidence breakdown
- Confidence and uncertainty flags
- Shared concepts lists

### Layer 8: Evaluation Baseline Ladder

**Location**: `neural_search/evaluation/baseline_ladder.py`

Progressive evaluation from simplest to full system:

1. **Lexical Only**: Pure keyword matching
2. **Metadata Only**: Structured field matching
3. **Embedding Only**: Dense retrieval
4. **Lexical + Embedding**: Hybrid without ontology
5. **Lexical + Embedding + Ontology**: Standard multi-signal
6. **Full Without Graph**: All signals except graph
7. **Full With Graph**: Including graph expansion
8. **Full System**: With planner, awareness, calibration

Metrics per level:
- P@5, P@10, MRR, NDCG@10
- Recall@K, Coverage
- Latency (p50, p95)
- Lift over previous level

Analysis outputs:
- Best performing level
- Graph/ontology/embedding lift
- Queries benefiting from graph
- Queries hurt by graph

### Layer 9: Human Labeling Workflow

**Location**: `neural_search/evaluation/relevance.py`, `neural_search/evaluation/label_relevance.py`

**Relevance labels**:
- 6-level scale: exact, highly_relevant, relevant, partially, not_relevant, hard_negative
- Dimension scores: task_match, modality_match, species_match, analysis_fit (0-3)
- Reviewer confidence and notes

**Active learning sample selection**:
- `select_samples_for_labeling()`: Prioritize uncertain/diverse samples
- Strategies: uncertainty, diversity, hybrid
- Coverage tracking: queries labeled, labels per query

**Metrics computation**:
- P@K, Recall@K, MRR, NDCG
- Hard negative violation detection
- Per-query and aggregate reporting

### Layer 10: Corpus Ingestion

**Location**: `neural_search/ingestion/`, `neural_search/corpus/`

**Sources**:
- DANDI (~163 datasets)
- OpenNeuro (~190 datasets)
- Allen Brain Atlas
- NeMO Archive
- Curated/demo fixtures

**Pipeline**:
1. Source-specific fetchers (`dandi.py`, `openneuro.py`, etc.)
2. Normalization to `NormalizedDatasetRecord`/`NormalizedPaperRecord`
3. Scientific label extraction
4. Graph building
5. Embedding generation
6. Quality scoring

## Data Flow

```
Query → Planner → [QueryPlan]
                       ↓
              Candidate Generation
              (Lexical, Ontology, Embedding, Graph)
                       ↓
                    Fusion
              (Score combination)
                       ↓
                  Reranking
              (Provenance boost)
                       ↓
                 Explanation
              (Why matched)
                       ↓
              [RetrievalResult]
```

## Configuration

**Primary configs** (`data/config/`):
- `retrieval.yaml`: Scoring weights, penalties, thresholds
- `intent_profiles.yaml`: Weight adjustments per intent
- `retrieval_presets.yaml`: Pre-tuned weight sets

**Ontology** (`data/ontology/`):
- `behavioral_task_ontology.yaml`: 40+ task definitions with synonyms, modalities, regions

## Testing Strategy

Tests are classified as:
- **keep**: Validates still-relevant behavior
- **update**: Old expectation, useful concept
- **replace**: Tied to old architecture
- **quarantine**: Obsolete or low-value

Key test files:
- `tests/test_graph_*.py`: Graph building, querying, features
- `tests/test_retrieval_*.py`: Search scoring, ranking
- `tests/test_evaluation_*.py`: Benchmark, calibration
- `tests/test_core_*.py`: New architecture tests

## Next Steps

See `docs/HANDOFF_V05.md` for detailed next steps and priorities.
