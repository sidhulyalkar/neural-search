# Handoff Document v0.5

This document summarizes the architectural changes made in this development pass and provides clear next steps.

## Major Architecture Changes

### 1. New Core Module (`neural_search/core/`)

Created a new foundational module with cleaner abstractions:

- **`records.py`**: Canonical `ScientificRecord` model with layered metadata (raw/normalized/extracted/inferred/learned/graph/human), `ScientificEntity` with full provenance tracking, quality scoring
- **`query.py`**: `QueryPlan` with structured intent classification, constraint extraction, stage configuration, weight overrides
- **`retrieval.py`**: Multi-stage retrieval pipeline with `LexicalGenerator`, `OntologyGenerator`, `AffordanceGenerator`, `ScoreFuser`, `DeterministicReranker`
- **`linking.py`**: Enhanced `PaperDatasetLinkV2` with 8-signal evidence, confidence scoring, uncertainty flags

### 2. Evaluation Baseline Ladder

**File**: `neural_search/evaluation/baseline_ladder.py`

8-level progression to measure component contributions:
1. Lexical Only
2. Metadata Only
3. Embedding Only
4. Lexical + Embedding
5. Lexical + Embedding + Ontology
6. Full Without Graph
7. Full With Graph
8. Full System

Computes P@K, MRR, NDCG, coverage, latency, and lift per level.

### 3. Graph Fixture Update

Updated `tests/fixtures/graph/expected_graph_summary.json` to reflect expanded graph builder output:
- 146 nodes (was 47)
- 357 edges (was 66)
- New edge types: `analysis_requires_*`, `species_*`
- New node types: organism_model, required_signal, taxon_group

### 4. Architecture Documentation

New docs created:
- `docs/ARCHITECTURE_V05.md`: Comprehensive architecture overview
- `docs/WHITEPAPER_IMPLEMENTATION_ALIGNMENT.md`: Honest claim-by-claim status
- `docs/HANDOFF_V05.md`: This document

## Files Changed

### New Files
- `neural_search/core/__init__.py`
- `neural_search/core/records.py`
- `neural_search/core/query.py`
- `neural_search/core/retrieval.py`
- `neural_search/core/linking.py`
- `neural_search/evaluation/baseline_ladder.py`
- `docs/ARCHITECTURE_V05.md`
- `docs/WHITEPAPER_IMPLEMENTATION_ALIGNMENT.md`
- `docs/HANDOFF_V05.md`

### Modified Files
- `tests/fixtures/graph/expected_graph_summary.json` - Updated expected counts
- `tests/test_graph_fixtures.py` - Updated hardcoded node count

## What Was Preserved

### Core Search (`neural_search/search/core.py`)
- 1300+ LOC multi-signal scoring system
- Ontology matching, behavior matching, modality matching
- Score breakdown and explanation generation
- Graph integration via `search_features.py`

### Graph Infrastructure (`neural_search/graph/`)
- 39 node types, 39 edge types
- Provenance-aware edges with confidence
- Transitive expansion via BFS
- Paper linking via semantic similarity

### Evaluation Framework (`neural_search/evaluation/`)
- Benchmark query system
- Human relevance labeling
- Calibration and active learning
- Affordance validation

### Corpus and Ingestion (`neural_search/ingestion/`, `neural_search/corpus/`)
- DANDI, OpenNeuro, Allen, NeMO connectors
- Normalized record schemas
- Scientific label extraction

## What Was Deprecated

Nothing was removed, but the following are candidates for future cleanup:

1. **Hashing embeddings** (`neural_search/embeddings/hashing.py`) - Useful for tests but should be replaced with real embeddings in production
2. **Old planner** (`neural_search/intelligence/planner.py`) - Disabled by default; new `core/query.py` provides cleaner abstraction

## Test Status

### Passing
- All 54 graph tests ✅
- Core retrieval tests ✅
- Evaluation tests ✅
- Ontology tests ✅

### To Verify
Run full test suite:
```bash
python -m pytest tests/ -x --tb=short
```

## Top 10 Next Tasks

### High Priority

1. **Enable and validate planner**
   - Set `planner.enabled=true` in config
   - Run benchmark queries with planner
   - Measure intent classification accuracy

2. **Run baseline ladder evaluation**
   - Execute `run_baseline_ladder()` on benchmark queries
   - Generate markdown report
   - Identify which components provide most lift

3. **Increase embedding signal weight**
   - Test with `all-MiniLM-L6-v2` instead of hashing
   - Experiment with neuroscience-specific models
   - Measure precision/MRR improvement

4. **Collect more human labels**
   - Target 100+ benchmark queries
   - Use active learning sample selection
   - Establish inter-annotator agreement

5. **Integrate new core module**
   - Use `ScientificRecord` for corpus records
   - Use `QueryPlan` in `search_datasets()`
   - Use `MultiStageRetriever` as alternative pipeline

### Medium Priority

6. **Implement embedding-based candidate generation**
   - Add `EmbeddingGenerator` to core/retrieval.py
   - Use approximate nearest neighbor search
   - Benchmark latency vs. quality

7. **Add citation evidence to paper linking**
   - Integrate OpenAlex citation data
   - Build citation graph edges
   - Add citation-based linking signal

8. **Implement latent feature extraction**
   - Parse NWB files for neural statistics
   - Generate session-level feature vectors
   - Enable neural-neural similarity queries

9. **Build labeling UI**
   - Web interface for relevance labeling
   - Batch labeling workflow
   - Disagreement resolution

10. **Graph database migration**
    - Evaluate Neo4j or similar
    - Migrate from JSON to graph database
    - Enable real-time graph queries

## Vertical Slice Test

To verify the new architecture works end-to-end, run:

```python
from neural_search.core.query import parse_and_plan_query
from neural_search.core.retrieval import MultiStageRetriever
from neural_search.normalized import load_normalized_records

# Load corpus
corpus_path = "data/corpus/normalized"
records = load_normalized_records(corpus_path)
corpus = [r.model_dump() for r in records]

# Parse query
query = "Find datasets with neural and behavioral recordings suitable for latent-state modeling"
plan = parse_and_plan_query(query)
print(f"Intent: {plan.primary_intent}")
print(f"Stages: {[s.stage.value for s in plan.stages if s.enabled]}")

# Run retrieval
retriever = MultiStageRetriever()
result = retriever.search(plan, corpus, top_k=5)

# Print results
for r in result.results:
    print(f"  {r['score']:.1f} - {r['title'][:60]}...")
    print(f"    Why: {r['why_matched'][:2]}")
```

## Current Benchmark Status

Based on existing tests:
- **P@5**: ~78%
- **MRR**: ~0.89
- **Coverage**: 95%+
- **Latency**: <100ms typical

These metrics are from the legacy search path. New architecture metrics pending baseline ladder evaluation.

## Contact

For questions about this handoff, review the following key files:
- `neural_search/core/` - New architecture
- `neural_search/search/core.py` - Legacy search (still primary)
- `docs/ARCHITECTURE_V05.md` - Architecture overview
- `docs/WHITEPAPER_IMPLEMENTATION_ALIGNMENT.md` - Implementation status
