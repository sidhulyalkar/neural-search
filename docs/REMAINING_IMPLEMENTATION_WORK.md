# Remaining Implementation Work

This document summarizes implementation work remaining for Neural Search v0.4 and v0.5, organized for parallel development.

**Status as of 2026-05-24:**
- v0.3 Core: **COMPLETE** (scientific labels, analysis affordances, normalized schema)
- v0.4 Embedding Infrastructure: **COMPLETE** (providers, field embeddings, caches, ablation reports)

---

## V0.4 REMAINING WORK

### CLD1: Query Intent and Field Routing Design (Claude)

**Status:** NOT STARTED

**Deliverable:** `docs/QUERY_INTENT_AND_FIELD_ROUTING.md`

Design document covering:
- Query intent categories:
  - dataset_lookup
  - paper_lookup
  - task_search
  - modality_region_species_search
  - analysis_affordance_search
  - paper_to_dataset_linking
  - dataset_to_paper_linking
  - similar_dataset_search
  - negative_constraint_search
  - ambiguous_exploratory_search
- Which score heads to weight for each intent
- Which fields to embed/search for each intent
- How negative constraints should be parsed
- Fallback behavior for ambiguous queries

**Files to create:**
- `docs/QUERY_INTENT_AND_FIELD_ROUTING.md`

---

### CLD2: Real-Corpus Benchmark Expansion (Claude)

**Status:** NOT STARTED

**Deliverable:** 25+ real-corpus benchmark queries

Expand `data/eval/benchmark_queries_real_corpus.yaml`:
- 5 direct dataset/paper lookup queries
- 5 modality-region-species queries
- 5 task/behavior queries
- 5 analysis-affordance queries
- 5 paper-dataset linking or exploratory queries

**Files to update:**
- `data/eval/benchmark_queries_real_corpus.yaml`
- `docs/REAL_CORPUS_BENCHMARK_GUIDE.md` (new)

---

### CLD3: Embedding Model Strategy (Claude)

**Status:** NOT STARTED

**Deliverable:** `docs/EMBEDDING_MODEL_STRATEGY.md`

Document covering:
- Comparison of embedding models (sentence-transformers, SPECTER, SciBERT, biomedical)
- Recommended defaults for v0.4
- When NOT to trust semantic embeddings
- Evaluation criteria for future providers
- Why field-specific embeddings matter

**Files to create:**
- `docs/EMBEDDING_MODEL_STRATEGY.md`

---

### CLD4: Human Relevance Labeling Protocol (Claude)

**Status:** NOT STARTED

**Deliverable:** `docs/HUMAN_RELEVANCE_LABELING_PROTOCOL.md`

Define:
- Relevance labels: exact, relevant, partially_relevant, wrong_modality, wrong_task, wrong_species, missing_required_data, unclear
- Review format for top-k results
- Storage format (JSONL/YAML)
- How reviewed labels feed evaluation metrics

Schema:
```yaml
query_id: str
result_id: str
rank: int
relevance_label: str
scientific_rationale: str
reviewer: str
reviewed_at: str
```

**Files to create:**
- `docs/HUMAN_RELEVANCE_LABELING_PROTOCOL.md`

---

## V0.5 KNOWLEDGE GRAPH WORK

### Part 2: Graph Builder from Normalized/Enriched Corpus

**Status:** NOT STARTED

**Module:** `neural_search/graph/builder.py`

**Core Classes:**
```python
@dataclass
class GraphNode:
    node_id: str
    node_type: str  # dataset, paper, task, modality, brain_region, etc.
    label: str
    aliases: list[str]
    source: str | None
    confidence: float
    evidence: list[GraphEvidence]
    properties: dict[str, Any]

@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: str
    confidence: float
    evidence: list[GraphEvidence]
    properties: dict[str, Any]

@dataclass
class KnowledgeGraph:
    nodes: dict[str, GraphNode]
    edges: list[GraphEdge]
    metadata: dict[str, Any]
```

**Required Functions:**
- `build_graph_from_records(datasets, papers, min_confidence=0.5)`
- `build_dataset_subgraph(dataset)`
- `build_paper_subgraph(paper)`
- `merge_graphs(graphs)`

**Dataset-derived edges:**
- dataset_has_task
- dataset_has_modality
- dataset_records_region
- dataset_has_species
- dataset_has_behavioral_event
- dataset_supports_analysis
- dataset_uses_standard
- dataset_has_file_format

**Paper-derived edges:**
- paper_has_author
- paper_studies_task
- paper_uses_modality
- paper_mentions_region
- paper_mentions_dataset (weak)
- paper_uses_dataset (strong)

**Files to create:**
- `neural_search/graph/__init__.py`
- `neural_search/graph/schema.py`
- `neural_search/graph/builder.py`

---

### Part 3: Lightweight Graph Query Engine

**Status:** NOT STARTED

**Module:** `neural_search/graph/query.py`

**Required Functions:**
- `get_node(graph, node_id)`
- `get_neighbors(graph, node_id, edge_types=None, direction="both")`
- `get_edges_between(graph, source_id, target_id)`
- `find_nodes_by_type(graph, node_type)`
- `find_nodes_by_label(graph, label_or_alias)`
- `find_datasets_for_task(graph, task_id)`
- `find_datasets_for_analysis(graph, affordance_id)`
- `find_papers_for_dataset(graph, dataset_id)`
- `find_datasets_for_paper(graph, paper_id)`
- `find_datasets_with_constraints(graph, required_*, excluded_*)`
- `find_paths(graph, source_id, target_id, max_depth=3)`
- `explain_connection(graph, source_id, target_id)`
- `rank_related_datasets(graph, dataset_id, weights=None)`
- `rank_related_papers(graph, paper_id, weights=None)`

**Relatedness Weights:**
```python
DEFAULT_WEIGHTS = {
    "shared_task": 3.0,
    "shared_analysis_affordance": 3.0,
    "shared_behavioral_event": 2.5,
    "shared_brain_region": 2.0,
    "shared_modality": 1.5,
    "shared_species": 1.0,
    "shared_data_standard": 0.5,
    "shared_paper": 4.0,
}
```

**Files to create:**
- `neural_search/graph/query.py`

---

### Part 4: Graph Reports

**Status:** NOT STARTED

**Module:** `neural_search/graph/reports.py`

**Reports to Generate:**
1. `graph_summary_report.md` - node/edge counts, connectivity stats
2. `graph_scientific_coverage_report.md` - task/modality/region coverage
3. `graph_linking_report.md` - dataset-paper link analysis
4. `graph_gap_report.md` - missing data identification

**CLI:**
```bash
python -m neural_search.graph.build_graph \
  --datasets data/corpus/enriched/datasets.jsonl \
  --papers data/corpus/enriched/papers.jsonl \
  --out data/graph/neural_search_graph.json

python -m neural_search.graph.reports \
  --graph data/graph/neural_search_graph.json \
  --out data/reports/graph
```

**Files to create:**
- `neural_search/graph/reports.py`
- `neural_search/graph/build_graph.py` (CLI)

---

### Part 5: Graph-Augmented Search Hooks

**Status:** NOT STARTED

**Module:** `neural_search/graph/search_features.py`

**Functions:**
```python
def compute_graph_features_for_result(
    graph: KnowledgeGraph,
    result_id: str,
    query_context: dict | None = None
) -> dict:
    """Returns: graph_degree, linked_papers, affordances, etc."""

def graph_context_score(
    graph: KnowledgeGraph,
    result_id: str,
    query_context: dict | None = None,
    weights: dict | None = None
) -> float:
    """Optional score, 0.0 if graph absent."""
```

**Critical Requirements:**
- Search runs without graph
- No crash if graph file missing
- Graph score doesn't dominate by default

**Files to create:**
- `neural_search/graph/search_features.py`

---

### Part 6: Experimental Design Graph Seeds

**Status:** NOT STARTED

**Seed File:** `data/graph/experimental_design_seeds.yaml`

**Example Seeds (8+ required):**
1. `reversal_learning_ephys_experiment`
2. `motor_decoding_bci_experiment`
3. `speech_decoding_ecog_experiment`
4. `visual_decision_neuropixels_experiment`
5. `calcium_event_aligned_behavior_experiment`
6. `q_learning_behavior_neural_experiment`
7. `sleep_stage_eeg_experiment`
8. `seizure_detection_ieeg_experiment`

**Schema:**
```yaml
experimental_design: q_learning_behavior_neural_experiment
requires:
  - task: [reversal_learning, reward_learning]
  - behavioral_event: [choice, reward, trial_outcome]
  - analysis_affordance: q_learning_modeling
caveats:
  - reward alone is insufficient
  - reversal learning without choices is insufficient
```

**Functions:**
- `load_experimental_design_seeds(path)`
- `find_datasets_for_experimental_design(graph, design_id)`

**Files to create:**
- `data/graph/experimental_design_seeds.yaml`
- `neural_search/graph/experimental_design.py`

---

### Part 7: Test Fixtures

**Status:** NOT STARTED

**Fixture Files:**
- `tests/fixtures/graph/normalized_datasets.jsonl`
- `tests/fixtures/graph/normalized_papers.jsonl`
- `tests/fixtures/graph/expected_graph_summary.json`
- `tests/fixtures/graph/experimental_design_seeds.yaml`

**Required Fixture Datasets:**
1. Mouse OFC reversal learning electrophysiology
2. Human EEG motor imagery BCI
3. Human ECoG speech production
4. Mouse V1 Neuropixels visual decision-making
5. Human fMRI resting state (should NOT imply sleep staging)

**Required Fixture Papers:**
1. OFC reversal learning paper
2. Motor imagery BCI paper
3. Speech ECoG paper
4. Visual cortex Neuropixels paper

---

### Part 8: Documentation

**Files to create:**
- `docs/KNOWLEDGE_GRAPH_SCHEMA.md`
- `docs/GRAPH_AUGMENTED_SEARCH.md`
- `docs/EXPERIMENTAL_DESIGN_GRAPH.md`
- `docs/V0_5_IMPLEMENTATION_PLAN.md`

---

## IMPLEMENTATION STATUS SUMMARY

| Component | Status | Owner |
|-----------|--------|-------|
| v0.3 Scientific Labels | COMPLETE | - |
| v0.3 Analysis Affordances | COMPLETE | - |
| v0.4 Embedding Providers | COMPLETE | - |
| v0.4 Field Embeddings | COMPLETE | - |
| v0.4 Ablation Runner | COMPLETE | - |
| v0.4 Query Intent Docs | NOT STARTED | Claude |
| v0.4 Real-Corpus Benchmark | NOT STARTED | Claude |
| v0.4 Embedding Strategy Docs | NOT STARTED | Claude |
| v0.4 Human Relevance Protocol | NOT STARTED | Claude |
| v0.5 Graph Schema | NOT STARTED | Codex |
| v0.5 Graph Builder | NOT STARTED | Codex |
| v0.5 Graph Query Engine | NOT STARTED | Codex |
| v0.5 Graph Reports | NOT STARTED | Codex |
| v0.5 Search Features | NOT STARTED | Codex |
| v0.5 Experimental Design | NOT STARTED | Codex |
| v0.5 Graph Fixtures | NOT STARTED | Codex |
| v0.5 Graph Docs | NOT STARTED | Both |

---

## QUALITY GATES

All implementations must pass:
```bash
pytest -q
ruff check .
python -m neural_search.evaluation.run_benchmark --suite demo_v02
python -m neural_search.evaluation.run_benchmark --suite adversarial
```

v0.5 graph implementations must also pass:
```bash
python -m neural_search.graph.build_graph \
  --datasets tests/fixtures/graph/normalized_datasets.jsonl \
  --papers tests/fixtures/graph/normalized_papers.jsonl \
  --out /tmp/fixture_graph.json

python -m neural_search.graph.reports \
  --graph /tmp/fixture_graph.json \
  --out /tmp/graph_reports
```

---

## CRITICAL CONSTRAINTS

1. Do NOT build frontend features
2. Do NOT tune only the demo benchmark
3. Do NOT remove provenance/evidence requirements
4. Do NOT make graph score dominate retrieval
5. Do NOT require external graph database
6. Existing benchmark suites must continue to pass
7. No conflict with existing v0.4 embedding work
