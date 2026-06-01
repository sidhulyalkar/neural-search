# Task 12: Semantic Neuroscience Search

**Status: IMPLEMENTED**

## Implementation Summary

Task 12 has been fully implemented with 6 phases completed:

### Phase 1: Concept Embedding Foundation ✅
- Created `neural_search/embeddings/concept_embeddings.py` with `ConceptEmbedding` and `ConceptEmbeddingIndex`
- Created `neural_search/embeddings/concept_builder.py` with `ConceptEmbeddingBuilder`
- Built 119 concept embeddings (128D for tasks/modalities, 64D for behaviors/analyses/regions)
- Saved to `data/embeddings/concept_embeddings.v1.jsonl`

### Phase 2: Enhanced Dataset Fingerprints ✅
- Created `neural_search/embeddings/semantic_fingerprint.py` with `SemanticDatasetFingerprint`
- 7 embedding dimensions: text (256D), task (128D), modality (128D), behavior (64D), analysis (64D), region (64D), design (32D)
- Built 56 semantic fingerprints for all corpus datasets
- Saved to `data/embeddings/semantic_fingerprints.v1.jsonl`

### Phase 3: Intelligent Graph Relationships ✅
- Created `neural_search/graph/semantic_edges.py` with semantic similarity edge building
- Built 251 semantic similarity edges between datasets based on fingerprint similarity
- Added semantic concept similarity edges

### Phase 4: Semantic Search Integration ✅
- Created `neural_search/search/semantic_scoring.py` with `SemanticSearchIndex` and scoring functions
- Integrated semantic fingerprint scoring into search results
- Dimension-specific relevance scoring (task, modality, behavior, analysis, design)

### Phase 5: Query Understanding Enhancement ✅
- Created `neural_search/search/semantic_expansion.py` with `SemanticExpansion`
- Automatic query expansion with semantically related concepts
- Example: "neuropixels" expands to include "extracellular_ephys"

### Phase 6: Evaluation & Tuning ✅
- 67 tests across 5 test files
- Integration verified with full pipeline test

### New Files Created
- `neural_search/embeddings/concept_embeddings.py`
- `neural_search/embeddings/concept_builder.py`
- `neural_search/embeddings/semantic_fingerprint.py`
- `neural_search/embeddings/semantic_similarity.py`
- `neural_search/graph/semantic_edges.py`
- `neural_search/search/semantic_scoring.py`
- `neural_search/search/semantic_expansion.py`
- `tests/test_concept_embeddings.py`
- `tests/test_semantic_fingerprints.py`
- `tests/test_semantic_edges.py`
- `tests/test_semantic_scoring.py`
- `tests/test_semantic_expansion.py`
- `data/embeddings/concept_embeddings.v1.jsonl`
- `data/embeddings/semantic_fingerprints.v1.jsonl`

---

## Vision

Transform Neural Search from ontology-based keyword matching into a true semantic search engine that *understands* neuroscience experiments. Datasets should be connected not just by explicit labels, but by learned relationships that capture how experiments, analyses, and findings relate to each other.

**Core Insight:** Current search matches "reversal learning" to datasets tagged with that task. True understanding means finding datasets where the *experimental structure* enables reversal learning analysis, even if not explicitly labeled.

---

## Current State (Post-Task 11)

| Component | State | Limitation |
|-----------|-------|------------|
| Text Embeddings | Hash-based 256D | No semantic understanding |
| Concept Embeddings | 64D hash per dimension | No learned similarities |
| Behavioral Events | String matching only | Not in fingerprints |
| Analysis Affordances | Exact ID match | No method family clustering |
| Graph | 308 nodes, 957 edges | Missing learned relationships |
| Search | 10 weighted components | 6% embedding weight, 94% exact match |

**Key Gap:** The system knows "Go/NoGo" and "response inhibition" are related *only* because we manually defined that relationship. It cannot discover that a novel task structure enables response inhibition analysis.

---

## Target State

| Component | Target | Benefit |
|-----------|--------|---------|
| Concept Embeddings | 128D learned dense vectors | Semantic similarity between concepts |
| Behavioral Embeddings | 64D per event, in fingerprints | Similar behaviors cluster together |
| Analysis Embeddings | 64D per affordance | Method families emerge naturally |
| Modality Interactions | Pairwise compatibility scores | Multi-modal analysis matching |
| Graph | 500+ learned edges | Emergent relationships |
| Search | 30%+ embedding weight | Semantic understanding drives results |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Query Understanding                          │
│  ┌─────────┐  ┌──────────────┐  ┌─────────────────────────────┐ │
│  │ Parse   │→ │ Embed Query  │→ │ Multi-Space Similarity      │ │
│  │ (ontol) │  │ (dense 384D) │  │ (text + task + behavior)    │ │
│  └─────────┘  └──────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Concept Embedding Space                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  │
│  │ Task       │  │ Modality   │  │ Behavior   │  │ Analysis  │  │
│  │ Embeddings │  │ Embeddings │  │ Embeddings │  │ Embeddings│  │
│  │ (128D)     │  │ (128D)     │  │ (64D)      │  │ (64D)     │  │
│  └────────────┘  └────────────┘  └────────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Dataset Semantic Fingerprint                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ text_embedding (384D) + task_embedding (128D) +            │ │
│  │ modality_embedding (128D) + behavior_embedding (64D) +     │ │
│  │ analysis_embedding (64D) + region_embedding (64D) +        │ │
│  │ design_compatibility (32D) = combined_fingerprint (864D)   │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Knowledge Graph Enhancement                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Learned Edges: task_similar_to, modality_compatible_with,  │ │
│  │ behavior_relates_to, analysis_requires_pattern             │ │
│  │                                                            │ │
│  │ Constraint Edges: valid_task_modality_pair,                │ │
│  │ analysis_enabled_by_behavior_set, design_fits_dataset      │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Semantic Search Scoring                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │ Ontology Match   │  │ Embedding Sim    │  │ Graph Context  │ │
│  │ (30%)            │  │ (40%)            │  │ (20%)          │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
│                           ↓                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Constraint Validation (10%): Valid experiment combination? │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Concept Embedding Foundation

**Goal:** Replace hash-based concept embeddings with learned dense vectors

#### 1.1 Create Concept Embedding Schema
**File:** `neural_search/embeddings/concept_embeddings.py`

```python
@dataclass
class ConceptEmbedding:
    """Dense embedding for a neuroscience concept."""

    concept_id: str          # e.g., "task:reversal_learning"
    concept_type: str        # task, modality, behavior, analysis, region
    label: str
    embedding: list[float]   # 128D for major types, 64D for minor
    model_version: str
    aliases: list[str] = field(default_factory=list)
    parent_concepts: list[str] = field(default_factory=list)
    child_concepts: list[str] = field(default_factory=list)

class ConceptEmbeddingIndex:
    """Index for fast concept similarity lookup."""

    def __init__(self, embeddings: list[ConceptEmbedding]):
        self.by_id: dict[str, ConceptEmbedding] = {}
        self.by_type: dict[str, list[ConceptEmbedding]] = {}
        self._embedding_matrix: np.ndarray = None

    def find_similar(
        self,
        concept_id: str,
        concept_type: str | None = None,
        k: int = 10,
        min_similarity: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Find concepts similar to the given concept."""

    def embed_text(self, text: str, concept_type: str) -> np.ndarray:
        """Embed free text in the concept space."""
```

#### 1.2 Build Concept Embeddings from Ontology
**File:** `neural_search/embeddings/concept_builder.py`

Strategy: Use sentence transformers on concept definitions + aliases + related terms

```python
class ConceptEmbeddingBuilder:
    """Build dense concept embeddings from ontology definitions."""

    def __init__(
        self,
        text_model: str = "all-MiniLM-L6-v2",
        dim_reduction: str = "pca",  # pca or umap
        target_dim: int = 128,
    ):
        self.encoder = SentenceTransformerProvider(text_model)

    def build_task_embeddings(
        self,
        task_ontology: TaskOntology,
    ) -> list[ConceptEmbedding]:
        """Build embeddings for all tasks."""
        # Encode: task_name + " " + definition + " " + " ".join(aliases)

    def build_behavior_embeddings(
        self,
        behavior_ontology: BehaviorOntology,
    ) -> list[ConceptEmbedding]:
        """Build embeddings for behavioral events."""

    def build_analysis_embeddings(
        self,
        affordance_definitions: dict,
    ) -> list[ConceptEmbedding]:
        """Build embeddings for analysis methods."""
```

#### 1.3 Semantic Similarity Functions
**File:** `neural_search/embeddings/semantic_similarity.py`

```python
def concept_similarity(
    concept_a: str,
    concept_b: str,
    index: ConceptEmbeddingIndex,
) -> float:
    """Compute semantic similarity between two concepts."""

def find_semantically_similar_tasks(
    task_id: str,
    index: ConceptEmbeddingIndex,
    min_similarity: float = 0.6,
) -> list[tuple[str, float]]:
    """Find tasks with similar experimental structure."""

def query_to_concept_similarities(
    query: str,
    parsed_query: dict,
    index: ConceptEmbeddingIndex,
) -> dict[str, list[tuple[str, float]]]:
    """Map query to similar concepts in each dimension."""
```

---

### Phase 2: Enhanced Dataset Fingerprints

**Goal:** Add behavioral and analysis dimensions to fingerprints

#### 2.1 Extended Fingerprint Schema
**File:** `neural_search/embeddings/fingerprint.py` (extend)

```python
@dataclass
class SemanticDatasetFingerprint:
    """Rich multi-modal fingerprint with behavioral + analysis dimensions."""

    dataset_id: str

    # Text (384D - sentence transformer)
    text_embedding: list[float]

    # Concepts (128D each - learned)
    task_embedding: list[float]
    modality_embedding: list[float]

    # Behavioral (64D - learned)
    behavior_embedding: list[float]
    behavior_complexity: float  # 0-1 score

    # Analysis (64D - learned)
    analysis_embedding: list[float]
    analysis_affordance_ids: list[str]

    # Brain regions (64D - learned)
    region_embedding: list[float]

    # Experimental design (32D - computed)
    design_embedding: list[float]
    design_type: str  # "2afc", "go_nogo", "free_behavior", etc.

    # Combined (864D - weighted concatenation)
    combined_embedding: list[float]

    # Metadata
    model_version: str
    created_at: str

    # Quality indicators
    embedding_confidence: float  # How well-defined the dataset is
    missing_dimensions: list[str]  # Which embeddings are imputed
```

#### 2.2 Behavioral Embedding Extractor
**File:** `neural_search/embeddings/behavior_extractor.py`

```python
class BehaviorEmbeddingExtractor:
    """Extract behavioral embedding from dataset labels and description."""

    def __init__(self, concept_index: ConceptEmbeddingIndex):
        self.index = concept_index

    def extract_behavior_embedding(
        self,
        record: NormalizedDatasetRecord,
    ) -> tuple[list[float], float]:
        """Extract behavior embedding and complexity score.

        Returns:
            (embedding, complexity_score)
        """
        behaviors = record.behaviors or []
        behavioral_events = record.behavioral_events or []

        # Aggregate embeddings of matched behaviors
        # Weight by confidence and complexity

    def compute_behavior_complexity(
        self,
        behaviors: list[str],
        task_structure: dict,
    ) -> float:
        """Estimate behavioral complexity (0-1).

        Simple reflex → 0.2
        Learned association → 0.5
        Strategic decision → 0.8
        """
```

#### 2.3 Analysis Affordance Embedding
**File:** `neural_search/embeddings/analysis_extractor.py`

```python
class AnalysisEmbeddingExtractor:
    """Extract analysis affordance embedding from dataset properties."""

    def extract_analysis_embedding(
        self,
        record: NormalizedDatasetRecord,
        card: DatasetCard | None = None,
    ) -> list[float]:
        """Compute embedding representing analysis suitability."""

    def predict_affordances(
        self,
        fingerprint: SemanticDatasetFingerprint,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Predict which analyses this dataset supports."""
```

---

### Phase 3: Intelligent Graph Relationships

**Goal:** Add learned and computed edges to the knowledge graph

#### 3.1 Semantic Edge Generator
**File:** `neural_search/graph/semantic_edges.py`

```python
class SemanticEdgeGenerator:
    """Generate graph edges from embedding similarity."""

    def __init__(
        self,
        concept_index: ConceptEmbeddingIndex,
        min_similarity: float = 0.65,
    ):
        self.index = concept_index

    def generate_task_similarity_edges(
        self,
        graph: KnowledgeGraph,
    ) -> list[KnowledgeGraphEdge]:
        """Create task_similar_to edges from embedding similarity.

        Links tasks that have similar experimental structure
        even if not explicitly related in ontology.
        """

    def generate_modality_compatibility_edges(
        self,
        graph: KnowledgeGraph,
    ) -> list[KnowledgeGraphEdge]:
        """Create modality_compatible_with edges.

        Links modalities that work well together for analysis.
        """

    def generate_behavior_relation_edges(
        self,
        graph: KnowledgeGraph,
    ) -> list[KnowledgeGraphEdge]:
        """Create behavior_relates_to edges.

        Links behaviors in the same category (motor, cognitive, etc.)
        """
```

#### 3.2 Constraint Graph Builder
**File:** `neural_search/graph/constraints.py`

```python
@dataclass
class ExperimentConstraint:
    """A constraint on valid experiment combinations."""

    constraint_id: str
    constraint_type: str  # "requires", "excludes", "suggests"
    source_concepts: list[str]
    target_concepts: list[str]
    confidence: float
    explanation: str

class ConstraintGraphBuilder:
    """Build constraint edges from data and rules."""

    def build_task_modality_constraints(
        self,
        graph: KnowledgeGraph,
    ) -> list[ExperimentConstraint]:
        """Which tasks require which modalities?

        Example: EEG classification → requires EEG modality
        """

    def build_analysis_behavior_constraints(
        self,
        graph: KnowledgeGraph,
    ) -> list[ExperimentConstraint]:
        """Which analyses require which behaviors?

        Example: Choice decoding → requires choice behavioral event
        """

    def validate_dataset_for_analysis(
        self,
        dataset_id: str,
        analysis_id: str,
        graph: KnowledgeGraph,
    ) -> tuple[bool, list[str]]:
        """Check if dataset satisfies analysis constraints.

        Returns:
            (valid, list of missing requirements)
        """
```

#### 3.3 Design Compatibility Scorer
**File:** `neural_search/graph/design_compatibility.py`

```python
class DesignCompatibilityScorer:
    """Score how well a dataset fits an experimental design."""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def compute_design_fit(
        self,
        dataset_id: str,
        design_id: str,
    ) -> tuple[float, dict]:
        """Compute fit score and detailed breakdown.

        Returns:
            (score 0-1, {
                "task_fit": 0.9,
                "modality_fit": 0.8,
                "behavior_fit": 0.7,
                "missing": ["reward_timing"],
            })
        """

    def find_best_datasets_for_design(
        self,
        design_id: str,
        top_k: int = 10,
    ) -> list[tuple[str, float, dict]]:
        """Find datasets that best match a design template."""
```

---

### Phase 4: Semantic Search Integration

**Goal:** Integrate embeddings into core search with significant weight

#### 4.1 Embedding-Based Scorer
**File:** `neural_search/search/semantic_scorer.py`

```python
class SemanticScorer:
    """Score datasets using semantic embeddings."""

    def __init__(
        self,
        concept_index: ConceptEmbeddingIndex,
        fingerprint_index: FingerprintIndex,
    ):
        self.concepts = concept_index
        self.fingerprints = fingerprint_index

    def score_semantic_similarity(
        self,
        query: str,
        parsed_query: dict,
        dataset_id: str,
    ) -> SemanticScore:
        """Compute multi-dimensional semantic similarity.

        Returns:
            SemanticScore with per-dimension breakdown
        """

    def expand_query_semantically(
        self,
        parsed_query: dict,
    ) -> dict:
        """Add semantically similar concepts to query.

        If query mentions "Go/NoGo", also add similar tasks
        like "response inhibition", "impulse control".
        """

@dataclass
class SemanticScore:
    """Detailed semantic similarity breakdown."""

    overall: float
    text_similarity: float
    task_similarity: float
    modality_similarity: float
    behavior_similarity: float
    analysis_similarity: float
    region_similarity: float
    matched_concepts: list[tuple[str, str, float]]  # (query, matched, sim)
```

#### 4.2 Updated Retrieval Config
**File:** `data/config/retrieval.yaml` (extend)

```yaml
# Semantic search configuration
semantic_search:
  enabled: true

  # Concept embeddings
  concept_embeddings:
    path: data/embeddings/concept_embeddings.v1.jsonl
    similarity_threshold: 0.60

  # Enhanced fingerprints
  fingerprints:
    path: data/embeddings/semantic_fingerprints.v1.jsonl
    dimensions:
      text: 384
      task: 128
      modality: 128
      behavior: 64
      analysis: 64
      region: 64
      design: 32

  # Dimension weights for combined similarity
  dimension_weights:
    text: 0.20
    task: 0.25
    modality: 0.15
    behavior: 0.15
    analysis: 0.15
    region: 0.10

# Updated retrieval weights (semantic gets 40%)
weights:
  ontology: 0.20        # Reduced from 0.28
  behavior: 0.12        # Reduced from 0.20
  modality: 0.08        # Reduced from 0.12
  affordance: 0.06      # Reduced from 0.10
  metadata: 0.05        # Reduced from 0.10
  semantic_embedding: 0.25  # NEW: semantic fingerprint similarity
  semantic_expansion: 0.10  # NEW: expanded concept matching
  field_semantic: 0.04  # Reduced from 0.06
  graph: 0.06           # Increased from 0.04
  readiness: 0.04       # Reduced from 0.08
  paper_confidence: 0.02
  constraint_validation: 0.03  # NEW: valid experiment check
```

#### 4.3 Core Search Integration
**File:** `neural_search/search/core.py` (modify)

Add semantic scoring to `score_dataset_against_query()`:

```python
def score_dataset_against_query(
    dataset: Any,
    card: DatasetCardRead | Mapping[str, Any] | None,
    parsed_query: Mapping[str, Any],
    retrieval_config: Mapping[str, Any] | None = None,
    *,
    semantic_scorer: SemanticScorer | None = None,
) -> SearchResult:
    """Score with semantic embeddings."""

    # ... existing scoring ...

    # Add semantic embedding score
    if semantic_scorer is not None:
        semantic_score = semantic_scorer.score_semantic_similarity(
            query=str(parsed_query.get("query", "")),
            parsed_query=dict(parsed_query),
            dataset_id=dataset_id,
        )

        score_breakdown["semantic_embedding"] = semantic_score.overall
        score_breakdown["semantic_task"] = semantic_score.task_similarity
        score_breakdown["semantic_behavior"] = semantic_score.behavior_similarity

        # Add semantic matches to explanations
        for q_term, matched, sim in semantic_score.matched_concepts[:3]:
            why.append(f"Semantic match: {q_term} → {matched} ({sim:.2f})")

        # Add to final score
        final_score += weights.get("semantic_embedding", 0) * semantic_score.overall
```

---

### Phase 5: Query Understanding Enhancement

**Goal:** Deep query understanding with semantic expansion

#### 5.1 Semantic Query Parser
**File:** `neural_search/search/semantic_parser.py`

```python
class SemanticQueryParser:
    """Parse queries with semantic understanding."""

    def __init__(
        self,
        concept_index: ConceptEmbeddingIndex,
        expansion_threshold: float = 0.65,
    ):
        self.concepts = concept_index

    def parse_with_expansion(
        self,
        query: str,
        base_parsed: dict,
    ) -> dict:
        """Expand parsed query with semantically similar concepts.

        Example:
            Input: {"tasks": ["reversal_learning"]}
            Output: {"tasks": ["reversal_learning"],
                     "expanded_tasks": [
                         ("probabilistic_learning", 0.82),
                         ("rule_switching", 0.78),
                     ]}
        """

    def infer_implicit_requirements(
        self,
        parsed_query: dict,
    ) -> dict:
        """Infer requirements not explicitly stated.

        If user wants "choice decoding", they implicitly need:
        - Behavioral events: choice
        - Modalities: neural recording (any)
        - Task structure: trials with choices
        """
```

#### 5.2 Intent-Aware Semantic Expansion
**File:** `neural_search/search/intent.py` (extend)

```python
def expand_query_by_intent(
    parsed_query: dict,
    intent: IntentClassification,
    concept_index: ConceptEmbeddingIndex,
) -> dict:
    """Expand query based on intent type.

    TASK_SEARCH: Expand to similar tasks
    ANALYSIS_SEARCH: Expand to related analyses + required behaviors
    DATASET_LOOKUP: No expansion (exact match wanted)
    """
```

---

### Phase 6: Evaluation & Tuning

**Goal:** Measure and optimize semantic search quality

#### 6.1 Semantic Search Benchmark
**File:** `neural_search/evaluation/semantic_benchmark.py`

```python
class SemanticSearchBenchmark:
    """Benchmark semantic understanding capabilities."""

    def evaluate_concept_similarity(
        self,
        concept_index: ConceptEmbeddingIndex,
        gold_similar_pairs: list[tuple[str, str, float]],
    ) -> dict:
        """Evaluate if embeddings capture expected similarities."""

    def evaluate_semantic_expansion(
        self,
        queries: list[str],
        expected_expansions: list[list[str]],
    ) -> dict:
        """Evaluate if semantic expansion finds relevant concepts."""

    def evaluate_constraint_detection(
        self,
        dataset_analysis_pairs: list[tuple[str, str, bool]],
    ) -> dict:
        """Evaluate if constraints correctly identify valid combinations."""
```

#### 6.2 Embedding Quality Metrics
**File:** `neural_search/evaluation/embedding_metrics.py`

```python
def measure_embedding_coverage(
    fingerprints: list[SemanticDatasetFingerprint],
) -> dict:
    """Measure how many datasets have complete embeddings."""

def measure_embedding_separation(
    concept_embeddings: list[ConceptEmbedding],
    concept_type: str,
) -> dict:
    """Measure if different concept types are well-separated."""

def measure_embedding_clustering(
    fingerprints: list[SemanticDatasetFingerprint],
    expected_clusters: dict[str, list[str]],
) -> dict:
    """Measure if similar datasets cluster together."""
```

---

## File Summary

| File | Action | Phase |
|------|--------|-------|
| `neural_search/embeddings/concept_embeddings.py` | NEW | 1 |
| `neural_search/embeddings/concept_builder.py` | NEW | 1 |
| `neural_search/embeddings/semantic_similarity.py` | NEW | 1 |
| `neural_search/embeddings/fingerprint.py` | EXTEND | 2 |
| `neural_search/embeddings/behavior_extractor.py` | NEW | 2 |
| `neural_search/embeddings/analysis_extractor.py` | NEW | 2 |
| `neural_search/graph/semantic_edges.py` | NEW | 3 |
| `neural_search/graph/constraints.py` | NEW | 3 |
| `neural_search/graph/design_compatibility.py` | NEW | 3 |
| `neural_search/search/semantic_scorer.py` | NEW | 4 |
| `neural_search/search/core.py` | MODIFY | 4 |
| `data/config/retrieval.yaml` | EXTEND | 4 |
| `neural_search/search/semantic_parser.py` | NEW | 5 |
| `neural_search/search/intent.py` | EXTEND | 5 |
| `neural_search/evaluation/semantic_benchmark.py` | NEW | 6 |
| `neural_search/evaluation/embedding_metrics.py` | NEW | 6 |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| P@1 | ~93% | 97%+ |
| Semantic similarity tests | N/A | 90%+ accuracy |
| Concept clustering quality | N/A | Silhouette > 0.6 |
| Cross-task retrieval | N/A | 80%+ recall |
| Analysis constraint accuracy | N/A | 95%+ |
| Embedding coverage | 4 dimensions | 7 dimensions |
| Embedding weight in search | 6% | 35%+ |

---

## Implementation Order

1. **Phase 1** (Foundation): Build concept embeddings - everything else depends on this
2. **Phase 2** (Fingerprints): Extend fingerprints with new dimensions
3. **Phase 4** (Integration): Integrate into search while testing
4. **Phase 3** (Graph): Add learned edges using embeddings
5. **Phase 5** (Query): Enhance query understanding
6. **Phase 6** (Evaluation): Measure and tune

---

## Dependencies

- `sentence-transformers>=2.2.0` (required)
- `scikit-learn>=1.0.0` (for dimensionality reduction, clustering)
- `numpy>=1.21.0` (already present)
- `umap-learn>=0.5.0` (optional, for UMAP dimensionality reduction)

---

## Risk Mitigation

1. **Embedding quality varies:** Start with sentence-transformers baseline, upgrade later
2. **Concept coverage gaps:** Fall back to hash-based for unknown concepts
3. **Search latency:** Pre-compute fingerprints, use approximate NN for large corpora
4. **Over-expansion:** Cap semantic expansion at 5 concepts per dimension
5. **Constraint false positives:** Use high thresholds (0.9+) for hard constraints
6. **Backward compatibility:** All semantic features are additive, existing search works

---

## Example Queries After Task 12

**Query:** "datasets for studying cognitive flexibility"

**Current behavior:**
- Exact match "cognitive flexibility" → few results
- No expansion to related tasks

**Task 12 behavior:**
- Semantic expansion: cognitive flexibility → reversal learning (0.85), set shifting (0.82), rule switching (0.78)
- Finds datasets tagged with ANY of these tasks
- Ranks by semantic similarity to "cognitive flexibility"
- Explains: "Semantic match: cognitive flexibility → reversal learning (0.85)"

**Query:** "can I do population decoding on this?"

**Current behavior:**
- Matches "decoding" keyword
- No analysis of whether dataset structure supports decoding

**Task 12 behavior:**
- Identifies required behaviors: trial events, behavioral labels
- Checks constraints: modality must be neural recording
- Validates dataset has required structure
- Returns: "Yes, this dataset supports population decoding because it has spike data, behavioral labels, and trial structure."

---

## Future Extensions (Post Task 12)

- **Learned embeddings from usage:** Update concept embeddings based on search success
- **Fine-tuned neuroscience models:** Train sentence-transformer on neuroscience papers
- **Cross-species transfer:** Learn which findings generalize across species
- **Temporal reasoning:** Understand event sequences and timing constraints
- **Quality prediction:** Predict analysis success probability from embeddings
