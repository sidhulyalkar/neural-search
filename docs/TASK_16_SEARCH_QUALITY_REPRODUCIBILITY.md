# Task 16: Search Quality & Reproducibility

**Status: PLANNING**

## Vision

Establish Neural Search as a scientifically rigorous, reproducible search engine with documented quality metrics, ablation studies, and transparent decision-making. The search should be explainable, auditable, and demonstrate measurable improvements.

**Core Goals:**
1. **Reproducibility:** Any search can be reproduced with documented configuration
2. **Transparency:** Every ranking decision is explainable
3. **Measurability:** Quality improvements are quantified with ablation studies
4. **Robustness:** Performance is consistent across query types

---

## Current State

| Component | State | Gap |
|-----------|-------|-----|
| Query intent classification | Basic implementation | No routing to specialized handlers |
| Scoring transparency | 10+ weighted components | Weights not validated |
| Ablation infrastructure | Minimal | No systematic comparison |
| Reproducibility | Config-based | No versioned snapshots |
| Audit trails | None | Cannot trace ranking decisions |

---

## Implementation Phases

### Phase 1: Query Intent Router

**Goal:** Route queries to intent-specific scoring profiles for better results.

#### 1.1 Intent Categories

```python
class QueryIntent(Enum):
    """Query intent categories with specialized handling."""

    # Direct lookups (high precision needed)
    DATASET_LOOKUP = "dataset_lookup"       # "DANDI 000026"
    PAPER_LOOKUP = "paper_lookup"           # "Steinmetz 2019"

    # Structured searches (constraint satisfaction)
    TASK_SEARCH = "task_search"             # "reversal learning"
    MODALITY_SEARCH = "modality_search"     # "neuropixels recordings"
    SPECIES_REGION = "species_region"       # "mouse hippocampus"
    ANALYSIS_AFFORDANCE = "analysis_affordance"  # "decoding ready"

    # Linking queries (graph relationships)
    PAPER_TO_DATASET = "paper_to_dataset"   # "datasets from Steinmetz lab"
    DATASET_TO_PAPER = "dataset_to_paper"   # "papers using IBL data"
    SIMILAR_DATASET = "similar_dataset"     # "datasets like 000026"

    # Complex queries (multiple constraints)
    HARD_NEGATIVE = "hard_negative"         # "neuropixels NOT calcium"
    MULTI_CONSTRAINT = "multi_constraint"   # "mouse visual cortex go/nogo NWB"

    # Exploratory (recall over precision)
    EXPLORATORY = "exploratory"             # "what datasets study attention?"
```

#### 1.2 Intent Detection

```python
@dataclass
class IntentClassification:
    """Classification result for a query."""

    primary_intent: QueryIntent
    confidence: float  # 0-1
    secondary_intents: list[QueryIntent]
    detected_signals: dict[str, Any]  # What triggered classification


def classify_query_intent(
    query: str,
    parsed_query: dict[str, Any],
) -> IntentClassification:
    """Classify query intent for routing.

    Detection signals:
    - DATASET_LOOKUP: Matches dataset ID pattern (DANDI, ds, 6-digit)
    - PAPER_LOOKUP: Contains author names, journal, year pattern
    - TASK_SEARCH: Has task matches from ontology
    - MODALITY_SEARCH: Has modality matches
    - ANALYSIS_AFFORDANCE: Contains "for X analysis", "supports X"
    - HARD_NEGATIVE: Contains NOT, without, excluding
    - SIMILAR_DATASET: Contains "like", "similar to"
    - EXPLORATORY: Question words, broad terms
    """
```

#### 1.3 Intent-Specific Weight Profiles

```yaml
# data/config/intent_profiles.yaml

profiles:
  dataset_lookup:
    description: "Direct dataset ID searches"
    weights:
      metadata_exact: 0.95
      semantic: 0.05
    scoring_mode: "exact_match_priority"
    min_confidence: 0.9

  task_search:
    description: "Task-focused searches"
    weights:
      ontology: 0.35
      behavior: 0.25
      semantic: 0.15
      metadata: 0.10
      affordance: 0.10
      readiness: 0.05
    scoring_mode: "weighted_sum"

  analysis_affordance:
    description: "Analysis capability searches"
    weights:
      affordance: 0.40
      ontology: 0.20
      semantic: 0.15
      readiness: 0.15
      metadata: 0.10
    scoring_mode: "affordance_priority"

  hard_negative:
    description: "Queries with explicit exclusions"
    weights:
      negative_constraint: 0.30
      ontology: 0.25
      modality: 0.20
      semantic: 0.15
      metadata: 0.10
    scoring_mode: "constraint_filter_first"
    strict_exclusion: true

  similar_dataset:
    description: "Find similar datasets"
    weights:
      semantic_fingerprint: 0.40
      graph_similarity: 0.30
      ontology: 0.20
      metadata: 0.10
    scoring_mode: "similarity_ranking"

  exploratory:
    description: "Broad exploratory searches"
    weights:
      semantic: 0.30
      ontology: 0.25
      metadata: 0.20
      readiness: 0.15
      affordance: 0.10
    scoring_mode: "recall_priority"
    result_limit: 20
```

### Phase 2: Ablation Study Infrastructure

**Goal:** Systematically measure contribution of each scoring component.

#### 2.1 Ablation Framework

```python
@dataclass
class AblationConfig:
    """Configuration for ablation study."""

    name: str
    description: str
    baseline_config: dict[str, Any]
    ablated_components: list[str]  # Components to disable
    benchmark_suite: str
    metrics: list[str]


@dataclass
class AblationResult:
    """Results from an ablation study."""

    config: AblationConfig
    baseline_metrics: dict[str, float]
    ablated_metrics: dict[str, float]
    delta: dict[str, float]
    significant: bool  # Statistical significance


def run_ablation_study(
    baseline_config: dict[str, Any],
    component_to_ablate: str,
    benchmark_suite: str,
) -> AblationResult:
    """Run single-component ablation study.

    Process:
    1. Run benchmark with baseline config
    2. Set ablated component weight to 0
    3. Run benchmark with ablated config
    4. Compare metrics and compute delta
    5. Test statistical significance
    """
```

#### 2.2 Component Ablation Matrix

```python
ABLATION_COMPONENTS = [
    # Core scoring heads
    "ontology",
    "behavior",
    "modality",
    "affordance",
    "metadata",
    "semantic",
    "readiness",
    "paper_confidence",

    # Enhanced scoring
    "field_semantic",
    "graph_context",
    "semantic_fingerprint",
    "semantic_expansion",

    # Constraint handling
    "hard_negative",
    "modality_penalty",
    "missing_field_penalty",

    # Query understanding
    "intent_routing",
    "query_expansion",
    "transitive_matching",
]


def run_full_ablation_matrix(
    benchmark_suite: str = "real_v08",
) -> pd.DataFrame:
    """Run ablation for each component.

    Returns DataFrame:
    | Component | P@5 (baseline) | P@5 (ablated) | Delta | Significant |
    |-----------|----------------|---------------|-------|-------------|
    | ontology  | 0.65           | 0.45          | -0.20 | Yes         |
    | semantic  | 0.65           | 0.58          | -0.07 | Yes         |
    | ...       | ...            | ...           | ...   | ...         |
    """
```

#### 2.3 Ablation Reports

```markdown
# Ablation Study Report - v0.8

## Summary

Ablation study run on `real_v08` benchmark suite with 200 queries.

### Critical Components (>10% impact)

| Component | P@5 Delta | MRR Delta | Recommendation |
|-----------|-----------|-----------|----------------|
| ontology | -20% | -25% | KEEP - core matching |
| semantic | -7% | -8% | KEEP - semantic understanding |
| behavior | -5% | -6% | KEEP - behavioral matching |

### Marginal Components (<5% impact)

| Component | P@5 Delta | MRR Delta | Recommendation |
|-----------|-----------|-----------|----------------|
| readiness | -2% | -2% | OPTIONAL - metadata quality |
| paper_confidence | -1% | -1% | OPTIONAL - provenance |

### Negative Impact Components

| Component | P@5 Delta | MRR Delta | Recommendation |
|-----------|-----------|-----------|----------------|
| None | N/A | N/A | All components beneficial |
```

### Phase 3: Search Audit Trails

**Goal:** Enable complete traceability of ranking decisions.

#### 3.1 Audit Log Schema

```python
@dataclass
class SearchAuditEntry:
    """Complete audit trail for a search request."""

    # Request metadata
    audit_id: str
    timestamp: str
    query: str
    query_hash: str  # For reproducibility

    # Configuration snapshot
    config_version: str
    config_hash: str
    weights: dict[str, float]

    # Query understanding
    parsed_query: dict[str, Any]
    intent_classification: IntentClassification
    semantic_expansion: SemanticExpansion | None

    # Scoring details per result
    result_scores: list[ResultScoreBreakdown]

    # Final ranking
    final_ranking: list[str]  # dataset_ids in order
    ranking_method: str


@dataclass
class ResultScoreBreakdown:
    """Detailed score breakdown for a single result."""

    dataset_id: str
    rank: int
    final_score: float

    # Component scores
    component_scores: dict[str, float]

    # Why matched/penalized
    matched_terms: list[str]
    applied_penalties: list[str]
    expansion_boosts: list[str]

    # Evidence
    evidence_snippets: list[str]
```

#### 3.2 Audit CLI

```bash
# View audit for a search
python -m neural_search.audit view \
    --query "reversal learning neuropixels" \
    --format detailed

# Output:
# ┌─────────────────────────────────────────────────────┐
# │ Search Audit: reversal learning neuropixels         │
# ├─────────────────────────────────────────────────────┤
# │ Query ID: q_abc123                                  │
# │ Intent: task_search (confidence: 0.85)              │
# │ Config: retrieval_v08 (hash: a1b2c3)                │
# ├─────────────────────────────────────────────────────┤
# │ Rank 1: DEMO_REVERSAL_EPHYS (score: 85.2)           │
# │   Matched: task:reversal_learning (0.30)            │
# │            modality:neuropixels (0.20)              │
# │   Semantic: 0.65 (fingerprint match)                │
# │   No penalties applied                              │
# ├─────────────────────────────────────────────────────┤
# │ Rank 2: 000026 (score: 72.1)                        │
# │   Matched: modality:neuropixels (0.20)              │
# │   Semantic: 0.55 (similar task structure)           │
# │   Penalty: missing task label (-0.05)               │
# └─────────────────────────────────────────────────────┘
```

#### 3.3 Reproducibility Snapshots

```python
@dataclass
class ReproducibilitySnapshot:
    """Complete state for reproducing search results."""

    snapshot_id: str
    created_at: str

    # Code version
    git_commit: str
    git_branch: str

    # Configuration
    retrieval_config: dict[str, Any]
    intent_profiles: dict[str, Any]

    # Data artifacts
    corpus_hash: str  # Hash of normalized records
    ontology_hash: str  # Hash of ontology files
    embeddings_hash: str  # Hash of embedding files
    graph_hash: str  # Hash of knowledge graph

    # Benchmark reference
    benchmark_suite: str
    benchmark_results_hash: str


def create_snapshot(name: str) -> ReproducibilitySnapshot:
    """Create snapshot of current search state."""


def restore_snapshot(snapshot_id: str) -> None:
    """Restore search state from snapshot."""


def verify_snapshot(snapshot_id: str) -> bool:
    """Verify current state matches snapshot."""
```

### Phase 4: Quality Dashboard

**Goal:** Real-time visibility into search quality metrics.

#### 4.1 Metrics Collection

```python
@dataclass
class SearchQualityMetrics:
    """Aggregated search quality metrics."""

    # Overall performance
    mean_precision_at_5: float
    mean_mrr: float
    mean_ndcg_at_10: float

    # By query category
    category_metrics: dict[str, dict[str, float]]

    # By modality
    modality_metrics: dict[str, dict[str, float]]

    # Constraint satisfaction
    hard_negative_violation_rate: float
    constraint_satisfaction_rate: float

    # Human correlation
    human_system_correlation: float
    human_precision_at_5: float

    # Trends
    metric_history: list[MetricSnapshot]


def collect_quality_metrics(
    benchmark_suite: str,
    include_human_labels: bool = True,
) -> SearchQualityMetrics:
    """Collect comprehensive quality metrics."""
```

#### 4.2 Quality Report Generator

```python
def generate_quality_report(
    metrics: SearchQualityMetrics,
    format: Literal["markdown", "html", "json"] = "markdown",
) -> str:
    """Generate human-readable quality report.

    Sections:
    1. Executive Summary
    2. Overall Performance
    3. Performance by Query Category
    4. Performance by Modality
    5. Constraint Satisfaction
    6. Human Correlation Analysis
    7. Trends Over Time
    8. Recommendations
    """
```

#### 4.3 Quality Dashboard Output

```markdown
# Neural Search Quality Dashboard

## Executive Summary

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| P@5 (overall) | 0.62 | 0.65 | 🟡 |
| MRR | 0.71 | 0.75 | 🟡 |
| Hard Negative Violations | 0.02% | 0% | 🟢 |
| Human Correlation | 0.58 | 0.60 | 🟡 |

## Performance by Query Category

| Category | P@5 | MRR | Queries |
|----------|-----|-----|---------|
| dataset_lookup | 1.00 | 1.00 | 20 |
| task_search | 0.68 | 0.75 | 30 |
| analysis_affordance | 0.52 | 0.61 | 25 |
| hard_negative | 0.65 | 0.72 | 20 |
| exploratory | 0.45 | 0.55 | 15 |

## Performance by Modality

| Modality | P@5 | Coverage | Gap |
|----------|-----|----------|-----|
| neuropixels | 0.72 | 15 datasets | None |
| fmri | 0.48 | 8 datasets | +5 needed |
| eeg | 0.55 | 6 datasets | +4 needed |
| meg | 0.35 | 2 datasets | +8 needed |

## Recommendations

1. **Increase MEG corpus coverage** - Only 2 datasets, P@5 is 0.35
2. **Tune analysis affordance scoring** - P@5 of 0.52 below target
3. **Review exploratory query handling** - Consider recall-focused profile
```

---

## Implementation Checklist

### Phase 1: Query Intent Router
- [ ] Expand `QueryIntent` enum with all categories
- [ ] Implement robust intent detection signals
- [ ] Create `data/config/intent_profiles.yaml`
- [ ] Integrate intent routing into `search_datasets()`
- [ ] Add intent classification to SearchResult
- [ ] Write tests for intent detection

### Phase 2: Ablation Infrastructure
- [ ] Create `neural_search/evaluation/ablation.py`
- [ ] Implement `run_ablation_study()` function
- [ ] Implement `run_full_ablation_matrix()`
- [ ] Generate ablation report template
- [ ] Add CLI: `python -m neural_search.evaluation.ablation`
- [ ] Document component contribution rankings

### Phase 3: Audit Trails
- [ ] Create `neural_search/audit/` module
- [ ] Implement `SearchAuditEntry` schema
- [ ] Create audit logging during search
- [ ] Implement audit CLI viewer
- [ ] Create reproducibility snapshots
- [ ] Add snapshot verification

### Phase 4: Quality Dashboard
- [ ] Create `neural_search/evaluation/dashboard.py`
- [ ] Implement metrics collection
- [ ] Create markdown report generator
- [ ] Add trend tracking over time
- [ ] Integrate with CI for automated reports
- [ ] Create quality gate checks

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Intent detection accuracy | Unknown | >85% |
| Ablation coverage | 0 | 100% components |
| Audit trail coverage | 0% | 100% searches |
| Quality report automation | Manual | CI-integrated |
| Reproducibility verification | None | Snapshot-based |

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `neural_search/search/intent_router.py` | NEW |
| `data/config/intent_profiles.yaml` | NEW |
| `neural_search/evaluation/ablation.py` | NEW |
| `neural_search/audit/__init__.py` | NEW |
| `neural_search/audit/logger.py` | NEW |
| `neural_search/audit/snapshot.py` | NEW |
| `neural_search/evaluation/dashboard.py` | NEW |
| `neural_search/evaluation/quality_report.py` | NEW |
| `tests/test_intent_routing.py` | NEW |
| `tests/test_ablation.py` | NEW |
| `tests/test_audit.py` | NEW |

---

## Dependencies

- **Task 12** (Semantic Search): Uses semantic fingerprints for similarity
- **Task 13** (Neuroscience Awareness): Uses data-form awareness
- **Task 14** (Awareness-Integrated): Uses awareness scoring
- **Task 15** (Corpus Expansion): Uses expanded real corpus for validation
