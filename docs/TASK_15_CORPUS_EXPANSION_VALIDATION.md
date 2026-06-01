# Task 15: Real Corpus Expansion & Validation Pipeline

**Status: IN PROGRESS**

## Vision

Transform Neural Search from a demo-validated prototype into a production-validated search engine with real corpus coverage and human-verified relevance judgments. The current real corpus of 4 records cannot demonstrate search quality—we need 50-100+ real datasets with validated relevance labels.

**Core Problem:** Demo benchmarks (29/30 pass) may reflect overfitting between corpus, ontology, and queries. We need external validation with real data.

---

## Current State

| Metric | Current | Target |
|--------|---------|--------|
| Demo corpus | 26 datasets | Maintain |
| Real corpus | 4 records | 50-100+ records |
| Human relevance labels | 0 | 500+ judgments |
| Modality coverage (real) | 2 types | 8+ types |
| Benchmark queries (real) | 30 lines | 200+ queries |
| External validation | None | DANDI/OpenNeuro verified |

### Coverage Gaps in Real Corpus

| Modality | Demo Count | Real Count | Gap |
|----------|------------|------------|-----|
| Neuropixels/ephys | 13 | 2 | Critical |
| fMRI | 2 | 0 | Critical |
| MEG | 0 | 0 | Critical |
| EEG | 2 | 0 | High |
| Calcium imaging | 3 | 0 | High |
| Connectomics | 0 | 0 | Medium |
| Single-cell/molecular | 0 | 0 | Medium |
| Clinical | 0 | 0 | Medium |

---

## Implementation Phases

### Phase 1: Corpus Expansion Pipeline

**Goal:** Systematically ingest 50+ real datasets from DANDI, OpenNeuro, and OpenAlex.

#### 1.1 DANDI Expansion (20+ datasets)

Target dandisets covering underrepresented modalities:

```yaml
# High-priority DANDI datasets for ingestion
priority_dandisets:
  # Neuropixels
  - "000003"  # Allen Institute Visual Coding
  - "000004"  # Steinmetz 2019
  - "000005"  # Allen Institute Brain Observatory

  # Calcium imaging
  - "000037"  # Tolias lab calcium imaging
  - "000049"  # Svoboda lab calcium imaging

  # iEEG/ECoG
  - "000055"  # Human iEEG
  - "000060"  # ECoG motor cortex

  # Behavior-rich
  - "000045"  # IBL behavior + ephys
  - "000067"  # Decision making + neuropixels
```

**Implementation:**
- Create `scripts/expand_real_corpus.py` with DANDI API pagination
- Fetch dandiset metadata + asset file structure
- Normalize to `NormalizedRecord` schema
- Save to `data/corpus/normalized/real_v08.*.jsonl`

#### 1.2 OpenNeuro Expansion (15+ datasets)

Target datasets with structured BIDS metadata:

```yaml
priority_openneuro:
  # fMRI
  - "ds000001"  # Balloon Analog Risk Task fMRI
  - "ds000002"  # Classification learning fMRI
  - "ds000003"  # Rhyme judgment fMRI

  # EEG
  - "ds002778"  # ERP Core
  - "ds003061"  # LEMON EEG

  # MEG
  - "ds000117"  # MEG multimodal
  - "ds000248"  # MEG resting state

  # Multi-modal
  - "ds000030"  # UCLA Consortium for Neuropsychiatric Phenomics
  - "ds000228"  # MPI-Leipzig Mind-Brain-Body
```

**Implementation:**
- Create `scripts/expand_openneuro_corpus.py`
- Parse BIDS dataset_description.json and participants.tsv
- Extract modality from file extensions
- Normalize and save

#### 1.3 Manual Curation (10+ landmark datasets)

Add well-known datasets not in repositories:

```yaml
landmark_datasets:
  - name: "Steinmetz et al. 2019 IBL"
    doi: "10.1038/s41586-019-1787-x"
    modalities: ["neuropixels"]
    tasks: ["visual_discrimination"]

  - name: "Allen Brain Observatory"
    url: "https://observatory.brain-map.org"
    modalities: ["calcium_imaging", "neuropixels"]
    tasks: ["visual_coding"]

  - name: "Human Connectome Project"
    url: "https://www.humanconnectome.org"
    modalities: ["fmri", "dwi", "meg"]
    tasks: ["resting_state", "task_fmri"]
```

### Phase 2: Human Relevance Labeling Infrastructure

**Goal:** Enable human annotators to judge search result relevance.

#### 2.1 Relevance Schema

```python
@dataclass
class RelevanceJudgment:
    """Human relevance judgment for a search result."""

    judgment_id: str
    query_id: str
    query_text: str
    dataset_id: str
    dataset_title: str

    # Core relevance
    relevance: Literal[
        "exact",           # Perfect match for query intent
        "highly_relevant", # Good match, minor gaps
        "relevant",        # Matches query, some limitations
        "partially",       # Some relevance, significant gaps
        "not_relevant",    # Wrong domain/type/species
        "hard_negative",   # Explicitly should NOT match
    ]

    # Dimension-specific scores (0-3)
    task_match: int      # 0=wrong, 1=related, 2=close, 3=exact
    modality_match: int  # 0=wrong, 1=related, 2=compatible, 3=exact
    species_match: int   # 0=wrong, 1=related, 2=compatible, 3=exact
    analysis_fit: int    # 0=impossible, 1=maybe, 2=good, 3=excellent

    # Metadata
    reviewer_id: str
    review_timestamp: str
    review_notes: str
    confidence: float  # 0-1 reviewer confidence
```

#### 2.2 Relevance Labeling Tool

Create simple CLI for relevance annotation:

```bash
# Label top-10 results for a query
python -m neural_search.evaluation.label_relevance \
    --query "neuropixels visual cortex mouse" \
    --reviewer "sidso" \
    --output data/eval/relevance_labels.jsonl
```

Output format:
```jsonl
{"judgment_id": "j001", "query_id": "q_visual_ephys_001", "query_text": "neuropixels visual cortex mouse", "dataset_id": "000026", "relevance": "exact", "task_match": 3, "modality_match": 3, "species_match": 3, "analysis_fit": 3, "reviewer_id": "sidso", "review_timestamp": "2024-01-15T10:30:00Z", "review_notes": "Perfect match - Steinmetz data", "confidence": 0.95}
```

#### 2.3 Human-Validated Metrics

```python
def compute_human_precision(
    results: list[SearchResult],
    judgments: list[RelevanceJudgment],
    k: int = 5,
    min_relevance: str = "relevant",
) -> float:
    """Compute precision using human relevance labels."""

    relevant_levels = {"exact", "highly_relevant", "relevant"}
    if min_relevance == "highly_relevant":
        relevant_levels = {"exact", "highly_relevant"}
    elif min_relevance == "exact":
        relevant_levels = {"exact"}

    judgment_map = {j.dataset_id: j.relevance for j in judgments}

    top_k = results[:k]
    relevant_count = sum(
        1 for r in top_k
        if judgment_map.get(r.dataset_id) in relevant_levels
    )

    return relevant_count / k
```

### Phase 3: Comprehensive Benchmark Expansion

**Goal:** Create 200+ benchmark queries covering all search scenarios.

#### 3.1 Query Categories

```yaml
benchmark_categories:
  direct_lookup:
    count: 20
    examples:
      - "DANDI 000026"
      - "Steinmetz 2019 dataset"
      - "OpenNeuro ds000001"
    expected: exact_match_by_id

  task_modality_search:
    count: 30
    examples:
      - "reversal learning neuropixels"
      - "decision making calcium imaging"
      - "motor imagery EEG"
    expected: task_and_modality_match

  species_region_search:
    count: 25
    examples:
      - "mouse hippocampus electrophysiology"
      - "human visual cortex fMRI"
      - "macaque motor cortex"
    expected: species_region_modality_match

  analysis_affordance_search:
    count: 25
    examples:
      - "datasets for decoding analysis"
      - "Q-learning compatible recordings"
      - "spike sorting ready"
    expected: affordance_match

  hard_negative_search:
    count: 20
    examples:
      - "neuropixels NOT calcium imaging"
      - "mouse decision making, not human"
      - "visual cortex excluding auditory"
    expected: negative_constraint_respected

  cross_modal_search:
    count: 15
    examples:
      - "simultaneous ephys and behavior video"
      - "EEG with eye tracking"
      - "fMRI with behavioral measures"
    expected: multi_modal_match

  exploratory_search:
    count: 15
    examples:
      - "what datasets study attention?"
      - "recordings from prefrontal cortex"
      - "behavioral neuroscience datasets"
    expected: broad_relevant_results

  paper_linking_search:
    count: 15
    examples:
      - "datasets from Steinmetz lab"
      - "data supporting Nature 2019 publications"
      - "IBL consortium datasets"
    expected: paper_dataset_links

  underrepresented_modality_search:
    count: 20
    examples:
      - "MEG visual perception"
      - "connectomics mouse brain"
      - "single-cell RNA-seq brain"
      - "clinical prediction neuroimaging"
    expected: modality_specific_results

  complex_constraint_search:
    count: 15
    examples:
      - "mouse visual cortex neuropixels go/nogo task NWB format"
      - "human fMRI decision making BIDS"
      - "primate motor cortex reaching task"
    expected: all_constraints_satisfied
```

#### 3.2 Benchmark YAML Template

```yaml
# data/eval/benchmark_queries_v08_real.yaml

# Direct Lookup Queries
- id: lookup_001
  query: "DANDI 000026"
  query_category: direct_lookup
  expected_dataset_ids: ["000026"]
  minimum_precision_at_1: 1.0
  hard_negative_dataset_ids: []

# Task-Modality Queries
- id: task_mod_001
  query: "reversal learning neuropixels recordings"
  query_category: task_modality_search
  expected_tasks: ["reversal_learning"]
  expected_modalities_any: ["neuropixels", "extracellular_ephys"]
  minimum_precision_at_5: 0.6
  minimum_label_recall_at_10: 0.5
  hard_negative_modalities: ["calcium_imaging", "fmri"]

# Hard Negative Queries
- id: hard_neg_001
  query: "mouse neuropixels NOT calcium imaging"
  query_category: hard_negative_search
  expected_species: ["mouse"]
  expected_modalities_any: ["neuropixels"]
  hard_negative_modalities: ["calcium_imaging"]
  hard_negative_violation_tolerance: 0

# Analysis Affordance Queries
- id: affordance_001
  query: "datasets ready for neural decoding analysis"
  query_category: analysis_affordance_search
  expected_analysis_affordances: ["neural_decoding", "population_decoding"]
  minimum_precision_at_5: 0.4
```

### Phase 4: Quality Gates & CI Integration

#### 4.1 Regression Test Suite

```python
# tests/test_search_quality.py

class TestSearchQualityGates:
    """Quality gates that must pass before release."""

    def test_hard_negative_zero_violations(self):
        """Hard negative constraints must have 0 violations."""
        results = run_benchmark("adversarial")
        for query_result in results:
            assert query_result.hard_negative_violations == 0, (
                f"Query {query_result.query_id} had hard negative violations"
            )

    def test_direct_lookup_perfect_precision(self):
        """Direct ID lookups must return exact match at position 1."""
        results = run_benchmark("direct_lookup")
        for query_result in results:
            assert query_result.precision_at_1 == 1.0

    def test_minimum_modality_precision(self):
        """Each modality must achieve minimum precision threshold."""
        thresholds = {
            "neuropixels": 0.6,
            "fmri": 0.4,
            "eeg": 0.4,
            "calcium_imaging": 0.5,
        }
        results = run_benchmark_by_modality("real_v08")
        for modality, threshold in thresholds.items():
            assert results[modality].precision_at_5 >= threshold

    def test_human_relevance_correlation(self):
        """System rankings must correlate with human judgments."""
        system_results = search_datasets("reversal learning neuropixels")
        human_labels = load_relevance_labels("q_reversal_learning")

        # Compute rank correlation
        correlation = compute_rank_correlation(system_results, human_labels)
        assert correlation >= 0.5, "Low correlation with human judgments"
```

#### 4.2 CI Quality Report

```yaml
# .github/workflows/search_quality.yml

quality_check:
  runs-on: ubuntu-latest
  steps:
    - name: Run benchmark suite
      run: |
        python -m neural_search.evaluation.run_benchmark \
          --suite real_v08 \
          --output reports/benchmark_results.json

    - name: Check quality gates
      run: |
        python -m neural_search.evaluation.check_quality_gates \
          --results reports/benchmark_results.json \
          --gates config/quality_gates.yaml

    - name: Generate quality report
      run: |
        python -m neural_search.evaluation.generate_report \
          --results reports/benchmark_results.json \
          --format markdown \
          --output reports/SEARCH_QUALITY_REPORT.md
```

---

## Implementation Checklist

### Phase 1: Corpus Expansion
- [ ] Create `scripts/expand_real_corpus.py`
- [ ] Ingest 20+ DANDI datasets
- [ ] Ingest 15+ OpenNeuro datasets
- [ ] Add 10+ landmark manual datasets
- [ ] Generate `real_v08.datasets.jsonl` (50+ records)
- [ ] Generate `real_v08.papers.jsonl` (25+ records)
- [ ] Update corpus coverage reports

### Phase 2: Relevance Labeling
- [ ] Create `RelevanceJudgment` schema
- [ ] Build CLI labeling tool
- [ ] Label 20 queries with top-10 results each (200 judgments)
- [ ] Compute human-validated precision metrics
- [ ] Store in `data/eval/relevance_labels.jsonl`

### Phase 3: Benchmark Expansion
- [ ] Create `benchmark_queries_v08_real.yaml` (200+ queries)
- [ ] Cover all 10 query categories
- [ ] Add hard negative expectations
- [ ] Add human relevance expectations
- [ ] Document expected corpus coverage per category

### Phase 4: Quality Gates
- [ ] Create `tests/test_search_quality.py`
- [ ] Add hard negative violation tests
- [ ] Add modality precision thresholds
- [ ] Add human correlation tests
- [ ] Integrate into CI pipeline

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Real corpus size | 4 | 50+ |
| Human relevance labels | 0 | 500+ |
| Benchmark queries (real) | 30 | 200+ |
| Hard negative violations | Unknown | 0 |
| Human-system correlation | Unknown | >0.5 |
| Modality coverage (real) | 2 | 8+ |

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `scripts/expand_real_corpus.py` | NEW |
| `neural_search/evaluation/relevance.py` | NEW |
| `neural_search/evaluation/label_relevance.py` | NEW (CLI) |
| `data/eval/benchmark_queries_v08_real.yaml` | NEW |
| `data/eval/relevance_labels.jsonl` | NEW |
| `data/corpus/normalized/real_v08.*.jsonl` | NEW |
| `tests/test_search_quality.py` | NEW |
| `config/quality_gates.yaml` | NEW |
