# Neural Search v0.3 Implementation Plan

**Codename:** Real Corpus Alpha
**Status:** Complete
**Created:** 2026-05-23

---

## Executive Summary

v0.3 transforms Neural Search from a curated demo retrieval engine into a **provenance-aware, real-corpus neuroscience search system** with:

- Real dataset/paper ingestion from DANDI, OpenNeuro, OpenAlex
- Evidence-backed labels with confidence scores
- Adversarial scientific evaluation benchmarks
- Interpretable multi-head scoring
- Analysis affordance detection

**Critical Constraint:** The v0.2 benchmark (29/30 pass rate) is frozen as a regression suite. Do not optimize against it.

---

## Current Checkpoint: 2026-05-24 (COMPLETE)

v0.3 core implementation is now complete. Implemented features:

- `data/eval/benchmark_queries_demo_v02.yaml` freezes the current v0.2 benchmark without semantic edits.
- `data/eval/benchmark_queries_real_corpus.yaml` and `data/eval/benchmark_queries_adversarial.yaml` provide schema-valid starting suites.
- `neural_search/evaluation/run_benchmark.py` accepts `--suite demo_v02|real_corpus|adversarial|all` and writes suite-specific reports under `data/eval/results/{suite}/`, including `latest.json`.
- `neural_search/schemas.py` now includes `EvidenceLabel`, `UsabilityFlags`, `AnalysisAffordance`, `NormalizedDatasetRecord`, `NormalizedPaperRecord`, and `ScoreBreakdown`.
- `neural_search/normalized.py` adds stable normalized IDs plus JSON/JSONL serialization helpers.
- DANDI, OpenNeuro, and OpenAlex ingestion modules now expose additive v0.3 normalization functions that return provenance-aware records while preserving the existing database-oriented normalization functions.
- `neural_search/reports/corpus_report.py` generates the v0.3 corpus QA reports from normalized JSON/JSONL records.
- Search results now expose v0.3 score heads, missing requirements, negative constraint matches, evidence, and explanations while preserving existing score fields.
- `tests/test_v0_3_foundations.py` covers suite selection, evidence validation, JSONL roundtrip, fixture-backed normalizers, corpus reports, and negative-constraint score visibility.
- `neural_search/scientific_labels.py` implements conservative evidence-backed scientific label extraction across dataset and paper records.
- `neural_search/analysis_affordances.py` implements rule-based analysis affordance detection for normalized datasets.
- `neural_search/enrich_corpus.py` provides a backend CLI that enriches normalized records with labels and dataset affordances.
- `docs/SCIENTIFIC_LABEL_EXTRACTION_RULES.md` and `docs/ANALYSIS_AFFORDANCE_TAXONOMY.md` document the extraction and affordance rules.

Quality gate state from this checkpoint:

- `pytest -q`: 79 passed.
- `python -m neural_search.evaluation.run_benchmark --suite demo_v02`: 30 queries, 29/30 pass shape preserved, mean P@5 76.7%, label recall@10 87.8%.
- `python -m neural_search.evaluation.run_benchmark --suite real_corpus`: smoke suite runs and writes reports.
- `python -m neural_search.evaluation.run_benchmark --suite adversarial`: runs and intentionally exposes current hard-negative/scientific-fit failures.
- `python -m neural_search.reports.corpus_report --input data/corpus/normalized --out data/reports`: report generation succeeds.
- `ruff check .`: currently blocked by pre-existing generated notebook and migration lint outside this backend pass; scoped ruff over touched Python files passes.

Remaining must-do v0.3 work:

- Expand real-corpus normalization toward 100+ external records using saved raw payloads.
- Add reviewed, non-smoke relevance judgments for `real_corpus`.
- Expand adversarial benchmark toward 30+ scientifically documented queries.
- Make extraction rules natively produce `EvidenceLabel` objects across the full pipeline, not only additive normalization outputs.
- Add conservative analysis affordance detection and make `analysis_fit_score` consume it.
- Decide whether to migrate normalized records into database tables or keep them as a corpus artifact for v0.3 alpha.

---

## Phase 1: Freeze and Plan

### 1.1 Benchmark Versioning (WS2)

**Current State:**
- Single benchmark file: `data/eval/benchmark_queries.yaml`
- Results written to: `data/eval/results/`
- Runner: `neural_search/evaluation/run_benchmark.py`

**Required Changes:**

| Action | File | Description |
|--------|------|-------------|
| Rename | `data/eval/benchmark_queries.yaml` | Copy to `benchmark_queries_demo_v02.yaml` |
| Create | `data/eval/benchmark_queries_real_corpus.yaml` | Placeholder with schema-valid examples |
| Create | `data/eval/benchmark_queries_adversarial.yaml` | Hard-negative scientific queries |
| Modify | `neural_search/evaluation/benchmark.py` | Add `suite` parameter to `run_benchmark()` |
| Modify | `neural_search/evaluation/run_benchmark.py` | Add `--suite` CLI flag |
| Create | `data/eval/results/demo_v02/` | Suite-specific output directory |
| Create | `data/eval/results/real_corpus/` | Suite-specific output directory |
| Create | `data/eval/results/adversarial/` | Suite-specific output directory |
| Add | `tests/test_benchmark_suites.py` | Test suite selection and output paths |

**CLI Interface:**
```bash
python -m neural_search.evaluation.run_benchmark --suite demo_v02
python -m neural_search.evaluation.run_benchmark --suite real_corpus
python -m neural_search.evaluation.run_benchmark --suite adversarial
python -m neural_search.evaluation.run_benchmark --suite all
```

**Files to Leave Alone:**
- `neural_search/evaluation/benchmark.py` core logic (extend, don't rewrite)
- Existing benchmark query structure/schema

---

## Phase 2: Schema and Ingestion

### 2.1 Provenance-Aware Schema (WS3)

**Current State:**
- `neural_search/models.py` - SQLAlchemy ORM with basic metadata fields
- `neural_search/schemas.py` - Pydantic schemas (ExtractionResult, LabelEvidence)
- Labels stored as simple string lists without confidence/evidence

**Required Changes:**

| Action | File | Description |
|--------|------|-------------|
| Create | `neural_search/schemas/evidence.py` | EvidenceLabel, UsabilityFlags, AnalysisAffordance models |
| Create | `neural_search/schemas/normalized.py` | NormalizedDatasetRecord, NormalizedPaperRecord |
| Create | `neural_search/schemas/__init__.py` | Re-export all schema classes |
| Modify | `neural_search/schemas.py` | Import from new modules for backward compat |
| Create | `neural_search/utils/stable_id.py` | Deterministic UUID generation helpers |
| Create | `neural_search/utils/serialization.py` | JSON/JSONL serialization utilities |
| Add | `tests/test_evidence_schema.py` | Validation and roundtrip tests |

**New Schema Models:**

```python
# EvidenceLabel - provenance for every extracted label
class EvidenceLabel(BaseModel):
    id: str
    label: str
    label_type: str  # species, modality, brain_region, task, etc.
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: str | None = None
    source_field: str | None = None  # which metadata field
    source_value: str | None = None  # original value
    extractor_name: str = "ontology_matcher"
    extractor_version: str = "v0.3.0"

# NormalizedDatasetRecord - unified dataset representation
class NormalizedDatasetRecord(BaseModel):
    dataset_id: str
    source: Literal["dandi", "openneuro", "manual", "synthetic", "other"]
    source_id: str
    title: str
    description: str | None
    url: str | None
    raw_payload_path: str | None
    species: list[EvidenceLabel]
    modalities: list[EvidenceLabel]
    brain_regions: list[EvidenceLabel]
    tasks: list[EvidenceLabel]
    behavioral_events: list[EvidenceLabel]
    analysis_goals: list[EvidenceLabel]
    data_standards: list[EvidenceLabel]
    file_formats: list[EvidenceLabel]
    linked_papers: list[str]
    usability_flags: UsabilityFlags
    missing_fields: list[str]
    created_at: str
    extractor_version: str
```

**Files to Leave Alone:**
- `neural_search/models.py` ORM tables (extend, don't break existing DB)
- `neural_search/ontology/models.py` (keep LabelMatch for search layer)

---

### 2.2 Ingestion Hardening (WS4)

**Current State:**
- `neural_search/ingestion/dandi.py` - Basic DANDI fetch + normalize
- `neural_search/ingestion/openneuro.py` - GraphQL client
- `neural_search/ingestion/openalex.py` - REST API client
- `neural_search/ingestion/live.py` - DB persistence utilities
- Raw responses saved but not systematically

**Required Changes:**

| Action | File | Description |
|--------|------|-------------|
| Modify | `neural_search/ingestion/dandi.py` | Add `--dry-run`, `--save-raw`, `--limit`; output NormalizedDatasetRecord |
| Modify | `neural_search/ingestion/openneuro.py` | Same CLI flags; output NormalizedDatasetRecord |
| Modify | `neural_search/ingestion/openalex.py` | Same CLI flags; output NormalizedPaperRecord |
| Create | `neural_search/ingestion/normalizer.py` | Shared normalization logic with EvidenceLabel extraction |
| Create | `data/fixtures/dandi/` | Representative raw API payloads |
| Create | `data/fixtures/openneuro/` | Representative raw API payloads |
| Create | `data/fixtures/openalex/` | Representative raw API payloads |
| Modify | `neural_search/ingestion/live.py` | Deduplication by source/source_id |
| Add | `tests/test_ingestion_normalization.py` | Fixture-backed normalization tests |

**CLI Interface:**
```bash
python -m neural_search.ingestion.dandi --query "reversal learning" --limit 25 --dry-run --save-raw
python -m neural_search.ingestion.openneuro --query "motor imagery EEG" --limit 25 --dry-run --save-raw
python -m neural_search.ingestion.openalex --query "OFC reversal learning" --limit 50 --dry-run --save-raw
```

**Normalization Flow:**
1. Fetch raw payload from API
2. Save raw payload to `data/raw/{source}/{timestamp}_{id}.json`
3. Extract structured fields with EvidenceLabel provenance
4. Generate stable UUID from source + source_id
5. Validate against NormalizedDatasetRecord schema
6. Return or persist normalized record

**Files to Leave Alone:**
- Existing API client code (extend, don't rewrite)
- `data/seed/demo_datasets.yaml` (keep demo fixtures separate)

---

### 2.3 Label Extraction Rules (WS5)

**Current State:**
- `neural_search/ontology/matcher.py` - Returns LabelMatch with confidence
- `neural_search/ontology/loader.py` - Loads ontology YAML
- Confidence tiers exist but not consistently applied

**Required Changes:**

| Action | File | Description |
|--------|------|-------------|
| Modify | `neural_search/ontology/matcher.py` | Return EvidenceLabel instead of LabelMatch |
| Create | `neural_search/extraction/rules.py` | Confidence tier logic with documentation |
| Create | `neural_search/extraction/false_positives.py` | Known false positive patterns |
| Modify | `data/ontology/behavioral_task_ontology.yaml` | Add confidence_base to terms where appropriate |
| Add | `tests/test_extraction_confidence.py` | Confidence tier and false positive tests |

**Confidence Tiers:**
| Tier | Confidence | Criteria |
|------|------------|----------|
| exact_metadata_match | 0.90-1.00 | Label in structured field or controlled vocabulary |
| strong_synonym_match | 0.75-0.90 | Curated synonym or common abbreviation |
| description_phrase_match | 0.60-0.80 | Found in free-text description/abstract |
| weak_inference | 0.35-0.60 | Indirect inference, must expose uncertainty |
| unsupported | 0.00-0.30 | Do not include unless debug output |

**Guardrails:**
- Do not infer behavioral tasks from brain region alone
- Do not infer analysis support without required metadata fields
- Always include evidence_text and source_field

---

## Phase 3: Reports and Scoring

### 3.1 Corpus QA Reports (WS6)

**Current State:**
- `neural_search/reports/dataset_compilation.py` - Basic compilation report
- No corpus-wide QA/coverage reports

**Required Changes:**

| Action | File | Description |
|--------|------|-------------|
| Create | `neural_search/reports/corpus_report.py` | Main report generator |
| Create | `neural_search/reports/coverage.py` | Modality/species/task/region coverage |
| Create | `neural_search/reports/missing_metadata.py` | Missing field analysis by source |
| Create | `neural_search/reports/label_confidence.py` | Low-confidence label listing |
| Create | `neural_search/reports/linking.py` | Dataset-paper link candidates |
| Create | `data/reports/corpus_coverage_report.md` | Generated output |
| Create | `data/reports/missing_metadata_report.md` | Generated output |
| Create | `data/reports/label_confidence_report.md` | Generated output |
| Create | `data/reports/low_confidence_labels.json` | Machine-readable output |
| Add | `tests/test_corpus_reports.py` | Fixture-backed report generation |

**CLI Interface:**
```bash
python -m neural_search.reports.corpus_report --input data/corpus/normalized --out data/reports
```

**Report Contents:**
1. **Coverage Report:** Records by source, modality distribution, species, tasks, regions
2. **Missing Metadata Report:** Fields missing by source (%, examples)
3. **Label Confidence Report:** Distribution of confidence tiers, low-confidence samples
4. **Source Distribution Report:** Records per source, overlap analysis
5. **Linking Report:** Dataset-paper link candidates with evidence

---

### 3.2 Interpretable Multi-Head Scoring (WS7)

**Current State:**
- `neural_search/search/core.py` - Combined scoring with weights from `data/config/retrieval.yaml`
- Score components computed but not exposed in results

**Required Changes:**

| Action | File | Description |
|--------|------|-------------|
| Create | `neural_search/search/scoring.py` | ScoreBreakdown model and scoring head functions |
| Modify | `neural_search/search/core.py` | Use ScoreBreakdown in SearchResult |
| Modify | `neural_search/schemas.py` | Add ScoreBreakdown to SearchResult |
| Modify | `data/config/retrieval.yaml` | Add default weights for all heads |
| Add | `tests/test_score_decomposition.py` | Tests for each scoring head |

**Score Heads:**
| Head | Description |
|------|-------------|
| lexical_score | BM25/keyword match on title, description |
| ontology_score | Task, behavior, modality, region overlap |
| semantic_score | Embedding similarity |
| provenance_score | Source quality, evidence confidence, linked papers |
| usability_score | Trials, events, neural data, standard formats |
| analysis_fit_score | Whether dataset supports requested analysis |
| negative_constraint_score | Penalty for excluded modalities/species/tasks |
| final_score | Weighted combination |

**Result Shape:**
```python
class ScoreBreakdown(BaseModel):
    lexical_score: float
    ontology_score: float
    semantic_score: float
    provenance_score: float
    usability_score: float
    analysis_fit_score: float
    negative_constraint_score: float
    weights_used: dict[str, float]

class SearchResult(BaseModel):
    result_id: str
    title: str
    source: str
    final_score: float
    score_breakdown: ScoreBreakdown
    matched_labels: list[str]
    missing_requirements: list[str]
    negative_constraint_matches: list[str]
    evidence: list[str]
    explanation: str
```

---

## Phase 4: Scientific Evaluation

### 4.1 Adversarial Benchmark Design (WS8)

**Required Changes:**

| Action | File | Description |
|--------|------|-------------|
| Create | `data/eval/benchmark_queries_adversarial.yaml` | 30+ adversarial queries |
| Create | `docs/ADVERSARIAL_BENCHMARK_RATIONALE.md` | Scientific rationale per query |
| Modify | `neural_search/evaluation/benchmark.py` | Add exclusion correctness metrics |
| Modify | `neural_search/evaluation/benchmark.py` | Add missingness awareness metrics |

**Query Categories:**
1. direct_metadata_lookup
2. task_based_search
3. modality_region_species_search
4. analysis_goal_search
5. hard_negative_search
6. paper_to_dataset_linking
7. dataset_to_paper_linking
8. ambiguous_natural_language_search
9. missing_metadata_awareness

**Example Adversarial Queries:**
```yaml
- id: adv_001
  query: "Find OFC reversal learning datasets, but exclude fMRI and pure behavior-only studies"
  expected_positives:
    - datasets with OFC ephys/calcium + reversal task
  expected_negatives:
    - fMRI studies
    - behavior-only studies (no neural recording)
  rationale: "Tests exclusion parsing + modality filtering"

- id: adv_002
  query: "Find datasets suitable for fitting Q-learning models where choices, rewards, and trial outcomes are explicitly available"
  required_fields: [has_trials, choices, rewards, outcomes]
  rationale: "Tests analysis affordance detection"
```

**New Metrics:**
- `exclusion_correctness`: Were excluded modalities/species correctly filtered?
- `missingness_awareness`: Did results expose missing required fields?

---

### 4.2 Analysis Affordance Detection (WS9)

**Required Changes:**

| Action | File | Description |
|--------|------|-------------|
| Create | `neural_search/analysis/affordance.py` | AnalysisAffordance model and detector |
| Create | `neural_search/analysis/rules.py` | Rule-based affordance detection |
| Modify | `neural_search/search/core.py` | Integrate affordance into analysis_fit_score |
| Add | `tests/test_analysis_affordance.py` | High/medium/low support cases |

**Affordance Model:**
```python
class AnalysisAffordance(BaseModel):
    analysis_id: str  # e.g., "q_learning_modeling"
    support_level: Literal["high", "medium", "low", "unsupported", "unknown"]
    confidence: float
    required_fields_present: list[str]
    missing_fields: list[str]
    evidence: list[str]
```

**Supported Analyses:**
- event_aligned_activity
- trial_averaged_response
- choice_decoding
- motor_decoding
- q_learning_modeling
- state_space_modeling
- speech_decoding
- functional_connectivity
- sleep_stage_classification
- seizure_detection

---

## Phase 5: Latent Foundation

### 5.1 Latent Search Roadmap (WS10)

**Current State:**
- `neural_search/latent/search.py` - Prototype similarity search
- `neural_search/latent/embedding_schema.py` - FeatureType enum and models
- `neural_search/latent/summary_features.py` - Deterministic feature extraction

**Required Changes:**

| Action | File | Description |
|--------|------|-------------|
| Create | `docs/LATENT_SEARCH_V0_3_ROADMAP.md` | Roadmap document |
| Create | `neural_search/latent/fingerprint.py` | Deterministic session/dataset fingerprints |
| Modify | `neural_search/latent/summary_features.py` | Add NWB/BIDS structural features |
| Add | `tests/test_latent_fingerprints.py` | Fingerprint determinism tests |

**Fingerprint Features (v0.3):**
- number_of_units
- number_of_trials
- event_type_names
- sampling_rates
- brain_regions
- electrode_groups
- behavioral_timeseries_present
- trial_interval_tables_present
- modality_composition
- file_format_composition
- events_tsv_columns (BIDS)
- subject_count
- run_count

**v0.3 Scope:**
- Deterministic fingerprints only
- No learned embeddings
- Interface prepared for v0.4+ model integration

---

## Files to Modify (Summary)

| File | Changes |
|------|---------|
| `neural_search/evaluation/benchmark.py` | Suite selection, new metrics |
| `neural_search/evaluation/run_benchmark.py` | `--suite` CLI flag |
| `neural_search/ingestion/dandi.py` | `--dry-run`, `--save-raw`, normalized output |
| `neural_search/ingestion/openneuro.py` | Same flags, normalized output |
| `neural_search/ingestion/openalex.py` | Same flags, normalized output |
| `neural_search/ingestion/live.py` | Deduplication logic |
| `neural_search/ontology/matcher.py` | Return EvidenceLabel |
| `neural_search/search/core.py` | Score decomposition, affordance integration |
| `neural_search/schemas.py` | Import new models, add ScoreBreakdown |
| `data/config/retrieval.yaml` | Add all score head weights |

## Files to Create (Summary)

| File | Purpose |
|------|---------|
| `neural_search/schemas/evidence.py` | EvidenceLabel, UsabilityFlags, AnalysisAffordance |
| `neural_search/schemas/normalized.py` | NormalizedDatasetRecord, NormalizedPaperRecord |
| `neural_search/search/scoring.py` | ScoreBreakdown and scoring head functions |
| `neural_search/ingestion/normalizer.py` | Shared normalization with provenance |
| `neural_search/extraction/rules.py` | Confidence tier logic |
| `neural_search/reports/corpus_report.py` | Corpus QA report generator |
| `neural_search/analysis/affordance.py` | Analysis affordance detection |
| `neural_search/latent/fingerprint.py` | Deterministic fingerprints |
| `data/eval/benchmark_queries_demo_v02.yaml` | Frozen v0.2 benchmark |
| `data/eval/benchmark_queries_real_corpus.yaml` | Real corpus benchmark |
| `data/eval/benchmark_queries_adversarial.yaml` | Adversarial benchmark |
| `data/fixtures/{dandi,openneuro,openalex}/` | API fixture payloads |

## Files to Leave Alone

| File | Reason |
|------|--------|
| `neural_search/models.py` | ORM stability, extend only |
| `neural_search/cards/generator.py` | Card generation working |
| `neural_search/notebooks/generator.py` | Notebook generation working |
| `apps/web/*` | No frontend work in v0.3 |
| `data/seed/demo_datasets.yaml` | Keep demo corpus separate |
| `data/ontology/behavioral_task_ontology.yaml` | Extend conservatively |

---

## Quality Gates

```bash
# Install dev dependencies
python -m pip install -e ".[dev]"

# Run tests
pytest -q

# Lint check
ruff check .

# Run benchmarks (all suites)
python -m neural_search.evaluation.run_benchmark --suite demo_v02
python -m neural_search.evaluation.run_benchmark --suite real_corpus
python -m neural_search.evaluation.run_benchmark --suite adversarial

# Generate corpus reports
python -m neural_search.reports.corpus_report --input data/corpus/normalized --out data/reports
```

---

## Success Criteria

1. **v0.2 regression suite passes** (29/30 or better)
2. **100+ real records normalized** from DANDI/OpenNeuro/OpenAlex (dry-run or persisted)
3. **All labels include provenance** (confidence, evidence, extractor)
4. **Score decomposition visible** in search results
5. **Adversarial benchmark exists** (30+ queries, failure modes documented)
6. **Corpus QA reports generated** for any normalized corpus
7. **All new code has tests**
8. **Quality gates pass locally**

---

## Timeline and Phases

| Phase | Workstreams | Deliverables |
|-------|-------------|--------------|
| 1. Freeze and Plan | WS1, WS2 | Implementation plan, versioned benchmarks |
| 2. Schema and Ingestion | WS3, WS4, WS5 | Provenance schema, hardened ingestion, extraction rules |
| 3. Reports and Scoring | WS6, WS7 | Corpus reports, score decomposition |
| 4. Scientific Evaluation | WS8, WS9 | Adversarial benchmark, affordance detection |
| 5. Latent Foundation | WS10 | Latent search roadmap, deterministic fingerprints |

---

## Deferred to v0.4/v0.5

- BM25 retrieval integration
- SentenceTransformer/SPECTER embeddings
- Learned latent embeddings
- Manual relevance labels
- NWB/BIDS file inspection
- Session-level fingerprints
- Retrieval ablation reports
