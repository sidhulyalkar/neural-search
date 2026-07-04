# Core Search Audit - Neural Search v0.2

Generated: 2026-05-23

## 1. Current Corpus Files and Schema

### Corpus Location
- **Seed datasets**: `data/seed/demo_datasets.yaml` (5 datasets)
- **Linked papers**: `data/seed/demo_papers.yaml` (5 papers)
- **Ontology**: `data/ontology/behavioral_task_ontology.yaml`
- **Raw ingested**: `data/raw/{dandi,openneuro,openalex}/`

### Dataset Schema (from `neural_search/schemas.py`)
```yaml
source: str           # demo, dandi, openneuro
source_id: str        # unique ID
title: str
description: str
url: str (optional)
species: list[str]
modalities: list[str]
brain_regions: list[str]
tasks: list[str]
behaviors: list[str]
data_standards: list[str]
has_behavior: bool
has_trials: bool
license: str
linked_paper_ids: list[str]
```

### Current Demo Datasets (5 total)
| ID | Task | Modality | Species | Regions |
|----|------|----------|---------|---------|
| DEMO_GONOGO_CALCIUM | go_nogo | calcium_imaging, behavior_video | mouse | mPFC |
| DEMO_REVERSAL_EPHYS | reversal_learning | extracellular_ephys | mouse | OFC, striatum |
| DEMO_DELAY_DISCOUNTING | delay_discounting | fiber_photometry, behavior_video | rat | mPFC, nucleus_accumbens |
| DEMO_REACHING_ECOG_IEEG | reaching, grasping | ecog, ieeg, pose_tracking, bci | human | motor_cortex, parietal_cortex |
| DEMO_VISUAL_DECISION_NEUROPIXELS | visual_decision_making | neuropixels, extracellular_ephys | mouse | visual_cortex, PPC |

---

## 2. Current Search Pipeline

### Location: `neural_search/search/core.py`

### Pipeline Steps:
1. **Query Parsing** (`parse_query()`)
   - Match tasks via ontology fuzzy matching
   - Match behaviors via ontology
   - Expand modalities from generic phrases ("neural recordings" → 7 modalities)
   - Detect species aliases
   - Extract brain region mentions

2. **Dataset Scoring** (`score_dataset_against_query()`)
   - Multi-signal weighted scoring (0-100)
   - Configurable weights from `data/config/retrieval.yaml`

3. **Filter Application** (`_passes_filters()`)
   - Min readiness score
   - Reviewed/trusted only
   - Key-value filters

4. **Result Sorting**
   - Top-K by score descending
   - Includes explanations

---

## 3. Current Query Parsing Behavior

### Parsing Functions:
- `match_tasks()` - fuzzy/exact match against task labels and synonyms
- `match_behavior_labels()` - exact/fuzzy match against behavior ontology
- `match_brain_regions()` - alias-based matching
- `match_modalities()` - generic phrase expansion

### Query Expansion:
- "neural recordings" → [extracellular_ephys, calcium_imaging, neuropixels, fiber_photometry, ecog, ieeg, eeg]
- "electrophysiology" → [extracellular_ephys, neuropixels, ecog, ieeg, eeg]

### Normalization:
- Casefold + replace `[/_-]+` with space
- Remove non-alphanumeric
- Collapse whitespace

---

## 4. Current Scoring Behavior

### Weight Configuration (default from retrieval.yaml):
| Signal | Weight |
|--------|--------|
| Ontology (task) | 30% |
| Behavior | 22% |
| Modality | 14% |
| Metadata (species/region/analysis) | 10% |
| Semantic similarity | 10% |
| Analysis readiness | 10% |
| Paper confidence | 4% |

### Penalties:
- Modality mismatch: -18%
- Missing required field: -4% per field (max 5)

### Score Formula:
```
final_score = Σ(weight[signal] × score[signal]) - penalties
clamped to [0, 1], then × 100
```

---

## 5. Current Benchmark Format

### Location: `data/eval/benchmark_queries.yaml`

### Query Schema:
```yaml
- id: q001
  query: "Free-text query"
  expected_tasks: [task_id]        # All must match
  expected_modalities_any: [mod]   # Any can match (OR)
  expected_regions_any: [region]
  expected_species: [species]
  expected_data_standards: [std]
  expected_sources: [source]
  expected_analysis_any: [analysis]
  hard_negative_modalities: [bad]  # Must NOT be in results
  minimum_precision_at_5: 0.6
  minimum_label_recall_at_10: 0.5
```

### Pass Criteria:
- Precision@5 ≥ 40%
- Label Recall@10 ≥ 50%

---

## 6. Current Failure Analysis (17 Failing Queries)

### Baseline Metrics:
- **Total queries**: 30
- **Passed**: 13 (43%)
- **Failed**: 17 (57%)
- **Mean P@5**: 53.3%
- **Mean Label Recall@10**: 53.0%

### Failure Categories:

#### A. Missing Corpus Coverage (10 queries)
These queries ask for datasets that simply don't exist in the corpus:

| Query | Missing Content |
|-------|-----------------|
| q007 | naturalistic_vision, pupil_arousal tasks |
| q008 | motor_imagery task, EEG modality |
| q014 | spatial_navigation, hippocampus, place_cells |
| q018 | fMRI modality, stroop/flanker tasks |
| q021 | BIDS/OpenNeuro source, fMRI, EEG |
| q023 | trial_averaging, psth analysis goals |
| q024 | BCI/closed-loop tasks |
| q027 | non-human primate species |
| q028 | speech/language tasks |

#### B. Weak Synonym/Normalization (4 queries)
These queries have datasets that could match but synonyms aren't resolving:

| Query | Issue |
|-------|-------|
| q012 | "mPFC" in dataset but "medial prefrontal cortex" not normalized |
| q013 | "striatum" exists but "dorsal_striatum", "ventral_striatum" not recognized as children |
| q015 | "motor_cortex" exists but "M1", "primary_motor_cortex" not linked |
| q017 | "calcium_imaging" exists but "two_photon" not a synonym |

#### C. Missing Region/Modality Fields (2 queries)
| Query | Issue |
|-------|-------|
| q009 | seizure_monitoring task missing, no EEG dataset |
| q019 | deeplabcut/sleap not in modality list, pose_estimation analysis missing |

#### D. Scoring/Ranking Issues (1 query)
| Query | Issue |
|-------|-------|
| q004 | P@5=20% despite top result matching - only 1 result matches so other 4 slots filled with weak matches |

---

## 7. Root Cause Analysis

### Primary Causes:

1. **Corpus Too Small** (50% of failures)
   - Only 5 demo datasets
   - Missing entire task categories (motor_imagery, spatial_navigation, speech, seizure)
   - Missing modalities (fMRI, standalone EEG)
   - Missing species (macaque, non-human primate)

2. **Weak Synonym Coverage** (25% of failures)
   - Brain regions not normalized (mPFC ↔ medial prefrontal cortex)
   - Modality variants not linked (two_photon ↔ calcium_imaging)
   - Hierarchical relationships not modeled (striatum → dorsal_striatum, ventral_striatum)

3. **Ontology Gaps** (15% of failures)
   - Tasks not defined: motor_imagery, spatial_navigation, speech, seizure_monitoring
   - Analysis goals not populated: psth, trial_averaging, motor_decoding

4. **No Latent Model Yet** (10% of failures)
   - Analysis-goal queries can't be matched semantically
   - No feature-based similarity

### Not Primary Causes:
- Scoring weights appear reasonable
- Hard negative logic exists but rarely triggered (corpus too small)
- Embedding fallback provides some semantic coverage

---

## 8. Prioritized Fix List

### High Priority (Corpus Expansion)
1. Add 15-20 demo datasets covering missing task categories
2. Add datasets with fMRI, EEG, hippocampus, primate species
3. Add datasets from different sources (DANDI, OpenNeuro references)

### High Priority (Ontology)
4. Add synonym mappings for brain regions (mPFC ↔ medial_prefrontal_cortex ↔ prelimbic)
5. Add hierarchical region relationships (striatum parent of dorsal/ventral_striatum)
6. Add modality synonyms (two_photon ↔ 2p ↔ calcium_imaging variant)
7. Add missing task definitions (motor_imagery, spatial_navigation, speech, seizure_monitoring)

### Medium Priority (Normalization)
8. Improve query parser to recognize region aliases
9. Add species aliases (macaque ↔ rhesus ↔ non_human_primate)
10. Add analysis goal ontology with synonyms

### Lower Priority (Scoring)
11. Increase ontology match weight vs semantic
12. Add explicit region/modality match signals
13. Improve hard-negative penalty application

### Future (Latent Search)
14. Build feature extraction from synthetic data
15. Build embedding-based analysis goal matching

---

## 9. Files to Modify

### Corpus:
- `data/seed/demo_datasets.yaml` → expand to 20+ datasets
- Create `data/corpus/` with better organization

### Ontology:
- `data/ontology/behavioral_task_ontology.yaml` → add missing tasks
- Create `data/ontology/brain_regions.yaml` → region hierarchy
- Create `data/ontology/modalities.yaml` → modality synonyms
- Create `data/ontology/species.yaml` → species hierarchy
- Create `data/ontology/analysis_goals.yaml` → analysis goal ontology

### Search:
- `neural_search/ontology/matcher.py` → improve region/modality matching
- `neural_search/search/core.py` → add region/species signals

### Config:
- `data/config/retrieval.yaml` → tune weights after corpus expansion
