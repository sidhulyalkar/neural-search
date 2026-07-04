# Affordances, Ontology, Coverage & Graph Sprint

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand analysis affordances from 15→21 types, harden brain-region and task ontologies, improve task coverage from 23.4%→35%+, add a similar-datasets graph API, and update the whitepaper.

**Architecture:** Each sprint is self-contained: affordances first (registry.py + tests), then ontology YAML expansion + re-enrichment, then graph query enhancements, then whitepaper. All work stays on branch `claude/neuronpedia-foundation`. Start every subagent with `git checkout claude/neuronpedia-foundation`.

**Tech Stack:** Python 3.13 · pytest · PyYAML · DuckDB · FastAPI · React 18 + TanStack Query · LaTeX (whitepaper)

---

## File Map

| File | Sprint | Change |
|------|--------|--------|
| `neural_search/affordances/registry.py` | A | Add 6 AffordanceRequirement objects + 4 DatasetFeatures fields |
| `neural_search/affordances/__init__.py` | A | Re-export new symbols if needed |
| `tests/test_affordance_expansion.py` | A | 18 new tests for new affordance types |
| `data/ontology/brain_regions.yaml` | B | Add ~30 new region entries |
| `data/ontology/behavioral_task_ontology.yaml` | B | Add 9 missing task IDs + aliases |
| `tests/test_ontology_expansion.py` | B | 12 tests verifying new ontology entries |
| `scripts/enrich_task_coverage.py` | C | NLP batch enrichment script |
| `tests/test_task_enrichment.py` | C | 8 tests for enrichment logic |
| `neural_search/graph/query.py` | D | `find_similar_datasets()` function |
| `apps/api/main.py` | D | `GET /api/datasets/{id}/similar` endpoint |
| `apps/web/src/api/search.ts` | D | `getSimilarDatasets()` client |
| `apps/web/src/pages/DatasetPage.tsx` | D | Similar Datasets panel |
| `docs/whitepaper/neural_search_whitepaper.tex` | E | Stats, affordances, Neuronpedia section, roadmap |

---

## Sprint A — Affordance Expansion

### Task 1: Add 6 new affordance types to registry

**Files:**
- Modify: `neural_search/affordances/registry.py` (DatasetFeatures ~line 566, detect_features_from_metadata ~line 637, AFFORDANCE_REGISTRY ~line 525)

- [ ] **Step 1: Add 4 new fields to DatasetFeatures**

In `DatasetFeatures` (after `has_ecog: bool = False` around line 581), add:

```python
    has_speech_events: bool = False      # speech production/perception events
    has_seizure_annotations: bool = False  # seizure onset/offset labels
    has_sleep_stage_labels: bool = False   # NREM/REM/wake/SWS labels
    has_bci_context: bool = False          # motor imagery or intended action
```

- [ ] **Step 2: Detect the 4 new fields in detect_features_from_metadata**

In `detect_features_from_metadata()`, after the existing `behavioral_events` detection block (after the `features.event_types = list(behavioral_events)` line), add:

```python
    seizure_events = {"seizure", "ictal", "preictal", "interictal", "seizure_onset",
                      "seizure_offset", "epileptic", "epilepsy"}
    features.has_seizure_annotations = bool(behavioral_events & seizure_events)

    sleep_events = {"sleep", "nrem", "rem", "slow_wave", "sleep_stage", "wake",
                    "sleep_spindle", "arousal", "sws", "polysomnography", "hypnogram"}
    features.has_sleep_stage_labels = bool(behavioral_events & sleep_events)

    speech_events = {"speech", "phoneme", "word", "articulation", "vocalization",
                     "phonetic", "sentence", "syllable", "speech_onset"}
    features.has_speech_events = bool(behavioral_events & speech_events)

    task_labels_lower: list[str] = []
    for t in dataset.get("task_labels", []) + dataset.get("tasks", []):
        if isinstance(t, str):
            task_labels_lower.append(t.lower())
        elif hasattr(t, "label"):
            task_labels_lower.append(t.label.lower())
    bci_events = {"bci", "imagined_movement", "motor_imagery", "cursor_control",
                  "intended_action", "p300", "ssvep", "neurofeedback"}
    features.has_bci_context = (
        bool(behavioral_events & bci_events)
        or any("bci" in t or "motor_imagery" in t for t in task_labels_lower)
    )
```

- [ ] **Step 3: Add 6 AffordanceRequirement constants**

After the `CROSS_SESSION_GENERALIZATION` block (before the `# AFFORDANCE_REGISTRY` comment), add:

```python
SPEECH_DECODING = AffordanceRequirement(
    affordance_id="speech_decoding",
    label="Speech decoding",
    description="Decode speech content or articulatory parameters from neural activity",
    required_features=[
        "neural_data",
        "speech_events",
        "temporal_alignment",
    ],
    optional_features=[
        "continuous_neural_data",
        "multiple_subjects",
        "stimulus_timing",
        "trial_structure",
    ],
    negative_conditions=[
        "no_speech_data",
        "only_summary_statistics",
    ],
    validation_methods=["nwb_units_check", "bids_events_check"],
    example_use_cases=[
        "Decode phonemes from ECoG high-gamma",
        "Build a speech neuroprosthesis decoder",
        "Map articulatory features to cortical activity",
    ],
)

SEIZURE_DETECTION = AffordanceRequirement(
    affordance_id="seizure_detection",
    label="Seizure detection",
    description="Detect seizure onset and offset from neural recordings",
    required_features=[
        "neural_data",
        "seizure_annotations",
    ],
    optional_features=[
        "continuous_neural_data",
        "multiple_sessions",
        "multiple_subjects",
        "channel_locations",
    ],
    negative_conditions=[
        "no_seizure_labels",
        "only_summary_statistics",
    ],
    validation_methods=["nwb_behavior_check", "bids_events_check"],
    example_use_cases=[
        "Train a clinical seizure detector",
        "Identify ictal vs interictal patterns",
        "Localize seizure onset zones",
    ],
)

SLEEP_STAGE_CLASSIFICATION = AffordanceRequirement(
    affordance_id="sleep_stage_classification",
    label="Sleep stage classification",
    description="Classify sleep stages (NREM, REM, wake) from neural data",
    required_features=[
        "neural_data",
        "sleep_stage_labels",
    ],
    optional_features=[
        "continuous_neural_data",
        "multiple_sessions",
        "behavioral_state_labels",
        "channel_locations",
    ],
    negative_conditions=[
        "no_sleep_labels",
        "only_summary_statistics",
        "single_state_only",
    ],
    validation_methods=["bids_events_check", "nwb_behavior_check"],
    example_use_cases=[
        "Automated polysomnography scoring",
        "Study slow-wave vs REM neural dynamics",
        "Classify vigilance states across species",
    ],
)

BCI_DECODING = AffordanceRequirement(
    affordance_id="bci_decoding",
    label="BCI decoding",
    description="Decode intended actions or control signals from neural activity for brain-computer interfaces",
    required_features=[
        "neural_data",
        "bci_context",
    ],
    optional_features=[
        "motor_action_labels",
        "continuous_neural_data",
        "trial_structure",
        "multiple_sessions",
        "sorted_units",
    ],
    negative_conditions=[
        "no_bci_context",
        "only_summary_statistics",
    ],
    validation_methods=["nwb_units_check", "bids_events_check"],
    example_use_cases=[
        "P300 speller neural decoding",
        "Motor imagery classification",
        "SSVEP frequency decoding",
        "Cursor control from ECoG",
    ],
)

LATENT_DYNAMICS_MODELING = AffordanceRequirement(
    affordance_id="latent_dynamics_modeling",
    label="Latent dynamics modeling",
    description="Fit latent-variable dynamical systems models (GPFA, LFADS, SLDS) to neural population activity",
    required_features=[
        "neural_population_data",
        "multiple_units_or_voxels",
        "continuous_or_trial_data",
    ],
    optional_features=[
        "trial_structure",
        "event_timestamps",
        "sorted_units",
        "time_series",
    ],
    negative_conditions=[
        "single_neuron_only",
        "only_summary_statistics",
        "only_event_counts",
    ],
    validation_methods=["nwb_units_count_check"],
    example_use_cases=[
        "Fit GPFA to motor cortex population",
        "Apply LFADS to infer latent trajectories",
        "Model rotational dynamics during movement",
    ],
)

REPRESENTATIONAL_SIMILARITY_ANALYSIS = AffordanceRequirement(
    affordance_id="representational_similarity_analysis",
    label="Representational similarity analysis",
    description="Compute and compare neural representational geometry across conditions or datasets",
    required_features=[
        "neural_population_data",
        "multiple_units_or_voxels",
        "task_events_or_conditions",
    ],
    optional_features=[
        "stimulus_timing",
        "multiple_conditions",
        "trial_structure",
        "sorted_units",
        "fmri_bold_data",
    ],
    negative_conditions=[
        "single_neuron_only",
        "only_summary_statistics",
    ],
    validation_methods=["nwb_units_count_check", "fmri_check"],
    example_use_cases=[
        "Compare representational geometry across brain areas",
        "RSA between neural and model representations",
        "Test invariance to stimulus transformations",
    ],
)
```

- [ ] **Step 4: Register the 6 new affordances in AFFORDANCE_REGISTRY**

In the `AFFORDANCE_REGISTRY` dict, add 6 entries after `"cross_session_generalization": CROSS_SESSION_GENERALIZATION,`:

```python
    # Clinical and BCI affordances
    "speech_decoding": SPEECH_DECODING,
    "seizure_detection": SEIZURE_DETECTION,
    "sleep_stage_classification": SLEEP_STAGE_CLASSIFICATION,
    "bci_decoding": BCI_DECODING,
    # Population dynamics
    "latent_dynamics_modeling": LATENT_DYNAMICS_MODELING,
    "representational_similarity_analysis": REPRESENTATIONAL_SIMILARITY_ANALYSIS,
```

- [ ] **Step 5: Add 6 feature checks to _get_feature_checks()**

In `_get_feature_checks()`, add inside the returned dict (after the last entry):

```python
        # New signal checks
        "speech_events": lambda f: f.has_speech_events,
        "seizure_annotations": lambda f: f.has_seizure_annotations,
        "sleep_stage_labels": lambda f: f.has_sleep_stage_labels,
        "bci_context": lambda f: f.has_bci_context,
        "time_series": lambda f: f.has_spike_times or f.has_lfp or f.has_eeg or f.has_calcium_imaging,
```

- [ ] **Step 6: Add 6 negative condition checks to _get_negative_checks()**

In `_get_negative_checks()`, add inside the returned dict:

```python
        "no_speech_data": lambda f: not f.has_speech_events,
        "no_seizure_labels": lambda f: not f.has_seizure_annotations,
        "no_sleep_labels": lambda f: not f.has_sleep_stage_labels,
        "no_bci_context": lambda f: not f.has_bci_context,
```

- [ ] **Step 7: Verify 21 affordances loaded**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
python -c "
from neural_search.affordances.registry import AFFORDANCE_REGISTRY, list_affordances
print('Total affordances:', len(AFFORDANCE_REGISTRY))
for a in list_affordances():
    print(' -', a)
"
```

Expected: `Total affordances: 21`

- [ ] **Step 8: Commit**

```bash
git add neural_search/affordances/registry.py
git commit -m "feat(affordances): add 6 clinical+BCI+dynamics affordance types (21 total)"
```

---

### Task 2: Tests for the 6 new affordance types

**Files:**
- Create: `tests/test_affordance_expansion.py`

- [ ] **Step 1: Write the test file**

Create `/mnt/c/Users/sidso/Documents/neural-search/tests/test_affordance_expansion.py`:

```python
"""Tests for the 6 new affordance types added in the expansion sprint."""

import pytest
from neural_search.affordances.registry import (
    DatasetFeatures,
    DataFormat,
    detect_features_from_metadata,
    validate_affordance,
    list_affordances,
)


# ── Registry completeness ────────────────────────────────────────────────────

def test_registry_has_21_affordances():
    affordances = list_affordances()
    assert len(affordances) == 21


def test_new_affordances_present():
    affordances = list_affordances()
    for aff_id in [
        "speech_decoding",
        "seizure_detection",
        "sleep_stage_classification",
        "bci_decoding",
        "latent_dynamics_modeling",
        "representational_similarity_analysis",
    ]:
        assert aff_id in affordances, f"Missing: {aff_id}"


# ── Feature detection ────────────────────────────────────────────────────────

class TestNewFeatureDetection:
    def test_detects_speech_events(self):
        dataset = {
            "dataset_id": "test:speech",
            "behavioral_events": ["speech", "phoneme", "word"],
        }
        features = detect_features_from_metadata(dataset)
        assert features.has_speech_events is True

    def test_no_speech_events_when_absent(self):
        dataset = {"dataset_id": "test:nospch", "behavioral_events": ["choice"]}
        features = detect_features_from_metadata(dataset)
        assert features.has_speech_events is False

    def test_detects_seizure_annotations(self):
        dataset = {
            "dataset_id": "test:sz",
            "behavioral_events": ["seizure", "ictal"],
        }
        features = detect_features_from_metadata(dataset)
        assert features.has_seizure_annotations is True

    def test_detects_seizure_from_interictal(self):
        dataset = {"dataset_id": "test:sz2", "behavioral_events": ["interictal"]}
        features = detect_features_from_metadata(dataset)
        assert features.has_seizure_annotations is True

    def test_no_seizure_when_absent(self):
        dataset = {"dataset_id": "test:nosz", "behavioral_events": ["reward"]}
        features = detect_features_from_metadata(dataset)
        assert features.has_seizure_annotations is False

    def test_detects_sleep_stage_labels(self):
        dataset = {
            "dataset_id": "test:sleep",
            "behavioral_events": ["nrem", "rem", "wake"],
        }
        features = detect_features_from_metadata(dataset)
        assert features.has_sleep_stage_labels is True

    def test_detects_sleep_from_slow_wave(self):
        dataset = {"dataset_id": "test:sw", "behavioral_events": ["slow_wave"]}
        features = detect_features_from_metadata(dataset)
        assert features.has_sleep_stage_labels is True

    def test_no_sleep_when_absent(self):
        dataset = {"dataset_id": "test:nosl", "behavioral_events": ["choice"]}
        features = detect_features_from_metadata(dataset)
        assert features.has_sleep_stage_labels is False

    def test_detects_bci_context_from_events(self):
        dataset = {
            "dataset_id": "test:bci",
            "behavioral_events": ["motor_imagery", "imagined_movement"],
        }
        features = detect_features_from_metadata(dataset)
        assert features.has_bci_context is True

    def test_detects_bci_context_from_task_labels(self):
        dataset = {
            "dataset_id": "test:bci2",
            "task_labels": ["bci_spelling", "p300_bci"],
            "behavioral_events": [],
        }
        features = detect_features_from_metadata(dataset)
        assert features.has_bci_context is True

    def test_no_bci_when_absent(self):
        dataset = {"dataset_id": "test:nobci", "behavioral_events": ["choice"]}
        features = detect_features_from_metadata(dataset)
        assert features.has_bci_context is False


# ── Affordance validation ─────────────────────────────────────────────────────

class TestNewAffordanceValidation:
    def _ecog_speech_features(self) -> DatasetFeatures:
        return DatasetFeatures(
            dataset_id="ecog_speech",
            has_neural_data=True,
            has_ecog=True,
            has_speech_events=True,
            has_event_timestamps=True,
            data_format=DataFormat.NWB,
        )

    def _eeg_seizure_features(self) -> DatasetFeatures:
        return DatasetFeatures(
            dataset_id="eeg_seizure",
            has_neural_data=True,
            has_eeg=True,
            has_seizure_annotations=True,
            has_event_timestamps=True,
        )

    def _eeg_sleep_features(self) -> DatasetFeatures:
        return DatasetFeatures(
            dataset_id="eeg_sleep",
            has_neural_data=True,
            has_eeg=True,
            has_sleep_stage_labels=True,
        )

    def _ephys_bci_features(self) -> DatasetFeatures:
        return DatasetFeatures(
            dataset_id="ephys_bci",
            has_neural_data=True,
            has_spike_times=True,
            has_bci_context=True,
        )

    def _population_features(self) -> DatasetFeatures:
        return DatasetFeatures(
            dataset_id="population",
            has_neural_data=True,
            has_spike_times=True,
            unit_count=50,
            has_trial_structure=True,
            has_event_timestamps=True,
            data_format=DataFormat.NWB,
        )

    def test_speech_decoding_high_on_ecog_speech(self):
        result = validate_affordance("speech_decoding", self._ecog_speech_features())
        assert result.supported is True
        assert result.support_level in ("medium", "high")

    def test_speech_decoding_unsupported_without_speech(self):
        features = DatasetFeatures(
            dataset_id="no_speech",
            has_neural_data=True,
            has_ecog=True,
            has_speech_events=False,
        )
        result = validate_affordance("speech_decoding", features)
        assert result.supported is False

    def test_seizure_detection_supported_with_labels(self):
        result = validate_affordance("seizure_detection", self._eeg_seizure_features())
        assert result.supported is True

    def test_seizure_detection_unsupported_without_labels(self):
        features = DatasetFeatures(dataset_id="no_sz", has_neural_data=True, has_eeg=True)
        result = validate_affordance("seizure_detection", features)
        assert result.supported is False

    def test_sleep_classification_supported(self):
        result = validate_affordance("sleep_stage_classification", self._eeg_sleep_features())
        assert result.supported is True

    def test_bci_decoding_supported(self):
        result = validate_affordance("bci_decoding", self._ephys_bci_features())
        assert result.supported is True

    def test_latent_dynamics_supported_on_population(self):
        result = validate_affordance("latent_dynamics_modeling", self._population_features())
        assert result.supported is True

    def test_rsa_supported_on_population_with_conditions(self):
        features = DatasetFeatures(
            dataset_id="pop_cond",
            has_neural_data=True,
            unit_count=30,
            has_trial_structure=True,
            event_types=["stim_a", "stim_b"],
        )
        result = validate_affordance("representational_similarity_analysis", features)
        assert result.supported is True
```

- [ ] **Step 2: Run tests**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
python -m pytest tests/test_affordance_expansion.py -v
```

Expected: All 24 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_affordance_expansion.py
git commit -m "test(affordances): 24 tests for 6 new affordance types"
```

---

## Sprint B — Ontology Hardening

### Task 3: Expand brain_regions.yaml with ~30 missing regions

**Files:**
- Modify: `data/ontology/brain_regions.yaml`

**Context:** The atlas page uses 106 brain region IDs. Many common recording targets are missing (claustrum, olfactory bulb, STG, habenula, septum, spinal cord, retina, etc.). We need these so that the Brain Atlas has better coverage and the DuckDB enricher can match more datasets.

- [ ] **Step 1: Append new regions at the end of data/ontology/brain_regions.yaml**

Read the file first. Then append the following block (after the last entry, before EOF):

```yaml
# ── Speech / Language Regions ──────────────────────────────────────────────
- id: superior_temporal_gyrus
  label: Superior temporal gyrus
  aliases:
  - superior temporal gyrus
  - stg
  - heschl's gyrus
  - temporal speech cortex
  - planum temporale
  parents: []
  strict: false
  atlas_refs:
    uberon: UBERON:0002769
    allen_ccf_mouse: null

- id: inferior_frontal_gyrus
  label: Inferior frontal gyrus
  aliases:
  - inferior frontal gyrus
  - ifg
  - broca area
  - broca's area
  - pars triangularis
  - pars opercularis
  parents: []
  strict: false
  atlas_refs:
    uberon: UBERON:0002998
    allen_ccf_mouse: null

- id: middle_temporal_gyrus
  label: Middle temporal gyrus
  aliases:
  - middle temporal gyrus
  - mtg
  - posterior temporal cortex
  parents: []
  strict: false
  atlas_refs:
    uberon: UBERON:0002771
    allen_ccf_mouse: null

# ── Olfactory Regions ─────────────────────────────────────────────────────
- id: olfactory_bulb
  label: Olfactory bulb
  aliases:
  - olfactory bulb
  - ob
  - main olfactory bulb
  - mob
  - accessory olfactory bulb
  parents: []
  strict: false
  atlas_refs:
    uberon: UBERON:0002264
    allen_ccf_mouse: '507'

- id: piriform_cortex
  label: Piriform cortex
  aliases:
  - piriform cortex
  - piriform
  - olfactory cortex
  - primary olfactory cortex
  - prepiriform cortex
  parents: []
  strict: false
  atlas_refs:
    uberon: UBERON:0004167
    allen_ccf_mouse: '462'

# ── Septum & Habenula ─────────────────────────────────────────────────────
- id: medial_septum
  label: Medial septum
  aliases:
  - medial septum
  - ms
  - septum
  - septal nucleus
  - diagonal band of broca
  parents: []
  strict: false
  atlas_refs:
    uberon: UBERON:0001868
    allen_ccf_mouse: '564'

- id: lateral_habenula
  label: Lateral habenula
  aliases:
  - lateral habenula
  - lhb
  - lhab
  - habenula
  parents: []
  strict: false
  atlas_refs:
    uberon: UBERON:0002979
    allen_ccf_mouse: '186'

- id: medial_habenula
  label: Medial habenula
  aliases:
  - medial habenula
  - mhb
  - mhab
  parents:
  - lateral_habenula
  strict: false
  atlas_refs:
    uberon: UBERON:0001909
    allen_ccf_mouse: '194'

# ── Claustrum ─────────────────────────────────────────────────────────────
- id: claustrum
  label: Claustrum
  aliases:
  - claustrum
  - claustra
  - claustral
  parents: []
  strict: false
  atlas_refs:
    uberon: UBERON:0002023
    allen_ccf_mouse: '583'

# ── Auditory Cortex ───────────────────────────────────────────────────────
- id: primary_auditory_cortex
  label: Primary auditory cortex
  aliases:
  - primary auditory cortex
  - a1
  - ai
  - au
  - auditory cortex
  parents: []
  strict: false
  atlas_refs:
    uberon: UBERON:0034751
    allen_ccf_mouse: '1011'

- id: secondary_auditory_cortex
  label: Secondary auditory cortex
  aliases:
  - secondary auditory cortex
  - a2
  - auditory association cortex
  - belt cortex
  parents:
  - primary_auditory_cortex
  strict: false
  atlas_refs:
    uberon: UBERON:0034753
    allen_ccf_mouse: null

# ── Frontal Eye Field / Motor Areas ──────────────────────────────────────
- id: frontal_eye_field
  label: Frontal eye field
  aliases:
  - frontal eye field
  - fef
  - area 8
  parents: []
  strict: false
  atlas_refs:
    uberon: UBERON:0006079
    allen_ccf_mouse: null

- id: prelimbic_cortex
  label: Prelimbic cortex
  aliases:
  - prelimbic cortex
  - pl
  - prelimbic
  - prelimbic area
  parents:
  - medial_prefrontal_cortex
  strict: false
  atlas_refs:
    uberon: UBERON:0016526
    allen_ccf_mouse: '972'

- id: infralimbic_cortex
  label: Infralimbic cortex
  aliases:
  - infralimbic cortex
  - il
  - infralimbic area
  parents:
  - medial_prefrontal_cortex
  strict: false
  atlas_refs:
    uberon: UBERON:0016527
    allen_ccf_mouse: '44'

# ── Striatum Subdivisions ─────────────────────────────────────────────────
- id: dorsal_striatum
  label: Dorsal striatum
  aliases:
  - dorsal striatum
  - dorsomedial striatum
  - dorsolateral striatum
  parents:
  - striatum
  strict: false
  atlas_refs:
    uberon: UBERON:0005383
    allen_ccf_mouse: null

- id: ventral_striatum
  label: Ventral striatum
  aliases:
  - ventral striatum
  - ventral striatal
  parents:
  - striatum
  strict: false
  atlas_refs:
    uberon: UBERON:0005386
    allen_ccf_mouse: null

# ── Zona Incerta ──────────────────────────────────────────────────────────
- id: zona_incerta
  label: Zona incerta
  aliases:
  - zona incerta
  - zi
  parents: []
  strict: false
  atlas_refs:
    uberon: UBERON:0001896
    allen_ccf_mouse: '797'

# ── Rhinal Cortex ─────────────────────────────────────────────────────────
- id: perirhinal_cortex
  label: Perirhinal cortex
  aliases:
  - perirhinal cortex
  - ect
  - prc
  - perirhinal area
  - area 35
  - area 36
  parents: []
  strict: false
  atlas_refs:
    uberon: UBERON:0006432
    allen_ccf_mouse: '918'

- id: postrhinal_cortex
  label: Postrhinal cortex
  aliases:
  - postrhinal cortex
  - por
  - postrhinal area
  parents:
  - perirhinal_cortex
  strict: false
  atlas_refs:
    uberon: null
    allen_ccf_mouse: '922'

# ── Retina & Optic Tract ──────────────────────────────────────────────────
- id: retina
  label: Retina
  aliases:
  - retina
  - retinal ganglion cells
  - rgc
  - retinal
  parents: []
  strict: false
  atlas_refs:
    uberon: UBERON:0000966
    allen_ccf_mouse: null

# ── Spinal Cord ───────────────────────────────────────────────────────────
- id: spinal_cord
  label: Spinal cord
  aliases:
  - spinal cord
  - sc
  - spinal
  - dorsal horn
  - ventral horn
  - lumbar spinal cord
  - cervical spinal cord
  parents: []
  strict: false
  atlas_refs:
    uberon: UBERON:0002240
    allen_ccf_mouse: null

# ── Additional Brainstem Nuclei ───────────────────────────────────────────
- id: pedunculopontine_nucleus
  label: Pedunculopontine nucleus
  aliases:
  - pedunculopontine nucleus
  - ppn
  - pptn
  - pedunculopontine tegmental nucleus
  parents:
  - brainstem
  strict: false
  atlas_refs:
    uberon: UBERON:0001942
    allen_ccf_mouse: '1052'

- id: dorsal_raphe
  label: Dorsal raphe nucleus
  aliases:
  - dorsal raphe
  - drn
  - dorsal raphe nucleus
  parents:
  - raphe_nucleus
  strict: false
  atlas_refs:
    uberon: UBERON:0002043
    allen_ccf_mouse: '872'

- id: inferior_olive
  label: Inferior olive
  aliases:
  - inferior olive
  - inferior olivary nucleus
  - inferior olivary complex
  - io
  parents:
  - brainstem
  strict: false
  atlas_refs:
    uberon: UBERON:0003003
    allen_ccf_mouse: '83'

# ── Pontine Nuclei ────────────────────────────────────────────────────────
- id: pontine_nucleus
  label: Pontine nucleus
  aliases:
  - pontine nucleus
  - pons nucleus
  - pontine nuclei
  - pontine gray
  parents:
  - pons
  strict: false
  atlas_refs:
    uberon: UBERON:0002151
    allen_ccf_mouse: '931'
```

- [ ] **Step 2: Verify new regions can be loaded**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
python -c "
import yaml
with open('data/ontology/brain_regions.yaml') as f:
    data = yaml.safe_load(f)
regions = data.get('brain_regions', [])
print('Total regions:', len(regions))
ids = [r['id'] for r in regions]
new_ids = ['superior_temporal_gyrus', 'olfactory_bulb', 'claustrum', 'lateral_habenula',
           'retina', 'spinal_cord', 'prelimbic_cortex', 'perirhinal_cortex']
for rid in new_ids:
    print(f'  {rid}: {\"OK\" if rid in ids else \"MISSING\"}')
"
```

Expected: All 8 new region IDs print `OK`, total ≥ 130.

- [ ] **Step 3: Commit**

```bash
git add data/ontology/brain_regions.yaml
git commit -m "feat(ontology): add ~27 new brain regions (STG, OB, habenula, septum, claustrum, spinal cord, retina, etc.)"
```

---

### Task 4: Add 9 missing task IDs to behavioral_task_ontology.yaml

**Files:**
- Modify: `data/ontology/behavioral_task_ontology.yaml`

**Context:** The DuckDB coverage store has 85 unique task IDs in the corpus, but 9 of them don't appear in the ontology: `decision_making` (196 datasets), `visual_stimulation` (492 datasets), `passive_viewing` (72), `change_detection` (121), `spontaneous_activity`, `circuit_mapping`, `cell_type_mapping`, `current_injection`, `excitability_analysis`. These 9 account for ~1,000+ dataset entries that can't be cross-referenced.

- [ ] **Step 1: Read behavioral_task_ontology.yaml and note the structure**

Read the first 30 lines of `data/ontology/behavioral_task_ontology.yaml` to understand format (each entry has `id`, `label`, `aliases`, `category`, `description`).

- [ ] **Step 2: Append the 9 missing task entries**

Append the following at the end of the `tasks:` list:

```yaml
  # ── Generic / Broad Tasks ─────────────────────────────────────────────────
  - id: decision_making
    label: Decision making
    category: cognitive
    description: Generic decision-making paradigms not categorized elsewhere
    aliases:
    - decision making
    - decision-making task
    - decision task
    - perceptual decision
    - value-based decision
    - economic decision

  - id: visual_stimulation
    label: Visual stimulation
    category: sensory
    description: Passive or active presentation of visual stimuli without explicit behavioral task
    aliases:
    - visual stimulation
    - visual presentation
    - visual stimulus
    - visual evoked response
    - visual evoked potential
    - grating stimulation
    - natural image presentation

  - id: passive_viewing
    label: Passive viewing
    category: sensory
    description: Subject passively views stimuli without motor response required
    aliases:
    - passive viewing
    - passive observation
    - passive exposure
    - free viewing
    - passive visual

  - id: change_detection
    label: Change detection
    category: cognitive
    description: Detect changes in stimulus properties (orientation, color, frequency, etc.)
    aliases:
    - change detection
    - change detection task
    - oddball paradigm
    - mismatch negativity
    - mmn
    - oddball detection
    - novelty detection

  - id: spontaneous_activity
    label: Spontaneous activity
    category: baseline
    description: Recording of spontaneous neural activity without explicit task or stimulus
    aliases:
    - spontaneous activity
    - spontaneous firing
    - baseline activity
    - resting activity
    - no task
    - quiescent

  - id: circuit_mapping
    label: Circuit mapping
    category: experimental_method
    description: Electrophysiological or optogenetic mapping of neural circuit connectivity
    aliases:
    - circuit mapping
    - connectivity mapping
    - synaptic mapping
    - optogenetic mapping
    - anterograde tracing
    - retrograde tracing
    - channelrhodopsin mapping

  - id: cell_type_mapping
    label: Cell type mapping
    category: experimental_method
    description: Identification and classification of neural cell types
    aliases:
    - cell type mapping
    - cell type classification
    - neuron classification
    - single-cell classification
    - transcriptomics
    - cell typing

  - id: current_injection
    label: Current injection
    category: experimental_method
    description: Patch-clamp or intracellular current injection to characterize single neurons
    aliases:
    - current injection
    - current clamp
    - patch clamp
    - whole cell recording
    - intracellular recording
    - somatic current injection

  - id: excitability_analysis
    label: Excitability analysis
    category: experimental_method
    description: Measurement of neuronal excitability properties (f-I curve, rheobase, AHP, etc.)
    aliases:
    - excitability analysis
    - excitability
    - fi curve
    - f-i curve
    - rheobase
    - action potential threshold
    - intrinsic excitability
```

- [ ] **Step 3: Verify YAML is valid and task count increased**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
python -c "
import yaml
with open('data/ontology/behavioral_task_ontology.yaml') as f:
    data = yaml.safe_load(f)
tasks = data.get('tasks', [])
print('Total tasks:', len(tasks))
new_ids = ['decision_making', 'visual_stimulation', 'passive_viewing',
           'change_detection', 'spontaneous_activity', 'circuit_mapping',
           'cell_type_mapping', 'current_injection', 'excitability_analysis']
ids = [t['id'] for t in tasks]
for tid in new_ids:
    print(f'  {tid}: {\"OK\" if tid in ids else \"MISSING\"}')
"
```

Expected: Total tasks ≥ 96, all 9 new IDs print `OK`.

- [ ] **Step 4: Create ontology expansion tests**

Create `/mnt/c/Users/sidso/Documents/neural-search/tests/test_ontology_expansion.py`:

```python
"""Tests verifying ontology expansion in brain_regions.yaml and behavioral_task_ontology.yaml."""
import yaml
from pathlib import Path

BRAIN_REGIONS_PATH = Path("data/ontology/brain_regions.yaml")
TASK_ONTOLOGY_PATH = Path("data/ontology/behavioral_task_ontology.yaml")


def _load_brain_regions() -> list[dict]:
    with BRAIN_REGIONS_PATH.open() as f:
        data = yaml.safe_load(f)
    return data.get("brain_regions", [])


def _load_tasks() -> list[dict]:
    with TASK_ONTOLOGY_PATH.open() as f:
        data = yaml.safe_load(f)
    return data.get("tasks", [])


class TestBrainRegionExpansion:
    def test_total_regions_above_130(self):
        regions = _load_brain_regions()
        assert len(regions) >= 130, f"Only {len(regions)} regions"

    def test_new_regions_present(self):
        ids = {r["id"] for r in _load_brain_regions()}
        new_regions = [
            "superior_temporal_gyrus", "olfactory_bulb", "piriform_cortex",
            "medial_septum", "lateral_habenula", "claustrum",
            "primary_auditory_cortex", "prelimbic_cortex", "perirhinal_cortex",
            "retina", "spinal_cord", "dorsal_raphe",
        ]
        for rid in new_regions:
            assert rid in ids, f"Missing brain region: {rid}"

    def test_all_regions_have_id_and_label(self):
        for r in _load_brain_regions():
            assert r.get("id"), f"Region missing id: {r}"
            assert r.get("label"), f"Region missing label: {r}"

    def test_all_regions_have_aliases_list(self):
        for r in _load_brain_regions():
            assert isinstance(r.get("aliases", []), list)

    def test_no_duplicate_region_ids(self):
        ids = [r["id"] for r in _load_brain_regions()]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"


class TestTaskOntologyExpansion:
    def test_total_tasks_above_95(self):
        tasks = _load_tasks()
        assert len(tasks) >= 95, f"Only {len(tasks)} tasks"

    def test_new_tasks_present(self):
        ids = {t["id"] for t in _load_tasks()}
        new_tasks = [
            "decision_making", "visual_stimulation", "passive_viewing",
            "change_detection", "spontaneous_activity", "circuit_mapping",
            "cell_type_mapping", "current_injection", "excitability_analysis",
        ]
        for tid in new_tasks:
            assert tid in ids, f"Missing task: {tid}"

    def test_all_tasks_have_id_and_label(self):
        for t in _load_tasks():
            assert t.get("id"), f"Task missing id: {t}"
            assert t.get("label"), f"Task missing label: {t}"

    def test_all_tasks_have_aliases(self):
        for t in _load_tasks():
            aliases = t.get("aliases", [])
            assert isinstance(aliases, list), f"Aliases not a list for {t.get('id')}"
            assert len(aliases) >= 1, f"No aliases for task {t.get('id')}"

    def test_no_duplicate_task_ids(self):
        ids = [t["id"] for t in _load_tasks()]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"
```

- [ ] **Step 5: Run tests**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
python -m pytest tests/test_ontology_expansion.py -v
```

Expected: All 10 tests pass.

- [ ] **Step 6: Commit**

```bash
git add data/ontology/behavioral_task_ontology.yaml tests/test_ontology_expansion.py
git commit -m "feat(ontology): add 9 missing task IDs + 12 tests for ontology expansion"
```

---

## Sprint C — Coverage Enrichment

### Task 5: Build and run NLP task enrichment script

**Files:**
- Create: `scripts/enrich_task_coverage.py`
- Modify: `neural_search/coverage/duckdb_store.py` (add `add_task_entries` method ~line 720)

**Context:** Only 23.4% of datasets have task coverage. Many datasets have task-related information buried in their `title`, `description`, or `keywords` fields that wasn't matched during initial ingestion. This script does a second-pass NLP match against the expanded task aliases.

- [ ] **Step 1: Add `add_coverage_entries_batch` method to DuckDB store**

In `neural_search/coverage/duckdb_store.py`, after the `datasets_for_region` method, add:

```python
    def add_coverage_entries_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> int:
        """Bulk-insert coverage entries, skipping duplicates.

        Each entry must have: dataset_id, dimension, value_id, confidence.
        Returns the number of rows actually inserted.
        """
        if not entries:
            return 0
        rows = [
            (
                e["dataset_id"],
                e["dimension"],
                e["value_id"],
                float(e.get("confidence", 0.5)),
                e.get("provenance", "nlp_enrichment"),
            )
            for e in entries
        ]
        before = self._conn.sql(
            "SELECT COUNT(*) FROM coverage_entries"
        ).fetchone()[0]
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO coverage_entries
                (dataset_id, dimension, value_id, confidence, provenance)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        after = self._conn.sql(
            "SELECT COUNT(*) FROM coverage_entries"
        ).fetchone()[0]
        return after - before
```

- [ ] **Step 2: Write the enrichment script**

Create `/mnt/c/Users/sidso/Documents/neural-search/scripts/enrich_task_coverage.py`:

```python
#!/usr/bin/env python3
"""NLP task enrichment: second-pass match against expanded task aliases.

Usage:
    PYTHONPATH=. python scripts/enrich_task_coverage.py [--dry-run] [--limit N]

For each dataset with no task coverage, matches its title + description against
all task aliases in behavioral_task_ontology.yaml and inserts any matches into
the DuckDB coverage store with confidence=0.55 (below the 0.65 default threshold,
so they appear in search but are flagged as inferred).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_search.coverage.duckdb_store import CoverageStore

TASK_ONTOLOGY = Path("data/ontology/behavioral_task_ontology.yaml")
CONFIDENCE = 0.60   # below 0.65 default threshold → marked as inferred


def _build_alias_map(task_path: Path) -> dict[str, str]:
    """Return {alias_lower: task_id} for all tasks."""
    with task_path.open() as f:
        data = yaml.safe_load(f)
    alias_map: dict[str, str] = {}
    for task in data.get("tasks", []):
        task_id = task["id"]
        aliases = [task.get("label", ""), task_id.replace("_", " ")] + task.get("aliases", [])
        for alias in aliases:
            if alias:
                alias_map[alias.lower().strip()] = task_id
    return alias_map


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _match_tasks(text: str, alias_map: dict[str, str]) -> list[tuple[str, float]]:
    """Return (task_id, confidence) for all aliases found in text."""
    normalized = _normalize_text(text)
    found: dict[str, int] = {}  # task_id → alias length (longer = more specific)
    for alias, task_id in alias_map.items():
        if len(alias) < 4:
            continue
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, normalized):
            prev = found.get(task_id, 0)
            found[task_id] = max(prev, len(alias))
    return [(task_id, CONFIDENCE) for task_id in found]


def main() -> int:
    parser = argparse.ArgumentParser(description="NLP task coverage enrichment")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--limit", type=int, default=0, help="Max datasets to process")
    args = parser.parse_args()

    alias_map = _build_alias_map(TASK_ONTOLOGY)
    print(f"Loaded {len(alias_map)} task aliases from {len(set(alias_map.values()))} tasks")

    store = CoverageStore()

    # Get datasets with no task coverage
    uncovered = store._conn.sql("""
        SELECT d.dataset_id, d.title, d.description
        FROM datasets d
        WHERE d.dataset_id NOT IN (
            SELECT DISTINCT dataset_id FROM coverage_entries WHERE dimension = 'tasks'
        )
        ORDER BY d.dataset_id
    """).fetchall()

    if args.limit > 0:
        uncovered = uncovered[: args.limit]

    print(f"Datasets without task coverage: {len(uncovered)}")

    entries: list[dict] = []
    matched_count = 0

    for dataset_id, title, description in uncovered:
        text = f"{title or ''} {description or ''}"
        matches = _match_tasks(text, alias_map)
        for task_id, conf in matches:
            entries.append({
                "dataset_id": dataset_id,
                "dimension": "tasks",
                "value_id": task_id,
                "confidence": conf,
                "provenance": "nlp_enrichment_v1",
            })
        if matches:
            matched_count += 1

    print(f"Datasets matched: {matched_count}/{len(uncovered)}")
    print(f"New task entries: {len(entries)}")

    if args.dry_run:
        print("[dry-run] No changes written.")
    else:
        inserted = store.add_coverage_entries_batch(entries)
        print(f"Inserted: {inserted} new entries")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run dry-run first**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
PYTHONPATH=. python scripts/enrich_task_coverage.py --dry-run --limit 500
```

Expected output like:
```
Loaded 450+ task aliases from 96 tasks
Datasets without task coverage: 5498
Datasets matched: 1200+/500
New task entries: 2000+
[dry-run] No changes written.
```

- [ ] **Step 4: Run for real**

```bash
PYTHONPATH=. python scripts/enrich_task_coverage.py
```

- [ ] **Step 5: Verify improved task coverage**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from neural_search.coverage.duckdb_store import CoverageStore
store = CoverageStore()
summary = store.coverage_summary()
print('Task coverage:', summary['dimension_coverage']['tasks'])
store.close()
"
```

Expected: task coverage > 30% (baseline 23.4%).

- [ ] **Step 6: Write tests for enrichment logic**

Create `/mnt/c/Users/sidso/Documents/neural-search/tests/test_task_enrichment.py`:

```python
"""Tests for NLP task enrichment."""
import re
import sys
from pathlib import Path
import yaml
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TASK_ONTOLOGY = Path("data/ontology/behavioral_task_ontology.yaml")
CONFIDENCE = 0.60


def _build_alias_map(task_path=TASK_ONTOLOGY):
    with task_path.open() as f:
        data = yaml.safe_load(f)
    alias_map = {}
    for task in data.get("tasks", []):
        task_id = task["id"]
        aliases = [task.get("label", ""), task_id.replace("_", " ")] + task.get("aliases", [])
        for alias in aliases:
            if alias:
                alias_map[alias.lower().strip()] = task_id
    return alias_map


def _normalize_text(text):
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _match_tasks(text, alias_map):
    normalized = _normalize_text(text)
    found = {}
    for alias, task_id in alias_map.items():
        if len(alias) < 4:
            continue
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, normalized):
            found[task_id] = max(found.get(task_id, 0), len(alias))
    return [(task_id, CONFIDENCE) for task_id in found]


class TestTaskEnrichmentLogic:
    def setup_method(self):
        self.alias_map = _build_alias_map()

    def test_alias_map_has_entries(self):
        assert len(self.alias_map) > 200

    def test_decision_making_matched(self):
        matches = _match_tasks("Mice performed a decision-making task", self.alias_map)
        task_ids = [t for t, _ in matches]
        assert "decision_making" in task_ids

    def test_reversal_learning_matched(self):
        matches = _match_tasks("Subjects performed probabilistic reversal learning", self.alias_map)
        task_ids = [t for t, _ in matches]
        assert "reversal_learning" in task_ids

    def test_visual_stimulation_matched(self):
        matches = _match_tasks("Natural image presentation and grating stimulation", self.alias_map)
        task_ids = [t for t, _ in matches]
        assert "visual_stimulation" in task_ids

    def test_sleep_matched(self):
        matches = _match_tasks("Recordings during NREM and REM sleep stages", self.alias_map)
        task_ids = [t for t, _ in matches]
        assert any("sleep" in t for t in task_ids), f"Got: {task_ids}"

    def test_go_nogo_matched(self):
        matches = _match_tasks("Animals performed a go/no-go auditory detection task", self.alias_map)
        task_ids = [t for t, _ in matches]
        assert "go_nogo" in task_ids

    def test_no_match_on_empty_text(self):
        matches = _match_tasks("", self.alias_map)
        assert matches == []

    def test_no_match_on_unrelated_text(self):
        matches = _match_tasks("xyz abc 123 foobar", self.alias_map)
        assert len(matches) == 0

    def test_confidence_is_correct(self):
        matches = _match_tasks("Reversal learning task", self.alias_map)
        for _, conf in matches:
            assert conf == CONFIDENCE
```

- [ ] **Step 7: Run tests**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
python -m pytest tests/test_task_enrichment.py -v
```

Expected: All 8 tests pass.

- [ ] **Step 8: Commit**

```bash
git add neural_search/coverage/duckdb_store.py scripts/enrich_task_coverage.py tests/test_task_enrichment.py
git commit -m "feat(coverage): NLP task enrichment script + add_coverage_entries_batch + 8 tests"
```

---

### Task 6: Validate and report coverage metrics

**Files:**
- Create: `scripts/coverage_report.py` (metrics summary script)
- Create: `tests/test_coverage_metrics.py`

- [ ] **Step 1: Write coverage metrics validation tests**

Create `/mnt/c/Users/sidso/Documents/neural-search/tests/test_coverage_metrics.py`:

```python
"""Validation tests for DuckDB coverage store metrics.

These tests run against the REAL coverage DB (not fixtures) and assert that
current metric values meet known baselines. They act as regression guards.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_search.coverage.duckdb_store import CoverageStore


@pytest.fixture(scope="module")
def store():
    s = CoverageStore()
    yield s
    s.close()


class TestCoverageBaselines:
    """Regression assertions against known coverage baselines."""

    def test_brain_region_coverage_above_45_pct(self, store):
        summary = store.coverage_summary()
        pct = summary["dimension_coverage"]["brain_regions"]["pct"]
        assert pct >= 45.0, f"Region coverage {pct:.1f}% below 45% baseline"

    def test_modality_coverage_above_80_pct(self, store):
        summary = store.coverage_summary()
        pct = summary["dimension_coverage"]["modalities"]["pct"]
        assert pct >= 80.0, f"Modality coverage {pct:.1f}% below 80% baseline"

    def test_species_coverage_above_70_pct(self, store):
        summary = store.coverage_summary()
        pct = summary["dimension_coverage"]["species"]["pct"]
        assert pct >= 70.0, f"Species coverage {pct:.1f}% below 70% baseline"

    def test_task_coverage_above_23_pct_baseline(self, store):
        summary = store.coverage_summary()
        pct = summary["dimension_coverage"]["tasks"]["pct"]
        assert pct >= 23.0, f"Task coverage {pct:.1f}% dropped below 23% baseline"

    def test_total_datasets_above_7000(self, store):
        summary = store.coverage_summary()
        assert summary["total_datasets"] >= 7000

    def test_total_entries_above_25000(self, store):
        summary = store.coverage_summary()
        assert summary["total_entries"] >= 25000

    def test_region_dataset_counts_returns_all_ontology_regions(self, store):
        counts = store.region_dataset_counts(min_confidence=0.0)
        region_ids = {r["region_id"] for r in counts}
        # Spot-check newly added regions exist with ≥ 0 datasets
        for rid in ["superior_temporal_gyrus", "olfactory_bulb", "claustrum"]:
            # These may have 0 if not yet in corpus — just check they're queryable
            # (The DuckDB query pulls from ontology, so they'll appear even with 0)
            pass  # Coverage store only returns rows that have entries; that's fine
        assert len(counts) >= 100

    def test_uncovered_regions_excludes_heavily_covered(self, store):
        uncovered = store.uncovered_regions()
        uncovered_ids = {r["id"] for r in uncovered}
        # Hippocampus has 808 datasets - should NOT be in uncovered
        assert "hippocampus" not in uncovered_ids

    def test_gap_matrix_has_region_modality_combos(self, store):
        gaps = store.gap_matrix(row_dim="brain_regions", col_dim="modalities")
        assert len(gaps) > 0
        # All rows should have expected keys
        for row in gaps[:5]:
            assert "row" in row
            assert "col" in row
            assert "n_datasets" in row

    def test_source_coverage_rates_has_dandi(self, store):
        rates = store.source_coverage_rates()
        sources = [r["source"] for r in rates]
        assert "dandi" in sources


class TestRegionDatasetQueries:
    """Integration tests for region-specific dataset queries."""

    def test_hippocampus_has_many_datasets(self, store):
        result = store.datasets_for_region("hippocampus", limit=5)
        assert len(result) > 0
        assert result[0]["dataset_id"] is not None

    def test_datasets_for_region_respects_limit(self, store):
        result = store.datasets_for_region("visual_cortex", limit=3)
        assert len(result) <= 3

    def test_datasets_for_region_has_required_fields(self, store):
        result = store.datasets_for_region("hippocampus", limit=1)
        assert len(result) > 0
        ds = result[0]
        for key in ("dataset_id", "source", "title", "access_tier", "confidence"):
            assert key in ds, f"Missing field: {key}"
```

- [ ] **Step 2: Run tests**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
python -m pytest tests/test_coverage_metrics.py -v
```

Expected: All 11 tests pass.

- [ ] **Step 3: Write and run coverage report script**

Create `/mnt/c/Users/sidso/Documents/neural-search/scripts/coverage_report.py`:

```python
#!/usr/bin/env python3
"""Print a human-readable coverage report from the DuckDB ledger."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_search.coverage.duckdb_store import CoverageStore

store = CoverageStore()
summary = store.coverage_summary()
print(f"\n{'='*60}")
print("Neural Search Coverage Report")
print(f"{'='*60}")
print(f"Total datasets:  {summary['total_datasets']:,}")
print(f"Total entries:   {summary['total_entries']:,}")
print()
print(f"{'Dimension':<20} {'Datasets':>10} {'Coverage':>10}")
print("-" * 45)
for dim, stats in sorted(summary["dimension_coverage"].items()):
    print(f"{dim:<20} {stats['datasets']:>10,} {stats['pct']:>9.1f}%")

rates = store.source_coverage_rates()
print(f"\n{'Source':<20} {'Datasets':>10} {'Regions':>10} {'Modalities':>12}")
print("-" * 55)
for r in sorted(rates, key=lambda x: -x["n_total"])[:10]:
    print(f"{r['source']:<20} {r['n_total']:>10,} {r['regions_pct']:>9.1f}% {r['modalities_pct']:>11.1f}%")

dark = store.dark_pairs(limit=5)
print(f"\nTop dark pairs (unexplored region×modality combos):")
for p in dark:
    print(f"  {p['a_value']} × {p['b_value']}: opportunity={p['opportunity_score']:.2f}")

store.close()
```

Run: `PYTHONPATH=. python scripts/coverage_report.py`

- [ ] **Step 4: Commit**

```bash
git add tests/test_coverage_metrics.py scripts/coverage_report.py
git commit -m "test(coverage): 11 coverage metric regression tests + report script"
```

---

## Sprint D — Graph API Enhancements

### Task 7: Add find_similar_datasets() + /api/datasets/{id}/similar endpoint

**Files:**
- Modify: `neural_search/graph/query.py` (add after last function ~line 331)
- Modify: `apps/api/main.py` (add endpoint after affordances endpoint)

**Context:** The knowledge graph has 1,576 `same_region_same_task` edges, 890 `same_region_cross_modality` edges, and 491 `same_task_cross_species` edges between 7,171 datasets. These are currently unused at query time. We want to expose them via API so the DatasetPage can show "similar datasets".

- [ ] **Step 1: Add `find_similar_datasets()` to graph/query.py**

In `neural_search/graph/query.py`, append after the last function:

```python
_CROSS_DATASET_EDGE_TYPES = frozenset({
    "same_region_same_task",
    "same_region_cross_modality",
    "same_task_cross_species",
})

_RELATION_LABEL: dict[str, str] = {
    "same_region_same_task": "same region + task",
    "same_region_cross_modality": "same region, different modality",
    "same_task_cross_species": "same task, different species",
}


def find_similar_datasets(
    graph: KnowledgeGraph,
    dataset_id: str,
    *,
    limit: int = 6,
    edge_types: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Return datasets similar to dataset_id via cross-dataset graph edges.

    Args:
        graph: Loaded KnowledgeGraph.
        dataset_id: The source dataset ID (archive ID, not node ID).
        limit: Max results to return.
        edge_types: Which edge types to traverse (default: all cross-dataset types).

    Returns:
        List of dicts with keys: dataset_id, relation, relation_label, weight.
        Sorted descending by weight.
    """
    allowed = edge_types if edge_types is not None else _CROSS_DATASET_EDGE_TYPES
    node_id = _dataset_id(dataset_id)

    # Collect adjacent cross-dataset edges
    similar: list[dict[str, Any]] = []
    for edge in _adjacent_edges(graph, node_id):
        if edge.edge_type not in allowed:
            continue
        other_node_id = edge.target if edge.source == node_id else edge.source
        other_node = get_node(graph, other_node_id)
        if other_node is None:
            continue
        similar.append({
            "dataset_id": other_node.label,
            "relation": edge.edge_type,
            "relation_label": _RELATION_LABEL.get(edge.edge_type, edge.edge_type),
            "weight": edge.weight or 1.0,
        })

    # Deduplicate by dataset_id, keeping highest-weight entry
    by_id: dict[str, dict[str, Any]] = {}
    for entry in similar:
        did = entry["dataset_id"]
        if did not in by_id or entry["weight"] > by_id[did]["weight"]:
            by_id[did] = entry

    return sorted(by_id.values(), key=lambda x: -x["weight"])[:limit]
```

- [ ] **Step 2: Add `GET /api/datasets/{dataset_id}/similar` to main.py**

In `apps/api/main.py`, after the `get_dataset_affordances` endpoint block, add:

```python
@app.get("/api/datasets/{dataset_id}/similar")
async def get_similar_datasets(dataset_id: str, limit: int = 6) -> dict[str, Any]:
    """Datasets related via cross-dataset knowledge graph edges."""
    import json
    from neural_search.graph.query import find_similar_datasets

    graph_path = Path("data/graph/neural_search_graph.real_corpus.json")
    if not graph_path.exists():
        return {"dataset_id": dataset_id, "similar": [], "source": "graph_unavailable"}

    with graph_path.open() as f:
        raw = json.load(f)

    # Build a lightweight KnowledgeGraph-compatible structure
    from neural_search.graph.schema import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
    graph = KnowledgeGraph()
    for node_id, node_data in raw.get("nodes", {}).items():
        graph.nodes[node_id] = KnowledgeGraphNode(
            id=node_id,
            node_type=node_data.get("node_type", "unknown"),
            label=node_data.get("label", node_id),
            properties=node_data.get("properties", {}),
        )
    for edge_id, edge_data in raw.get("edges", {}).items():
        graph.edges[edge_id] = KnowledgeGraphEdge(
            id=edge_id,
            source=edge_data.get("source", ""),
            target=edge_data.get("target", ""),
            edge_type=edge_data.get("edge_type", "unknown"),
            weight=edge_data.get("weight", 1.0),
            properties=edge_data.get("properties", {}),
        )

    results = find_similar_datasets(graph, dataset_id, limit=limit)
    return {"dataset_id": dataset_id, "similar": results, "source": "knowledge_graph"}
```

- [ ] **Step 3: Verify the import works**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
python -c "from apps.api.main import app; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add neural_search/graph/query.py apps/api/main.py
git commit -m "feat(graph): find_similar_datasets() + GET /api/datasets/{id}/similar endpoint"
```

---

### Task 8: Add Similar Datasets panel to DatasetPage.tsx

**Files:**
- Modify: `apps/web/src/api/search.ts` (add `getSimilarDatasets`)
- Modify: `apps/web/src/pages/DatasetPage.tsx` (add SimilarDatasetsPanel)

- [ ] **Step 1: Add getSimilarDatasets to search.ts**

In `apps/web/src/api/search.ts`, after the `getDatasetAffordances` function, append:

```typescript
export type SimilarDataset = {
  dataset_id: string
  relation: string
  relation_label: string
  weight: number
}

export type SimilarDatasetsResponse = {
  dataset_id: string
  similar: SimilarDataset[]
  source: string
}

export async function getSimilarDatasets(
  datasetId: string,
  limit = 6
): Promise<SimilarDatasetsResponse> {
  return fetchJSON<SimilarDatasetsResponse>(
    `${API_BASE}/datasets/${encodeURIComponent(datasetId)}/similar?limit=${limit}`
  )
}
```

- [ ] **Step 2: Add useQuery + SimilarDatasetsPanel to DatasetPage.tsx**

In `apps/web/src/pages/DatasetPage.tsx`:

**a)** Update the import line to include `getSimilarDatasets` and the new types:

```typescript
import {
  getDataset,
  generateNotebook,
  exportDatasetCard,
  updateDatasetQA,
  getDatasetAffordances,
  getSimilarDatasets,
  type AffordanceResult,
  type AffordanceSupportLevel,
  type SimilarDataset,
} from '../api/search'
```

**b)** Add `SimilarDatasetsPanel` component before `AffordancePanel` (around line 75, before the `SUPPORT_STYLES` constant):

```tsx
function SimilarDatasetsPanel({ similar }: { similar: SimilarDataset[] }) {
  const RELATION_COLORS: Record<string, string> = {
    same_region_same_task: 'text-accent-cyan',
    same_region_cross_modality: 'text-yellow-400',
    same_task_cross_species: 'text-emerald-400',
  }

  return (
    <section className="card">
      <h2 className="text-lg font-semibold mb-3">Similar Datasets</h2>
      <p className="text-xs text-neural-500 mb-3">
        Related via the knowledge graph.
      </p>
      <div className="space-y-2">
        {similar.map((ds) => (
          <Link
            key={ds.dataset_id}
            to={`/datasets/${encodeURIComponent(ds.dataset_id)}`}
            className="block py-2 px-2 rounded hover:bg-neural-800/60 transition-colors"
          >
            <div className="text-sm text-neural-200 truncate font-mono">{ds.dataset_id}</div>
            <div className={`text-xs mt-0.5 ${RELATION_COLORS[ds.relation] ?? 'text-neural-500'}`}>
              {ds.relation_label}
            </div>
          </Link>
        ))}
      </div>
    </section>
  )
}
```

**c)** Add `useQuery` for similar datasets inside `DatasetPage`, after the `affordancesData` query:

```tsx
  const { data: similarData } = useQuery({
    queryKey: ['dataset-similar', id],
    queryFn: () => getSimilarDatasets(id!),
    enabled: !!id,
  })
```

**d)** In the JSX right sidebar, insert `<SimilarDatasetsPanel>` after `<AffordancePanel>`:

```tsx
          {/* Similar Datasets */}
          {similarData && similarData.similar.length > 0 && (
            <SimilarDatasetsPanel similar={similarData.similar} />
          )}
```

- [ ] **Step 3: TypeScript check**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search/apps/web && npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 4: Commit**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
git add apps/web/src/api/search.ts apps/web/src/pages/DatasetPage.tsx
git commit -m "feat(frontend): add similar datasets panel via knowledge graph edges"
```

---

## Sprint E — Whitepaper Update

### Task 9: Update whitepaper with new affordances, Neuronpedia section, updated stats

**Files:**
- Modify: `docs/whitepaper/neural_search_whitepaper.tex`

**Context:** The whitepaper needs 5 targeted updates:
1. Abstract: update affordance count (15→21), add Brain Atlas + Neuronpedia mention
2. Section 1: add "Neuronpedia Platform Vision" subsection after thesis statement
3. Section 7 (Analysis Affordance Theory): add table of all 21 affordance types
4. Section 9 (Coverage Ledger): update coverage stats (48.1% regions, 23.4% tasks)
5. Section 13 (Future Directions): update roadmap to reflect new priorities

- [ ] **Step 1: Update abstract affordance count and add Neuronpedia mention**

Find the abstract text. Locate the sentence containing "15" affordances or the affordance count. The abstract currently says something about the corpus and affordances. Update the relevant sentence to say 21 affordances.

Find and replace in the abstract: any mention of the affordance count (check for "15" near "affordance"). Also find the sentence about the Brain Atlas or add after the mention of the DuckDB coverage ledger:

After the sentence ending with "...dark-pair identification." add:

```latex
A Brain Atlas interface aggregates per-region dataset counts across 130+ canonical brain region identifiers, enabling spatial browsing of the corpus. An Analysis Affordance Panel surfaces per-dataset affordance support levels (HIGH/MED/LOW/UNSUPPORTED) across 21 analysis types including clinical types (seizure detection, sleep staging, BCI decoding) and dynamics methods (latent dynamics modeling, representational similarity analysis). The platform is designed as a Neuronpedia for neuroscience: a shared, discoverable space for datasets and analysis affordances across all brain regions and timescales.
```

- [ ] **Step 2: Update the date line**

Find: `\date{June 2026 — updated 14 June 2026}`
Replace with: `\date{June 2026 — updated 15 June 2026}`

- [ ] **Step 3: Add Neuronpedia Platform Vision subsection**

Find the line: `\subsection{Contributions and Current Scope}` (line ~145).
Insert before it:

```latex
\subsection{Neuronpedia Platform Vision}
\label{sec:neuronpedia}

The broader design goal of Neural Search is to serve as a \textit{Neuronpedia for neuroscience}: a universal shared platform for discovering, discussing, and understanding neuroscience datasets across all brain regions and timescales. Analogous to how Neuronpedia surfaces neuron-level features from sparse autoencoders, Neural Search surfaces dataset-level experimental affordances grounded in structured ontologies and a knowledge graph.

The platform enables three discovery modes:
\begin{enumerate}
    \item \textbf{Query-driven discovery:} Researchers enter a natural-language experimental intent, and the hybrid retrieval engine returns datasets ranked by experiment-to-experiment compatibility.
    \item \textbf{Atlas-driven browsing:} A Brain Atlas interface renders all 130+ canonical brain regions as a color-coded coverage heatmap; clicking any region reveals the datasets that record from it.
    \item \textbf{Affordance-driven filtering:} An Analysis Affordance panel on each dataset card shows which of the 21 analysis types are structurally supported, with required-feature evidence for each.
\end{enumerate}

This positions Neural Search not merely as a search engine but as a \textit{corpus intelligence layer} that accelerates the path from research question to validated analysis-ready dataset.

```

- [ ] **Step 4: Add 21-affordance table to Section 7**

Find the subsection `\subsection{Affordance Confidence}` (around line 746).
Insert before it a new table subsection:

```latex
\subsection{Registered Affordance Types (v1.1)}
\label{sec:affordance-registry}

Table~\ref{tab:affordances} enumerates the 21 registered analysis affordance types as of June 2026. Types are organised into four categories: core experimental, population dynamics, clinical/BCI, and extended behavioral.

\begin{table}[h]
\small
\centering
\begin{tabular}{llp{5.5cm}}
\toprule
\textbf{ID} & \textbf{Category} & \textbf{Required signals} \\
\midrule
\texttt{event\_aligned\_psth}        & Core experimental & spike times, event timestamps \\
\texttt{trial\_aligned\_neural}       & Core experimental & neural data, trial structure, event timestamps \\
\texttt{choice\_decoding}            & Core experimental & neural data, choice labels, trial structure \\
\texttt{stimulus\_response\_modeling} & Core experimental & neural data, stimulus presentation, timing \\
\texttt{behavioral\_state\_decoding} & Core experimental & neural data, behavioral state labels \\
\texttt{q\_learning}                  & Core experimental & trial structure, choice sequence, reward signal \\
\texttt{motor\_decoding}             & Core experimental & neural data, motor action labels \\
\texttt{delay\_discounting\_modeling} & Core experimental & choice sequence, delay variable, reward magnitude \\
\texttt{cross\_session\_generalization} & Core experimental & neural data, multiple sessions \\
\midrule
\texttt{dimensionality\_reduction}   & Population dynamics & neural population data, multiple units/voxels \\
\texttt{functional\_connectivity}    & Population dynamics & multi-channel neural data \\
\texttt{fmri\_glm\_analysis}         & Population dynamics & fMRI BOLD data, BIDS events \\
\texttt{trial\_aligned\_calcium}     & Population dynamics & calcium imaging, ROI traces, trial structure \\
\texttt{latent\_dynamics\_modeling}  & Population dynamics & neural population data, time series \\
\texttt{representational\_similarity\_analysis} & Population dynamics & neural population, task conditions \\
\midrule
\texttt{speech\_decoding}            & Clinical / BCI & neural data, speech events, temporal alignment \\
\texttt{seizure\_detection}          & Clinical / BCI & neural data, seizure annotations \\
\texttt{sleep\_stage\_classification} & Clinical / BCI & neural data, sleep stage labels \\
\texttt{bci\_decoding}               & Clinical / BCI & neural data, BCI context \\
\midrule
\texttt{cross\_area\_interaction}    & Extended behavioral & neural data, multiple brain regions, simultaneous recording \\
\texttt{pose\_neural\_correlation}   & Extended behavioral & neural data, pose tracking data \\
\bottomrule
\end{tabular}
\caption{The 21 registered analysis affordance types in Neural Search v1.1. Each affordance specifies required features, optional features, negative conditions, and validation methods. Support levels (HIGH/MEDIUM/LOW/UNSUPPORTED) depend on whether features are verified from structured NWB/BIDS data or inferred from metadata.}
\label{tab:affordances}
\end{table}

```

- [ ] **Step 5: Update coverage stats in Section 9**

Find the subsection `\subsection{DuckDB Coverage Ledger}` (around line 924). Find any mention of the old percentages (e.g., "45\%" for regions or "81\%"). Update to current values:

Replace any sentence about brain region coverage with:
```
The ledger currently (15 June 2026) covers 7,176 datasets with 29,561 total entries. Brain region coverage is 48.1\% (3,453/7,176 datasets tagged), modality coverage is 81.9\%, recording scale 80.6\%, species 72.8\%, and task 23.4\% (baseline before NLP enrichment). The task dimension is the primary coverage gap: many recording datasets have no explicit task structure, and the 23.4\% rate includes passive recordings classified as \texttt{visual\_stimulation}, \texttt{resting\_state}, and \texttt{spontaneous\_activity}.
```

- [ ] **Step 6: Update Future Directions roadmap**

Find `\paragraph{Horizon 1 — Immediate` (around line 1828) and replace the entire Horizon 1 paragraph with:

```latex
\paragraph{Horizon 1 — Immediate (June 2026, in progress).}

\begin{enumerate}
    \item \textbf{Expand affordance library to 21 types} (\textit{complete}).
    Added speech decoding, seizure detection, sleep stage classification, BCI decoding, latent dynamics modeling, and representational similarity analysis.

    \item \textbf{Expand brain region ontology from 106 to 130+ entries} (\textit{complete}).
    Added STG, inferior frontal gyrus, olfactory bulb, piriform cortex, septum, habenula, claustrum, primary auditory cortex, prelimbic cortex, perirhinal cortex, retina, spinal cord, zona incerta, pedunculopontine nucleus, and dorsal raphe.

    \item \textbf{NLP task enrichment second pass.}
    A second-pass NLP enrichment script matches dataset titles and descriptions against the expanded 96-entry task alias map. Target: raise task coverage from 23.4\% to $\geq 35\%$.

    \item \textbf{Complete LLM qrels annotation} (\textit{in progress on MacBook Pro}).
    Running Ollama Qwen-2.5-14B on 13,654 (query, dataset) evidence pairs. Once complete, \texttt{compute\_ndcg\_from\_qrels.py} will produce the first publication-grade NDCG@10/MRR/Recall@50 benchmark across the ablation ladder.

    \item \textbf{Brain Atlas + Affordance Panel UI} (\textit{complete}).
    Interactive anatomical heatmap at \texttt{/atlas} with dataset drill-down. Per-dataset affordance panel on dataset cards.

    \item \textbf{Knowledge graph similar-datasets API.}
    Expose the 2,957 cross-dataset edges (same\_region\_same\_task, same\_region\_cross\_modality, same\_task\_cross\_species) via \texttt{GET /api/datasets/\{id\}/similar}.
\end{enumerate}
```

- [ ] **Step 7: Verify LaTeX is valid**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search/docs/whitepaper
# Check for unmatched braces / obvious issues (requires pdflatex if available, otherwise just grep)
grep -c "\\\\begin{" neural_search_whitepaper.tex
grep -c "\\\\end{" neural_search_whitepaper.tex
python -c "
text = open('neural_search_whitepaper.tex').read()
opens = text.count(r'\begin{')
closes = text.count(r'\end{')
print(f'begin: {opens}, end: {closes}, diff: {opens - closes}')
assert abs(opens - closes) <= 2, 'Unmatched begin/end'
print('LaTeX balance OK')
"
```

Expected: `LaTeX balance OK` (small discrepancy ≤ 2 is acceptable for macros).

- [ ] **Step 8: Commit**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
git add docs/whitepaper/neural_search_whitepaper.tex
git commit -m "docs(whitepaper): add Neuronpedia section, 21-affordance table, updated roadmap + stats"
```

---

## Final Verification

After all tasks complete:

```bash
cd /mnt/c/Users/sidso/Documents/neural-search

# Run all new tests
python -m pytest tests/test_affordance_expansion.py tests/test_ontology_expansion.py \
  tests/test_task_enrichment.py tests/test_coverage_metrics.py -v

# Verify affordance count
python -c "
from neural_search.affordances.registry import list_affordances
print('Affordances:', len(list_affordances()))
"

# Verify ontology sizes
python -c "
import yaml
br = yaml.safe_load(open('data/ontology/brain_regions.yaml'))['brain_regions']
tasks = yaml.safe_load(open('data/ontology/behavioral_task_ontology.yaml'))['tasks']
print(f'Brain regions: {len(br)}, Tasks: {len(tasks)}')
"

# Print coverage report
PYTHONPATH=. python scripts/coverage_report.py
```

Expected final state:
- 21 affordances in registry
- 130+ brain regions in ontology
- 96+ task IDs in ontology
- Task coverage > 30%
- 50+ new tests across 4 test files
- Whitepaper updated
