# Scientific Label Extraction Rules

Neural Search v0.3 extracts scientific labels into `EvidenceLabel` objects. The extractor is rule-based, deterministic, and intentionally conservative.

Implementation:

- Module: `neural_search/scientific_labels.py`
- Extractor name: `rule_based_scientific_label_extractor`
- Extractor version: `v0.3.0`

## Label Types

The extractor supports:

- `species`
- `modality`
- `brain_region`
- `task`
- `behavioral_event`
- `analysis_goal`
- `data_standard`
- `file_format`
- `modeling_method`
- `disease_state`
- `clinical_condition`
- `stimulus_type`
- `recording_context`
- `subject_state`

Dataset schema fields exist for the core label families. Extra label families are preserved as `EvidenceLabel`s in `analysis_goals` so reports and later scoring can inspect them without a database migration.

## EvidenceLabel Population

Each extracted label includes:

- `id`: stable label ID, for example `label:modality:neuropixels`
- `label`: human-readable label
- `label_type`: normalized label family
- `confidence`: `0.0` to `1.0`
- `evidence_text`: matching snippet or source value
- `source_field`: field where the evidence came from
- `source_value`: original field value
- `extractor_name`
- `extractor_version`

## Confidence Scale

| Confidence | Meaning |
|------------|---------|
| `0.90-1.00` | Existing structured metadata or direct source field match. |
| `0.75-0.90` | Curated synonym or strong title phrase. |
| `0.60-0.80` | Free-text description, abstract, URL, or raw path evidence. |
| `0.35-0.60` | Weak inference; generally avoided for positive labels. |
| `<0.35` | Unsupported; not emitted as a positive label. |

Duplicate labels are merged by `label_type + id`, keeping the highest-confidence evidence.

## Examples

Title: `Two-photon imaging during go/no-go behavior`

Emits:

- `label:modality:calcium_imaging`
- `label:task:go_nogo`

Description: `Fiber photometry during reward delivery`

Emits:

- `label:modality:fiber_photometry`
- `label:behavioral_event:reward`

Paper abstract: `Neuropixels recordings during reward prediction error tasks`

Emits:

- `label:modality:neuropixels`
- `label:analysis_goal:q_learning_modeling`

## False-Positive Traps

The extractor deliberately does not infer:

- `reversal_learning` from orbitofrontal cortex alone.
- `q_learning_modeling` from reward alone.
- `motor_decoding` from motor cortex alone.
- `speech_decoding` from auditory cortex alone.
- Behavioral task labels from disease keywords alone.
- Continuous kinematics from movement unless pose, trajectory, position, or kinematic evidence exists.
- `neuropixels` from generic extracellular electrophysiology.
- `human` from `clinical` alone.
- `bci_decoding` from EEG alone.
- `fmri` from generic brain imaging.
- `sleep_staging` from resting state.

## Usage

Use `extract_scientific_labels(record)` for inspection, or `enrich_record_with_scientific_labels(record)` to return a copy of a normalized dataset or paper with labels attached.

Ingestion should run extraction after raw payload normalization and before corpus reports or search scoring. Search can then explain what matched, where the label came from, and how confident the extractor was.
