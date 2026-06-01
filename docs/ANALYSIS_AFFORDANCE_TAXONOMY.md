# Analysis Affordance Taxonomy

Analysis affordances estimate what a normalized dataset can support. They are conservative rule-based hints, not guarantees.

Implementation:

- Module: `neural_search/analysis_affordances.py`
- Detector name: `rule_based_analysis_affordance_detector`
- Detector version: `v0.3.0`

Each `AnalysisAffordance` includes:

- `analysis_id`
- `support_level`: `high`, `medium`, `low`, `unsupported`, or `unknown`
- `confidence`
- `required_fields_present`
- `helpful_fields_present`
- `missing_fields`
- `evidence`
- detector provenance

## Supported Affordances

| Affordance | Required Evidence | Conservative Traps |
|------------|-------------------|--------------------|
| `event_aligned_activity` | neural data plus event timestamps or event labels | neural data alone is low support |
| `trial_averaged_response` | trial structure plus neural data | task labels without trials are medium at best |
| `choice_decoding` | neural data plus choice or response labels | decision task without behavior is low |
| `motor_decoding` | neural data plus behavior tracking and movement/kinematic evidence | motor cortex alone is low |
| `speech_decoding` | ECoG/iEEG/ephys/EEG plus speech events or speech production task | auditory cortex alone is low |
| `q_learning_modeling` | choice, reward, and trial outcome labels | reward alone is low |
| `state_space_modeling` | neural time series, ideally with trials/events | metadata-only is low |
| `cross_modal_prediction` | neural data plus behavior tracking or stimulus labels | one modality is low |
| `brain_behavior_alignment` | neural data plus behavior | neural data alone is low |
| `seizure_detection` | seizure/epilepsy label plus neural modality | epilepsy paper metadata without neural data is medium at best |
| `sleep_stage_classification` | sleep-stage labels plus neural modality | resting state is not sleep staging |
| `functional_connectivity` | suitable neural modality plus connectivity labels or multiple regions | single-region neural data is low |
| `representational_similarity_analysis` | neural responses plus stimulus/task conditions | neural data alone is low |
| `encoding_modeling` | neural data plus stimulus or behavior covariates | neural data alone is low |
| `bci_decoding` | BCI context or motor/speech task plus neural data | EEG alone is low |
| `latent_dynamics_modeling` | neural time series plus trials/events | neural data without events is medium |

## Example Support Levels

High `q_learning_modeling`:

- behavioral events: `choice`, `reward`, `trial_outcome`

Low `q_learning_modeling`:

- behavioral events: `reward`
- missing: `choice`, `trial_outcome`

High `event_aligned_activity`:

- usability flags: `has_neural_data=true`, `has_event_timestamps=true`

Low `event_aligned_activity`:

- neural modality exists
- missing: `event_timestamps`

## Usage

Run `detect_analysis_affordances(record)` on a `NormalizedDatasetRecord`, or `enrich_record_with_affordances(record)` to attach results to the record.

The combined CLI can enrich a normalized corpus:

```bash
python -m neural_search.enrich_corpus --input data/corpus/normalized --out data/corpus/enriched
```

Future scoring can consume affordances for `analysis_fit_score`, and corpus reports can count available affordances across sources.
