# Spectral Phenotype Knowledge-Graph Schema

`neural_search.spectral.kg.build_spectral_subgraph(bundle)` turns one
`SpectralFeatureBundle` into a `neural_search.graph.schema.KnowledgeGraph`
subgraph. This document is the reference for the node and edge types it
emits, defined in `neural_search/graph/schema.py`.

## Node types

| Node type | Represents | Key properties |
|---|---|---|
| `spectral_feature_bundle` | One reanalysis run's full output for a dataset | `overall_qc_status`, `n_estimates`, `eligibility_support_level`, `interpretation_cautions` |
| `spectral_run` | The method/settings provenance (`SpectralRunConfig`) shared by all estimates in a bundle | backend, backend_version, freq_range_hz, aperiodic_mode, peak settings |
| `spectral_estimate` | One aperiodic+periodic fit for one channel/region/task-state | `fit_r_squared`, `fit_error`, `n_frequency_bins` |
| `aperiodic_component` | The aperiodic part of one estimate | `offset`, `exponent`, `knee_hz`, `mode` |
| `periodic_peak` | One oscillatory peak fit on top of the aperiodic component | `center_frequency_hz`, `power`, `bandwidth_hz` |
| `spectral_qc_assessment` | The QC outcome for one estimate | `status` (`pass`/`warn`/`fail`), `flags`, `notes` |
| `task_state_epoch` | The behavioral/task state during which an estimate was measured | placeholder — resolve against the canonical `task`/`subject_state` graph |
| `channel`, `electrode`, `probe` | Recording hardware context for an estimate | placeholder — resolve against asset/device metadata when available |

`dataset` and `brain_region` nodes referenced by this subgraph are emitted
as **low-confidence placeholders** (`properties.placeholder = True`,
`confidence = 0.35`), matching the convention already used by
`neural_search.graph.builder` for cross-references it cannot fully resolve
on its own. Merge this subgraph with the dataset's primary subgraph via
`neural_search.graph.builder.merge_graphs([dataset_subgraph,
spectral_subgraph])` so placeholders resolve against the richer canonical
nodes.

## Edge types

| Edge type | From -> To | Meaning |
|---|---|---|
| `dataset_has_spectral_feature_bundle` | dataset -> spectral_feature_bundle | This dataset has this reanalysis run's output |
| `dataset_has_spectral_estimate` | dataset / spectral_feature_bundle -> spectral_estimate | Direct dataset-to-estimate shortcut (in addition to the bundle path) |
| `spectral_estimate_generated_by_run` | spectral_estimate -> spectral_run | Provenance: which method/settings produced this estimate |
| `spectral_estimate_has_aperiodic_component` | spectral_estimate -> aperiodic_component | |
| `spectral_estimate_has_periodic_peak` | spectral_estimate -> periodic_peak | |
| `spectral_estimate_has_qc_assessment` | spectral_estimate -> spectral_qc_assessment | |
| `spectral_estimate_measured_in_region` | spectral_estimate -> brain_region | |
| `spectral_estimate_measured_during_state` | spectral_estimate -> task_state_epoch | |
| `spectral_estimate_measured_from_channel` | spectral_estimate -> channel | |
| `aperiodic_component_estimated_by_method` | aperiodic_component -> spectral_run | |
| `periodic_peak_estimated_by_method` | periodic_peak -> spectral_run | |
| `dataset_reanalyzable_by_pipeline` | dataset -> spectral_feature_bundle | Emitted only when eligibility support_level is `high`/`medium` |
| `dataset_supports_aperiodic_reanalysis` | dataset -> spectral_feature_bundle | Emitted only when eligibility support_level is `high`/`medium` |
| `dataset_missing_aperiodic_requirement` | dataset -> spectral_feature_bundle | Emitted when `eligibility.missing_fields` is non-empty; `properties.missing_fields` lists them |

## Confidence semantics

* `spectral_feature_bundle` node confidence = `eligibility.confidence` (how
  sure we are this dataset *can* be analyzed this way).
* `aperiodic_component` node confidence = that estimate's `fit_r_squared`
  (how well the model fit the data) — **not** a claim about scientific
  interpretability.
* `periodic_peak` nodes use a flat 0.7 placeholder confidence; refine this
  once peak-level uncertainty estimates are available from a real
  specparam/FOOOF fit (`peak_params_` confidence is not currently exposed
  by that package).
* `spectral_qc_assessment` nodes always have confidence 1.0 — the QC
  *assessment itself* is deterministic, even when it reports `fail`.

## Example

```python
from neural_search.spectral.kg import build_spectral_subgraph
from neural_search.graph.builder import merge_graphs, build_dataset_subgraph

spectral_graph = build_spectral_subgraph(bundle)
dataset_graph = build_dataset_subgraph(normalized_record)
full_graph = merge_graphs([dataset_graph, spectral_graph])
```
