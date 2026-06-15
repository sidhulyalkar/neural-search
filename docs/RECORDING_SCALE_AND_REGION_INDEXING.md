# Recording Scale and Region Indexing

Neural Search separates broad modality from sampling scale.

- `modalities`: acquisition family, such as `extracellular_ephys`, `neuropixels`, `calcium_imaging`, `fmri`, or `eeg`.
- `recording_scales`: how brain information is sampled, such as `single_unit_spikes`, `local_field_potential`, `raw_extracellular_voltage`, `calcium_roi_fluorescence`, or `bold_voxel_timeseries`.
- `brain_regions`: anatomical target labels from `data/ontology/brain_regions.yaml`.
- region index categories: derived labels from `neural_search.corpus.brain_region_index`, including `brain_system:*`, `parent_region:*`, `child_region:*`, `species_scope:*`, and `atlas:*`.

This distinction matters for reuse. Two datasets can both be electrophysiology while supporting different analyses: single-unit spikes support spike-train decoding and spike sorting; LFP supports oscillation, phase, and spectral analyses; raw voltage supports preprocessing and spike sorting; ECoG/SEEG support field-potential and high-gamma analyses at human intracranial electrode scales.

Dataset relationships should compare:

- task and behavior overlap;
- species and brain-region overlap;
- modality compatibility;
- recording-scale overlap;
- raw versus processed signal availability;
- analysis affordance requirements.

The recording-scale ontology is in `data/ontology/recording_scales.yaml`. It is loaded and matched by `neural_search.ontology`, extracted into dataset cards, exposed through search parsing, added to graph nodes through `dataset_has_recording_scale`, and shown in the frontend result filters.
