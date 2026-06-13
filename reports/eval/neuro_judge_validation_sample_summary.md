# Neuro-Judge Validation Sample Summary

> Validation sample only. These packets are for real LLM judging and human audit; they are not gold qrels.

- Requested n: `150`
- Selected n: `150`
- Seed: `42`
- Require diversity: `True`
- Include high-impact proxy: `False`
- Include missing evidence: `True`

## Label Counts

| Label | Count |
|---|---:|
| 0 | 24 |
| 1 | 35 |
| 2 | 69 |
| 3 | 22 |

## Selection Reasons

| Reason | Count |
|---|---:|
| random_fill | 17 |
| label_ge_2_missing_evidence | 7 |
| label_ge_2_missing_information | 7 |
| near_threshold_confidence | 7 |
| raw_data_required | 7 |
| high_ranked_candidate | 7 |
| low_ranked_candidate | 7 |
| missing_description | 7 |
| raw_data_absent_or_uncertain | 7 |
| abstain_recommended | 6 |
| missing_modality | 6 |
| missing_species | 6 |
| warning_heavy | 2 |
| diversity:abstain_recommended | 1 |
| diversity:intent:CROSS_DATASET_COMPARISON | 1 |
| diversity:intent:EXPLORATION | 1 |
| diversity:intent:METHOD_TRANSFER | 1 |
| diversity:intent:MODEL_VALIDATION | 1 |
| diversity:intent:PIPELINE_REUSE | 1 |
| diversity:intent:REANALYSIS_FEASIBILITY | 1 |
| diversity:intent:REPLICATION | 1 |
| diversity:label:0 | 1 |
| diversity:label:1 | 1 |
| diversity:label:2 | 1 |
| diversity:label:3 | 1 |
| diversity:label_ge_2_missing_evidence | 1 |
| diversity:label_ge_2_missing_information | 1 |
| diversity:low_ranked_candidate | 1 |
| diversity:missing_description | 1 |
| diversity:missing_modality | 1 |
| diversity:missing_species | 1 |
| diversity:modality:beh | 1 |
| diversity:modality:behavior | 1 |
| diversity:modality:behavior_video | 1 |
| diversity:modality:bold | 1 |
| diversity:modality:calcium_imaging | 1 |
| diversity:modality:ecog | 1 |
| diversity:modality:eeg | 1 |
| diversity:modality:extracellular_ephys | 1 |
| diversity:modality:fiber_photometry | 1 |
| diversity:modality:ieeg | 1 |
| diversity:modality:lfp | 1 |
| diversity:modality:meg | 1 |
| diversity:modality:mri | 1 |
| diversity:modality:neuropixels | 1 |
| diversity:modality:patch_clamp | 1 |
| diversity:modality:pet | 1 |
| diversity:modality:pose_tracking | 1 |
| diversity:modality:single_cell_atacseq | 1 |
| diversity:modality:spatial_transcriptomics | 1 |
| diversity:modality:two_photon_calcium_imaging | 1 |
| diversity:modality:visium | 1 |
| diversity:raw_data_absent_or_uncertain | 1 |
| diversity:source:allen | 1 |
| diversity:source:crcns | 1 |
| diversity:source:dandi | 1 |
| diversity:source:figshare | 1 |
| diversity:source:gin | 1 |
| diversity:source:neurovault | 1 |
| diversity:source:openneuro | 1 |
| diversity:source:osf | 1 |
| diversity:species:drosophila | 1 |
| diversity:species:human | 1 |
| diversity:species:macaque | 1 |
| diversity:species:monkey | 1 |
| diversity:species:mouse | 1 |
| diversity:species:primate | 1 |
| diversity:species:rat | 1 |
| diversity:species:unknown_species | 1 |
| diversity:species:zebrafish | 1 |

## Category Coverage

| Category | Count |
|---|---:|
| raw_data_absent_or_uncertain | 150 |
| low_ranked_candidate | 132 |
| near_threshold_confidence | 115 |
| abstain_recommended | 69 |
| label:2 | 69 |
| label_ge_2_missing_evidence | 69 |
| label_ge_2_missing_information | 69 |
| missing_species | 51 |
| species:unknown_species | 51 |
| source:zenodo | 45 |
| modality:fmri | 40 |
| species:mouse | 37 |
| label:1 | 35 |
| species:human | 34 |
| modality:extracellular_electrophysiology | 31 |
| intent:META_ANALYSIS | 29 |
| intent:PIPELINE_REUSE | 29 |
| missing_description | 28 |
| label:0 | 24 |
| source:openneuro | 24 |
| missing_modality | 24 |
| modality:unknown_modality | 24 |
| raw_data_required | 23 |
| label:3 | 22 |
| intent:MODEL_VALIDATION | 22 |
| intent:REANALYSIS_FEASIBILITY | 20 |
| intent:CROSS_DATASET_COMPARISON | 19 |
| modality:neuropixels | 18 |
| modality:mri | 17 |
| species:rat | 17 |
| source:figshare | 14 |
| modality:eeg | 13 |
| source:allen | 13 |
| source:crcns | 13 |
| source:dandi | 12 |
| modality:extracellular_ephys | 12 |
| modality:calcium_imaging | 11 |
| intent:REPLICATION | 11 |
| modality:fiber_photometry | 10 |
| modality:behavior | 10 |
| source:ibl | 10 |
| intent:METHOD_TRANSFER | 10 |
| intent:EXPLORATION | 10 |
| source:neurovault | 9 |
| high_ranked_candidate | 8 |
| modality:meg | 6 |
| modality:lfp | 6 |
| species:monkey | 5 |
| species:macaque | 5 |
| source:osf | 5 |
| modality:two_photon_calcium_imaging | 5 |
| source:gin | 3 |
| modality:emg | 2 |
| modality:spatial_transcriptomics | 2 |
| warning_heavy | 2 |
| modality:neuron_morphology | 2 |
| source:neuromorpho | 2 |
| modality:behavior_video | 2 |
| modality:pose_tracking | 2 |
| modality:ieeg | 2 |
| species:primate | 1 |
| modality:bold | 1 |
| modality:events | 1 |
| modality:t1w | 1 |
| species:zebrafish | 1 |
| modality:patch_clamp | 1 |
| species:drosophila | 1 |
| modality:pet | 1 |
| modality:single_cell_atacseq | 1 |
| modality:ecog | 1 |
| modality:visium | 1 |
| modality:audio | 1 |
| modality:beh | 1 |
