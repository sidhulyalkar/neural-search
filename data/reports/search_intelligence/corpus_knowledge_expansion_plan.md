# Corpus and Knowledge Base Expansion Plan

- Dataset records: 6
- Paper records: 1
- Expansion tasks: 12

## Source Counts

- cellxgene: 1
- dandi: 2
- microns: 1
- modeldb: 1
- openalex: 1
- openneuro: 1

## Graph Coverage

### Node Types

- analysis_affordance: 36
- analysis_goal: 1
- author: 4
- behavioral_event: 7
- brain_region: 1
- data_standard: 17
- dataset: 6
- file_format: 1
- modality: 34
- paper: 1
- required_signal: 22
- species: 5
- task: 2

### Edge Types

- analysis_requires_behavioral_event: 23
- analysis_requires_modality: 99
- analysis_requires_task_structure: 159
- dataset_has_behavioral_event: 4
- dataset_has_file_format: 6
- dataset_has_modality: 9
- dataset_has_species: 13
- dataset_has_task: 6
- dataset_records_region: 6
- dataset_supports_analysis: 36
- dataset_uses_standard: 9
- paper_has_author: 4
- paper_mentions_analysis_goal: 1
- paper_mentions_behavioral_event: 1
- paper_mentions_dataset: 1
- paper_mentions_species: 1
- paper_studies_task: 1
- paper_uses_dataset: 1
- paper_uses_modality: 1

## Expansion Tasks

| Priority | Task | Track | Targets |
|---|---|---|---|
| critical | Expand Behavior and task events corpus and benchmark coverage | corpus_and_benchmark | DANDI, OpenNeuro, manual landmark corpus, behavior_video, pose_tracking, running_wheel, events, dataset_has_modality, dataset_supports_analysis, dataset_uses_standard |
| critical | Expand Clinical neuroscience corpus and benchmark coverage | corpus_and_benchmark | OpenNeuro, clinical BIDS/EDF repositories, clinical, polysomnography, eeg, mri, dataset_has_modality, dataset_supports_analysis, dataset_uses_standard |
| critical | Expand Computational model or simulation corpus and benchmark coverage | corpus_and_benchmark | ModelDB, Open Source Brain, model_output, simulation, json, hdf5, dataset_has_modality, dataset_supports_analysis, dataset_uses_standard |
| critical | Expand Connectomics and morphology corpus and benchmark coverage | corpus_and_benchmark | MICrONS, DANDI, NeuroMorpho, electron_microscopy, morphology, tracing, swc, dataset_has_modality, dataset_supports_analysis, dataset_uses_standard |
| critical | Expand EEG and MEG corpus and benchmark coverage | corpus_and_benchmark | OpenNeuro, clinical EDF/BIDS repositories, eeg, meg, polysomnography, bids, dataset_has_modality, dataset_supports_analysis, dataset_uses_standard |
| critical | Expand Extracellular electrophysiology corpus and benchmark coverage | corpus_and_benchmark | DANDI, IBL, Allen Brain Observatory, neuropixels, extracellular_ephys, lfp, nwb, dataset_has_modality, dataset_supports_analysis, dataset_uses_standard |
| critical | Expand Fiber photometry corpus and benchmark coverage | corpus_and_benchmark | DANDI, fiber_photometry, nwb, dandi, event_aligned_analysis, dataset_has_modality, dataset_supports_analysis, dataset_uses_standard |
| critical | Expand Intracellular electrophysiology corpus and benchmark coverage | corpus_and_benchmark | DANDI, NeuroElectro-style curated tables, patch_clamp, intracellular_ephys, nwb, cellular_physiology, dataset_has_modality, dataset_supports_analysis, dataset_uses_standard |
| critical | Expand Human intracranial electrophysiology corpus and benchmark coverage | corpus_and_benchmark | OpenNeuro iEEG, DANDI, ecog, ieeg, seeg, bids, dataset_has_modality, dataset_supports_analysis, dataset_uses_standard |
| critical | Expand Molecular and single-cell neuroscience corpus and benchmark coverage | corpus_and_benchmark | cellxgene, Allen Brain Cell Atlas, single_cell_rna, transcriptomics, proteomics, h5ad, dataset_has_modality, dataset_supports_analysis, dataset_uses_standard |
| critical | Expand MRI and fMRI corpus and benchmark coverage | corpus_and_benchmark | OpenNeuro, Human Connectome Project, fmri, mri, diffusion_mri, bids, dataset_has_modality, dataset_supports_analysis, dataset_uses_standard |
| critical | Expand Optical neural imaging corpus and benchmark coverage | corpus_and_benchmark | DANDI, Allen Brain Observatory, calcium_imaging, two_photon, widefield_imaging, nwb, dataset_has_modality, dataset_supports_analysis, dataset_uses_standard |

## Acceptance Checks

### task24_expand_behavior
Behavior and task events has 3/5 target corpus records and 1/3 target benchmark queries.
- normalized records include source, source_id, title, labels, and provenance
- graph artifact contains modality, analysis, and standard edges for new records
- benchmark seeds include reviewed expected IDs or explicit review_required notes

### task24_expand_clinical
Clinical neuroscience has 1/5 target corpus records and 1/3 target benchmark queries.
- normalized records include source, source_id, title, labels, and provenance
- graph artifact contains modality, analysis, and standard edges for new records
- benchmark seeds include reviewed expected IDs or explicit review_required notes

### task24_expand_computational_model
Computational model or simulation has 1/5 target corpus records and 1/3 target benchmark queries.
- normalized records include source, source_id, title, labels, and provenance
- graph artifact contains modality, analysis, and standard edges for new records
- benchmark seeds include reviewed expected IDs or explicit review_required notes

### task24_expand_connectomics
Connectomics and morphology has 1/5 target corpus records and 1/3 target benchmark queries.
- normalized records include source, source_id, title, labels, and provenance
- graph artifact contains modality, analysis, and standard edges for new records
- benchmark seeds include reviewed expected IDs or explicit review_required notes

### task24_expand_eeg_meg
EEG and MEG has 1/5 target corpus records and 1/3 target benchmark queries.
- normalized records include source, source_id, title, labels, and provenance
- graph artifact contains modality, analysis, and standard edges for new records
- benchmark seeds include reviewed expected IDs or explicit review_required notes

### task24_expand_extracellular_ephys
Extracellular electrophysiology has 2/5 target corpus records and 2/3 target benchmark queries.
- normalized records include source, source_id, title, labels, and provenance
- graph artifact contains modality, analysis, and standard edges for new records
- benchmark seeds include reviewed expected IDs or explicit review_required notes

### task24_expand_fiber_photometry
Fiber photometry has 0/5 target corpus records and 0/3 target benchmark queries.
- normalized records include source, source_id, title, labels, and provenance
- graph artifact contains modality, analysis, and standard edges for new records
- benchmark seeds include reviewed expected IDs or explicit review_required notes

### task24_expand_intracellular_ephys
Intracellular electrophysiology has 0/5 target corpus records and 0/3 target benchmark queries.
- normalized records include source, source_id, title, labels, and provenance
- graph artifact contains modality, analysis, and standard edges for new records
- benchmark seeds include reviewed expected IDs or explicit review_required notes

### task24_expand_intracranial_human_ephys
Human intracranial electrophysiology has 0/5 target corpus records and 0/3 target benchmark queries.
- normalized records include source, source_id, title, labels, and provenance
- graph artifact contains modality, analysis, and standard edges for new records
- benchmark seeds include reviewed expected IDs or explicit review_required notes

### task24_expand_molecular
Molecular and single-cell neuroscience has 1/5 target corpus records and 1/3 target benchmark queries.
- normalized records include source, source_id, title, labels, and provenance
- graph artifact contains modality, analysis, and standard edges for new records
- benchmark seeds include reviewed expected IDs or explicit review_required notes

### task24_expand_mri
MRI and fMRI has 0/5 target corpus records and 0/3 target benchmark queries.
- normalized records include source, source_id, title, labels, and provenance
- graph artifact contains modality, analysis, and standard edges for new records
- benchmark seeds include reviewed expected IDs or explicit review_required notes

### task24_expand_optical_imaging
Optical neural imaging has 0/5 target corpus records and 0/3 target benchmark queries.
- normalized records include source, source_id, title, labels, and provenance
- graph artifact contains modality, analysis, and standard edges for new records
- benchmark seeds include reviewed expected IDs or explicit review_required notes
