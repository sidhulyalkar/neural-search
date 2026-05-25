# Corpus and Knowledge Base Expansion Plan

- Dataset records: 3
- Paper records: 1
- Expansion tasks: 13

## Source Counts

- dandi: 2
- openalex: 1
- openneuro: 1

## Graph Coverage

### Node Types

- analysis_affordance: 13
- analysis_goal: 1
- author: 4
- behavioral_event: 3
- brain_region: 1
- data_standard: 3
- dataset: 3
- file_format: 1
- modality: 4
- paper: 1
- species: 5
- task: 2

### Edge Types

- dataset_has_behavioral_event: 4
- dataset_has_file_format: 3
- dataset_has_modality: 6
- dataset_has_species: 9
- dataset_has_task: 3
- dataset_records_region: 3
- dataset_supports_analysis: 33
- dataset_uses_standard: 6
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
| critical | Add missing source families to corpus intake | source_intake | modeldb, cellxgene, microns, computational model coverage, single-cell molecular coverage, connectomics coverage |

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
Computational model or simulation has 0/5 target corpus records and 0/3 target benchmark queries.
- normalized records include source, source_id, title, labels, and provenance
- graph artifact contains modality, analysis, and standard edges for new records
- benchmark seeds include reviewed expected IDs or explicit review_required notes

### task24_expand_connectomics
Connectomics and morphology has 0/5 target corpus records and 0/3 target benchmark queries.
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
Molecular and single-cell neuroscience has 0/5 target corpus records and 0/3 target benchmark queries.
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

### task26_add_source_families
A general neuroscience search corpus needs public-source coverage across physiology, imaging, literature, models, molecular data, and connectomics.
- each source family has fixture-backed normalized records
- network-backed ingestion remains optional outside CI
- source counts appear in expansion and corpus reports
