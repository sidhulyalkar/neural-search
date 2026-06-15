# Coverage Ledger Gap Report

- Snapshot: `full_corpus_v09`
- Generated: 2026-06-13T20:00:25.711722+00:00
- Datasets: 7171
- Coverage entries: 61098

## Executive Review

- Structured species coverage: 72.8%
- Structured brain-region coverage: 48.1%
- Structured modality coverage: 81.9%
- Recording-scale coverage after conservative backfill: 81.6%
- State coverage: species=100.0%, brain_regions=100.0%, modalities=100.0%, recording_scales=100.0%, tasks=100.0%, behavioral_events=100.0%
- Unknown state slots still needing source/file/paper enrichment: 10576
- Recording-scale coverage is mostly inferred from modality and text; treat it as sortable silver metadata until file-level validation.
- Behavioral-event coverage remains sparse; HED/event-table enrichment should be a priority.

## Dimension Coverage

| Dimension | Value Coverage | State Coverage | Actionable State | Entries | Missing Values | Unknown State | N/A | Review Needed | Inferred Entries | Open Access Entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| species | 72.8% | 100.0% | 72.8% | 5393 | 1950 | 1950 | 0 | 39 | 0 | 5373 |
| brain_regions | 48.1% | 100.0% | 48.1% | 7104 | 3721 | 3721 | 0 | 152 | 0 | 7086 |
| modalities | 81.9% | 100.0% | 81.9% | 8083 | 1295 | 1295 | 0 | 78 | 0 | 8061 |
| recording_scales | 81.6% | 100.0% | 81.6% | 6729 | 1317 | 1317 | 0 | 0 | 6729 | 6718 |
| tasks | 23.3% | 100.0% | 88.5% | 2423 | 5499 | 821 | 4612 | 31 | 0 | 2408 |
| behavioral_events | 1.7% | 100.0% | 79.8% | 138 | 7053 | 1452 | 5443 | 7 | 0 | 121 |
| analysis_levels | 100.0% | 100.0% | 100.0% | 24057 | 0 | 0 | 0 | 0 | 24057 | 23989 |
| access_tiers | 100.0% | 100.0% | 99.7% | 7171 | 0 | 20 | 0 | 20 | 7171 | 7151 |

## Coverage States

| Dimension | Observed | File-Derived | Source Default | Inferred Silver | Not Applicable | Restricted | Unknown |
|---|---:|---:|---:|---:|---:|---:|---:|
| species | 5105 | 77 | 0 | 39 | 0 | 0 | 1950 |
| brain_regions | 3363 | 0 | 0 | 87 | 0 | 0 | 3721 |
| modalities | 5798 | 18 | 0 | 60 | 0 | 0 | 1295 |
| recording_scales | 0 | 0 | 5785 | 69 | 0 | 0 | 1317 |
| tasks | 1642 | 9 | 66 | 21 | 4612 | 0 | 821 |
| behavioral_events | 67 | 45 | 158 | 6 | 5443 | 0 | 1452 |
| analysis_levels | 0 | 0 | 7171 | 0 | 0 | 0 | 0 |
| access_tiers | 0 | 0 | 7151 | 0 | 0 | 0 | 20 |

## Evidence Tiers

- inferred_metadata: 37957
- declared_metadata: 22666
- silver_inferred: 307
- structured_metadata: 168

## Access Tiers

- open_access: 7151
- unknown_access: 20

## Analysis Levels

- raw_signal: 5161
- standardized_container: 4877
- behavior_correlation: 4637
- processed_derivative: 3975
- voxelwise_imaging: 2479
- circuit_structure: 1368
- population_dynamics: 870
- mesoscale_field_potential: 427
- spike_sorted: 114
- unit_activity: 114
- molecular_profile: 35

## Source Gap Hotspots

| Source | Datasets | Species | Regions | Modalities | Scales | Tasks | Events |
|---|---:|---:|---:|---:|---:|---:|---:|
| neurovault | 2000 | 100.0% | 12.6% | 100.0% | 100.0% | 8.6% | 0.0% |
| dandi | 848 | 53.9% | 59.7% | 61.1% | 59.8% | 24.9% | 4.5% |
| neuromorpho | 1000 | 87.5% | 68.3% | 100.0% | 100.0% | 0.0% | 0.0% |
| harvard_dataverse | 500 | 32.2% | 76.6% | 39.4% | 36.6% | 28.4% | 0.0% |
| gin | 408 | 35.8% | 27.5% | 42.2% | 44.1% | 22.3% | 0.0% |
| zenodo | 500 | 52.6% | 51.6% | 61.2% | 61.4% | 32.6% | 0.0% |
| openneuro | 299 | 26.4% | 20.1% | 99.3% | 99.3% | 18.4% | 23.1% |
| figshare | 200 | 21.5% | 100.0% | 32.0% | 33.0% | 11.5% | 0.0% |
| osf | 200 | 29.0% | 30.0% | 54.5% | 56.0% | 33.0% | 0.0% |
| brain_image_library | 200 | 100.0% | 1.0% | 100.0% | 100.0% | 0.5% | 0.0% |
| allen | 500 | 100.0% | 99.6% | 100.0% | 100.0% | 98.4% | 0.0% |
| crcns | 153 | 70.6% | 89.5% | 99.4% | 97.4% | 32.0% | 0.0% |
| bluebrain | 100 | 68.0% | 47.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| ibl | 198 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| buzsaki | 35 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |

## Source State Hotspots

| Source | Datasets | Actionable State | Unknown Total | Species Unknown | Regions Unknown | Modalities Unknown | Scales Unknown | Tasks Unknown | Events Unknown | N/A Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neurovault | 2000 | 89.1% | 1749 | 0 | 1749 | 0 | 0 | 0 | 0 | 3828 |
| zenodo | 500 | 57.5% | 1701 | 237 | 242 | 194 | 193 | 336 | 499 | 2 |
| dandi | 848 | 77.8% | 1506 | 391 | 342 | 330 | 341 | 102 | 0 | 1208 |
| harvard_dataverse | 500 | 62.5% | 1499 | 339 | 117 | 303 | 317 | 164 | 259 | 435 |
| gin | 408 | 64.4% | 1163 | 262 | 296 | 236 | 228 | 58 | 83 | 584 |
| osf | 200 | 59.8% | 644 | 142 | 140 | 91 | 88 | 69 | 114 | 151 |
| figshare | 200 | 69.4% | 489 | 157 | 0 | 136 | 134 | 27 | 35 | 315 |
| openneuro | 299 | 80.6% | 463 | 220 | 239 | 2 | 2 | 0 | 0 | 387 |
| neuromorpho | 1000 | 94.5% | 442 | 125 | 317 | 0 | 0 | 0 | 0 | 2000 |
| brain_image_library | 200 | 87.6% | 198 | 0 | 198 | 0 | 0 | 0 | 0 | 399 |
| ibl | 198 | 87.5% | 198 | 0 | 0 | 0 | 0 | 0 | 198 | 0 |
| allen | 500 | 95.1% | 195 | 0 | 2 | 0 | 0 | 0 | 193 | 315 |
| crcns | 153 | 90.1% | 121 | 45 | 16 | 1 | 4 | 22 | 33 | 202 |
| bluebrain | 100 | 89.4% | 85 | 32 | 53 | 0 | 0 | 0 | 0 | 200 |
| buzsaki | 35 | 75.0% | 70 | 0 | 0 | 0 | 0 | 35 | 35 | 0 |

## Completion Worklist Summary

| Source | Dimension | Items | Max Priority | Recommended Action |
|---|---|---:|---:|---|
| neurovault | brain_regions | 1749 | 100 | reverse-map image masks/coordinates to atlas regions and mine collection cognitive atlas tags |
| zenodo | behavioral_events | 499 | 85 | resolve behavioral events from BIDS events.tsv/HED, NWB trials/epochs/stimulus tables, or protocol text |
| dandi | species | 391 | 100 | query DANDI dandiset assets and NWB subject/specimen metadata |
| dandi | brain_regions | 342 | 100 | inspect NWB electrodes/optical physiology imaging_plane locations and DANDI metadata |
| dandi | recording_scales | 341 | 96 | inspect NWB acquisition objects, units tables, electrodes, imaging planes, and processing modules |
| harvard_dataverse | species | 339 | 98 | resolve species from source specimen metadata, participants.tsv, or linked publication organism terms |
| zenodo | tasks | 336 | 83 | resolve task labels from task-* filenames, NWB trials, stimulus tables, protocols, or linked publications |
| dandi | modalities | 330 | 98 | inspect NWB acquisition/processing groups and dandiset measurementTechnique metadata |
| neuromorpho | brain_regions | 317 | 100 | resolve brain region from NeuroMorpho archive metadata and normalize to atlas IDs |
| harvard_dataverse | recording_scales | 317 | 91 | resolve recording scale from NWB/BIDS/file contents or modality-specific sampling defaults |
| harvard_dataverse | modalities | 303 | 93 | resolve modality from file inventory, source technique metadata, or dataset methods text |
| gin | brain_regions | 296 | 100 | resolve regions from atlas IDs, electrode coordinates, specimen metadata, masks, or linked paper methods |
| gin | species | 262 | 98 | resolve species from source specimen metadata, participants.tsv, or linked publication organism terms |
| harvard_dataverse | behavioral_events | 259 | 85 | resolve behavioral events from BIDS events.tsv/HED, NWB trials/epochs/stimulus tables, or protocol text |
| zenodo | brain_regions | 242 | 100 | resolve regions from atlas IDs, electrode coordinates, specimen metadata, masks, or linked paper methods |
| openneuro | brain_regions | 239 | 100 | parse BIDS derivatives, masks, coordsystem/electrodes files, and linked paper region terms |
| zenodo | species | 237 | 98 | resolve species from source specimen metadata, participants.tsv, or linked publication organism terms |
| gin | modalities | 236 | 93 | resolve modality from file inventory, source technique metadata, or dataset methods text |
| gin | recording_scales | 228 | 91 | resolve recording scale from NWB/BIDS/file contents or modality-specific sampling defaults |
| openneuro | species | 220 | 100 | parse BIDS participants.tsv, samples.tsv, dataset_description.json, and linked publication metadata |

## Top Covered Values

- species: human (2646), mouse (1962), rat (477), macaque (152), zebrafish (60), drosophila (43), marmoset (11), cat (11), c_elegans (9), other (6)
- brain_regions: visual_cortex (1271), hippocampus (806), prefrontal_cortex (475), v1 (430), striatum (393), thalamus (371), brainstem (292), v2 (288), parietal_cortex (222), motor_cortex (201)
- modalities: fmri (2473), extracellular_ephys (1044), neuron_morphology (1000), calcium_imaging (849), two_photon_calcium_imaging (434), neuropixels (309), eeg (249), microscopy (216), behavior (197), population_imaging (192)
- recording_scales: bold_voxel_timeseries (2473), connectomic_edge (1368), raw_extracellular_voltage (1122), calcium_roi_fluorescence (849), eeg_sensor_timeseries (256), intracellular_membrane_signal (194), local_field_potential (159), single_unit_spikes (99), ecog_surface_potential (44), meg_sensor_timeseries (34)
- tasks: visual_stimulation (492), two_alternative_forced_choice (201), decision_making (196), auditory_processing (169), resting_state (154), change_detection (121), locomotion (96), passive_viewing (72), sleep_wake (67), reaching (65)

## Thin Species-Region Pairs

- human × periaqueductal_gray: observed=1/3, opportunity=2655
- human × piriform_cortex: observed=1/3, opportunity=2652
- human × cerebellar_cortex: observed=1/3, opportunity=2648
- human × s1: observed=1/3, opportunity=2648
- human × anterior_thalamic_nuclei: observed=1/3, opportunity=2647
- human × cerebellar_vermis: observed=1/3, opportunity=2647
- human × globus_pallidus_internal: observed=1/3, opportunity=2647
- human × sts: observed=1/3, opportunity=2647
- human × bnst: observed=1/3, opportunity=2646
- human × cervical_spinal_cord: observed=1/3, opportunity=2646
- human × fusiform_face_area: observed=1/3, opportunity=2646
- human × globus_pallidus_external: observed=1/3, opportunity=2646
- human × gustatory_cortex: observed=1/3, opportunity=2646
- human × medial_septum: observed=1/3, opportunity=2646
- human × mgb: observed=1/3, opportunity=2646
- human × perirhinal_cortex: observed=1/3, opportunity=2646
- human × pulvinar: observed=1/3, opportunity=2646
- human × seizure_focus: observed=1/3, opportunity=2646
- human × sensory_cortex: observed=1/3, opportunity=2646
- human × suprachiasmatic_nucleus: observed=1/3, opportunity=2646

## Thin Species-Modality Pairs

- human × chip_seq: observed=1/3, opportunity=2646
- human × diffusion_mri: observed=1/3, opportunity=2646
- human × methylation: observed=1/3, opportunity=2646
- human × t2w: observed=1/3, opportunity=2646
- mouse × ieeg: observed=1/3, opportunity=1981
- mouse × morphology: observed=1/3, opportunity=1968
- mouse × bulk_rnaseq: observed=1/3, opportunity=1966
- mouse × connectivity: observed=1/3, opportunity=1962
- mouse × methylation: observed=1/3, opportunity=1962
- mouse × viral_tracing: observed=1/3, opportunity=1962
- cat × extracellular_ephys: observed=1/3, opportunity=1054
- aplysia × extracellular_ephys: observed=1/3, opportunity=1044
- primate × extracellular_ephys: observed=1/3, opportunity=1044
- rat × audio: observed=1/3, opportunity=515
- rat × meg: observed=1/3, opportunity=510
- rat × ieeg: observed=1/3, opportunity=496
- drosophila × patch_clamp: observed=1/3, opportunity=224
- clonal_raider_ant × microscopy: observed=1/3, opportunity=216
- macaque × ecog: observed=1/3, opportunity=176
- macaque × eye_tracking: observed=1/3, opportunity=166

## Thin Region-Modality Pairs

- olfactory_bulb × fmri: observed=1/3, opportunity=2522
- basolateral_amygdala × fmri: observed=1/3, opportunity=2493
- pmd × fmri: observed=1/3, opportunity=2487
- ca3 × fmri: observed=1/3, opportunity=2483
- subiculum × fmri: observed=1/3, opportunity=2480
- central_amygdala × fmri: observed=1/3, opportunity=2478
- globus_pallidus × fmri: observed=1/3, opportunity=2478
- substantia_nigra_pars_compacta × fmri: observed=1/3, opportunity=2478
- medial_geniculate_nucleus × fmri: observed=1/3, opportunity=2477
- s1 × fmri: observed=1/3, opportunity=2475
- cerebellar_vermis × fmri: observed=1/3, opportunity=2474
- globus_pallidus_internal × fmri: observed=1/3, opportunity=2474
- medulla × fmri: observed=1/3, opportunity=2474
- sts × fmri: observed=1/3, opportunity=2474
- bnst × fmri: observed=1/3, opportunity=2473
- fusiform_face_area × fmri: observed=1/3, opportunity=2473
- globus_pallidus_external × fmri: observed=1/3, opportunity=2473
- gustatory_cortex × fmri: observed=1/3, opportunity=2473
- mgb × fmri: observed=1/3, opportunity=2473
- perirhinal_cortex × fmri: observed=1/3, opportunity=2473

## Largest Dark Species-Region Pairs

These are candidate acquisition/enrichment gaps based on marginal coverage; they are not yet filtered by species-specific anatomical validity.

- human × medial_entorhinal_cortex: observed=0, opportunity=2657
- human × barrel_cortex: observed=0, opportunity=2655
- human × dorsolateral_striatum: observed=0, opportunity=2652
- human × lateral_geniculate_nucleus: observed=0, opportunity=2652
- human × dorsomedial_striatum: observed=0, opportunity=2651
- human × arcuate_nucleus: observed=0, opportunity=2650
- human × dorsal_raphe: observed=0, opportunity=2650
- human × lateral_entorhinal_cortex: observed=0, opportunity=2649
- human × lateral_septum: observed=0, opportunity=2648
- human × dorsal_horn: observed=0, opportunity=2647
- human × frontal_eye_field: observed=0, opportunity=2647
- human × mst: observed=0, opportunity=2647
- human × nucleus_basalis: observed=0, opportunity=2647
- human × paraventricular_hypothalamic_nucleus: observed=0, opportunity=2647
- human × ventral_posteromedial_thalamus: observed=0, opportunity=2647
- mouse × insula: observed=0, opportunity=2006
- mouse × dlpfc: observed=0, opportunity=1988
- mouse × temporal_lobe: observed=0, opportunity=1986
- mouse × broca_area: observed=0, opportunity=1978
- mouse × pmd: observed=0, opportunity=1977

## Largest Dark Species-Modality Pairs

These pairs need curator review because some modality/species combinations are scientifically impossible or intentionally absent.

- human × two_photon_calcium_imaging: observed=0, opportunity=3080
- human × microscopy: observed=0, opportunity=2862
- human × behavior: observed=0, opportunity=2843
- human × population_imaging: observed=0, opportunity=2838
- human × two_photon: observed=0, opportunity=2838
- human × extracellular_electrophysiology: observed=0, opportunity=2704
- human × merfish: observed=0, opportunity=2652
- human × channels: observed=0, opportunity=2651
- human × pet: observed=0, opportunity=2650
- human × multiome: observed=0, opportunity=2648
- human × confocal_microscopy: observed=0, opportunity=2647
- human × connectivity: observed=0, opportunity=2647
- human × coordsystem: observed=0, opportunity=2647
- human × viral_tracing: observed=0, opportunity=2647
- zebrafish × fmri: observed=0, opportunity=2533
- drosophila × fmri: observed=0, opportunity=2516
- cat × fmri: observed=0, opportunity=2484
- c_elegans × fmri: observed=0, opportunity=2482
- other × fmri: observed=0, opportunity=2479
- rabbit × fmri: observed=0, opportunity=2477

## Largest Dark Region-Modality Pairs

These mostly identify where atlas-level regions and modality families do not co-occur in the current corpus.

- retina × fmri: observed=0, opportunity=2560
- medial_entorhinal_cortex × fmri: observed=0, opportunity=2484
- septum × fmri: observed=0, opportunity=2484
- barrel_cortex × fmri: observed=0, opportunity=2482
- piriform_cortex × fmri: observed=0, opportunity=2480
- dorsolateral_striatum × fmri: observed=0, opportunity=2479
- lateral_geniculate_nucleus × fmri: observed=0, opportunity=2479
- dorsomedial_striatum × fmri: observed=0, opportunity=2478
- arcuate_nucleus × fmri: observed=0, opportunity=2477
- dorsal_raphe × fmri: observed=0, opportunity=2477
- cerebellar_cortex × fmri: observed=0, opportunity=2476
- lateral_entorhinal_cortex × fmri: observed=0, opportunity=2476
- somatosensory_area_2 × fmri: observed=0, opportunity=2476
- anterior_thalamic_nuclei × fmri: observed=0, opportunity=2475
- lateral_septum × fmri: observed=0, opportunity=2475
- cervical_spinal_cord × fmri: observed=0, opportunity=2474
- dorsal_horn × fmri: observed=0, opportunity=2474
- frontal_eye_field × fmri: observed=0, opportunity=2474
- medial_septum × fmri: observed=0, opportunity=2474
- mst × fmri: observed=0, opportunity=2474

## Recommendations

- Resolve unknown brain_regions states: 3721/7171 dataset slots still need source/file/paper enrichment.
- Resolve unknown species states: 1950/7171 dataset slots still need source/file/paper enrichment.
- Resolve unknown behavioral_events states: 1452/7171 dataset slots still need source/file/paper enrichment.
- Resolve unknown recording_scales states: 1317/7171 dataset slots still need source/file/paper enrichment.
- Resolve unknown modalities states: 1295/7171 dataset slots still need source/file/paper enrichment.
- Resolve unknown tasks states: 821/7171 dataset slots still need source/file/paper enrichment.
- Normalize licenses/access metadata so open-access gaps are actionable.
- Backfill behavioral_events for `dandi` (810/848 missing).
- Backfill behavioral_events for `harvard_dataverse` (500/500 missing).
- Backfill behavioral_events for `gin` (408/408 missing).
- Review brain_regions silver/low-confidence labels before treating coverage as gold.
- Review modalities silver/low-confidence labels before treating coverage as gold.
- Review species silver/low-confidence labels before treating coverage as gold.
- Review tasks silver/low-confidence labels before treating coverage as gold.
- Review access_tiers silver/low-confidence labels before treating coverage as gold.
- Review behavioral_events silver/low-confidence labels before treating coverage as gold.
