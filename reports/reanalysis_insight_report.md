# Reanalysis Insight Report

Synthesized by the `reanalysis-insight-synthesizer` agent (`artifacts/agents/playbooks/reanalysis_insight_synthesizer.md`) from the live production graph. Ranks existing reanalysis/reinterpretation signal instead of just counting it -- see `reports/reanalysis_candidates_report.md` and `reports/methodology_coverage_report.md` for the raw aggregate counts this builds on.

- `dataset_old_dataset_new_method_candidate` edges live: 59126
- `dataset_reanalysis_bridge_dataset` edges live: 2517
- `dataset_reinterpretation_candidate` edges live: 0

## Top 15 evidence-backed reuse opportunities (`dataset_reanalysis_bridge_dataset`, ranked by confidence)

A similar dataset was actually analyzed with the named method, per a real linked paper; the candidate dataset has no such evidence yet, per the corpus's own linked-paper coverage.

| Candidate dataset | Precedent dataset | Method | Confidence |
|---|---|---|---|
| NeuroMorpho: Acsady (13 neurons) | NeuroMorpho: Timofeev (11 neurons) | Eeg Analysis | 0.36 |
| NeuroMorpho: Althammer (1394 neurons) | NeuroMorpho: Timofeev (11 neurons) | Eeg Analysis | 0.36 |
| Zhang: Hippocampal-neocortical dialogue during memory consolidation | Neural coding in barrel cortex during whisker-guided locomotion | Calcium Imaging Analysis | 0.342 |
| cai-2: jRGECO1a and jRCaMP1a characterization in the intact mouse visual cortex, using AAV-based gene transfer, 2-photon imaging and loose-seal cell attached recordings, as described in Dana et al 2016 | Layer-Specific Physiological Features and Interlaminar Interactions in the Primary Visual Cortex of the Mouse | Coherence | 0.342 |
| cai-3: Simultaneous two-photon imaging and electrophysiological recordings from neurons in the mouse retina and primary visual cortex using OGB and GCamp6 | Layer-Specific Physiological Features and Interlaminar Interactions in the Primary Visual Cortex of the Mouse | Coherence | 0.342 |
| hc-26: Longitudinal LFP recordings from female apoE3-KI and apoE4-KI mice at rest and subsequent spatial memory performance | Neural coding in barrel cortex during whisker-guided locomotion | Calcium Imaging Analysis | 0.342 |
| ofc-4: fMRI time series in orbitofrontal cortex during face-house state-space task | The hierarchical organization of the lateral prefrontal cortex | Dcm Spectral | 0.342 |
| Human ECoG speaking consonant-vowel syllables | Decoding the Role of the Insula in Human Cognition: Functional Parcellation and Large-Scale Reverse Inference | Decoding | 0.342 |
| Human fNIRS recordings of motor cortex during finger-tapping task | Simultaneous EEG-fMRI Reveals Temporal Evolution of Coupling between Supramodal Cortical Attention Networks and the Brainstem | Dcm Spectral | 0.342 |
| Simultaneous electroencephalography, extracellular electrophysiology, and cortical electrical stimulation in head-fixed mice | Decoding the Role of the Insula in Human Cognition: Functional Parcellation and Large-Scale Reverse Inference | Decoding | 0.342 |
| Cellular Mechanisms of State-Dependent Processing in Visual Cortex (preliminary data) | Fork of Sleep spindles mediate hippocampal-neocortical coupling during long-duration ripples | Coherence | 0.342 |
| Dataset of human medial temporal lobe neurons, scalp and intracranial EEG during a verbal working memory task | Fork of Sleep spindles mediate hippocampal-neocortical coupling during long-duration ripples | Coherence | 0.342 |
| Neurovascular impulse response function (IRF) during spontaneous activity differentially reflects intrinsic neuromodulation across cortical regions | The hippocampus supports deliberation during value based decisions | Fmri Analysis | 0.342 |
| Neurovascular impulse response function (IRF) during spontaneous activity differentially reflects intrinsic neuromodulation across cortical regions | The hippocampus supports deliberation during value based decisions | Fmri Analysis | 0.342 |
| MRI and histological data for: Deep brain stimulation induces white matter remodeling and functional changes to brain-wide networks | The cerebellum is involved in processing of predictions and prediction errors in a fear conditioning paradigm | Fmri Analysis | 0.342 |

## Top 9 high-confidence, genuinely unexplored candidates (confidence >= 0.85, zero linked papers)

Heuristic (`dataset_old_dataset_new_method_candidate`), but the strongest current proxy for 'nobody has published anything on this dataset yet, and its profile strongly matches this technique's requirements.' Still `requires_human_review=True`.

| Dataset | Technique | Analysis family | Confidence |
|---|---|---|---|
| eeg-1: 64-channel human scalp EEG from 24 subjects examining spontaneous thought using experience sampling | fft | time_frequency | 0.9 |
| fcx-2: Intracranial EEG recordings of medial temporal, lateral frontal, and orbitofrontal regions in 10 human adults performing a visuospatial working memory task | fft | time_frequency | 0.9 |
| fcx-3: Intracranial EEG recordings of lateral frontal and parietal regions in 7 human adults performing a visuospatial working memory task | fft | time_frequency | 0.9 |
| Allen Cell Types Database - Electrophysiology | dcm | connectivity | 0.85 |
| Allen Mouse Brain Connectivity Atlas | dcm | connectivity | 0.85 |
| Allen Visual Coding Neuropixels: brain_observatory_1.1 session 715093703 | burst_analysis | spike_train_analysis | 0.85 |
| Allen Visual Coding Neuropixels: brain_observatory_1.1 session 719161530 | burst_analysis | spike_train_analysis | 0.85 |
| Allen Visual Coding Neuropixels: brain_observatory_1.1 session 721123822 | burst_analysis | spike_train_analysis | 0.85 |
| Experimental Data - Literature curated data - Bouton density - Human - Cortex - shapsonCoe data - ShapsonCoe data - random nonais pure axon selection 5000 per layer per type from agg20200916c3 pruned skeletons flat | dcm | connectivity | 0.85 |

## Why `dataset_reinterpretation_candidate` is still 0, precisely

Growing paper-dataset linkage from 403 to 2,510 real matches (5 sources) did **not** change this. Re-measured 2026-07-04: resolving DataCite/Crossref/PubMed matches to an OpenAlex ID by shared DOI finds only 8 additional usable matches (393 -> 401), and adds zero new `dataset_reanalysis_bridge_dataset` edges (2,517 either way). The bottleneck is not paper-dataset linkage breadth; it's that `artifacts/ner/ner_kg.jsonl`'s method-mention extraction has only ever run against OpenAlex-ingested paper text. Closing this needs NER extraction against Crossref/PubMed/DataCite paper records directly -- a real, scoped, not-yet-attempted next step, not a re-run of existing linkers.

## Cross-reference: methodology registry gaps limiting candidate generation

See `reports/methodology_coverage_report.md` for the full list. As of the last coverage run: 10/27 analysis families have no technique mapping yet (including `biomarker_discovery`, `cell_type_mapping`, `phenotyping`, `spatial_mapping`), and `intracellular_ephys`/`molecular` data forms have zero candidate-eligible analysis families at all -- any dataset in those data forms cannot generate a candidate edge regardless of its actual analysis potential, until the registry is extended.
