# Neuro-Judge Preflight Report

> PRELIMINARY NEURO-JUDGE EVALUATION — RAG-GROUNDED LLM LABELS, NOT PURE HUMAN GOLD

## Summary

| Metric | Value |
|--------|-------|
| Total packets | 675 |
| Unique queries | 15 |
| Min packets/query | 37 |
| Max packets/query | 56 |
| Avg packets/query | 45 |
| Missing description | 109 (16%) |
| Missing title | 0 |
| Missing source_url | 0 |
| Missing intent | 0 |
| Empty hard_negatives | 0 |
| Missing modalities | 130 |
| Missing species | 185 |
| Suitable for real judge (has desc) | 566 (83%) |

## Stop-Condition Check

⚠ **STOP CONDITION MET**: 16.1% packets missing descriptions (threshold: 10%).
**Action**: Real LLM judge restricted to the {n_good} packets with descriptions. Mock backend will run on all 675.

## Source Distribution

| Source | Count |
|--------|-------|
| zenodo | 181 |
| openneuro | 147 |
| allen | 59 |
| dandi | 56 |
| neurovault | 52 |
| crcns | 51 |
| figshare | 41 |
| osf | 37 |
| ibl | 25 |
| gin | 24 |
| neuromorpho | 2 |

## Modality Distribution

| Modality | Count |
|----------|-------|
| fmri | 206 |
| extracellular_electrophysiology | 109 |
| mri | 106 |
| calcium_imaging | 76 |
| eeg | 66 |
| neuropixels | 62 |
| extracellular_ephys | 48 |
| fiber_photometry | 30 |
| two_photon_calcium_imaging | 26 |
| behavior | 24 |
| lfp | 22 |
| meg | 15 |
| pose_tracking | 13 |
| ieeg | 10 |
| emg | 9 |

## Intent Distribution

| Intent | Pairs |
|--------|-------|
| METHOD_TRANSFER | 97 |
| PIPELINE_REUSE | 93 |
| REPLICATION | 92 |
| CROSS_DATASET_COMPARISON | 90 |
| META_ANALYSIS | 87 |
| REANALYSIS_FEASIBILITY | 87 |
| MODEL_VALIDATION | 86 |
| EXPLORATION | 43 |

## Top 20 Missing Metadata Problems

- missing_brain_regions: 321
- missing_tasks: 210
- missing_species: 185
- missing_modalities: 130
- missing_description: 109

## Data Quality Notes

- 109/675 records have no description in either the pooled candidates or the enrichment corpus.
- These records are Zenodo/figshare/osf datasets that lack text descriptions.
- They still have modalities, species, and source metadata in many cases.
- The judge prompt will indicate missing description; confidence will be penalised.
- Real judge will still run but confidence expected to be lower for these.
