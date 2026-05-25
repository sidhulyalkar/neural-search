# Query Plan Evaluation

- Queries: 6
- Corpus: data/corpus/normalized/search_intelligence_task23.datasets.jsonl (6 records)
- Promotion safe: true
- Mean hit@5 delta: 0.0
- Mean MRR delta: 0.1111
- Hard-negative violation delta: 0.0

## By Intent

| Intent | Queries | Hit@5 Delta | MRR Delta | Hard Neg Delta | Promotion Safe |
|---|---:|---:|---:|---:|---|
| analysis_affordance | 1 | 0.0 | 0.6667 | 0 | true |
| cross_modal | 1 | 0.0 | 0.0 | 0 | true |
| data_form_search | 1 | 0.0 | 0.0 | 0 | true |
| hard_negative | 3 | 0.0 | 0.0 | 0 | true |

## By Data Form

| Data Form | Queries | Hit@5 Delta | MRR Delta | Hard Neg Delta | Promotion Safe |
|---|---:|---:|---:|---:|---|
| behavior | 2 | 0.0 | 0.0 | 0 | true |
| computational_model | 1 | 0.0 | 0.0 | 0 | true |
| connectomics | 1 | 0.0 | 0.0 | 0 | true |
| eeg_meg | 1 | 0.0 | 0.0 | 0 | true |
| intracellular_ephys | 1 | 0.0 | 0.0 | 0 | true |
| molecular | 1 | 0.0 | 0.6667 | 0 | true |
| mri | 1 | 0.0 | 0.0 | 0 | true |

## Queries

- `task23_connectomics_mapping` hard_negative/constraint_filter_first: hit@5 delta 0.0, MRR delta 0.0, hard-neg delta 0.0
- `task23_single_cell_mapping` analysis_affordance/analysis_readiness: hit@5 delta 0.0, MRR delta 0.6667, hard-neg delta 0.0
- `task23_patch_clamp_excitability` hard_negative/constraint_filter_first: hit@5 delta 0.0, MRR delta 0.0, hard-neg delta 0.0
- `task23_computational_model` data_form_search/balanced: hit@5 delta 0.0, MRR delta 0.0, hard-neg delta 0.0
- `task23_meg_time_frequency` cross_modal/cross_modal_fit: hit@5 delta 0.0, MRR delta 0.0, hard-neg delta 0.0
- `task23_fmri_connectivity_no_eeg` hard_negative/constraint_filter_first: hit@5 delta 0.0, MRR delta 0.0, hard-neg delta 0.0
