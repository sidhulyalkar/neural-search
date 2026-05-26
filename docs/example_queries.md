# Example Queries

Use these queries for demos, screenshots, benchmark discussion, and manual QA.

## Public Demo Queries

- `Find reversal learning datasets with reward omission and trial outcomes`
- `Go/NoGo task with calcium imaging in mPFC and lick events`
- `Visual decision-making with Neuropixels recordings`
- `Find datasets where I can decode choice from neural activity`
- `Human ECoG or iEEG reaching data for BCI classification`
- `Delay discounting with fiber photometry and reward choice`

## Structured Query Combinations

| Free text | Structured filters |
| --- | --- |
| `choice decoding from neural activity` | behavior: `choice`, analysis goal: `choice_decoding`, readiness >= 70 |
| `reward omission during reversal learning` | task: `reversal_learning`, behavior: `omission`, species: `mouse` |
| `human BCI reaching data` | task: `reaching`, modality: `ecog` or `ieeg`, species: `human` |
| `visual decisions with high-density electrophysiology` | task: `visual_decision_making`, modality: `neuropixels` |

## What Each Query Shows

- Reversal learning: ontology and behavior matching.
- Go/NoGo calcium imaging: task, modality, brain-region, and event matching.
- Visual decision-making Neuropixels: modality synonym and task matching.
- Choice decoding: analysis-intent search beyond literal dataset title.
- ECoG/iEEG reaching: source metadata plus clinical/neural recording labels.
- Delay discounting: behavioral task plus reward-choice constraints.
