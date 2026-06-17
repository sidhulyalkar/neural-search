---
annotation_id: q_0007__osf_ucda2
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: osf:ucda2
judge_version: lf_v1
label: 0
query_id: q_0007
source: bronze
tags:
- annotation
- audit
type: annotation
---

## Query
**human fMRI N-back working memory task prefrontal cortex BOLD signal**

**Scientific goal:** Find datasets suitable for reproducing a known N-back working memory fMRI study.

## Hard Negatives
- resting-state fMRI with no working memory task
- EEG N-back without fMRI modality

## Dataset
**A single graph quantity predicts working-memory fragility across circuits and species in the Drosophila connectome**

A previous study showed that the antennal lobe is the unique Drosophila brain circuit whose connectome topology is necessary for an abstract trace-conditioning working-memory task. Here we identify the static graph quantity that explains and generalises this differential fragility: the Newman-Girvan modularity that survives degree-preserving randomization (Q_shuf). Across the six FlyEM hemibrain circuits, Q_shuf predicts the shuffle-induced learning slowdown at Spearman rho = 1.00 (n = 6). Within each circuit, gradual interpolation between the real and the shuffled graph confirms monotonic fragility scaling, with a sharp threshold near Q_shuf ~ 0.17. The same prediction holds at three times the delay length (Pearson r = -0.99). The protocol also generalises across species: applied to the Drosophila larva connectome (Winding et al. 2023, Science), it pools to Pearson r = -0.93 (n = 9). The motor circuit reverses outcome between adult and larva — same anatomy, different topology, opposite shuffle response.

## Rule Votes
_No active votes_

## Ensemble
- Label: **0**, Confidence: 0.95

## Human Audit Checklist
- [ ] Reviewed query intent and hard negatives
- [ ] Checked dataset modality / species
- [ ] Verified or corrected label in frontmatter
- [ ] Set `audit_status: done` in frontmatter

> **Edit in frontmatter:** `label`, `confidence`, `audit_status`
