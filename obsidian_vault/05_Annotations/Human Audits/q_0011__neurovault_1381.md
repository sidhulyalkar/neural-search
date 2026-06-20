---
annotation_id: q_0011__neurovault_1381
audit_status: pending
confidence: 0.7
created: '2026-06-17'
dataset_id: neurovault:1381
judge_version: lf_v1
label: 2
query_id: q_0011
source: bronze
tags:
- annotation
- audit
type: annotation
---

## Query
**dopamine reward prediction error ventral striatum mesolimbic pathway rodent**

**Scientific goal:** Find datasets supporting comparison of dopamine signals across recording modalities and species.

## Hard Negatives
- fMRI BOLD signal mistaken for direct dopamine measurement
- striatum recordings without reward task

## Dataset
**Predicting local striatal reward signals from corticostriatal connectivity**

This collection contains the observed contrasts presented in the paper, contrast maps predicted by structural connectivity, and the Mean Absolute Error (MAE) between observed and predicted contrast. All maps were computed at the single-subject level, normalized to MNI space, and averaged. 

## Rule Votes
_No active votes_

## Ensemble
- Label: **2**, Confidence: 0.7

## Human Audit Checklist
- [ ] Reviewed query intent and hard negatives
- [ ] Checked dataset modality / species
- [ ] Verified or corrected label in frontmatter
- [ ] Set `audit_status: done` in frontmatter

> **Edit in frontmatter:** `label`, `confidence`, `audit_status`
