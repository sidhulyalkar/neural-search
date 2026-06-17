---
annotation_id: q_0007__crcns_pfc-5
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: crcns:pfc-5
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
**pfc-5: 64-channel human scalp EEG from 14 unilateral PFC patients and 20 healthy controls performing a lateralized visuospatial working memory task**

64-channel human scalp EEG from 14 unilateral PFC patients and 20 healthy controls performing a lateralized visuospatial working memory task. Contributed by Elizabeth Johnson.

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
