---
annotation_id: q_0009__openneuro_ds001168
audit_status: pending
confidence: 0.7
created: '2026-06-17'
dataset_id: openneuro:ds001168
judge_version: lf_v1
label: 2
query_id: q_0009
source: bronze
tags:
- annotation
- audit
type: annotation
---

## Query
**human resting-state fMRI functional connectivity default mode network**

**Scientific goal:** Find resting-state fMRI datasets for meta-analysis of default mode network connectivity.

## Hard Negatives
- task-based fMRI incorrectly labeled as resting state
- structural MRI without BOLD signal

## Dataset
**A high resolution 7-Tesla resting-state fMRI test-retest dataset with cognitive and physiological measures**

_No description_

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
