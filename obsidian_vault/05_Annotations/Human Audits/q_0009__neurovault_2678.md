---
annotation_id: q_0009__neurovault_2678
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: neurovault:2678
judge_version: lf_v1
label: 0
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
**Uncoupling protein 2 haplotype does not affect human brain structure and function in a sample of community-dwelling older adults**

In this collection you find unthresholded statistical maps for the image-based analyses run on all subjects that had suitable MRI data available (n=317). A full description of all analyses (TBSS, VBM, dual regression for resting-state fMRI) can be found in this paper: https://doi.org/10.1371/journal.pone.0181392
For each resting-state network the map that corresponds to the network is also included and has been thresholded at z=2.3 to improve visualisation.

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
