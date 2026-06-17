---
annotation_id: q_0009__neurovault_1883
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: neurovault:1883
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
**The Real-time fMRI Neurofeedback Based Stratification of Default Network Regulation Neuroimaging Data Repository.**

This describes cross-sectional functional magnetic resonance imaging (fMRI) data from two block-design tasks, a resting state fMRI scan, and a default mode network (DMN) neurofeedback paradigm, along with accompanying behavioral and cognitive measures in an ongoing study. We report technical validation from n=125 participants of the final targeted sample of 180 participants. Each session includes acquisition of one whole-brain anatomical scan and whole-brain echo-planar imaging (EPI) scans, acquired during the aforementioned tasks and resting state. 

DOI: http://dx.doi.org/10.1101/075275

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
