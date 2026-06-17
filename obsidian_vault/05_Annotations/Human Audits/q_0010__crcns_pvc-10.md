---
annotation_id: q_0010__crcns_pvc-10
audit_status: pending
confidence: 0.7667
created: '2026-06-17'
dataset_id: crcns:pvc-10
judge_version: lf_v1
label: 3
query_id: q_0010
source: bronze
tags:
- annotation
- audit
type: annotation
---

## Query
**two-photon calcium imaging GCaMP mouse cortex dF/F traces deconvolution**

**Scientific goal:** Find calcium imaging datasets suitable for testing fluorescence extraction and spike deconvolution pipelines.

## Hard Negatives
- fiber photometry bulk signal without single-cell imaging
- widefield imaging without single-cell resolution

## Dataset
**pvc-10: Disparity selectivity in mouse V1 measured using two-photon calcium imaging**

Disparity selectivity in mouse V1 measured using two-photon calcium imaging. Contributed by labs of Drs. Nicholas Priebe and Boris Zemelman, University of Texas at Austin.

## Rule Votes
_No active votes_

## Ensemble
- Label: **3**, Confidence: 0.7667

## Human Audit Checklist
- [ ] Reviewed query intent and hard negatives
- [ ] Checked dataset modality / species
- [ ] Verified or corrected label in frontmatter
- [ ] Set `audit_status: done` in frontmatter

> **Edit in frontmatter:** `label`, `confidence`, `audit_status`
