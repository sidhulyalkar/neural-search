---
annotation_id: q_0010__crcns_ssc-8
audit_status: pending
confidence: 0.7667
created: '2026-06-17'
dataset_id: crcns:ssc-8
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
**ssc-8: Two-photon calcium imaging recordings of spontaneous activity from mouse somatosensory cortex in wild-type and Fmr1 knock-out mice from three developmental age groups**

Two-photon calcium imaging recordings of spontaneous activity from mouse somatosensory cortex in wild-type and Fmr1 knock-out mice from three developmental age groups. Contributed by Gonçalves, J.T, O’Donnell, C., Sejnowski, T.J., and Portera-Cailliau, C.

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
