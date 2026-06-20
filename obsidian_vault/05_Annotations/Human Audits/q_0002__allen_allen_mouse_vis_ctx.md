---
annotation_id: q_0002__allen_allen_mouse_vis_ctx
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: allen:allen_mouse_vis_ctx
judge_version: lf_v1
label: 0
query_id: q_0002
source: bronze
tags:
- annotation
- audit
type: annotation
---

## Query
**mouse visual cortex calcium imaging population coding orientation selectivity**

**Scientific goal:** Find mouse visual cortex calcium imaging datasets appropriate for validating population coding models.

## Hard Negatives
- mouse electrophysiology visual cortex — modality mismatch
- human visual fMRI — species mismatch

## Dataset
**Allen Mouse Visual Cortex Cell Types**

Multi-modal cell type characterization of mouse visual cortex combining
            single-cell RNA-seq, single-nucleus RNA-seq, ATAC-seq, and Patch-seq
            electrophysiology. Defines canonical cortical cell type taxonomy with
            transcriptomic, epigenomic, and physiological signatures.

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
