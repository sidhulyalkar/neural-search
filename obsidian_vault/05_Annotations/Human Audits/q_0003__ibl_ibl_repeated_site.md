---
annotation_id: q_0003__ibl_ibl_repeated_site
audit_status: pending
confidence: 0.7286
created: '2026-06-17'
dataset_id: ibl:ibl_repeated_site
judge_version: lf_v1
label: 3
query_id: q_0003
source: bronze
tags:
- annotation
- audit
type: annotation
---

## Query
**extracellular electrophysiology spike sorting benchmark neuropixels single unit**

**Scientific goal:** Find electrophysiology datasets suitable for testing spike sorting or neural decoding pipelines.

## Hard Negatives
- EEG dataset with no single-unit or extracellular recordings
- calcium imaging with extracted spikes but no raw ephys

## Dataset
**IBL Repeated Site Dataset — Reproducible Ephys**

IBL repeated-site dataset: standardized Neuropixels probe insertion at an identical brain location across 10 labs, 59 mice. Designed to test reproducibility of in-vivo electrophysiology across labs, experimenters, and equipment. Probe traverses motor cortex (MOs/MOp), retrosplenial cortex, thalamus (VPM/VPL/LP), and hippocampus (CA1/DG). Data includes spike sorting output, LFP, and behavioral readouts.

## Rule Votes
_No active votes_

## Ensemble
- Label: **3**, Confidence: 0.7286

## Human Audit Checklist
- [ ] Reviewed query intent and hard negatives
- [ ] Checked dataset modality / species
- [ ] Verified or corrected label in frontmatter
- [ ] Set `audit_status: done` in frontmatter

> **Edit in frontmatter:** `label`, `confidence`, `audit_status`
