---
annotation_id: q_0003__ibl_ibl_np1_np2_kim
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: ibl:ibl_np1_np2_kim
judge_version: lf_v1
label: 0
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
**IBL NP1 vs NP2 Neuropixels Comparison (Kim et al.)**

IBL dataset comparing Neuropixels 1.0 and 2.0 probes in the same brain regions. Recordings from visual cortex, hippocampus, and brainstem. Provides matched NP1/NP2 sessions for benchmarking spike sorting algorithms and probe comparisons.

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
