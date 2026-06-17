---
annotation_id: q_0003__ibl_ibl_brain_wide_map_2023
audit_status: pending
confidence: 0.75
created: '2026-06-17'
dataset_id: ibl:ibl_brain_wide_map_2023
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
**IBL Brain Wide Map 2023 Q4 Release**

International Brain Laboratory Brain Wide Map second public data release (2023 Q4). Extended Neuropixels dataset with improved spike sorting, additional sessions, and updated brain region assignments. Covers the full mouse brain during the IBL perceptual decision-making task. ~700 sessions from 12 labs. Includes multi-region population dynamics data for decoding analyses.

## Rule Votes
_No active votes_

## Ensemble
- Label: **3**, Confidence: 0.75

## Human Audit Checklist
- [ ] Reviewed query intent and hard negatives
- [ ] Checked dataset modality / species
- [ ] Verified or corrected label in frontmatter
- [ ] Set `audit_status: done` in frontmatter

> **Edit in frontmatter:** `label`, `confidence`, `audit_status`
