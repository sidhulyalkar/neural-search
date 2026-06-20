---
annotation_id: q_0003__crcns_am-4
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: crcns:am-4
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
**am-4: Single unit electrophysiological recordings in Nucleus Interface in juvenile birds singing and listening to a tutor, including HVC-projectors**

Single unit electrophysiological recordings in Nucleus Interface in juvenile birds singing and listening to a tutor, including HVC-projectors. Contributed by: Mackevicius EL, Happ MTL, and Fee MS.

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
