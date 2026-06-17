---
annotation_id: q_0003__crcns_hc-17
audit_status: pending
confidence: 0.675
created: '2026-06-17'
dataset_id: crcns:hc-17
judge_version: lf_v1
label: 2
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
**hc-17: Extracellular DG spike trains recorded from wild-type mice during free exploration in a variety of environments that include a circle and a large box, and a circular rotating open field during an active place avoidance task with changing shock zone locations**

Extracellular DG spike trains recorded from wild-type mice during free exploration in a variety of environments that include a circle and a large box, and a circular rotating open field during an active place avoidance task with changing shock zone locations. Contributed by M.T. van Dijk and A.A. Fenton.

## Rule Votes
_No active votes_

## Ensemble
- Label: **2**, Confidence: 0.675

## Human Audit Checklist
- [ ] Reviewed query intent and hard negatives
- [ ] Checked dataset modality / species
- [ ] Verified or corrected label in frontmatter
- [ ] Set `audit_status: done` in frontmatter

> **Edit in frontmatter:** `label`, `confidence`, `audit_status`
