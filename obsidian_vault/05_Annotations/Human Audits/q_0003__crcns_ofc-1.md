---
annotation_id: q_0003__crcns_ofc-1
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: crcns:ofc-1
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
**ofc-1: Single-neuron spike train recordings from rat orbitofrontal cortex during an odor discrimination task**

Single-neuron spike train recordings from rat orbitofrontal cortex during an odor discrimination task.

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
