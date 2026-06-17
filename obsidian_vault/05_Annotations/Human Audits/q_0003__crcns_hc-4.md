---
annotation_id: q_0003__crcns_hc-4
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: crcns:hc-4
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
**hc-4: Extracellular recordings from muiti-site silicon probes used for clustering of neuron responses in rat hippocampal and entorhinal regions**

Extracellular recordings from muiti-site silicon probes used for clustering of neuron responses in rat hippocampal and entorhinal regions. Contains additional data (including raw data) used for spike sorting of some neuron responses provided in the hc-3 data set. Contributed by the lab of Gyorgy Buzsáki.

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
