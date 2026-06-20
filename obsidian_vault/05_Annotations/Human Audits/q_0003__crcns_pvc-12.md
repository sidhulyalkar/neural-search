---
annotation_id: q_0003__crcns_pvc-12
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: crcns:pvc-12
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
**pvc-12: Single-neuron extracellular recordings from area V1 of awake behaving rhesus monkeys viewing brightness illusions based on Victor Vasarely’s artworks**

Single-neuron extracellular recordings from area V1 of awake behaving rhesus monkeys viewing brightness illusions based on Victor Vasarely’s artworks. Contributed by: Susana Martinez-Conde, Jorge Otero-Millan, Xoana G. Troncoso, Michael B. McCamy and Stephen L. Macknik.

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
