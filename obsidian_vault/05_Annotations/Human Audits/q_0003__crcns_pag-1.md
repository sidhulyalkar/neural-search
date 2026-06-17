---
annotation_id: q_0003__crcns_pag-1
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: crcns:pag-1
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
**pag-1: Ventrolateral periaqueductal gray single-single unit activity during multi-cue fear discrimination in rats**

Ventrolateral periaqueductal gray single-single unit activity during multi-cue fear discrimination in rats. Contributed by KM Wright and MA McDannald.

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
