---
annotation_id: q_0003__crcns_cb-1
audit_status: pending
confidence: 0.675
created: '2026-06-17'
dataset_id: crcns:cb-1
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
**cb-1: Eyelid behavior and spike activity of cerebellar interpositus nucleus neurons during eyeblink conditioning in awake behaving mice**

Eyelid behavior and spike activity of cerebellar interpositus nucleus neurons during eyeblink conditioning in awake behaving mice. Contributed by Michiel M. ten Brinke, Shane A. Heiney, Xiaolu Wang, Martina Proietti-Onori, Henk-Jan Boele, Jacob Bakermans, Javier F. Medina, Zhenyu Gao, &amp; Christiaan I. De Zeeuw.

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
