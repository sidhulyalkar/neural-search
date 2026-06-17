---
annotation_id: q_0007__neurovault_1922
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: neurovault:1922
judge_version: lf_v1
label: 0
query_id: q_0007
source: bronze
tags:
- annotation
- audit
type: annotation
---

## Query
**human fMRI N-back working memory task prefrontal cortex BOLD signal**

**Scientific goal:** Find datasets suitable for reproducing a known N-back working memory fMRI study.

## Hard Negatives
- resting-state fMRI with no working memory task
- EEG N-back without fMRI modality

## Dataset
**Images from experiment 2 of: A comprehensive evaluation of multiband-accelerated sequences and their effects on statistical outcome measures in fMRI.**

These images are results from two tasks (a faces/places perception task, and an n-back working memory task), plus resting state data. Each set of three paradigms was repeated on the same subjects using a standard EPI sequence (ST-EPI) and multiband sequences with 2x (MB2) and 3x (acceleration). The TR for the standard sequence was 2s, with a 1s TR for the MB2 sequence and a 0.66s TR for the MB3 sequence. The experiment was also repeated on two different scanner platforms; Scanner 1 is a Siemens Trio (3T) and Scanner 2 is a Siemens Verio (3T). All other parameters were constant (see preprint on biorXiv for full details: http://biorxiv.org/content/early/2016/09/20/076307).

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
