---
annotation_id: q_0007__neurovault_1929
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: neurovault:1929
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
**Intensive Working Memory Training Produces Functional Changes in Large-scale Frontoparietal Networks**

Working memory is central to human cognition, and intensive cognitive training has been shown to expand working memory capacity in a given domain. It remains unknown, however, how the neural systems that support working memory are altered through intensive training to enable the expansion of working memory capacity. We used fMRI to measure plasticity in activations associated with complex working memory before and after 20 days of training. Healthy young adults were randomly assigned to train on either a dual n-back working memory task or a demanding visuospatial attention task. Training resulted in substantial and task-specific expansion of dual n-back abilities accompanied by changes in the relationship between working memory load and activation. Training differentially affected activations in two large-scale frontoparietal networks thought to underlie working memory: the executive control network and the dorsal attention network. Activations in both networks linearly scaled with working memory load before training, but training dissociated the role of the two networks and eliminated this relationship in the executive control network. Load-dependent functional connectivity both within and between these two networks increased following training, and the magnitudes of increased connectivity were positively correlated with improvements in task performance. These results provide insight into the adaptive neural systems that underlie large gains in working memory capacity through training.

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
