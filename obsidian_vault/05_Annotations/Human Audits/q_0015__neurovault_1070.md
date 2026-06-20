---
annotation_id: q_0015__neurovault_1070
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: neurovault:1070
judge_version: lf_v1
label: 0
query_id: q_0015
source: bronze
tags:
- annotation
- audit
type: annotation
---

## Query
**monkey LIP area lateral intraparietal decision making saccade accumulator model**

**Scientific goal:** Find non-human primate LIP electrophysiology datasets for replicating decision accumulator models.

## Hard Negatives
- human fMRI parietal cortex without single-unit resolution
- macaque PFC instead of LIP area

## Dataset
**Adaptive Engagement of Cognitive Control in Context-Dependent Decision Making**

Many decisions require a context-dependent mapping from sensory evidence to action. The capacity for flexible information processing of this sort is thought to depend on a cognitive control system in frontoparietal cortex, but the costs and limitations of control entail that its engagement should be minimized. Here, we show that humans reduce demands on control by exploiting statistical structure in their environment. Using a context-dependent perceptual discrimination task and model-based analyses of behavioral and neuroimaging data, we found that predictions about task context facilitated decision making and that a quantitative measure of context prediction error accounted for graded engagement of the frontoparietal control network. Within this network, multivariate analyses further showed that context prediction error enhanced the representation of task context. These results indicate that decision making is adaptively tuned by experience to minimize costs while maintaining flexibility.

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
