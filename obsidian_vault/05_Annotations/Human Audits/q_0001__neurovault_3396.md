---
annotation_id: q_0001__neurovault_3396
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: neurovault:3396
judge_version: lf_v1
label: 0
query_id: q_0001
source: bronze
tags:
- annotation
- audit
type: annotation
---

## Query
**human fMRI reward prediction error reinforcement learning task**

**Scientific goal:** Identify datasets suitable for cross-study reward-learning meta-analysis.

## Hard Negatives
- resting-state fMRI with reward words in description
- animal RL task when human-only requested

## Dataset
**Instructed knowledge shapes feedback-driven aversive learning in striatum and orbitofrontal cortex, but not the amygdala**

This includes all maps from Atlas et al., 2016. The experiment was an aversive reversal learning task (fear conditioning with reversals) in which participants viewed images (angry faces from the Ekman set) and one stimulus was paired with a shock on 30% of trials. There were 3 reversals across the task. One group (n = 30, Instructed Group) was informed about contingencies before learning & prior to each reversal, whereas a second group (n = 40, Uninstructed Group) learned only through experience. Neuroimaging analyses focused on correlations with dynamic expected value (EV) calculated based on an adapted Rescorla-Wagner learning model with an additional parameter to measure the effects of instructed reversals (see Atlas et al). This model was fit to skin conductance from learners in either the Instructed group (n=20) or the Uninstructed Group (n = 20) to generate Instructed and Feedback-driven EV. We used the best fits from the models fit to each group to generate parametric modulators for fMRI analyses and modeled EV on unreinforced (no shock) trials in our first level analyses and compared within and across groups at second level. We also used task-based fMRI to look at trials surrounding reversal within the Instructed Group to identify regions that update immediately with instruction and those that continue to respond to previous contingencies, and how well reversals correlated with responses to instructions in the DLPFC region that showed greater activation in the Instructed Group across all trials. 

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
