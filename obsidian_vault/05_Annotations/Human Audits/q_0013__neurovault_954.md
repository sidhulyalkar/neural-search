---
annotation_id: q_0013__neurovault_954
audit_status: pending
confidence: 0.95
created: '2026-06-17'
dataset_id: neurovault:954
judge_version: lf_v1
label: 0
query_id: q_0013
source: bronze
tags:
- annotation
- audit
type: annotation
---

## Query
**sleep EEG slow oscillations spindles memory consolidation overnight recording**

**Scientific goal:** Find overnight sleep EEG datasets with enough staging and event annotations for memory consolidation analysis.

## Hard Negatives
- EEG datasets from wake-only paradigms
- sleep datasets without spindle or slow-wave annotations

## Dataset
**REM Sleep Is Causal to Successful Consolidation of Dangerous and Safety Stimuli and Reduces Return of Fear after Extinction**

We use a split-night protocol to investigate the influence of different sleep phases on successful consolidation of conditioned fear and extinction. Such a protocol utilizes the fact that in humans the first half of the night is dominated by slow wave sleep, whereas during the second half, rapid eye movement (REM) sleep is more predominant. Our data show that only REM-rich sleep during the second half of the night promoted good discrimination between fear-relevant and neutral stimuli during recall, while staying awake led to a recovery of discrimination between extinguished and neutral stimuli. This suggests that sleep following extinction contributes independently to successful extinction memory consolidation.

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
