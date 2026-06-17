---
annotation_id: q_0011__neurovault_1408
audit_status: pending
confidence: 0.7
created: '2026-06-17'
dataset_id: neurovault:1408
judge_version: lf_v1
label: 2
query_id: q_0011
source: bronze
tags:
- annotation
- audit
type: annotation
---

## Query
**dopamine reward prediction error ventral striatum mesolimbic pathway rodent**

**Scientific goal:** Find datasets supporting comparison of dopamine signals across recording modalities and species.

## Hard Negatives
- fMRI BOLD signal mistaken for direct dopamine measurement
- striatum recordings without reward task

## Dataset
**Distinct Reward Properties are Encoded via Corticostriatal Interactions**

The striatum serves as a critical brain region for reward processing. Yet, understanding the link between striatum and reward presents a challenge because rewards are composed of multiple properties. Notably, affective properties modulate emotion while informative properties help obtain future rewards. We approached this problem by emphasizing affective and informative reward properties within two independent guessing games. We found that both reward properties evoked activation within the nucleus accumbens, a subregion of the striatum. Striatal responses to informative, but not affective, reward properties predicted subsequent utilization of information for obtaining monetary reward. We hypothesized that activation of the striatum may be necessary but not sufficient to encode distinct reward properties. To investigate this possibility, we examined whether affective and informative reward properties were differentially encoded in corticostriatal interactions. Strikingly, we found that the striatum exhibited dissociable connectivity patterns with the ventrolateral prefrontal cortex, with increasing connectivity for affective reward properties and decreasing connectivity for informative reward properties. Our results demonstrate that affective and informative reward properties are encoded via corticostriatal interactions. These findings highlight how corticostriatal systems contribute to reward processing, potentially advancing models linking striatal activation to behavior.

## Rule Votes
_No active votes_

## Ensemble
- Label: **2**, Confidence: 0.7

## Human Audit Checklist
- [ ] Reviewed query intent and hard negatives
- [ ] Checked dataset modality / species
- [ ] Verified or corrected label in frontmatter
- [ ] Set `audit_status: done` in frontmatter

> **Edit in frontmatter:** `label`, `confidence`, `audit_status`
