---
annotation_id: q_0011__osf_s7hf8
audit_status: pending
confidence: 0.775
created: '2026-06-17'
dataset_id: osf:s7hf8
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
**Cortico-hippocampal interactions during memory updating in highly dynamic environments**

Introduction

Human behaviour relies on continuous extracting of information from the environment, which is often dynamic and relationally complex. To select appropriate actions under changing statistics, internal beliefs must be flexibly updated as structures evolve. However, not always is all the information necessary available to the sensory systems and latent information needs to be leveraged in order to adapt successfully. 
In humans, several brain regions have been implicated in aiding flexible adaptation of behaviour. Among the most commonly reported is the anterior medial prefrontal cortex (amPFC), which has been shown to map the associations between items (Klein-Flügge et al., 2022). Notably, the medial prefrontal cortex, including its anterior part, was found to support navigation in physical (Doeller et al., 2010) as well as conceptual spaces (Constantinescu et al., 2016; Park et al., 2021). This suggests that the amPFC accommodates flexible ‘navigation’ through the space. Moving more posterior, the posterior medial frontal cortex (pmPFC) is involved in performance monitoring and error processing (Ridderinkhof et al., 2004). Signals from the pmPFC are relayed to the lateral PFC (lPFC) which controls adaptive behaviour (Kluen et al., 2019; Ridderinkhof et al., 2004). On the subcortical level, the hippocampus supports flexible behaviour through mechanisms that extend beyond simple memory. For example, Klein-Flugge and colleagues (2019) showed that the hippocampus stores the flexible statistical knowledge about the environment (Klein-Flügge et al., 2019). After having learned what sequences of stimuli lead to a reward, the hippocampus shows an adaptive signal that reflects this knowledge even when it is not explicitly required. The functional organization of the hippocampus also reflects the familiarity with the implicit statistical structure of the environment (Aitken &amp; Kok, 2022). Specifically, when the statistical structure has not yet been learned, unexpected items violating the expectations (based on the true underlying latent associations implicitly learned by the brain) elicit learning-like signals, whereas signals after having learned the structure resemble the expected item rather than the unexpected one. The connectivity within the hippocampal complex changes as a result of prediction errors, biasing the hippocampal state into encoding or retrieval mode (Bein et al., 2020). Furthermore, connectivity changes between the hippocampus and mPFC (Klein et al., 2007) and the hippocampus and the lPFC (Kluen et al., 2019) have been observed during prediction error processing and subsequent behavioural adaptation. 
	This body of work points to a coordinated system in which prefrontal and hippocampal circuits track both explicit prediction errors and implicit, latent signals about environmental structure to accommodate performance; yet, despite growing evidence for their behavioral relevance, we still lack a mechanistic account of how latent variables are neurally represented and integrated with prediction errors to drive belief updating. 
	Previous work on decision making using unobserved variables offers a mechanistic foothold on this question. Schuck and colleagues (2016) proposed that the orbitofrontal cortex (OFC) serves as a cognitive map of the task space, aiding in task performance through representing unobserved variables (or task states) that are relevant for the decision (Schuck et al., 2016). Similarly, during latent variable inference (knowing AB &amp; BC and inferring AC), the hippocampal formation uses the observed inputs (e.g. A from AB) to infer the new relationship (AC) through reactivation of the common part (B), leading to a recall of C and subsequent inference of AC (Koster et al., 2018). Moreover, when people encode BC pairs, reinstatement of A from prior AB episodes is expressed through hippocampal patterns (i.e. reactivation), and the strength of the reactivation predicts subsequent AC inference, linking representational reinstatement to successful relational generalization (Molitor et al., 2021). In this study, Molitor and colleagues showed that the reactivation was reflected in different ways, depending on the subfield of the hippocampus - dentate gyrus/CA3 and subiculum showed signs of pattern separation, whereas CA1 showed pattern completion. This is in line with other research implicating the subfields in different processes (Berron et al., 2016; Wanjia et al., 2021; Zou et al., 2023). Such integration allows adaptive inference when environmental contingencies shift. While these studies offer an insight into how the brain deals with making decisions using unobserved information, it is not clear how this information is used to update once’s beliefs. Arguably, the error processing regions need to integrate this information and allow the representation of the environment to be adapted. 
To answer this question, we developed a novel associative memory paradigm which requires subjects to dynamically update their beliefs about the underlying associations in the environment. To perform the update, they need to make use of previously learned relational structure which is, however, at the time of the update not observed.

## Rule Votes
_No active votes_

## Ensemble
- Label: **2**, Confidence: 0.775

## Human Audit Checklist
- [ ] Reviewed query intent and hard negatives
- [ ] Checked dataset modality / species
- [ ] Verified or corrected label in frontmatter
- [ ] Set `audit_status: done` in frontmatter

> **Edit in frontmatter:** `label`, `confidence`, `audit_status`
