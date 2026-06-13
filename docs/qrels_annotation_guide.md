# Qrels Annotation Guide

**Field:** neuroscience_dataset_reuse  
**Relevance scale:** 0 (not relevant) → 3 (highly relevant)

---

## Your Role

You are judging whether a dataset is reusable for a specific scientific goal. You are NOT judging overall dataset quality, prestige, sample size, or whether you would personally use it. The single question is:

> **Can a researcher reuse this dataset for the stated scientific goal?**

---

## Relevance Scale

| Score | Label | Meaning |
|-------|-------|---------|
| 3 | Highly relevant | Can be reused directly. Modality, species, task, brain region all match. Sufficient metadata. Clean provenance. |
| 2 | Partially relevant | Probably reusable with caveats. One key constraint is missing or uncertain. |
| 1 | Weakly relevant | Superficially related but not directly reusable for the stated goal. |
| 0 | Not relevant | Off-topic, wrong species/modality, or violates a hard negative. |

---

## What to Check for Each Query

Before scoring, confirm:

1. **Modality match** — Does the dataset's data modality match `expected_modalities`?
2. **Species match** — Does the dataset's species match `expected_species`?
3. **Task match** — Does the dataset's task align with `expected_tasks` or `must_have`?
4. **Brain region match** — Does it cover `expected_brain_regions` (when specified)?
5. **Analysis affordance** — Does the dataset have the metadata needed for the stated scientific goal?
6. **Hard negatives** — Does this dataset match any description in `hard_negatives`?

---

## 7 Worked Examples

### Example 1 — Score 3: Exact match

**Query:** "human fMRI reward prediction error reinforcement learning task"  
**Dataset:** OpenNeuro ds003351 — "Reward learning fMRI: n=54 healthy adults, blocked RL task, events files, preprocessed BOLD."

**Judgement:** Score **3**. Human species ✓, fMRI ✓, reward RL task ✓, events files ✓, preprocessed data ✓.  
**Rationale required:** No (score 2 and 3 both optional, but helpful).

---

### Example 2 — Score 2: Missing one key piece

**Query:** "human fMRI reward prediction error reinforcement learning task"  
**Dataset:** NeuroVault 1200 — "Activation maps from reward learning study in 30 adults. No raw data, no events files."

**Judgement:** Score **2**. Species and task match. However, no raw BOLD data and no events files — only statistical maps. Reusable for some meta-analyses but not for re-analysis requiring trial-level data.  
**Rationale:** "Statistical maps only — no raw BOLD or events file. Limits reuse for trial-level analyses."

---

### Example 3 — Score 1: Correct domain, wrong modality

**Query:** "human fMRI reward prediction error reinforcement learning task"  
**Dataset:** OpenNeuro ds004000 — "EEG study of reward anticipation in 40 adults, reward cue paradigm."

**Judgement:** Score **1**. Human species ✓, reward task ✓, but modality is EEG not fMRI. Not directly reusable for the stated goal without substantial re-framing.  
**Rationale:** "EEG modality — query requires fMRI."

---

### Example 4 — Score 0 + hard-negative violation

**Query:** "human fMRI reward prediction error reinforcement learning task"  
**Hard negatives include:** "animal RL task when human-only requested"  
**Dataset:** DANDI 000010 — "Mouse ventral striatum calcium imaging, reward conditioning task, 10 mice."

**Judgement:** Score **0**, `hard_negative_violation=True`.  
**Rationale required:** "Mouse calcium imaging — wrong species and modality. Explicitly matches stated hard negative (animal RL task)."

---

### Example 5 — Score 0: No match, not a hard negative

**Query:** "human fMRI reward prediction error reinforcement learning task"  
**Dataset:** NeuroMorpho NMO_000001 — "Morphological reconstruction of rat hippocampal neurons."

**Judgement:** Score **0**, `hard_negative_violation=False`.  
**Rationale required:** "Morphological data, not neuroimaging. Completely off-topic."

---

### Example 6 — Score 2: Species implied but not labeled

**Query:** "mouse calcium imaging visual cortex orientation selectivity"  
**Dataset:** Allen Brain Observatory — "Two-photon calcium imaging, visual stimuli, no explicit species field in metadata."

**Judgement:** Score **2**. The Allen Brain Observatory data is from mouse — species is implied by the title. Modality and task match. Score 2 (not 3) because species is not explicitly labeled in metadata.  
**Rationale:** "Species implied (Allen Brain Observatory = mouse) but not labeled in metadata. Modality and task match."

---

### Example 7 — Ambiguous: default to lower score

**Query:** "primate electrophysiology prefrontal cortex working memory"  
**Dataset:** CRCNS pfc-2 — "Neural recordings from prefrontal cortex during oculomotor delayed response task. No species field."

**Judgement:** Score **1**. Prefrontal cortex electrophysiology matches. However, species is unlabeled and the title does not clearly indicate primate. The oculomotor DRS task is a working memory task, but alignment is uncertain.  
**When in doubt, prefer the lower score.** Flag in rationale.  
**Rationale:** "PFC ephys match. Working memory task inferred from ODRS paradigm. Species not labeled — could be monkey or rodent. Ambiguous — scoring 1."

---

## Common Annotation Mistakes

| Mistake | Correction |
|---------|-----------|
| Scoring a dataset 3 because it is from a famous lab | Fame ≠ reusability. Check metadata. |
| Scoring 0 because the dataset is small | Sample size is not a relevance criterion. |
| Marking hard_negative_violation=True for any irrelevant result | Only mark True when the dataset explicitly matches a listed hard-negative pattern. |
| Scoring 2 for a complete match because you are being conservative | If all constraints are met, score 3. |
| Giving high scores to datasets with no events file or raw data | Missing events/raw data is a real constraint for most reuse goals. Score 2 at most. |

---

## When to Flag for Adjudication

Request adjudication when:
- You scored 1 but it could plausibly be 2
- The species is ambiguous (e.g., "subjects" with no further context)
- The task label is absent but could be inferred from the title
- You have a conflict of interest (e.g., you are an author of this dataset)
- The hard-negative violation judgment is uncertain

---

## Rationale Requirements

| Relevance | Rationale |
|-----------|-----------|
| 3 | Optional but encouraged. Describe what makes it strongly reusable. |
| 2 | Encouraged. Explain the specific gap. |
| 1 | Encouraged. State what is wrong or missing. |
| 0 | **Required.** State why it is irrelevant or which hard negative it matches. |

---

## Field Guide: Modality Equivalences

| Acceptable for query | Matches |
|---------------------|---------|
| fMRI | fMRI, BOLD, functional MRI |
| EEG | EEG, electroencephalography, BCI-EEG |
| MEG | MEG, magnetoencephalography |
| electrophysiology | single-unit, multi-unit, LFP, ephys, spike recordings |
| calcium imaging | two-photon, GCaMP, widefield calcium |
| light sheet | light-sheet microscopy, SPIM |
| sMRI | structural MRI, T1, T2, diffusion MRI |

---

## Field Guide: Species Equivalences

| Query species | Matches |
|---------------|---------|
| human | human, Homo sapiens, participant, subject (implied) |
| mouse | mouse, Mus musculus |
| rat | rat, Rattus norvegicus |
| primate | macaque, Macaca mulatta, M. fascicularis, marmoset, NHP |
| zebrafish | zebrafish, Danio rerio |

---

## Using Silver Labels to Prioritise Your Review Queue

The v0.7 silver qrels system automatically pre-screens candidates before human annotation.
When you receive a review queue from `artifacts/qrels_review_queue.jsonl`, each entry includes:

- `silver_relevance` — the machine-generated guess (0–3)
- `silver_confidence` — how confident the system is (0.0–1.0)
- `why_selected` — why this entry was chosen for human review
- `disagreement_summary` — where labelers disagreed
- `hard_negative_violation` — whether a hard-negative pattern was triggered
- `fields_needed` — which fields the system could not find

**Important:** The silver label is a starting-point suggestion, not a pre-judgement.
You must make your own independent assessment.  If `hard_negative_violation` is True,
verify carefully — but do not automatically confirm the machine's 0 score.

See [docs/silver_qrels_protocol.md](silver_qrels_protocol.md) for full details.

`mouse` does **not** match `rat`. `primate` does not match `human`.
