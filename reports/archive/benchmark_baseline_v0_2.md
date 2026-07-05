# Neural Search Benchmark Evaluation Report

Generated: 2026-05-23T18:19:45.323159+00:00

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Queries | 30 |
| Queries with Results | 30 |
| Mean Precision@1 | 83.3% |
| Mean Precision@3 | 63.3% |
| **Mean Precision@5** | **53.3%** |
| Mean Precision@10 | 53.3% |
| Mean Recall@5 | 0.0% |
| Mean Recall@10 | 0.0% |
| **Mean Label Recall@10** | **53.0%** |
| Mean MRR | 0.850 |
| Mean NDCG@10 | 0.833 |
| Task Match Rate | 66.9% |
| Modality Match Rate | 85.0% |
| Behavior Match Rate | 81.4% |

## Recommendations

- Add ontology coverage for tasks: ['motor imagery', 'naturalistic vision', 'pupil arousal']
- Add synonym expansion for modalities: ['eeg', 'fmri', 'two photon']
- Add synonym expansion for behaviors: ['reward prediction', 'dopamine', 'pupil']
- Consider adjusting scoring weights for better precision

## Per-Query Results

### q001: PASS

**Query:** Find Go/NoGo datasets with neural recordings and lick events.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 100.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 87.5% |
| MRR | 1.000 |
| NDCG@10 | 0.960 |
| Task Match | 100.0% |
| Modality Match | 75.0% |
| Behavior Match | 100.0% |

**Matched tasks:** go nogo
**Matched modalities:** calcium imaging, extracellular ephys, neuropixels
**Matched behaviors:** choice, lick, reward
**Missing modalities:** eeg

**Warnings:**
- Expected modalities not found: ['eeg']

**Top Results:**

1. `DEMO_GONOGO_CALCIUM` (score: 71.21)
   - Task matched: go nogo
   - Behavior matched: lick
   - Modality matched: calcium imaging
2. `DEMO_REACHING_ECOG_IEEG` (score: 20.08)
   - Modality matched: ecog
   - Modality matched: ieeg
   - High analysis readiness: 95/100
3. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 19.3)
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
   - Keyword evidence: reward

---

### q002: PASS

**Query:** Find reversal learning datasets with reward omission and trial outcomes.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 100.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** reversal learning
**Matched behaviors:** choice, omission, reward

**Top Results:**

1. `DEMO_REVERSAL_EPHYS` (score: 76.79)
   - Task matched: reversal learning
   - Behavior matched: omission
   - Behavior matched: reward
2. `DEMO_GONOGO_CALCIUM` (score: 44.98)
   - Behavior matched: omission
   - Behavior matched: reward
   - Keyword evidence: omission, reward
3. `DEMO_DELAY_DISCOUNTING` (score: 34.75)
   - Behavior matched: reward
   - Keyword evidence: choice, reward
   - High analysis readiness: 95/100

---

### q003: PASS

**Query:** Find delay discounting datasets with neural activity and behavior.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 100.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** delay discounting
**Matched behaviors:** choice, reward

**Top Results:**

1. `DEMO_DELAY_DISCOUNTING` (score: 49.8)
   - Task matched: delay discounting
   - Modality matched: fiber photometry
   - Keyword evidence: choice, delay discounting, delay period, delayed reward, delayed reward choice, immediate reward
2. `DEMO_REACHING_ECOG_IEEG` (score: 19.28)
   - Modality matched: ecog
   - Modality matched: ieeg
   - Keyword evidence: choice
3. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 19.01)
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
   - Keyword evidence: choice, reward

---

### q004: PASS

**Query:** Find ECoG or iEEG datasets involving reaching or motor control.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 33.3% |
| Precision@5 | 20.0% |
| Precision@10 | 20.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** center out reaching, grasping, reaching
**Matched modalities:** ecog, ieeg

**Warnings:**
- Modality mismatch: query requested ecog, ieeg but dataset lists behavior video, calcium imaging.
- Modality mismatch: query requested ecog, ieeg but dataset lists behavior video, fiber photometry.
- Modality mismatch: query requested ecog, ieeg but dataset lists extracellular ephys, neuropixels.
- Modality mismatch: query requested ecog, ieeg but dataset lists extracellular ephys.

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 71.29)
   - Task matched: reaching
   - Modality matched: ecog
   - Modality matched: ieeg
2. `DEMO_GONOGO_CALCIUM` (score: 0.0)
   - Keyword evidence: reward
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_REVERSAL_EPHYS` (score: 0.0)
   - Keyword evidence: reward
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q005: PASS

**Query:** Find visual decision-making datasets with Neuropixels recordings.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 40.0% |
| Precision@10 | 40.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 0.950 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** visual decision making, visual discrimination
**Matched modalities:** extracellular ephys, neuropixels

**Warnings:**
- Modality mismatch: query requested neuropixels but dataset lists bci, ecog, ieeg, pose tracking.
- Modality mismatch: query requested neuropixels but dataset lists extracellular ephys.
- Modality mismatch: query requested neuropixels but dataset lists behavior video, fiber photometry.
- Modality mismatch: query requested neuropixels but dataset lists behavior video, calcium imaging.

**Top Results:**

1. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 84.58)
   - Task matched: visual decision making
   - Behavior matched: choice
   - Modality matched: neuropixels
2. `DEMO_DELAY_DISCOUNTING` (score: 21.23)
   - Behavior matched: choice
   - Keyword evidence: choice, reaction time, reward
   - High analysis readiness: 95/100
3. `DEMO_REVERSAL_EPHYS` (score: 20.88)
   - Behavior matched: choice
   - Keyword evidence: choice, reward
   - High analysis readiness: 95/100

---

### q006: PASS

**Query:** Find datasets where I can decode choice from neural activity.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 80.0% |
| Precision@10 | 80.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 66.7% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched behaviors:** choice

**Warnings:**
- Expected analyses not found: ['stimulus choice separation']

**Top Results:**

1. `DEMO_DELAY_DISCOUNTING` (score: 50.05)
   - Behavior matched: choice
   - Modality matched: fiber photometry
   - Analysis matched: choice decoding
2. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 41.66)
   - Behavior matched: choice
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
3. `DEMO_REACHING_ECOG_IEEG` (score: 41.27)
   - Behavior matched: choice
   - Modality matched: ecog
   - Modality matched: ieeg

---

### q007: FAIL

**Query:** Find naturalistic vision datasets with pupil or running arousal measurements.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 0.0% |
| Precision@3 | 0.0% |
| Precision@5 | 0.0% |
| Precision@10 | 0.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 0.0% |
| MRR | 0.000 |
| NDCG@10 | 0.000 |
| Task Match | 0.0% |
| Modality Match | 100.0% |
| Behavior Match | 0.0% |

**Missing tasks:** naturalistic vision, pupil arousal
**Missing behaviors:** pupil, running speed

**Warnings:**
- Expected behaviors not found: ['pupil', 'running speed']
- Expected tasks not found: ['naturalistic vision', 'pupil arousal']

**Top Results:**

1. `DEMO_GONOGO_CALCIUM` (score: 23.16)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
2. `DEMO_DELAY_DISCOUNTING` (score: 23.11)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 23.11)
   - Keyword evidence: stimulus onset
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q008: FAIL

**Query:** Find motor imagery EEG datasets for BCI classification.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 0.0% |
| Precision@3 | 0.0% |
| Precision@5 | 0.0% |
| Precision@10 | 0.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 0.0% |
| MRR | 0.000 |
| NDCG@10 | 0.000 |
| Task Match | 0.0% |
| Modality Match | 0.0% |
| Behavior Match | 100.0% |

**Missing tasks:** motor imagery
**Missing modalities:** eeg

**Warnings:**
- Modality mismatch: query requested bci, eeg, ieeg but dataset lists extracellular ephys.
- Expected tasks not found: ['motor imagery']
- Modality mismatch: query requested bci, eeg, ieeg but dataset lists behavior video, calcium imaging.
- Expected modalities not found: ['eeg']
- Modality mismatch: query requested bci, eeg, ieeg but dataset lists extracellular ephys, neuropixels.

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 24.81)
   - Modality matched: bci
   - Modality matched: ieeg
   - High analysis readiness: 95/100
2. `DEMO_GONOGO_CALCIUM` (score: 0.0)
   - Keyword evidence: cue onset
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_REVERSAL_EPHYS` (score: 0.0)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q009: PASS

**Query:** Find seizure monitoring iEEG datasets with annotated seizure onset.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 33.3% |
| Precision@5 | 20.0% |
| Precision@10 | 20.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 50.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 0.0% |
| Modality Match | 66.7% |
| Behavior Match | 100.0% |

**Matched modalities:** ecog, ieeg
**Missing tasks:** seizure monitoring
**Missing modalities:** eeg

**Warnings:**
- Modality mismatch: query requested ieeg but dataset lists behavior video, calcium imaging.
- Modality mismatch: query requested ieeg but dataset lists extracellular ephys.
- Expected tasks not found: ['seizure monitoring']
- Modality mismatch: query requested ieeg but dataset lists behavior video, fiber photometry.
- Modality mismatch: query requested ieeg but dataset lists extracellular ephys, neuropixels.

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 30.3)
   - Modality matched: ieeg
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
2. `DEMO_GONOGO_CALCIUM` (score: 0.0)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_REVERSAL_EPHYS` (score: 0.0)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q010: PASS

**Query:** Find social interaction datasets with behavior video and neural recordings.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 100.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 80.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 0.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** behavior video, calcium imaging, extracellular ephys, pose tracking
**Missing tasks:** social interaction

**Warnings:**
- Expected tasks not found: ['social interaction']

**Top Results:**

1. `DEMO_DELAY_DISCOUNTING` (score: 20.82)
   - Modality matched: behavior video
   - Modality matched: fiber photometry
   - High analysis readiness: 95/100
2. `DEMO_GONOGO_CALCIUM` (score: 20.7)
   - Modality matched: behavior video
   - Modality matched: calcium imaging
   - High analysis readiness: 95/100
3. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 19.44)
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
   - High analysis readiness: 95/100

---

### q011: PASS

**Query:** Find OFC recordings during value-based decision making.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 100.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 62.5% |
| MRR | 1.000 |
| NDCG@10 | 0.964 |
| Task Match | 66.7% |
| Modality Match | 100.0% |
| Behavior Match | 66.7% |

**Matched tasks:** delay discounting, reversal learning
**Matched behaviors:** choice, reward
**Missing tasks:** value based decision
**Missing behaviors:** value

**Warnings:**
- Expected behaviors not found: ['value']
- Expected tasks not found: ['value based decision']
- Expected regions not found: ['orbitofrontal cortex']

**Top Results:**

1. `DEMO_REVERSAL_EPHYS` (score: 56.09)
   - Behavior matched: choice
   - Brain region matched: ofc
   - Keyword evidence: choice
2. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 46.8)
   - Behavior matched: choice
   - Keyword evidence: choice, decision
   - High analysis readiness: 95/100
3. `DEMO_REACHING_ECOG_IEEG` (score: 45.88)
   - Behavior matched: choice
   - Keyword evidence: choice
   - High analysis readiness: 95/100

---

### q012: FAIL

**Query:** Find mPFC neural activity during working memory tasks.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 40.0% |
| Precision@10 | 40.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 20.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 0.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Missing tasks:** delayed match to sample, working memory

**Warnings:**
- Expected regions not found: ['medial prefrontal cortex', 'prefrontal cortex']
- Expected tasks not found: ['delayed match to sample', 'working memory']

**Top Results:**

1. `DEMO_GONOGO_CALCIUM` (score: 28.55)
   - Modality matched: calcium imaging
   - Brain region matched: mpfc
   - Keyword evidence: correct, response
2. `DEMO_DELAY_DISCOUNTING` (score: 28.33)
   - Modality matched: fiber photometry
   - Brain region matched: mpfc
   - Keyword evidence: reaction time
3. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 19.39)
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
   - Keyword evidence: correct, error, reaction time, stimulus onset

---

### q013: FAIL

**Query:** Find striatum recordings during reward learning with dopamine signals.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 80.0% |
| Precision@10 | 80.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 42.9% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 33.3% |

**Matched behaviors:** reward
**Missing behaviors:** dopamine, reward prediction

**Warnings:**
- Expected behaviors not found: ['dopamine', 'reward prediction']
- Expected regions not found: ['dorsal striatum', 'ventral striatum']

**Top Results:**

1. `DEMO_REVERSAL_EPHYS` (score: 56.24)
   - Behavior matched: reward
   - Brain region matched: striatum
   - Keyword evidence: omission, reward
2. `DEMO_DELAY_DISCOUNTING` (score: 46.06)
   - Behavior matched: reward
   - Keyword evidence: reward
   - High analysis readiness: 95/100
3. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 45.92)
   - Behavior matched: reward
   - Keyword evidence: reward
   - High analysis readiness: 95/100

---

### q014: FAIL

**Query:** Find hippocampus spatial navigation recordings with place cells.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 0.0% |
| Precision@3 | 0.0% |
| Precision@5 | 0.0% |
| Precision@10 | 0.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 0.0% |
| MRR | 0.000 |
| NDCG@10 | 0.000 |
| Task Match | 0.0% |
| Modality Match | 100.0% |
| Behavior Match | 0.0% |

**Missing tasks:** place cell recording, spatial navigation
**Missing behaviors:** navigation, spatial position

**Warnings:**
- Expected tasks not found: ['place cell recording', 'spatial navigation']
- Expected regions not found: ['ca1', 'ca3', 'dentate gyrus', 'hippocampus']
- Expected behaviors not found: ['navigation', 'spatial position']

**Top Results:**

1. `DEMO_DELAY_DISCOUNTING` (score: 24.16)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
2. `DEMO_REVERSAL_EPHYS` (score: 23.23)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 22.98)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q015: PASS

**Query:** Find M1 motor cortex recordings during reaching or grasping.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 33.3% |
| Precision@5 | 20.0% |
| Precision@10 | 20.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 66.7% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** center out reaching, grasping, reaching

**Warnings:**
- Expected regions not found: ['m1', 'primary motor cortex']

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 56.92)
   - Task matched: grasping
   - Task matched: reaching
   - Brain region matched: motor cortex
2. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 23.51)
   - Keyword evidence: reward
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_GONOGO_CALCIUM` (score: 23.14)
   - Keyword evidence: reward
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q016: PASS

**Query:** Find fiber photometry datasets with reward-related signals.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 80.0% |
| Precision@10 | 80.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 50.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 33.3% |

**Matched modalities:** fiber photometry
**Matched behaviors:** reward
**Missing behaviors:** dopamine, reward prediction

**Warnings:**
- Modality mismatch: query requested fiber photometry but dataset lists extracellular ephys, neuropixels.
- Expected behaviors not found: ['dopamine', 'reward prediction']
- Modality mismatch: query requested fiber photometry but dataset lists bci, ecog, ieeg, pose tracking.
- Modality mismatch: query requested fiber photometry but dataset lists extracellular ephys.
- Modality mismatch: query requested fiber photometry but dataset lists behavior video, calcium imaging.

**Top Results:**

1. `DEMO_DELAY_DISCOUNTING` (score: 53.88)
   - Behavior matched: reward
   - Modality matched: fiber photometry
   - Keyword evidence: reward
2. `DEMO_REVERSAL_EPHYS` (score: 20.27)
   - Behavior matched: reward
   - Keyword evidence: reward
   - High analysis readiness: 95/100
3. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 20.22)
   - Behavior matched: reward
   - Keyword evidence: reward
   - High analysis readiness: 95/100

---

### q017: FAIL

**Query:** Find two-photon calcium imaging datasets in visual cortex.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 40.0% |
| Precision@10 | 40.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 40.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 50.0% |
| Behavior Match | 100.0% |

**Matched modalities:** calcium imaging
**Missing modalities:** two photon

**Warnings:**
- Modality mismatch: query requested calcium imaging but dataset lists extracellular ephys.
- Modality mismatch: query requested calcium imaging but dataset lists extracellular ephys, neuropixels.
- Expected modalities not found: ['two photon']
- Modality mismatch: query requested calcium imaging but dataset lists behavior video, fiber photometry.
- Expected regions not found: ['primary visual cortex', 'v1']

**Top Results:**

1. `DEMO_GONOGO_CALCIUM` (score: 31.12)
   - Modality matched: calcium imaging
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
2. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 2.89)
   - Brain region matched: visual cortex
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_REVERSAL_EPHYS` (score: 0.0)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q018: FAIL

**Query:** Find fMRI datasets during cognitive control tasks.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 33.3% |
| Precision@5 | 20.0% |
| Precision@10 | 20.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 16.7% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 25.0% |
| Modality Match | 0.0% |
| Behavior Match | 100.0% |

**Matched tasks:** go nogo
**Missing tasks:** cognitive control, flanker, stroop
**Missing modalities:** fmri, functional mri

**Warnings:**
- Expected modalities not found: ['fmri', 'functional mri']
- Modality mismatch: query requested fmri but dataset lists behavior video, calcium imaging.
- Modality mismatch: query requested fmri but dataset lists bci, ecog, ieeg, pose tracking.
- Modality mismatch: query requested fmri but dataset lists behavior video, fiber photometry.
- Expected tasks not found: ['cognitive control', 'flanker', 'stroop']

**Top Results:**

1. `DEMO_GONOGO_CALCIUM` (score: 0.0)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
2. `DEMO_REVERSAL_EPHYS` (score: 0.0)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_DELAY_DISCOUNTING` (score: 0.0)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q019: FAIL

**Query:** Find pose tracking datasets for automated behavior analysis.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 60.0% |
| Precision@10 | 60.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 33.3% |
| MRR | 1.000 |
| NDCG@10 | 0.967 |
| Task Match | 100.0% |
| Modality Match | 50.0% |
| Behavior Match | 100.0% |

**Matched modalities:** behavior video, pose tracking
**Missing modalities:** deeplabcut, sleap

**Warnings:**
- Modality mismatch: query requested pose tracking but dataset lists behavior video, fiber photometry.
- Expected analyses not found: ['behavior classification', 'pose estimation']
- Modality mismatch: query requested pose tracking but dataset lists extracellular ephys, neuropixels.
- Modality mismatch: query requested pose tracking but dataset lists extracellular ephys.
- Modality mismatch: query requested pose tracking but dataset lists behavior video, calcium imaging.

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 29.83)
   - Modality matched: pose tracking
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
2. `DEMO_GONOGO_CALCIUM` (score: 0.0)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_REVERSAL_EPHYS` (score: 0.0)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q020: PASS

**Query:** Find NWB-formatted electrophysiology datasets from DANDI.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 80.0% |
| Precision@10 | 80.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 83.3% |
| MRR | 1.000 |
| NDCG@10 | 0.969 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** ecog, extracellular ephys, ieeg, neuropixels

**Warnings:**
- Expected sources not found: ['dandi']
- Modality mismatch: query requested ecog, eeg, extracellular ephys, ieeg, neuropixels but dataset lists behavior video, fiber photometry.
- Modality mismatch: query requested ecog, eeg, extracellular ephys, ieeg, neuropixels but dataset lists behavior video, calcium imaging.

**Top Results:**

1. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 21.9)
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
   - High analysis readiness: 95/100
2. `DEMO_REACHING_ECOG_IEEG` (score: 21.43)
   - Modality matched: ecog
   - Modality matched: ieeg
   - High analysis readiness: 95/100
3. `DEMO_REVERSAL_EPHYS` (score: 19.76)
   - Modality matched: extracellular ephys
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q021: FAIL

**Query:** Find BIDS-formatted neuroimaging datasets from OpenNeuro.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 33.3% |
| Precision@5 | 40.0% |
| Precision@10 | 40.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 40.0% |
| MRR | 1.000 |
| NDCG@10 | 0.674 |
| Task Match | 100.0% |
| Modality Match | 33.3% |
| Behavior Match | 100.0% |

**Matched modalities:** ieeg
**Missing modalities:** eeg, fmri

**Warnings:**
- Expected modalities not found: ['eeg', 'fmri']
- Expected sources not found: ['openneuro']

**Top Results:**

1. `DEMO_DELAY_DISCOUNTING` (score: 23.71)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
2. `DEMO_GONOGO_CALCIUM` (score: 23.44)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 23.3)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q022: FAIL

**Query:** Find datasets suitable for training reward prediction models.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 80.0% |
| Precision@10 | 80.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 20.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 33.3% |

**Matched behaviors:** reward
**Missing behaviors:** reward omission, reward prediction

**Warnings:**
- Expected behaviors not found: ['reward omission', 'reward prediction']
- Expected analyses not found: ['reward prediction', 'temporal difference', 'value estimation']

**Top Results:**

1. `DEMO_DELAY_DISCOUNTING` (score: 45.33)
   - Behavior matched: reward
   - Keyword evidence: reward
   - High analysis readiness: 95/100
2. `DEMO_GONOGO_CALCIUM` (score: 45.12)
   - Behavior matched: reward
   - Keyword evidence: reward
   - High analysis readiness: 95/100
3. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 45.08)
   - Behavior matched: reward
   - Keyword evidence: reward
   - High analysis readiness: 95/100

---

### q023: FAIL

**Query:** Find datasets with trial-aligned neural activity for event analysis.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 0.0% |
| Precision@3 | 0.0% |
| Precision@5 | 0.0% |
| Precision@10 | 0.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 0.0% |
| MRR | 0.000 |
| NDCG@10 | 0.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 0.0% |

**Missing behaviors:** outcome, response, stimulus onset, trial start

**Warnings:**
- Expected behaviors not found: ['outcome', 'response', 'stimulus onset', 'trial start']
- Expected analyses not found: ['event alignment', 'psth', 'trial averaging']

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 19.44)
   - Modality matched: ecog
   - Modality matched: ieeg
   - High analysis readiness: 95/100
2. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 18.84)
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
   - High analysis readiness: 95/100
3. `DEMO_DELAY_DISCOUNTING` (score: 18.17)
   - Modality matched: fiber photometry
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q024: FAIL

**Query:** Find closed-loop BCI datasets with real-time neural feedback.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 60.0% |
| Precision@10 | 60.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 33.3% |
| MRR | 1.000 |
| NDCG@10 | 0.885 |
| Task Match | 0.0% |
| Modality Match | 75.0% |
| Behavior Match | 100.0% |

**Matched modalities:** ecog, extracellular ephys, ieeg
**Missing tasks:** bci control, closed loop control, motor imagery
**Missing modalities:** eeg

**Warnings:**
- Modality mismatch: query requested bci but dataset lists extracellular ephys.
- Expected tasks not found: ['bci control', 'closed loop control', 'motor imagery']
- Modality mismatch: query requested bci but dataset lists extracellular ephys, neuropixels.
- Expected analyses not found: ['bci classification', 'real time decoding']
- Modality mismatch: query requested bci but dataset lists behavior video, fiber photometry.

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 29.71)
   - Modality matched: bci
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
2. `DEMO_GONOGO_CALCIUM` (score: 0.0)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_REVERSAL_EPHYS` (score: 0.0)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q025: PASS

**Query:** Find datasets ready for trial outcome prediction analysis.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 100.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 66.7% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 75.0% |

**Matched behaviors:** choice, correct, error
**Missing behaviors:** trial outcome

**Warnings:**
- Expected behaviors not found: ['trial outcome']
- Expected analyses not found: ['trial outcome prediction']

**Top Results:**

1. `DEMO_GONOGO_CALCIUM` (score: 23.05)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
2. `DEMO_REVERSAL_EPHYS` (score: 22.99)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 22.93)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q026: PASS

**Query:** Find mouse Neuropixels datasets during decision-making.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 60.0% |
| Precision@10 | 60.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 80.0% |
| MRR | 1.000 |
| NDCG@10 | 0.925 |
| Task Match | 66.7% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** go nogo, visual decision making
**Matched modalities:** neuropixels
**Missing tasks:** decision making

**Warnings:**
- Expected tasks not found: ['decision making']
- Modality mismatch: query requested neuropixels but dataset lists extracellular ephys.
- Modality mismatch: query requested neuropixels but dataset lists behavior video, fiber photometry.
- Modality mismatch: query requested neuropixels but dataset lists behavior video, calcium imaging.
- Modality mismatch: query requested neuropixels but dataset lists bci, ecog, ieeg, pose tracking.

**Top Results:**

1. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 64.11)
   - Behavior matched: choice
   - Modality matched: neuropixels
   - Species matched: mouse
2. `DEMO_REVERSAL_EPHYS` (score: 30.82)
   - Behavior matched: choice
   - Species matched: mouse
   - Keyword evidence: choice
3. `DEMO_DELAY_DISCOUNTING` (score: 20.71)
   - Behavior matched: choice
   - Keyword evidence: choice
   - High analysis readiness: 95/100

---

### q027: FAIL

**Query:** Find non-human primate reaching datasets for motor decoding.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 33.3% |
| Precision@5 | 20.0% |
| Precision@10 | 20.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 37.5% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** center out reaching, grasping, reaching

**Warnings:**
- Expected species not found: ['macaque', 'non human primate', 'rhesus']
- Expected analyses not found: ['motor decoding', 'trajectory prediction']

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 58.1)
   - Task matched: reaching
   - Species matched: human
   - Keyword evidence: center out reaching, grasp, movement onset, reach, reaching, target onset
2. `DEMO_DELAY_DISCOUNTING` (score: 22.44)
   - Keyword evidence: reward
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_GONOGO_CALCIUM` (score: 22.42)
   - Keyword evidence: reward
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q028: PASS

**Query:** Find human iEEG datasets with speech or language tasks.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 33.3% |
| Precision@5 | 20.0% |
| Precision@10 | 20.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 50.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 0.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** ecog, ieeg
**Missing tasks:** auditory processing, language, speech

**Warnings:**
- Modality mismatch: query requested ieeg but dataset lists behavior video, calcium imaging.
- Modality mismatch: query requested ieeg but dataset lists extracellular ephys.
- Expected tasks not found: ['auditory processing', 'language', 'speech']
- Modality mismatch: query requested ieeg but dataset lists behavior video, fiber photometry.
- Modality mismatch: query requested ieeg but dataset lists extracellular ephys, neuropixels.

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 40.82)
   - Modality matched: ieeg
   - Species matched: human
   - High analysis readiness: 95/100
2. `DEMO_GONOGO_CALCIUM` (score: 0.0)
   - Keyword evidence: response
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_REVERSAL_EPHYS` (score: 0.0)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q029: PASS

**Query:** Find probabilistic reversal learning datasets NOT using fMRI.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 80.0% |
| Precision@10 | 80.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 80.0% |
| MRR | 1.000 |
| NDCG@10 | 0.988 |
| Task Match | 50.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** reversal learning
**Matched modalities:** calcium imaging, extracellular ephys, fiber photometry
**Missing tasks:** probabilistic reversal learning

**Warnings:**
- Modality mismatch: query requested fmri but dataset lists behavior video, calcium imaging.
- Modality mismatch: query requested fmri but dataset lists bci, ecog, ieeg, pose tracking.
- Modality mismatch: query requested fmri but dataset lists behavior video, fiber photometry.
- Expected tasks not found: ['probabilistic reversal learning']
- Modality mismatch: query requested fmri but dataset lists extracellular ephys.

**Top Results:**

1. `DEMO_REVERSAL_EPHYS` (score: 29.76)
   - Task matched: reversal learning
   - Keyword evidence: choice, error, omission, probabilistic reversal learning, reversal learning, reversal point
   - High analysis readiness: 95/100
2. `DEMO_GONOGO_CALCIUM` (score: 0.0)
   - Keyword evidence: omission, reward
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_DELAY_DISCOUNTING` (score: 0.0)
   - Keyword evidence: choice, reward
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q030: PASS

**Query:** Find visual cortex recordings from mouse NOT using EEG.

| Metric | Value |
|--------|-------|
| Results | 5 |
| Precision@1 | 0.0% |
| Precision@3 | 66.7% |
| Precision@5 | 60.0% |
| Precision@10 | 60.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 83.3% |
| MRR | 0.500 |
| NDCG@10 | 0.713 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** calcium imaging, extracellular ephys, neuropixels

**Warnings:**
- Modality mismatch: query requested eeg, ieeg but dataset lists behavior video, fiber photometry.
- Expected regions not found: ['v1']
- Modality mismatch: query requested eeg, ieeg but dataset lists extracellular ephys, neuropixels.
- Modality mismatch: query requested eeg, ieeg but dataset lists extracellular ephys.
- Modality mismatch: query requested eeg, ieeg but dataset lists behavior video, calcium imaging.

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 23.18)
   - Modality matched: ieeg
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
2. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 7.2)
   - Species matched: mouse
   - Brain region matched: visual cortex
   - High analysis readiness: 95/100
3. `DEMO_REVERSAL_EPHYS` (score: 3.91)
   - Species matched: mouse
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---
