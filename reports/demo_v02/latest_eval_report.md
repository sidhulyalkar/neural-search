# Neural Search Benchmark Evaluation Report

Generated: 2026-05-27T05:38:08.382769+00:00
Suite: demo_v02

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Queries | 30 |
| Queries with Results | 30 |
| Mean Precision@1 | 83.3% |
| Mean Precision@3 | 82.2% |
| **Mean Precision@5** | **78.0%** |
| Mean Precision@10 | 64.3% |
| Mean Recall@5 | 0.0% |
| Mean Recall@10 | 0.0% |
| **Mean Label Recall@10** | **88.5%** |
| Mean MRR | 0.894 |
| Mean NDCG@10 | 0.921 |
| Task Match Rate | 97.8% |
| Modality Match Rate | 98.1% |
| Behavior Match Rate | 95.6% |

## Recommendations

- Add ontology coverage for tasks: ['value based decision', 'auditory processing']
- Add synonym expansion for modalities: ['eeg']
- Add synonym expansion for behaviors: ['reward prediction', 'dopamine', 'reward omission']

## Per-Query Results

### q001: PASS

**Query:** Find Go/NoGo datasets with neural recordings and lick events.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 80.0% |
| Precision@10 | 90.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 87.5% |
| MRR | 1.000 |
| NDCG@10 | 0.958 |
| Task Match | 100.0% |
| Modality Match | 75.0% |
| Behavior Match | 100.0% |

**Matched tasks:** go nogo
**Matched modalities:** calcium imaging, extracellular ephys, neuropixels
**Matched behaviors:** choice, lick, reward
**Missing modalities:** eeg

**Warnings:**
- Modality mismatch: query requested calcium imaging, ecog, eeg, extracellular ephys, fiber photometry, ieeg, neuropixels but dataset lists bold, fmri, functional mri.
- Expected modalities not found: ['eeg']

**Top Results:**

1. `DEMO_GONOGO_CALCIUM` (score: 66.77)
   - Task matched: go nogo
   - Behavior matched: lick
   - Modality matched: calcium imaging
2. `DEMO_TRIAL_ALIGNED_EPHYS` (score: 48.52)
   - Task matched: go nogo
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
3. `DEMO_COGNITIVE_CONTROL_FMRI` (score: 32.33)
   - Task matched: go nogo
   - Keyword evidence: go nogo, nogo task, response, response inhibition
   - High analysis readiness: 95/100

---

### q002: PASS

**Query:** Find reversal learning datasets with reward omission and trial outcomes.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 80.0% |
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

1. `DEMO_REVERSAL_EPHYS` (score: 52.58)
   - Task matched: reversal learning
   - Behavior matched: learning
   - Behavior matched: omission
2. `DEMO_DOPAMINE_PHOTOMETRY` (score: 47.94)
   - Task matched: temporal difference learning
   - Behavior matched: learning
   - Behavior matched: omission
3. `DEMO_GONOGO_CALCIUM` (score: 34.84)
   - Behavior matched: omission
   - Behavior matched: reward
   - Behavior matched: trial outcome

---

### q003: PASS

**Query:** Find delay discounting datasets with neural activity and behavior.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 80.0% |
| Precision@10 | 60.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 0.876 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** delay discounting
**Matched behaviors:** choice, reward

**Top Results:**

1. `DEMO_DOPAMINE_PHOTOMETRY` (score: 48.2)
   - Modality matched: fiber photometry
   - Analysis matched: temporal difference modeling
   - Affordance matched: temporal difference modeling
2. `DEMO_DELAY_DISCOUNTING` (score: 48.05)
   - Task matched: delay discounting
   - Modality matched: fiber photometry
   - Keyword evidence: choice, delay discounting, delay period, delayed reward, immediate reward, intertemporal choice
3. `DEMO_WORKING_MEMORY_EPHYS` (score: 21.99)
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
   - Keyword evidence: choice, delay period, reward

---

### q004: PASS

**Query:** Find ECoG or iEEG datasets involving reaching or motor control.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 50.0% |
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
- Modality mismatch: query requested ecog, ieeg but dataset lists extracellular ephys, spikes.
- Missing metadata field: behaviors
- Modality mismatch: query requested ecog, ieeg but dataset lists eeg, scalp eeg.
- Modality mismatch: query requested ecog, ieeg but dataset lists extracellular ephys, neuropixels, spikes.
- Missing metadata field: tasks

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 68.46)
   - Task matched: reaching
   - Modality matched: ecog
   - Modality matched: ieeg
2. `DEMO_NHP_REACHING_UTAH` (score: 42.1)
   - Task matched: reaching
   - Brain region matched: motor cortex
   - Keyword evidence: center out reaching, movement onset, reach, reaching
3. `DEMO_SPEECH_ECOG` (score: 40.29)
   - Modality matched: ecog
   - Modality matched: ieeg
   - Brain region matched: motor cortex

---

### q005: PASS

**Query:** Find visual decision-making datasets with Neuropixels recordings.

| Metric | Value |
|--------|-------|
| Results | 10 |
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

**Matched tasks:** visual decision making, visual discrimination
**Matched modalities:** extracellular ephys, neuropixels

**Warnings:**
- Modality mismatch: query requested neuropixels but dataset lists extracellular ephys, lfp, spikes.
- Modality mismatch: query requested neuropixels but dataset lists extracellular ephys, lfp, position tracking, spikes, tetrode.
- Modality mismatch: query requested neuropixels but dataset lists extracellular ephys, spikes.

**Top Results:**

1. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 71.36)
   - Task matched: visual decision making
   - Behavior matched: choice
   - Modality matched: neuropixels
2. `DEMO_DANDI_NWB_EPHYS` (score: 70.16)
   - Task matched: visual decision making
   - Behavior matched: choice
   - Modality matched: neuropixels
3. `DEMO_WORKING_MEMORY_EPHYS` (score: 58.22)
   - Behavior matched: choice
   - Modality matched: neuropixels
   - Keyword evidence: choice, reward

---

### q006: PASS

**Query:** Find datasets where I can decode choice from neural activity.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 0.0% |
| Precision@3 | 66.7% |
| Precision@5 | 80.0% |
| Precision@10 | 70.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 0.500 |
| NDCG@10 | 0.812 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched behaviors:** choice

**Top Results:**

1. `DEMO_DOPAMINE_PHOTOMETRY` (score: 43.43)
   - Modality matched: fiber photometry
   - Analysis matched: temporal difference modeling
   - Affordance matched: temporal difference modeling
2. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 41.16)
   - Behavior matched: choice
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
3. `DEMO_TRIAL_OUTCOME_PREDICTION` (score: 41.15)
   - Behavior matched: choice
   - Modality matched: extracellular ephys
   - Analysis matched: choice decoding

---

### q007: PASS

**Query:** Find naturalistic vision datasets with pupil or running arousal measurements.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 60.0% |
| Precision@10 | 30.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** naturalistic vision, pupil arousal
**Matched behaviors:** pupil, running speed

**Top Results:**

1. `DEMO_NATURALISTIC_VISION` (score: 63.07)
   - Task matched: locomotion
   - Task matched: naturalistic vision
   - Task matched: pupil arousal
2. `DEMO_FACEMAP_PUPIL` (score: 55.51)
   - Task matched: locomotion
   - Task matched: pupil arousal
   - Behavior matched: arousal
3. `DEMO_VIRTUAL_NAVIGATION_NEUROPIXELS` (score: 48.83)
   - Task matched: locomotion
   - Task matched: pupil arousal
   - Behavior matched: pupil

---

### q008: PASS

**Query:** Find motor imagery EEG datasets for BCI classification.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 40.0% |
| Precision@10 | 20.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** motor imagery
**Matched modalities:** eeg

**Warnings:**
- Modality mismatch: query requested bci, eeg, ieeg, seeg but dataset lists behavior video, fiber photometry.
- Missing metadata field: behaviors
- Modality mismatch: query requested bci, eeg, ieeg, seeg but dataset lists extracellular ephys, lfp, position tracking, spikes, tetrode.
- Missing metadata field: tasks
- Modality mismatch: query requested bci, eeg, ieeg, seeg but dataset lists extracellular ephys, spikes.

**Top Results:**

1. `DEMO_MOTOR_IMAGERY_EEG` (score: 42.23)
   - Task matched: motor imagery
   - Modality matched: eeg
   - Analysis matched: motor imagery classification
2. `DEMO_OPENNEURO_BIDS_IEEG` (score: 31.39)
   - Modality matched: eeg
   - Modality matched: ieeg
   - Modality matched: seeg
3. `DEMO_SEIZURE_IEEG` (score: 27.26)
   - Modality matched: ieeg
   - Modality matched: seeg
   - High analysis readiness: 95/100

---

### q009: PASS

**Query:** Find seizure monitoring iEEG datasets with annotated seizure onset.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 80.0% |
| Precision@10 | 40.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** seizure monitoring
**Matched modalities:** ecog, eeg, ieeg

**Warnings:**
- Modality mismatch: query requested ieeg but dataset lists behavior video, fiber photometry.
- Modality mismatch: query requested ieeg but dataset lists extracellular ephys, spikes.
- Modality mismatch: query requested ieeg but dataset lists extracellular ephys, neuropixels, spikes.
- Modality mismatch: query requested ieeg but dataset lists bold, fmri, functional mri.
- Missing metadata field: tasks

**Top Results:**

1. `DEMO_SEIZURE_IEEG` (score: 77.72)
   - Task matched: interictal monitoring
   - Task matched: seizure monitoring
   - Behavior matched: seizure onset
2. `DEMO_REACHING_ECOG_IEEG` (score: 29.3)
   - Modality matched: ieeg
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_SPEECH_ECOG` (score: 29.25)
   - Modality matched: ieeg
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q010: PASS

**Query:** Find social interaction datasets with behavior video and neural recordings.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 90.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** social interaction
**Matched modalities:** behavior video, calcium imaging, extracellular ephys, pose tracking

**Warnings:**
- Missing metadata field: brain_regions

**Top Results:**

1. `DEMO_POSE_TRACKING_BEHAVIOR` (score: 70.41)
   - Task matched: social interaction
   - Modality matched: behavior video
   - Analysis matched: pose estimation
2. `DEMO_FACEMAP_PUPIL` (score: 50.99)
   - Modality matched: behavior video
   - Modality matched: calcium imaging
   - Analysis matched: pose estimation
3. `DEMO_DELAY_DISCOUNTING` (score: 23.24)
   - Modality matched: behavior video
   - Modality matched: fiber photometry
   - Keyword evidence: behavior video

---

### q011: PASS

**Query:** Find OFC recordings during value-based decision making.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 90.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 87.5% |
| MRR | 1.000 |
| NDCG@10 | 0.996 |
| Task Match | 66.7% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** delay discounting, reversal learning
**Matched behaviors:** choice, reward, value
**Missing tasks:** value based decision

**Warnings:**
- Expected tasks not found: ['value based decision']

**Top Results:**

1. `DEMO_REVERSAL_EPHYS` (score: 53.63)
   - Behavior matched: choice
   - Behavior matched: value
   - Brain region matched: ofc
2. `DEMO_FORAGING_EPHYS` (score: 43.65)
   - Behavior matched: value
   - Brain region matched: ofc
   - Keyword evidence: reward, value
3. `DEMO_DELAY_DISCOUNTING` (score: 42.9)
   - Behavior matched: choice
   - Behavior matched: value
   - Keyword evidence: choice, offer onset, reward, value

---

### q012: PASS

**Query:** Find mPFC neural activity during working memory tasks.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 0.0% |
| Precision@3 | 33.3% |
| Precision@5 | 20.0% |
| Precision@10 | 20.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 0.500 |
| NDCG@10 | 0.600 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** delayed match to sample, working memory

**Warnings:**
- Missing metadata field: tasks

**Top Results:**

1. `DEMO_DOPAMINE_PHOTOMETRY` (score: 43.19)
   - Modality matched: fiber photometry
   - Analysis matched: temporal difference modeling
   - Affordance matched: temporal difference modeling
2. `DEMO_WORKING_MEMORY_EPHYS` (score: 40.26)
   - Behavior matched: memory
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
3. `DEMO_AUDITORY_PROCESSING` (score: 36.12)
   - Behavior matched: memory
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels

---

### q013: PASS

**Query:** Find striatum recordings during reward learning with dopamine signals.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 90.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 85.7% |
| MRR | 1.000 |
| NDCG@10 | 0.996 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 66.7% |

**Matched behaviors:** dopamine, reward
**Missing behaviors:** reward prediction

**Warnings:**
- Expected behaviors not found: ['reward prediction']

**Top Results:**

1. `DEMO_DOPAMINE_PHOTOMETRY` (score: 81.55)
   - Task matched: classical conditioning
   - Task matched: temporal difference learning
   - Behavior matched: dopamine
2. `DEMO_REVERSAL_EPHYS` (score: 46.9)
   - Behavior matched: learning
   - Behavior matched: reward
   - Brain region matched: striatum
3. `DEMO_OPERANT_CONDITIONING` (score: 46.3)
   - Behavior matched: learning
   - Behavior matched: reward
   - Brain region matched: striatum

---

### q014: PASS

**Query:** Find hippocampus spatial navigation recordings with place cells.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 60.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** place cell recording, spatial navigation
**Matched behaviors:** navigation, spatial position

**Warnings:**
- Missing metadata field: tasks

**Top Results:**

1. `DEMO_HIPPOCAMPUS_NAVIGATION` (score: 62.49)
   - Task matched: spatial navigation
   - Brain region matched: hippocampus
   - Keyword evidence: navigation, position, spatial navigation
2. `DEMO_VIRTUAL_NAVIGATION_NEUROPIXELS` (score: 60.87)
   - Task matched: spatial navigation
   - Brain region matched: hippocampus
   - Keyword evidence: navigation, position, spatial navigation, virtual navigation
3. `DEMO_SLEEP_EPHYS` (score: 32.97)
   - Brain region matched: hippocampus
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q015: PASS

**Query:** Find M1 motor cortex recordings during reaching or grasping.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 50.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** center out reaching, grasping, reaching

**Warnings:**
- Missing metadata field: behaviors

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 56.3)
   - Task matched: grasping
   - Task matched: reaching
   - Brain region matched: m1
2. `DEMO_NHP_REACHING_UTAH` (score: 45.24)
   - Task matched: reaching
   - Brain region matched: m1
   - Brain region matched: motor cortex
3. `DEMO_MOTOR_IMAGERY_EEG` (score: 26.97)
   - Brain region matched: m1
   - Brain region matched: motor cortex
   - Brain region matched: premotor cortex

---

### q016: PASS

**Query:** Find fiber photometry datasets with reward-related signals.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 90.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 50.0% |
| MRR | 1.000 |
| NDCG@10 | 0.996 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 33.3% |

**Matched modalities:** fiber photometry
**Matched behaviors:** reward
**Missing behaviors:** dopamine, reward prediction

**Warnings:**
- Modality mismatch: query requested fiber photometry but dataset lists behavior video, calcium imaging, two photon.
- Expected behaviors not found: ['dopamine', 'reward prediction']
- Modality mismatch: query requested fiber photometry but dataset lists extracellular ephys, neuropixels, pupil tracking, running wheel, spikes.
- Modality mismatch: query requested fiber photometry but dataset lists extracellular ephys, lfp, spikes.
- Modality mismatch: query requested fiber photometry but dataset lists extracellular ephys, lfp, position tracking, spikes, tetrode.

**Top Results:**

1. `DEMO_DELAY_DISCOUNTING` (score: 57.58)
   - Behavior matched: reward
   - Modality matched: fiber photometry
   - Keyword evidence: reward
2. `DEMO_DOPAMINE_PHOTOMETRY` (score: 57.42)
   - Behavior matched: reward
   - Modality matched: fiber photometry
   - Keyword evidence: reward
3. `DEMO_WORKING_MEMORY_EPHYS` (score: 18.93)
   - Behavior matched: reward
   - Keyword evidence: reward
   - High analysis readiness: 95/100

---

### q017: PASS

**Query:** Find two-photon calcium imaging datasets in visual cortex.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 80.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** calcium imaging, two photon

**Warnings:**
- Modality mismatch: query requested calcium imaging but dataset lists eeg, ieeg, seeg.
- Modality mismatch: query requested calcium imaging but dataset lists extracellular ephys, neuropixels, spikes.
- Modality mismatch: query requested calcium imaging but dataset lists emg, extracellular ephys, lfp, spikes.
- Modality mismatch: query requested calcium imaging but dataset lists bold, fmri, functional mri.
- Missing metadata field: tasks

**Top Results:**

1. `DEMO_NATURALISTIC_VISION` (score: 45.76)
   - Modality matched: calcium imaging
   - Brain region matched: visual cortex
   - High analysis readiness: 95/100
2. `DEMO_FACEMAP_PUPIL` (score: 45.03)
   - Modality matched: calcium imaging
   - Brain region matched: visual cortex
   - High analysis readiness: 95/100
3. `DEMO_GONOGO_CALCIUM` (score: 42.07)
   - Modality matched: calcium imaging
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q018: PASS

**Query:** Find fMRI datasets during cognitive control tasks.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 60.0% |
| Precision@10 | 30.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 0.982 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** cognitive control, flanker, go nogo, stroop
**Matched modalities:** fmri, functional mri

**Warnings:**
- Modality mismatch: query requested fmri but dataset lists emg, extracellular ephys, lfp, spikes.
- Modality mismatch: query requested fmri but dataset lists behavior video, calcium imaging, two photon.
- Modality mismatch: query requested fmri but dataset lists extracellular ephys, spikes.
- Modality mismatch: query requested fmri but dataset lists behavior video, calcium imaging, facemap, pupil tracking, whisker tracking.
- Modality mismatch: query requested fmri but dataset lists behavior video, fiber photometry.

**Top Results:**

1. `DEMO_COGNITIVE_CONTROL_FMRI` (score: 42.87)
   - Modality matched: fmri
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
2. `DEMO_OPENNEURO_BIDS_FMRI` (score: 39.81)
   - Modality matched: fmri
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_TRIAL_OUTCOME_PREDICTION` (score: 5.54)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: title, title, combined_scientific_summary

---

### q019: PASS

**Query:** Find pose tracking datasets for automated behavior analysis.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 60.0% |
| Precision@10 | 40.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 66.7% |
| MRR | 1.000 |
| NDCG@10 | 0.935 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** behavior video, deeplabcut, pose tracking, sleap

**Warnings:**
- Modality mismatch: query requested pose tracking but dataset lists calcium imaging, pupil tracking, running wheel, two photon.
- Missing metadata field: brain_regions
- Modality mismatch: query requested pose tracking but dataset lists bold, fmri, functional mri.
- Modality mismatch: query requested pose tracking but dataset lists extracellular ephys, neuropixels, spikes.
- Modality mismatch: query requested pose tracking but dataset lists behavior video, calcium imaging, facemap, pupil tracking, whisker tracking.

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 29.31)
   - Modality matched: pose tracking
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
2. `DEMO_POSE_TRACKING_BEHAVIOR` (score: 25.53)
   - Modality matched: pose tracking
   - Keyword evidence: approach
   - High analysis readiness: 85/100
3. `DEMO_OPENNEURO_BIDS_FMRI` (score: 5.45)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: title, title, combined_scientific_summary

---

### q020: PASS

**Query:** Find NWB-formatted electrophysiology datasets from DANDI.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 100.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 0.991 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** ecog, extracellular ephys, ieeg, neuropixels

**Top Results:**

1. `DEMO_DANDI_NWB_EPHYS` (score: 22.46)
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
   - High analysis readiness: 95/100
2. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 21.74)
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
   - High analysis readiness: 95/100
3. `DEMO_FORAGING_EPHYS` (score: 21.73)
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
   - High analysis readiness: 95/100

---

### q021: PASS

**Query:** Find BIDS-formatted neuroimaging datasets from OpenNeuro.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 0.0% |
| Precision@3 | 0.0% |
| Precision@5 | 40.0% |
| Precision@10 | 30.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 60.0% |
| MRR | 0.250 |
| NDCG@10 | 0.515 |
| Task Match | 100.0% |
| Modality Match | 66.7% |
| Behavior Match | 100.0% |

**Matched modalities:** fmri, ieeg
**Missing modalities:** eeg

**Warnings:**
- Expected sources not found: ['openneuro']
- Expected modalities not found: ['eeg']

**Top Results:**

1. `DEMO_TRIAL_OUTCOME_PREDICTION` (score: 22.69)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: title, title, combined_scientific_summary
2. `DEMO_DOPAMINE_PHOTOMETRY` (score: 22.57)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: combined_scientific_summary, combined_scientific_summary, description
3. `DEMO_HIPPOCAMPUS_NAVIGATION` (score: 22.51)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: combined_scientific_summary, combined_scientific_summary, title

---

### q022: FAIL

**Query:** Find datasets suitable for training reward prediction models.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 100.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 40.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 66.7% |

**Matched behaviors:** reward, reward prediction
**Missing behaviors:** reward omission

**Warnings:**
- Expected analyses not found: ['reward prediction', 'temporal difference', 'value estimation']
- Expected behaviors not found: ['reward omission']

**Top Results:**

1. `DEMO_DOPAMINE_PHOTOMETRY` (score: 41.58)
   - Behavior matched: learning
   - Behavior matched: reward
   - Behavior matched: reward prediction
2. `DEMO_REVERSAL_EPHYS` (score: 41.39)
   - Behavior matched: learning
   - Behavior matched: reward
   - Behavior matched: reward prediction
3. `DEMO_OPERANT_CONDITIONING` (score: 31.96)
   - Behavior matched: learning
   - Behavior matched: reward
   - Keyword evidence: learning, reward

---

### q023: PASS

**Query:** Find datasets with trial-aligned neural activity for event analysis.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 0.0% |
| Precision@3 | 33.3% |
| Precision@5 | 40.0% |
| Precision@10 | 30.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 57.1% |
| MRR | 0.333 |
| NDCG@10 | 0.552 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched behaviors:** outcome, response, stimulus onset, trial start

**Warnings:**
- Expected analyses not found: ['event alignment', 'psth', 'trial averaging']

**Top Results:**

1. `DEMO_TRIAL_OUTCOME_PREDICTION` (score: 35.76)
   - Modality matched: extracellular ephys
   - Analysis matched: choice decoding
   - Affordance matched: choice decoding
2. `DEMO_DELAY_DISCOUNTING` (score: 34.74)
   - Modality matched: fiber photometry
   - Analysis matched: choice decoding
   - Affordance matched: choice decoding
3. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 34.49)
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
   - Analysis matched: choice decoding

---

### q024: PASS

**Query:** Find closed-loop BCI datasets with real-time neural feedback.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 80.0% |
| Precision@10 | 60.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 77.8% |
| MRR | 1.000 |
| NDCG@10 | 0.922 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** bci control, closed loop control, motor imagery
**Matched modalities:** ecog, eeg, extracellular ephys, ieeg

**Warnings:**
- Modality mismatch: query requested bci but dataset lists eeg, scalp eeg.
- Modality mismatch: query requested bci but dataset lists behavior video, fiber photometry.
- Modality mismatch: query requested bci but dataset lists eeg, ieeg, seeg.
- Modality mismatch: query requested bci but dataset lists extracellular ephys, spikes.
- Modality mismatch: query requested bci but dataset lists extracellular ephys, lfp, position tracking, spikes, tetrode.

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 57.47)
   - Task matched: closed loop control
   - Modality matched: bci
   - Keyword evidence: closed loop, closed loop control, target onset
2. `DEMO_OPENNEURO_BIDS_IEEG` (score: 5.93)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: title, title, combined_scientific_summary
3. `DEMO_OPERANT_CONDITIONING` (score: 5.78)
   - Keyword evidence: reward
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### q025: PASS

**Query:** Find datasets ready for trial outcome prediction analysis.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 60.0% |
| Precision@10 | 70.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 83.3% |
| MRR | 1.000 |
| NDCG@10 | 0.886 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched behaviors:** choice, correct, error, trial outcome

**Warnings:**
- Expected analyses not found: ['trial outcome prediction']

**Top Results:**

1. `DEMO_TRIAL_OUTCOME_PREDICTION` (score: 37.18)
   - Behavior matched: trial outcome
   - Keyword evidence: outcome, trial outcome, trial outcome prediction
   - High analysis readiness: 95/100
2. `DEMO_TRIAL_ALIGNED_EPHYS` (score: 35.27)
   - Behavior matched: trial outcome
   - Keyword evidence: outcome, trial outcome
   - High analysis readiness: 95/100
3. `DEMO_GONOGO_CALCIUM` (score: 35.05)
   - Behavior matched: trial outcome
   - Keyword evidence: outcome, trial outcome
   - High analysis readiness: 95/100

---

### q026: PASS

**Query:** Find mouse Neuropixels datasets during decision-making.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 100.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 0.972 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** decision making, go nogo, visual decision making
**Matched modalities:** neuropixels

**Warnings:**
- Modality mismatch: query requested neuropixels but dataset lists extracellular ephys, lfp, spikes.
- Modality mismatch: query requested neuropixels but dataset lists extracellular ephys, lfp, position tracking, spikes, tetrode.
- Modality mismatch: query requested neuropixels but dataset lists extracellular ephys, spikes.

**Top Results:**

1. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 60.84)
   - Behavior matched: choice
   - Modality matched: neuropixels
   - Species matched: mouse
2. `DEMO_WORKING_MEMORY_EPHYS` (score: 60.32)
   - Behavior matched: choice
   - Modality matched: neuropixels
   - Species matched: mouse
3. `DEMO_DANDI_NWB_EPHYS` (score: 59.9)
   - Behavior matched: choice
   - Modality matched: neuropixels
   - Species matched: mouse

---

### q027: PASS

**Query:** Find non-human primate reaching datasets for motor decoding.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 40.0% |
| Precision@10 | 20.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 75.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** center out reaching, grasping, reaching

**Warnings:**
- Missing metadata field: tasks
- Expected analyses not found: ['motor decoding', 'trajectory prediction']

**Top Results:**

1. `DEMO_NHP_REACHING_UTAH` (score: 55.61)
   - Task matched: reaching
   - Species matched: human
   - Species matched: macaque
2. `DEMO_REACHING_ECOG_IEEG` (score: 51.96)
   - Task matched: reaching
   - Species matched: human
   - Species matched: primate
3. `DEMO_SPEECH_ECOG` (score: 39.92)
   - Behavior matched: memory
   - Species matched: human
   - Species matched: primate

---

### q028: PASS

**Query:** Find human iEEG datasets with speech or language tasks.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 80.0% |
| Precision@10 | 70.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 83.3% |
| MRR | 1.000 |
| NDCG@10 | 0.990 |
| Task Match | 66.7% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** language, speech
**Matched modalities:** ecog, ieeg
**Missing tasks:** auditory processing

**Warnings:**
- Modality mismatch: query requested ieeg but dataset lists behavior video, fiber photometry.
- Expected tasks not found: ['auditory processing']
- Modality mismatch: query requested ieeg but dataset lists eeg, scalp eeg.
- Missing metadata field: behaviors
- Modality mismatch: query requested ieeg but dataset lists emg, extracellular ephys, spikes, utah array.

**Top Results:**

1. `DEMO_SPEECH_ECOG` (score: 41.08)
   - Modality matched: ieeg
   - Species matched: human
   - Keyword evidence: phoneme, word onset
2. `DEMO_SEIZURE_IEEG` (score: 40.68)
   - Modality matched: ieeg
   - Species matched: human
   - High analysis readiness: 95/100
3. `DEMO_REACHING_ECOG_IEEG` (score: 40.04)
   - Modality matched: ieeg
   - Species matched: human
   - High analysis readiness: 95/100

---

### q029: PASS

**Query:** Find probabilistic reversal learning datasets NOT using fMRI.

| Metric | Value |
|--------|-------|
| Results | 10 |
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

**Matched tasks:** probabilistic reversal learning, reversal learning
**Matched modalities:** calcium imaging, extracellular ephys, fiber photometry

**Top Results:**

1. `DEMO_REVERSAL_EPHYS` (score: 58.31)
   - Task matched: reversal learning
   - Behavior matched: learning
   - Keyword evidence: choice, error, learning, omission, prediction error, probabilistic reversal learning
2. `DEMO_DOPAMINE_PHOTOMETRY` (score: 56.77)
   - Task matched: temporal difference learning
   - Behavior matched: learning
   - Keyword evidence: error, learning, omission, prediction error, reward, temporal difference
3. `DEMO_OPERANT_CONDITIONING` (score: 43.88)
   - Behavior matched: learning
   - Keyword evidence: learning, reward
   - High analysis readiness: 95/100

---

### q030: PASS

**Query:** Find visual cortex recordings from mouse NOT using EEG.

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 0.0% |
| Precision@3 | 0.0% |
| Precision@5 | 40.0% |
| Precision@10 | 70.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 0.250 |
| NDCG@10 | 0.657 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** calcium imaging, extracellular ephys, neuropixels

**Warnings:**
- Modality mismatch: query requested ieeg, seeg but dataset lists emg, extracellular ephys, lfp, spikes.
- Modality mismatch: query requested ieeg, seeg but dataset lists behavior video, calcium imaging, facemap, pupil tracking, whisker tracking.
- Modality mismatch: query requested ieeg, seeg but dataset lists extracellular ephys, neuropixels, spikes.
- Modality mismatch: query requested ieeg, seeg but dataset lists calcium imaging, pupil tracking, running wheel, two photon.

**Top Results:**

1. `DEMO_SEIZURE_IEEG` (score: 36.08)
   - Modality matched: ieeg
   - Modality matched: seeg
   - High analysis readiness: 95/100
2. `DEMO_SPEECH_ECOG` (score: 27.66)
   - Modality matched: ieeg
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_REACHING_ECOG_IEEG` (score: 26.72)
   - Modality matched: ieeg
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---
