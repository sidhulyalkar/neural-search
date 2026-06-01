# Neural Search Benchmark Evaluation Report

Generated: 2026-06-01T15:43:09.785610+00:00
Suite: real_corpus

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Queries | 30 |
| Queries with Results | 30 |
| Mean Precision@1 | 80.0% |
| Mean Precision@3 | 73.3% |
| **Mean Precision@5** | **69.3%** |
| Mean Precision@10 | 61.0% |
| Mean Recall@5 | 0.0% |
| Mean Recall@10 | 0.0% |
| **Mean Label Recall@10** | **85.5%** |
| Mean MRR | 0.839 |
| Mean NDCG@10 | 0.822 |
| Task Match Rate | 95.0% |
| Modality Match Rate | 94.2% |
| Behavior Match Rate | 91.7% |

## Recommendations

- Add ontology coverage for tasks: ['visual decision making', 'motor imagery']
- Add synonym expansion for modalities: ['behavior tracking', 'ecog']
- Add synonym expansion for behaviors: ['reach onset', 'speech onset', 'response']

## Per-Query Results

### rc_lookup_001: FAIL

**Query:** Find the Steinmetz 2019 Neuropixels visual coding dataset

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 0.0% |
| Precision@3 | 0.0% |
| Precision@5 | 0.0% |
| Precision@10 | 0.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 0.000 |
| NDCG@10 | 0.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** extracellular ephys, neuropixels
**Missed datasets:** dataset:dandi:000026

**Why failed:**
- Precision@5 0.0% below minimum 60.0%
- Missed expected datasets: ['dataset:dandi:000026']

**Warnings:**
- Modality mismatch: query requested neuropixels but dataset lists emg, extracellular ephys, lfp, spikes.
- Modality mismatch: query requested neuropixels but dataset lists extracellular ephys, lfp, position tracking, spikes, tetrode.
- Modality mismatch: query requested neuropixels but dataset lists extracellular ephys, spikes.
- Awareness: Missing query-required signals: spike_times, units
- Awareness: Missing query-required signals: events, spike_times, units

**Top Results:**

1. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 29.49)
   - Behavior matched: memory
   - Modality matched: neuropixels
   - Keyword evidence: encoding, memory
2. `DEMO_AUDITORY_PROCESSING` (score: 28.53)
   - Behavior matched: memory
   - Modality matched: neuropixels
   - Keyword evidence: encoding, memory
3. `DEMO_WORKING_MEMORY_EPHYS` (score: 28.2)
   - Behavior matched: memory
   - Modality matched: neuropixels
   - Keyword evidence: memory

---

### rc_lookup_002: FAIL

**Query:** DANDI dataset 000020 mouse hippocampus

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 0.0% |
| Precision@3 | 0.0% |
| Precision@5 | 0.0% |
| Precision@10 | 0.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 0.000 |
| NDCG@10 | 0.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Missed datasets:** dataset:dandi:000020

**Why failed:**
- Precision@5 0.0% below minimum 40.0%
- Missed expected datasets: ['dataset:dandi:000020']

**Warnings:**
- Expected datasets not returned: ['dataset:dandi:000020']

**Top Results:**

1. `DEMO_HIPPOCAMPUS_NAVIGATION` (score: 49.45)
   - Species matched: mouse
   - Brain region matched: hippocampus
   - High analysis readiness: 95/100
2. `DEMO_DANDI_NWB_EPHYS` (score: 48.96)
   - Species matched: mouse
   - Brain region matched: hippocampus
   - High analysis readiness: 95/100
3. `DEMO_SLEEP_EPHYS` (score: 47.55)
   - Species matched: mouse
   - Brain region matched: hippocampus
   - High analysis readiness: 95/100

---

### rc_lookup_003: FAIL

**Query:** OpenNeuro motor imagery EEG BCI dataset ds003505

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 0.0% |
| Precision@3 | 0.0% |
| Precision@5 | 0.0% |
| Precision@10 | 0.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 0.000 |
| NDCG@10 | 0.000 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** motor imagery
**Matched modalities:** eeg
**Missed datasets:** dataset:openneuro:ds003505

**Why failed:**
- Precision@5 0.0% below minimum 40.0%
- Missed expected datasets: ['dataset:openneuro:ds003505']

**Warnings:**
- Awareness: Missing query-required signals: channels, diagnosis, participants, sampling_rate, sessions
- Awareness: Missing query-required signals: diagnosis, events, participants, sessions
- Expected datasets not returned: ['dataset:openneuro:ds003505']
- Awareness: Missing query-required signals: channels, diagnosis, participants, sessions
- Awareness: Missing query-required signals: channels, diagnosis, events, participants, sampling_rate, sessions

**Top Results:**

1. `DEMO_MOTOR_IMAGERY_EEG` (score: 24.17)
   - Task matched: motor imagery
   - Modality matched: eeg
   - Keyword evidence: imagined movement, motor imagery, motor imagery bci, right hand
2. `DEMO_OPENNEURO_BIDS_IEEG` (score: 15.7)
   - Modality matched: eeg
   - Modality matched: ieeg
   - Modality matched: seeg
3. `DEMO_SEIZURE_IEEG` (score: 14.08)
   - Modality matched: ieeg
   - Modality matched: seeg
   - High analysis readiness: 95/100

---

### rc_lookup_004: PASS

**Query:** International Brain Laboratory brain-wide map datasets

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 33.3% |
| Precision@5 | 60.0% |
| Precision@10 | 70.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 60.0% |
| MRR | 1.000 |
| NDCG@10 | 0.703 |
| Task Match | 0.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** extracellular ephys, neuropixels
**Missing tasks:** visual decision making

**Warnings:**
- Expected tasks not found: ['visual decision making']
- Expected sources not found: ['dandi']

**Top Results:**

1. `DEMO_OPERANT_CONDITIONING` (score: 35.58)
   - Species matched: rat
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
2. `DEMO_DELAY_DISCOUNTING` (score: 35.39)
   - Species matched: rat
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_COGNITIVE_CONTROL_FMRI` (score: 26.43)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: title, title, combined_scientific_summary

---

### rc_lookup_005: PASS

**Query:** Allen Institute visual coding Neuropixels NWB

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
| NDCG@10 | 0.940 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** neuropixels

**Warnings:**
- Modality mismatch: query requested neuropixels but dataset lists emg, extracellular ephys, lfp, spikes.
- Modality mismatch: query requested neuropixels but dataset lists extracellular ephys, lfp, position tracking, spikes, tetrode.
- Modality mismatch: query requested neuropixels but dataset lists extracellular ephys, spikes.
- Awareness: Missing query-required signals: spike_times, units
- Awareness: Missing query-required signals: events, spike_times, units

**Top Results:**

1. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 37.1)
   - Behavior matched: memory
   - Modality matched: neuropixels
   - Keyword evidence: encoding, memory
2. `DEMO_AUDITORY_PROCESSING` (score: 36.3)
   - Behavior matched: memory
   - Modality matched: neuropixels
   - Keyword evidence: encoding, memory
3. `DEMO_WORKING_MEMORY_EPHYS` (score: 35.13)
   - Behavior matched: memory
   - Modality matched: neuropixels
   - Keyword evidence: memory

---

### rc_modality_001: PASS

**Query:** Human ECoG recordings in motor cortex

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
| NDCG@10 | 0.980 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** ecog, ieeg

**Warnings:**
- Modality mismatch: query requested ecog but dataset lists eeg, scalp eeg.
- Awareness: Missing query-required signals: channels, electrodes
- Modality mismatch: query requested ecog but dataset lists ieeg, lfp, seeg.
- Modality mismatch: query requested ecog but dataset lists bold, fmri, functional mri.
- Modality mismatch: query requested ecog but dataset lists emg, extracellular ephys, spikes, utah array.

**Top Results:**

1. `DEMO_SPEECH_ECOG` (score: 43.32)
   - Modality matched: ecog
   - Species matched: human
   - Brain region matched: motor cortex
2. `DEMO_REACHING_ECOG_IEEG` (score: 42.43)
   - Modality matched: ecog
   - Species matched: human
   - Brain region matched: motor cortex
3. `DEMO_NHP_REACHING_UTAH` (score: 20.07)
   - Species matched: human
   - Brain region matched: motor cortex
   - Brain region matched: premotor cortex

---

### rc_modality_002: PASS

**Query:** Mouse calcium imaging prefrontal cortex two-photon

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

**Matched modalities:** calcium imaging

**Warnings:**
- Modality mismatch: query requested calcium imaging but dataset lists extracellular ephys, spikes.
- Modality mismatch: query requested calcium imaging but dataset lists extracellular ephys, lfp, spikes.
- Modality mismatch: query requested calcium imaging but dataset lists extracellular ephys, neuropixels, pupil tracking, running wheel, spikes.
- Modality mismatch: query requested calcium imaging but dataset lists emg, extracellular ephys, lfp, spikes.
- Awareness: Missing query-required signals: events, fluorescence, roi_masks

**Top Results:**

1. `DEMO_GONOGO_CALCIUM` (score: 41.46)
   - Modality matched: calcium imaging
   - Species matched: mouse
   - High analysis readiness: 95/100
2. `DEMO_NATURALISTIC_VISION` (score: 40.38)
   - Modality matched: calcium imaging
   - Species matched: mouse
   - High analysis readiness: 95/100
3. `DEMO_FACEMAP_PUPIL` (score: 39.18)
   - Modality matched: calcium imaging
   - Species matched: mouse
   - High analysis readiness: 95/100

---

### rc_modality_003: PASS

**Query:** Macaque electrophysiology V1 visual cortex

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 90.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 66.7% |
| MRR | 1.000 |
| NDCG@10 | 0.954 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** extracellular ephys

**Warnings:**
- Expected species not found: ['nonhuman primate']

**Top Results:**

1. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 31.56)
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
   - Brain region matched: v1
2. `DEMO_NHP_REACHING_UTAH` (score: 31.06)
   - Modality matched: extracellular ephys
   - Species matched: macaque
   - High analysis readiness: 95/100
3. `DEMO_DANDI_NWB_EPHYS` (score: 29.46)
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
   - Brain region matched: visual cortex

---

### rc_modality_004: PASS

**Query:** Human intracranial EEG hippocampus memory

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
| NDCG@10 | 0.993 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** ecog, ieeg

**Warnings:**
- Modality mismatch: query requested eeg, ieeg, seeg but dataset lists extracellular ephys, neuropixels, spikes.
- Awareness: Missing query-required signals: channels, diagnosis, participants, sampling_rate, sessions
- Modality mismatch: query requested eeg, ieeg, seeg but dataset lists extracellular ephys, lfp, position tracking, spikes, tetrode.
- Awareness: Missing query-required signals: diagnosis, events, participants, sessions
- Modality mismatch: query requested eeg, ieeg, seeg but dataset lists emg, extracellular ephys, lfp, spikes.

**Top Results:**

1. `DEMO_OPENNEURO_BIDS_IEEG` (score: 38.61)
   - Behavior matched: memory
   - Modality matched: eeg
   - Modality matched: ieeg
2. `DEMO_SEIZURE_IEEG` (score: 31.12)
   - Modality matched: ieeg
   - Modality matched: seeg
   - Species matched: human
3. `DEMO_SPEECH_ECOG` (score: 30.97)
   - Behavior matched: memory
   - Modality matched: ieeg
   - Species matched: human

---

### rc_modality_005: PASS

**Query:** Mouse fiber photometry striatum reward

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
| NDCG@10 | 0.989 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** fiber photometry
**Matched behaviors:** reward

**Warnings:**
- Modality mismatch: query requested fiber photometry but dataset lists extracellular ephys, lfp, position tracking, spikes, tetrode.
- Modality mismatch: query requested fiber photometry but dataset lists extracellular ephys, neuropixels, spikes.
- Awareness: Missing query-required signals: events
- Awareness: Missing query-required signals: events, fluorescence
- Modality mismatch: query requested fiber photometry but dataset lists extracellular ephys, lfp, spikes.

**Top Results:**

1. `DEMO_DOPAMINE_PHOTOMETRY` (score: 54.16)
   - Behavior matched: reward
   - Modality matched: fiber photometry
   - Species matched: mouse
2. `DEMO_DELAY_DISCOUNTING` (score: 46.25)
   - Behavior matched: reward
   - Modality matched: fiber photometry
   - Brain region matched: striatum
3. `DEMO_REVERSAL_EPHYS` (score: 27.58)
   - Behavior matched: reward
   - Species matched: mouse
   - Brain region matched: striatum

---

### rc_task_001: PASS

**Query:** Go/no-go task with licking behavior and neural recordings

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
| NDCG@10 | 0.993 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** go nogo
**Matched modalities:** calcium imaging, extracellular ephys, neuropixels
**Matched behaviors:** lick, response

**Warnings:**
- Awareness: Dataset contains excluded data forms: behavior
- Modality mismatch: query requested calcium imaging, ecog, eeg, extracellular ephys, fiber photometry, ieeg, neuropixels but dataset lists bold, fmri, functional mri.

**Top Results:**

1. `DEMO_GONOGO_CALCIUM` (score: 42.42)
   - Task matched: go nogo
   - Behavior matched: lick
   - Modality matched: calcium imaging
2. `DEMO_TRIAL_ALIGNED_EPHYS` (score: 39.28)
   - Task matched: go nogo
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
3. `DEMO_COGNITIVE_CONTROL_FMRI` (score: 21.04)
   - Task matched: go nogo
   - Keyword evidence: go nogo, nogo task, response, response inhibition
   - High analysis readiness: 95/100

---

### rc_task_002: PASS

**Query:** Reversal learning task with choice and reward behavioral events

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 80.0% |
| Precision@10 | 90.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 0.976 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** reversal learning, reward learning
**Matched behaviors:** choice, reward, trial outcome

**Warnings:**
- Awareness: Missing query-required signals: trials
- Awareness: Missing query-required signals: events
- Awareness: Missing query-required signals: events, trials

**Top Results:**

1. `DEMO_DOPAMINE_PHOTOMETRY` (score: 40.46)
   - Task matched: temporal difference learning
   - Behavior matched: learning
   - Behavior matched: omission
2. `DEMO_REVERSAL_EPHYS` (score: 39.76)
   - Task matched: reversal learning
   - Behavior matched: choice
   - Behavior matched: learning
3. `DEMO_DELAY_DISCOUNTING` (score: 31.89)
   - Behavior matched: choice
   - Behavior matched: reward
   - Keyword evidence: choice, reward

---

### rc_task_003: PASS

**Query:** Visual decision-making orientation discrimination

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 80.0% |
| Precision@10 | 80.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 0.988 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** visual decision making
**Matched behaviors:** choice, stimulus onset

**Top Results:**

1. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 44.53)
   - Task matched: visual decision making
   - Task matched: visual discrimination
   - Behavior matched: choice
2. `DEMO_DANDI_NWB_EPHYS` (score: 38.96)
   - Task matched: visual decision making
   - Behavior matched: choice
   - Keyword evidence: choice, decision, reward, visual decision making
3. `DEMO_REVERSAL_EPHYS` (score: 34.21)
   - Behavior matched: choice
   - Keyword evidence: choice, error, outcome, reward
   - High analysis readiness: 95/100

---

### rc_task_004: PASS

**Query:** Reaching and grasping motor task with kinematics

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 40.0% |
| Precision@10 | 20.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 50.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 0.0% |
| Behavior Match | 50.0% |

**Matched tasks:** reaching
**Matched behaviors:** movement onset
**Missing modalities:** behavior tracking
**Missing behaviors:** reach onset

**Warnings:**
- Expected behaviors not found: ['reach onset']
- Awareness: Missing query-required signals: trials
- Modality mismatch: query requested pose tracking but dataset lists emg, extracellular ephys, lfp, spikes.
- Modality mismatch: query requested pose tracking but dataset lists emg, extracellular ephys, spikes, utah array.
- Modality mismatch: query requested pose tracking but dataset lists behavior video, fiber photometry.

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 50.34)
   - Task matched: grasping
   - Task matched: reaching
   - Behavior matched: kinematics
2. `DEMO_NHP_REACHING_UTAH` (score: 28.93)
   - Task matched: reaching
   - Behavior matched: kinematics
   - Keyword evidence: center out reaching, hand position, kinematics, movement onset, position, reach
3. `DEMO_POSE_TRACKING_BEHAVIOR` (score: 22.81)
   - Modality matched: pose tracking
   - High analysis readiness: 85/100
   - Linked papers increase confidence in provenance.

---

### rc_task_005: PASS

**Query:** Speech production task with ECoG and phoneme events

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 80.0% |
| Precision@10 | 40.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 80.0% |
| MRR | 1.000 |
| NDCG@10 | 0.963 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 0.0% |

**Matched tasks:** speech production
**Matched modalities:** ecog, ieeg
**Missing behaviors:** speech onset

**Warnings:**
- Awareness: Missing query-required signals: trials
- Awareness: Missing query-required signals: channels, electrodes, trials
- Awareness: Missing query-required signals: events, trials
- Modality mismatch: query requested ecog but dataset lists extracellular ephys, neuropixels, pupil tracking, running wheel, spikes.
- Expected behaviors not found: ['speech onset']

**Top Results:**

1. `DEMO_SPEECH_ECOG` (score: 54.44)
   - Task matched: speech production
   - Behavior matched: phoneme
   - Modality matched: ecog
2. `DEMO_REACHING_ECOG_IEEG` (score: 25.58)
   - Modality matched: ecog
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_DOPAMINE_PHOTOMETRY` (score: 11.26)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: combined_scientific_summary, combined_scientific_summary, description

---

### rc_affordance_001: PASS

**Query:** Datasets suitable for choice decoding analysis

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 100.0% |
| Precision@10 | 100.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 62.5% |
| MRR | 1.000 |
| NDCG@10 | 0.974 |
| Task Match | 100.0% |
| Modality Match | 75.0% |
| Behavior Match | 50.0% |

**Matched modalities:** calcium imaging, extracellular ephys, neuropixels
**Matched behaviors:** choice
**Missing modalities:** ecog
**Missing behaviors:** response

**Warnings:**
- Expected modalities not found: ['ecog']
- Expected behaviors not found: ['response']
- Expected analyses not found: ['decoding']

**Top Results:**

1. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 57.65)
   - Task matched: two alternative forced choice
   - Behavior matched: choice
   - Behavior matched: memory
2. `DEMO_TRIAL_OUTCOME_PREDICTION` (score: 56.2)
   - Task matched: two alternative forced choice
   - Behavior matched: choice
   - Analysis matched: choice decoding
3. `DEMO_DELAY_DISCOUNTING` (score: 43.8)
   - Behavior matched: choice
   - Analysis matched: choice decoding
   - Affordance matched: choice decoding

---

### rc_affordance_002: PASS

**Query:** Data for Q-learning reinforcement learning modeling with trial outcomes

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 80.0% |
| Precision@10 | 80.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 0.972 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** reversal learning, reward learning
**Matched behaviors:** choice, reward, trial outcome

**Warnings:**
- Awareness: Missing query-required signals: trials

**Top Results:**

1. `DEMO_DOPAMINE_PHOTOMETRY` (score: 50.04)
   - Task matched: temporal difference learning
   - Behavior matched: learning
   - Behavior matched: reward
2. `DEMO_REVERSAL_EPHYS` (score: 34.66)
   - Behavior matched: learning
   - Behavior matched: reward
   - Behavior matched: trial outcome
3. `DEMO_GONOGO_CALCIUM` (score: 32.19)
   - Behavior matched: reward
   - Behavior matched: trial outcome
   - Keyword evidence: cue onset, omission, outcome, reward, trial outcome

---

### rc_affordance_003: PASS

**Query:** Datasets for event-aligned neural activity PSTH analysis

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
| NDCG@10 | 0.961 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** calcium imaging, extracellular ephys, neuropixels
**Matched behaviors:** response, reward, stimulus onset

**Warnings:**
- Expected analyses not found: ['event aligned analysis']

**Top Results:**

1. `DEMO_TRIAL_ALIGNED_EPHYS` (score: 43.91)
   - Modality matched: extracellular ephys
   - Modality matched: neuropixels
   - Analysis matched: event aligned activity
2. `DEMO_GONOGO_CALCIUM` (score: 42.55)
   - Modality matched: calcium imaging
   - Analysis matched: event aligned activity
   - Affordance matched: event aligned activity
3. `DEMO_DOPAMINE_PHOTOMETRY` (score: 41.68)
   - Modality matched: fiber photometry
   - Analysis matched: temporal difference modeling
   - Affordance matched: temporal difference modeling

---

### rc_affordance_004: PASS

**Query:** Data suitable for encoding model analysis with visual stimuli

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 40.0% |
| Precision@10 | 40.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 50.0% |
| MRR | 1.000 |
| NDCG@10 | 0.816 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |


**Warnings:**
- Missing metadata field: brain_regions
- Expected analyses not found: ['encoding modeling']

**Top Results:**

1. `DEMO_FACEMAP_PUPIL` (score: 52.23)
   - Analysis matched: neural behavior correlation
   - Affordance matched: neural behavior correlation
   - Supports analysis: neural behavior correlation
2. `DEMO_POSE_TRACKING_BEHAVIOR` (score: 48.44)
   - Analysis matched: neural behavior correlation
   - Affordance matched: neural behavior correlation
   - Supports analysis: neural behavior correlation
3. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 33.17)
   - Behavior matched: memory
   - Keyword evidence: encoding, memory
   - High analysis readiness: 95/100

---

### rc_affordance_005: PASS

**Query:** Datasets for state-space latent dynamics modeling

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 80.0% |
| Precision@10 | 60.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 60.0% |
| MRR | 1.000 |
| NDCG@10 | 0.978 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** calcium imaging, extracellular ephys, neuropixels

**Warnings:**
- Expected analyses not found: ['latent dynamics modeling', 'state space modeling']

**Top Results:**

1. `DEMO_AUDITORY_PROCESSING` (score: 26.45)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: title, title, combined_scientific_summary
2. `DEMO_TRIAL_OUTCOME_PREDICTION` (score: 25.95)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: combined_scientific_summary, combined_scientific_summary, title
3. `DEMO_NATURALISTIC_VISION` (score: 25.56)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: combined_scientific_summary, combined_scientific_summary, description

---

### rc_linking_001: PASS

**Query:** What datasets are linked to papers about Neuropixels visual cortex?

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

**Matched modalities:** neuropixels

**Warnings:**
- Modality mismatch: query requested neuropixels but dataset lists behavior video, calcium imaging, facemap, pupil tracking, whisker tracking.
- Modality mismatch: query requested neuropixels but dataset lists emg, extracellular ephys, lfp, spikes.
- Modality mismatch: query requested neuropixels but dataset lists calcium imaging, pupil tracking, running wheel, two photon.
- Awareness: Missing query-required signals: spike_times, units
- Awareness: Missing query-required signals: events

**Top Results:**

1. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 39.09)
   - Modality matched: neuropixels
   - Brain region matched: visual cortex
   - High analysis readiness: 95/100
2. `DEMO_DANDI_NWB_EPHYS` (score: 38.24)
   - Modality matched: neuropixels
   - Brain region matched: visual cortex
   - High analysis readiness: 95/100
3. `DEMO_TRIAL_ALIGNED_EPHYS` (score: 36.36)
   - Modality matched: neuropixels
   - Brain region matched: visual cortex
   - High analysis readiness: 95/100

---

### rc_linking_002: PASS

**Query:** Find papers that used DANDI datasets for decision-making research

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 0.0% |
| Precision@3 | 66.7% |
| Precision@5 | 60.0% |
| Precision@10 | 30.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 0.500 |
| NDCG@10 | 0.686 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** reversal learning, visual decision making

**Top Results:**

1. `DEMO_TRIAL_OUTCOME_PREDICTION` (score: 24.55)
   - Behavior matched: choice
   - Keyword evidence: choice, decision, outcome
   - High analysis readiness: 95/100
2. `DEMO_DANDI_NWB_EPHYS` (score: 24.5)
   - Behavior matched: choice
   - Keyword evidence: choice, decision
   - High analysis readiness: 95/100
3. `DEMO_VISUAL_DECISION_NEUROPIXELS` (score: 24.4)
   - Behavior matched: choice
   - Keyword evidence: choice, decision
   - High analysis readiness: 95/100

---

### rc_linking_003: PASS

**Query:** Datasets similar to IBL brain-wide map with multi-region recordings

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 33.3% |
| Precision@5 | 60.0% |
| Precision@10 | 80.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 0.858 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** extracellular ephys, neuropixels

**Top Results:**

1. `DEMO_HIPPOCAMPUS_NAVIGATION` (score: 30.87)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: combined_scientific_summary, combined_scientific_summary, title
2. `DEMO_DELAY_DISCOUNTING` (score: 28.85)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: combined_scientific_summary, combined_scientific_summary, title
3. `DEMO_COGNITIVE_CONTROL_FMRI` (score: 28.48)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: combined_scientific_summary, combined_scientific_summary, title

---

### rc_linking_004: PASS

**Query:** Published reanalysis-friendly NWB datasets with high-quality metadata

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 80.0% |
| Precision@10 | 80.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 0.951 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |


**Top Results:**

1. `DEMO_DANDI_NWB_EPHYS` (score: 28.33)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: title, title, combined_scientific_summary
2. `DEMO_TRIAL_OUTCOME_PREDICTION` (score: 27.66)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: combined_scientific_summary, combined_scientific_summary, title
3. `DEMO_COGNITIVE_CONTROL_FMRI` (score: 27.43)
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
   - Field semantic matches: title, title, combined_scientific_summary

---

### rc_linking_005: PASS

**Query:** OpenAlex papers about calcium imaging and behavior tracking integration

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 100.0% |
| Precision@5 | 60.0% |
| Precision@10 | 30.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 50.0% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 100.0% |
| Modality Match | 50.0% |
| Behavior Match | 100.0% |

**Matched modalities:** calcium imaging
**Missing modalities:** behavior tracking

**Warnings:**
- Modality mismatch: query requested calcium imaging but dataset lists extracellular ephys, spikes.
- Awareness: Missing query-required signals: events, fluorescence, roi_masks, trials
- Awareness: Missing query-required signals: trials
- Modality mismatch: query requested calcium imaging but dataset lists bci, ecog, ieeg, pose tracking.
- Awareness: Missing query-required signals: events, fluorescence, trials

**Top Results:**

1. `DEMO_GONOGO_CALCIUM` (score: 30.27)
   - Modality matched: calcium imaging
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
2. `DEMO_FACEMAP_PUPIL` (score: 30.06)
   - Modality matched: calcium imaging
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
3. `DEMO_NATURALISTIC_VISION` (score: 29.82)
   - Modality matched: calcium imaging
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### rc_scientific_001: PASS

**Query:** Seizure detection datasets with EEG or iEEG and epilepsy labels

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

**Matched tasks:** seizure monitoring
**Matched modalities:** ecog, eeg, ieeg

**Warnings:**
- Modality mismatch: query requested eeg, ieeg, seeg but dataset lists extracellular ephys, neuropixels, spikes.
- Awareness: Missing query-required signals: diagnosis, events, participants, sessions
- Modality mismatch: query requested eeg, ieeg, seeg but dataset lists emg, extracellular ephys, lfp, spikes.
- Awareness: Missing query-required signals: channels, diagnosis, electrodes, participants, sessions
- Awareness: Missing query-required signals: channels, diagnosis, events, participants, sampling_rate, sessions

**Top Results:**

1. `DEMO_SEIZURE_IEEG` (score: 75.6)
   - Task matched: seizure monitoring
   - Modality matched: ieeg
   - Modality matched: seeg
2. `DEMO_OPENNEURO_BIDS_IEEG` (score: 29.5)
   - Modality matched: eeg
   - Modality matched: ieeg
   - Modality matched: seeg
3. `DEMO_REACHING_ECOG_IEEG` (score: 29.18)
   - Modality matched: ieeg
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### rc_scientific_002: FAIL

**Query:** Sleep staging datasets with EEG and polysomnography

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 0.0% |
| Precision@3 | 33.3% |
| Precision@5 | 20.0% |
| Precision@10 | 20.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 66.7% |
| MRR | 0.333 |
| NDCG@10 | 0.525 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched tasks:** sleep staging
**Matched modalities:** eeg

**Why failed:**
- Precision@5 20.0% below minimum 30.0%

**Warnings:**
- Awareness: Missing query-required signals: channels, diagnosis, participants, sampling_rate, sessions
- Awareness: Missing query-required signals: diagnosis, events, participants, sessions
- Modality mismatch: query requested eeg, ieeg, polysomnography, seeg but dataset lists behavior video, calcium imaging, facemap, pupil tracking, whisker tracking.
- Awareness: Missing query-required signals: channels, diagnosis, participants, sessions
- Modality mismatch: query requested eeg, ieeg, polysomnography, seeg but dataset lists calcium imaging, pupil tracking, running wheel, two photon.

**Top Results:**

1. `DEMO_SLEEP_EPHYS` (score: 26.29)
   - Task matched: sleep staging
   - Task matched: sleep wake
   - Keyword evidence: nrem, rem, sleep, sleep staging, sleep wake, sleep wake state
2. `DEMO_SEIZURE_IEEG` (score: 21.01)
   - Modality matched: ieeg
   - Modality matched: seeg
   - High analysis readiness: 95/100
3. `DEMO_OPENNEURO_BIDS_IEEG` (score: 18.44)
   - Modality matched: eeg
   - Modality matched: ieeg
   - Modality matched: seeg

---

### rc_scientific_003: PASS

**Query:** Functional connectivity fMRI resting state datasets

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 0.0% |
| Precision@3 | 33.3% |
| Precision@5 | 40.0% |
| Precision@10 | 20.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 0.333 |
| NDCG@10 | 0.571 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** fmri

**Warnings:**
- Modality mismatch: query requested fmri but dataset lists extracellular ephys, spikes.
- Modality mismatch: query requested fmri but dataset lists behavior video, calcium imaging, two photon.
- Modality mismatch: query requested fmri but dataset lists behavior video, calcium imaging, facemap, pupil tracking, whisker tracking.
- Modality mismatch: query requested fmri but dataset lists emg, extracellular ephys, lfp, spikes.
- Modality mismatch: query requested fmri but dataset lists ieeg, lfp, seeg.

**Top Results:**

1. `DEMO_SLEEP_EPHYS` (score: 56.86)
   - Task matched: resting state
   - Analysis matched: functional connectivity
   - Affordance matched: functional connectivity
2. `DEMO_SEIZURE_IEEG` (score: 56.41)
   - Task matched: resting state
   - Analysis matched: functional connectivity
   - Affordance matched: functional connectivity
3. `DEMO_COGNITIVE_CONTROL_FMRI` (score: 32.55)
   - Modality matched: fmri
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---

### rc_scientific_004: FAIL

**Query:** BCI motor decoding datasets with reaching kinematics NOT EEG

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 40.0% |
| Precision@10 | 20.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 33.3% |
| MRR | 1.000 |
| NDCG@10 | 1.000 |
| Task Match | 50.0% |
| Modality Match | 100.0% |
| Behavior Match | 50.0% |

**Matched tasks:** reaching
**Matched behaviors:** movement onset
**Missing tasks:** motor imagery
**Missing behaviors:** reach onset

**Why failed:**
- Label recall@10 33.3% below minimum 40.0%

**Warnings:**
- Expected behaviors not found: ['reach onset']
- Awareness: Missing query-required signals: trials
- Modality mismatch: query requested bci, ieeg, pose tracking, seeg but dataset lists emg, extracellular ephys, spikes, utah array.
- Awareness: Missing query-required signals: events, trials
- Modality mismatch: query requested bci, ieeg, pose tracking, seeg but dataset lists behavior video, calcium imaging, facemap, pupil tracking, whisker tracking.

**Top Results:**

1. `DEMO_REACHING_ECOG_IEEG` (score: 50.46)
   - Task matched: reaching
   - Behavior matched: kinematics
   - Modality matched: bci
2. `DEMO_NHP_REACHING_UTAH` (score: 27.76)
   - Task matched: reaching
   - Behavior matched: kinematics
   - Keyword evidence: center out reaching, hand position, kinematics, movement onset, position, reach
3. `DEMO_SEIZURE_IEEG` (score: 19.56)
   - Modality matched: ieeg
   - Modality matched: seeg
   - High analysis readiness: 95/100

---

### rc_scientific_005: PASS

**Query:** Optogenetic stimulation datasets with neural recordings

| Metric | Value |
|--------|-------|
| Results | 10 |
| Precision@1 | 100.0% |
| Precision@3 | 66.7% |
| Precision@5 | 80.0% |
| Precision@10 | 80.0% |
| Recall@5 | 0.0% |
| Recall@10 | 0.0% |
| Label Recall@10 | 100.0% |
| MRR | 1.000 |
| NDCG@10 | 0.900 |
| Task Match | 100.0% |
| Modality Match | 100.0% |
| Behavior Match | 100.0% |

**Matched modalities:** calcium imaging, extracellular ephys

**Top Results:**

1. `DEMO_NHP_REACHING_UTAH` (score: 27.96)
   - Modality matched: extracellular ephys
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.
2. `DEMO_REACHING_ECOG_IEEG` (score: 27.95)
   - Modality matched: ecog
   - Modality matched: ieeg
   - High analysis readiness: 95/100
3. `DEMO_HIPPOCAMPUS_NAVIGATION` (score: 27.43)
   - Modality matched: extracellular ephys
   - High analysis readiness: 95/100
   - Linked papers increase confidence in provenance.

---
