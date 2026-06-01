# Neuroscience Awareness Report

- Records: `data/corpus/normalized`
- Datasets: 30

## Data Forms

- `behavior`: 18
- `clinical`: 7
- `eeg_meg`: 3
- `extracellular_ephys`: 16
- `fiber_photometry`: 2
- `intracranial_human_ephys`: 4
- `mri`: 2
- `optical_imaging`: 4

## Analysis Families

- `bci_decoding`: 3
- `behavioral_modeling`: 18
- `biomarker_discovery`: 7
- `clinical_prediction`: 9
- `connectivity`: 5
- `decoding`: 20
- `encoding_modeling`: 2
- `event_aligned_analysis`: 22
- `kinematics`: 18
- `memory_analysis`: 4
- `neuromodulator_analysis`: 2
- `phenotyping`: 7
- `population_dynamics`: 4
- `reinforcement_learning`: 18
- `speech_decoding`: 4
- `spike_train_analysis`: 16
- `time_frequency`: 3

## Underrepresented Data Forms

- `intracellular_ephys`
- `connectomics`
- `molecular`
- `computational_model`

## Dataset Awareness

- `dataset:dandi:000001:fixture`: forms=['behavior', 'optical_imaging'], analysis=['behavioral_modeling', 'decoding', 'event_aligned_analysis', 'kinematics', 'population_dynamics', 'reinforcement_learning']
- `dataset:dandi:000020`: forms=['behavior', 'extracellular_ephys'], analysis=['behavioral_modeling', 'decoding', 'event_aligned_analysis', 'kinematics', 'reinforcement_learning', 'spike_train_analysis']
- `dataset:dandi:000026`: forms=['behavior', 'extracellular_ephys'], analysis=['behavioral_modeling', 'decoding', 'event_aligned_analysis', 'kinematics', 'reinforcement_learning', 'spike_train_analysis']
- `dataset:dandi:DEMO_DANDI_NWB_EPHYS`: forms=['extracellular_ephys'], analysis=['decoding', 'event_aligned_analysis', 'spike_train_analysis']
- `dataset:demo:DEMO_AUDITORY_PROCESSING`: forms=['extracellular_ephys'], analysis=['decoding', 'event_aligned_analysis', 'spike_train_analysis']
- `dataset:demo:DEMO_COGNITIVE_CONTROL_FMRI`: forms=['behavior', 'clinical', 'mri'], analysis=['behavioral_modeling', 'biomarker_discovery', 'clinical_prediction', 'connectivity', 'encoding_modeling', 'kinematics', 'phenotyping', 'reinforcement_learning']
- `dataset:demo:DEMO_DELAY_DISCOUNTING`: forms=['behavior', 'fiber_photometry'], analysis=['behavioral_modeling', 'event_aligned_analysis', 'kinematics', 'neuromodulator_analysis', 'reinforcement_learning']
- `dataset:demo:DEMO_DOPAMINE_PHOTOMETRY`: forms=['behavior', 'fiber_photometry'], analysis=['behavioral_modeling', 'event_aligned_analysis', 'kinematics', 'neuromodulator_analysis', 'reinforcement_learning']
- `dataset:demo:DEMO_FACEMAP_PUPIL`: forms=['behavior', 'optical_imaging'], analysis=['behavioral_modeling', 'decoding', 'event_aligned_analysis', 'kinematics', 'population_dynamics', 'reinforcement_learning']
- `dataset:demo:DEMO_FORAGING_EPHYS`: forms=['extracellular_ephys'], analysis=['decoding', 'event_aligned_analysis', 'spike_train_analysis']
- `dataset:demo:DEMO_GONOGO_CALCIUM`: forms=['behavior', 'optical_imaging'], analysis=['behavioral_modeling', 'decoding', 'event_aligned_analysis', 'kinematics', 'population_dynamics', 'reinforcement_learning']
- `dataset:demo:DEMO_HIPPOCAMPUS_NAVIGATION`: forms=['behavior', 'extracellular_ephys'], analysis=['behavioral_modeling', 'decoding', 'event_aligned_analysis', 'kinematics', 'reinforcement_learning', 'spike_train_analysis']
- `dataset:demo:DEMO_MOTOR_IMAGERY_EEG`: forms=['clinical', 'eeg_meg'], analysis=['bci_decoding', 'biomarker_discovery', 'clinical_prediction', 'connectivity', 'phenotyping', 'time_frequency']
- `dataset:demo:DEMO_NATURALISTIC_VISION`: forms=['behavior', 'optical_imaging'], analysis=['behavioral_modeling', 'decoding', 'event_aligned_analysis', 'kinematics', 'population_dynamics', 'reinforcement_learning']
- `dataset:demo:DEMO_NHP_REACHING_UTAH`: forms=['behavior', 'extracellular_ephys'], analysis=['behavioral_modeling', 'decoding', 'event_aligned_analysis', 'kinematics', 'reinforcement_learning', 'spike_train_analysis']
- `dataset:demo:DEMO_OPERANT_CONDITIONING`: forms=['extracellular_ephys'], analysis=['decoding', 'event_aligned_analysis', 'spike_train_analysis']
- `dataset:demo:DEMO_POSE_TRACKING_BEHAVIOR`: forms=['behavior'], analysis=['behavioral_modeling', 'kinematics', 'reinforcement_learning']
- `dataset:demo:DEMO_REACHING_ECOG_IEEG`: forms=['behavior', 'intracranial_human_ephys'], analysis=['behavioral_modeling', 'clinical_prediction', 'kinematics', 'memory_analysis', 'reinforcement_learning', 'speech_decoding']
- `dataset:demo:DEMO_REVERSAL_EPHYS`: forms=['extracellular_ephys'], analysis=['decoding', 'event_aligned_analysis', 'spike_train_analysis']
- `dataset:demo:DEMO_SEIZURE_IEEG`: forms=['behavior', 'clinical', 'extracellular_ephys', 'intracranial_human_ephys'], analysis=['behavioral_modeling', 'biomarker_discovery', 'clinical_prediction', 'decoding', 'event_aligned_analysis', 'kinematics', 'memory_analysis', 'phenotyping', 'reinforcement_learning', 'speech_decoding', 'spike_train_analysis']
- `dataset:demo:DEMO_SLEEP_EPHYS`: forms=['behavior', 'clinical', 'extracellular_ephys'], analysis=['behavioral_modeling', 'biomarker_discovery', 'clinical_prediction', 'decoding', 'event_aligned_analysis', 'kinematics', 'phenotyping', 'reinforcement_learning', 'spike_train_analysis']
- `dataset:demo:DEMO_SPEECH_ECOG`: forms=['intracranial_human_ephys'], analysis=['clinical_prediction', 'memory_analysis', 'speech_decoding']
- `dataset:demo:DEMO_TRIAL_ALIGNED_EPHYS`: forms=['extracellular_ephys'], analysis=['decoding', 'event_aligned_analysis', 'spike_train_analysis']
- `dataset:demo:DEMO_TRIAL_OUTCOME_PREDICTION`: forms=['extracellular_ephys'], analysis=['decoding', 'event_aligned_analysis', 'spike_train_analysis']
- `dataset:demo:DEMO_VIRTUAL_NAVIGATION_NEUROPIXELS`: forms=['behavior', 'extracellular_ephys'], analysis=['behavioral_modeling', 'decoding', 'event_aligned_analysis', 'kinematics', 'reinforcement_learning', 'spike_train_analysis']
- `dataset:demo:DEMO_VISUAL_DECISION_NEUROPIXELS`: forms=['extracellular_ephys'], analysis=['decoding', 'event_aligned_analysis', 'spike_train_analysis']
- `dataset:demo:DEMO_WORKING_MEMORY_EPHYS`: forms=['extracellular_ephys'], analysis=['decoding', 'event_aligned_analysis', 'spike_train_analysis']
- `dataset:openneuro:DEMO_OPENNEURO_BIDS_FMRI`: forms=['clinical', 'mri'], analysis=['biomarker_discovery', 'clinical_prediction', 'connectivity', 'encoding_modeling', 'phenotyping']
- `dataset:openneuro:DEMO_OPENNEURO_BIDS_IEEG`: forms=['behavior', 'clinical', 'eeg_meg', 'intracranial_human_ephys'], analysis=['bci_decoding', 'behavioral_modeling', 'biomarker_discovery', 'clinical_prediction', 'connectivity', 'kinematics', 'memory_analysis', 'phenotyping', 'reinforcement_learning', 'speech_decoding', 'time_frequency']
- `dataset:openneuro:ds003505`: forms=['behavior', 'clinical', 'eeg_meg'], analysis=['bci_decoding', 'behavioral_modeling', 'biomarker_discovery', 'clinical_prediction', 'connectivity', 'kinematics', 'phenotyping', 'reinforcement_learning', 'time_frequency']
