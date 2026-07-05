// NEATLabs static profile — Neural Engineering and Translation Labs, UCSD
// Data sourced from neatlabs.ucsd.edu + Google Scholar

export const NEATLABS = {
  id: 'neatlabs_ucsd',
  name: 'Neural Engineering and Translation Labs',
  acronym: 'NEATLabs',
  institution: 'UC San Diego — Department of Psychiatry',
  url: 'https://neatlabs.ucsd.edu/',
  tagline: 'Advancing neuro-cognitive circuits across the lifespan through neural engineering',
  mission:
    'NEATLabs bridges animal and human neuroscience to develop closed-loop technologies ' +
    'that stimulate brain plasticity and modify behavior — translating discoveries from ' +
    'rodent models into diagnostics and therapeutics for mental health disorders.',

  directors: [
    {
      name: 'Dr. Jyoti Mishra',
      role: 'Co-Director',
      focus: 'Neurotechnology in humans — EEG, neurofeedback, and ML for mental health diagnostics & therapeutics',
      scholar: 'https://scholar.google.com/citations?user=7cI2_e4AAAAJ&hl=en',
    },
    {
      name: 'Dr. Dhakshin Ramanathan',
      role: 'Co-Director',
      focus: 'Neural engineering in animal models of neuropsychiatric disorders — cortico-striatal circuits, oscillations',
      scholar: 'https://scholar.google.com/citations?user=ZUFLEBIAAAAJ&hl=en',
    },
  ],

  faculty: [
    { name: 'Dr. Matthew Herbert', focus: 'Precision health for chronic pain' },
    { name: 'Dr. Miranda Francoeur', focus: 'Animal neuro-cognition' },
    { name: 'Dr. Satish Jaiswal', focus: 'Human neuro-cognition' },
  ],

  // Which of the 26 KG topics NEATLabs actively publishes on
  kg_topics: [
    { id: 'spectral_dynamics', relevance: 'primary', note: 'Beta/gamma oscillations in reward certainty (2025)' },
    { id: 'reward_learning', relevance: 'primary', note: 'Intertemporal choice in rodents; reward prediction' },
    { id: 'neural_synchrony', relevance: 'primary', note: 'LFP oscillation synchrony across cortico-striatal circuits' },
    { id: 'attention_and_salience', relevance: 'primary', note: 'Selective attention, interference resolution, EEG markers' },
    { id: 'working_memory', relevance: 'primary', note: 'WM training, neurofeedback enhancement, aging' },
    { id: 'cognitive_control', relevance: 'primary', note: 'Mindfulness + brain stimulation for cognitive control' },
    { id: 'executive_function', relevance: 'high', note: 'Task-switching, inhibition, TBI-impaired EF' },
    { id: 'decision_making', relevance: 'high', note: 'Intertemporal choice, value computation in depression' },
    { id: 'fear_and_anxiety', relevance: 'high', note: 'PTSD (theta burst stimulation), anxiety disorders' },
    { id: 'sleep_and_oscillations', relevance: 'high', note: 'Sleep architecture disruption in TBI and depression' },
    { id: 'development_and_plasticity', relevance: 'high', note: 'TBI recovery, neuroplasticity through stimulation' },
    { id: 'neuromodulation', relevance: 'high', note: 'Ketamine, cannabis-antidepressant interactions, TMS/tDCS' },
    { id: 'emotional_processing', relevance: 'medium', note: 'Climate trauma, wildfire impacts on emotional regulation' },
    { id: 'social_behavior', relevance: 'medium', note: 'Psychosocial wellbeing, healthcare professional burnout' },
  ],

  // Which of the 20 circuits their research focuses on
  kg_circuits: [
    { id: 'spectral_dynamics', relevance: 'primary', note: 'Core methodology — beta/gamma LFP in striatum and PFC' },
    { id: 'basal_ganglia_loop', relevance: 'primary', note: 'Cortico-striatal connectivity across TBI, reward, depression' },
    { id: 'executive_control', relevance: 'primary', note: 'PFC-parietal attention circuits measured with EEG' },
    { id: 'default_mode', relevance: 'high', note: 'DMN disruption in depression and suicidal ideation' },
    { id: 'reward_addiction', relevance: 'high', note: 'Mesolimbic dopamine in decision-making and addiction' },
    { id: 'sleep_oscillations', relevance: 'high', note: 'Sleep spindle/SWS disruptions and memory consolidation in TBI' },
    { id: 'stress_hpa', relevance: 'high', note: 'HPA dysregulation in PTSD, depression, climate trauma' },
    { id: 'attention_salience', relevance: 'high', note: 'Salience network — interference resolution & mindfulness' },
    { id: 'neuromodulatory', relevance: 'medium', note: 'LC-NE (arousal), serotonin (mood) in depression treatment' },
  ],

  // Key brain regions in their work
  focus_regions: [
    { id: 'orbitofrontal_cortex', note: 'vmOFC modulation for reward/depression — 2023 key paper' },
    { id: 'striatum', note: 'Cortico-striatal connectivity, TBI, reward processing' },
    { id: 'anterior_cingulate_cortex', note: 'Error monitoring, conflict detection, mindfulness effects' },
    { id: 'dlpfc', note: 'Target for TMS/tDCS in working memory and depression' },
    { id: 'hippocampus', note: 'Memory consolidation disrupted in TBI; PTSD HPC volume' },
    { id: 'insular_cortex', note: 'Interoception, salience, mindfulness training effects' },
    { id: 'motor_cortex', note: 'TBI motor recovery, cortico-spinal plasticity' },
    { id: 'nucleus_accumbens', note: 'Reward certainty oscillations, dopamine in depression' },
  ],

  // Technologies / platforms
  technologies: [
    {
      id: 'BrainE',
      label: 'BrainE',
      species: 'human',
      description: 'Mobile wireless EEG platform for human cognitive assessment, neurofeedback, and brain stimulation',
      icon: '🧠',
    },
    {
      id: 'BrainER',
      label: 'BrainER',
      species: 'rodent',
      description: 'Rodent analogue of BrainE — LFP recording during operant cognitive tasks, enabling direct cross-species comparison',
      icon: '🐀',
    },
    {
      id: 'WellMind',
      label: 'WellMind',
      species: 'human',
      description: 'Personalized mood monitoring and digital intervention platform (Nature Translational Psychiatry 2021)',
      icon: '💚',
    },
    {
      id: 'SimBSI',
      label: 'SimBSI',
      species: 'both',
      description: 'Simulink Brain Signal Interface for real-time closed-loop experiments',
      icon: '⚡',
    },
    {
      id: 'DSI',
      label: 'DSI Toolbox',
      species: 'human',
      description: 'Distributed Source Imaging for electromagnetic neural signals',
      icon: '📡',
    },
  ],

  // Key publication highlights — real papers from Google Scholar
  highlight_papers: [
    {
      title: 'Low-frequency cortical activity is a neuromodulatory target that tracks recovery after stroke',
      year: 2018,
      venue: 'Nature Medicine 24(8)',
      citations: 159,
      pi: 'Ramanathan',
      topics: ['spectral_dynamics', 'development_and_plasticity', 'neuromodulation'],
      regions: ['motor_cortex', 'striatum'],
      species: 'rodent',
      significance: 'Low-frequency LFP oscillations as a biomarker and stimulation target for stroke/TBI recovery — directly bridges animal and human neuro-rehabilitation',
    },
    {
      title: 'Sleep-dependent reactivation of ensembles in motor cortex promotes skill consolidation',
      year: 2015,
      venue: 'PLoS Biology 13(9)',
      citations: 210,
      pi: 'Ramanathan',
      topics: ['sleep_and_oscillations', 'development_and_plasticity', 'neural_synchrony'],
      regions: ['motor_cortex', 'hippocampus'],
      species: 'rodent',
      significance: 'Demonstrates sleep replay of learned motor sequences — mechanistic basis for why sleep matters for skill learning, with direct human EEG translation',
    },
    {
      title: 'Neural reactivations during sleep determine network credit assignment',
      year: 2017,
      venue: 'Nature Neuroscience 20(9)',
      citations: 139,
      pi: 'Ramanathan',
      topics: ['sleep_and_oscillations', 'reward_learning', 'neural_synchrony'],
      regions: ['motor_cortex', 'hippocampus'],
      species: 'rodent',
      significance: 'Sleep replay selectively strengthens rewarded actions — a circuit-level mechanism for consolidation of value-based learning',
    },
    {
      title: 'Coupling between motor cortex and striatum increases during sleep over long-term skill learning',
      year: 2021,
      venue: 'eLife 10',
      citations: 64,
      pi: 'Ramanathan',
      topics: ['sleep_and_oscillations', 'spectral_dynamics', 'development_and_plasticity'],
      regions: ['motor_cortex', 'striatum'],
      species: 'rodent',
      significance: 'Cortico-striatal oscillatory coupling during sleep as a readout of consolidated motor skill — translationally measurable with EEG in humans post-TBI',
    },
    {
      title: 'Adaptive training diminishes distractibility in aging across species',
      year: 2014,
      venue: 'Neuron 84(5)',
      citations: 158,
      pi: 'Mishra',
      topics: ['attention_and_salience', 'cognitive_control', 'development_and_plasticity'],
      regions: ['prefrontal_cortex', 'anterior_cingulate_cortex'],
      species: 'both',
      significance: 'Same training protocol improves attention in both aging rats and humans — the canonical NEATLabs cross-species validation paper',
    },
    {
      title: 'Personalized machine learning of depressed mood using wearables',
      year: 2021,
      venue: 'Translational Psychiatry 11(1)',
      citations: 148,
      pi: 'both',
      topics: ['emotional_processing', 'cognitive_control', 'neuromodulation'],
      regions: ['prefrontal_cortex', 'anterior_cingulate_cortex'],
      species: 'human',
      significance: 'WellMind platform — ML models on wearable + EEG data predict day-to-day mood in depression, enabling personalized digital therapeutics',
    },
    {
      title: 'Closed-loop digital meditation improves sustained attention in young adults',
      year: 2019,
      venue: 'Nature Human Behaviour 3(7)',
      citations: 122,
      pi: 'Mishra',
      topics: ['attention_and_salience', 'cognitive_control', 'neural_synchrony'],
      regions: ['anterior_cingulate_cortex', 'insular_cortex', 'dlpfc'],
      species: 'human',
      significance: 'BrainE closed-loop neurofeedback during meditation significantly improves sustained attention — proof-of-concept for real-time cognitive augmentation',
    },
    {
      title: 'Open-source Raspberry Pi-based operant box for translational behavioral testing in rodents',
      year: 2020,
      venue: 'Journal of Neuroscience Methods 342',
      citations: 36,
      pi: 'Ramanathan',
      topics: ['reward_learning', 'decision_making', 'cognitive_control'],
      regions: ['orbitofrontal_cortex', 'striatum'],
      species: 'rodent',
      highlight: 'contributed',
      significance: 'Open-source BrainER platform for standardized rodent cognitive testing — enables cross-lab replication and direct comparison with human BrainE paradigms',
    },
    {
      title: 'Mapping cognitive brain functions at scale',
      year: 2021,
      venue: 'NeuroImage 231',
      citations: 43,
      pi: 'both',
      topics: ['attention_and_salience', 'working_memory', 'executive_function'],
      regions: ['dlpfc', 'anterior_cingulate_cortex', 'insular_cortex', 'posterior_parietal_cortex'],
      species: 'human',
      significance: 'Large-scale EEG mapping of cognitive function across tasks — directly comparable to this KG\'s coverage map by brain region and topic',
    },
    {
      title: 'Effects of intranasal (S)-ketamine on veterans with co-morbid treatment-resistant depression and PTSD',
      year: 2022,
      venue: 'EClinicalMedicine 48',
      citations: 41,
      pi: 'Ramanathan',
      topics: ['neuromodulation', 'fear_and_anxiety', 'emotional_processing'],
      regions: ['prefrontal_cortex', 'hippocampus', 'anterior_cingulate_cortex'],
      species: 'human',
      significance: 'Ketamine as rapid-acting antidepressant for dual-diagnosis veterans — neuroplasticity-based mechanism informed by animal circuit studies',
    },
    {
      title: 'Differences in interference processing and frontal brain function with climate trauma',
      year: 2023,
      venue: 'PLOS Climate 2(1)',
      citations: 53,
      pi: 'both',
      topics: ['attention_and_salience', 'emotional_processing', 'executive_function'],
      regions: ['anterior_cingulate_cortex', 'dlpfc'],
      species: 'human',
      significance: 'First EEG study of wildfire trauma impacts on frontal cortex — novel clinical frontier connecting environmental events to measurable circuit dysfunction',
    },
    {
      title: 'Video games for neuro-cognitive optimization',
      year: 2016,
      venue: 'Neuron 90(2)',
      citations: 203,
      pi: 'Mishra',
      topics: ['attention_and_salience', 'cognitive_control', 'working_memory'],
      regions: ['prefrontal_cortex', 'anterior_cingulate_cortex'],
      species: 'human',
      significance: 'Action game training improves multitasking and attention — behavioral evidence that closed-loop training generalizes to real-world cognition',
    },
  ],

  // The core translational argument — why cross-species matters
  translational_rationale: {
    title: 'Animal → Human Translation',
    steps: [
      {
        species: 'rodent',
        platform: 'BrainER',
        what: 'Causal circuit manipulation (optogenetics, LFP recording)',
        why: 'Establish ground-truth circuit mechanisms — impossible in humans',
      },
      {
        species: 'both',
        platform: 'Shared paradigms',
        what: 'Identical cognitive tasks (reward, attention, working memory) in rats & humans',
        why: 'Align neural signatures across species to validate rodent-to-human translation',
      },
      {
        species: 'human',
        platform: 'BrainE',
        what: 'Non-invasive EEG + closed-loop stimulation',
        why: 'Apply mechanistic knowledge to diagnose, predict, and treat human disorders',
      },
    ],
    insight:
      'By standardizing cognitive paradigms and neural metrics across species, NEATLabs creates ' +
      'a two-way bridge: rodent models reveal mechanisms, human data reveals individual variability — ' +
      'together building the foundation for precision psychiatry.',
  },

  // Clinical trials
  clinical_trials: [
    { nct: 'NCT06399406', title: 'Mindfulness Engaged Brain Stimulation' },
    { nct: 'NCT06675240', title: 'Meditative Neurofeedback' },
  ],

  funding: ['NIMH', 'Veterans Affairs', 'Burroughs Wellcome Fund', 'NINDS', 'Tang Prize Foundation', 'Hope for Depression Research Foundation'],

  // What this KG enables for NEATLab — the pitch
  kg_value_propositions: [
    {
      icon: '🗺️',
      title: 'Map your work in context',
      description:
        'Every NEATLab paper plotted on the global topic timeline — see how your findings on beta oscillations, ' +
        'reward, and TBI fit the arc of the field from 1990 to today.',
    },
    {
      icon: '🐀↔️🧠',
      title: 'Cross-species dataset discovery',
      description:
        'Query datasets by circuit (e.g., cortico-striatal), filter by species (rat vs. human), and ' +
        'identify which rodent LFP studies share paradigms with your BrainER experiments.',
    },
    {
      icon: '🕳️',
      title: 'Identify research gaps',
      description:
        'The coverage gap analysis flags brain regions your circuits target that have fewer than ' +
        '5 datasets — your next replication target or funding story.',
    },
    {
      icon: '🔗',
      title: 'Citation ancestry',
      description:
        'Trace which foundational papers the field cites most in reward, oscillations, and TBI — ' +
        'understand who you are building on and who is building on you.',
    },
    {
      icon: '🏷️',
      title: 'Standardize cross-lab comparison',
      description:
        'The KG\'s shared ontology (region IDs, task labels, modality tags) creates a common vocabulary ' +
        'across your rodent and human datasets — enabling automated meta-analysis.',
    },
    {
      icon: '📡',
      title: 'Neurostimulation target discovery',
      description:
        'Filter the corpus for datasets where TMS, tDCS, or optogenetics was applied to a target region — ' +
        'discover unpublished or under-cited evidence for your stimulation protocols.',
    },
  ],
} as const

// ── Sid Hulyalkar's own publications from this lab ────────────────────────────

export const SID_PAPERS = [
  {
    title: 'Beta and high gamma oscillations in the cortico-striatal network reflect reward certainty on a probabilistic reversal learning task',
    year: 2025,
    venue: 'Journal of Neuroscience 45',
    citations: 4,
    topics: ['spectral_dynamics', 'reward_learning', 'cognitive_control'],
    regions: ['striatum', 'prefrontal_cortex', 'orbitofrontal_cortex'],
    species: 'rodent',
    significance: 'Core thesis paper: beta and high-gamma LFP in cortico-striatal circuits encode reward certainty probabilistically — mechanistic basis for translating oscillatory biomarkers to humans',
    co_authors: ['MF Koloski', 'M Salimi', 'T Tang', 'SA Barnes', 'J Mishra', 'DS Ramanathan'],
  },
  {
    title: 'Cortico-striatal beta oscillations as a reward-related signal',
    year: 2024,
    venue: 'Cognitive, Affective, & Behavioral Neuroscience',
    citations: 9,
    topics: ['spectral_dynamics', 'reward_learning'],
    regions: ['striatum', 'prefrontal_cortex'],
    species: 'rodent',
    significance: 'Establishes beta oscillations as a readout of learned reward value in cortico-striatal circuits — first formal characterization of the signal in this task context',
    co_authors: ['MF Koloski', 'SA Barnes', 'J Mishra', 'DS Ramanathan'],
  },
  {
    title: 'Electrophysiological correlates of rodent default-mode network suppression revealed by large-scale local field potential recordings',
    year: 2021,
    venue: 'Cerebral Cortex Communications',
    citations: 22,
    topics: ['neural_synchrony', 'spectral_dynamics', 'cognitive_control'],
    regions: ['prefrontal_cortex', 'posterior_parietal_cortex', 'hippocampus'],
    species: 'rodent',
    significance: 'Direct electrophysiological evidence for rodent DMN suppression during task engagement — validates the use of rodent LFP to model human fMRI default mode phenomena',
    co_authors: ['L Fakhraei', 'M Francoeur', 'PP Balasubramani', 'T Tang', 'DS Ramanathan'],
  },
  {
    title: 'Mapping large-scale networks associated with action, behavioral inhibition and impulsivity',
    year: 2021,
    venue: 'eNeuro',
    citations: 17,
    topics: ['attention_and_salience', 'cognitive_control', 'executive_function'],
    regions: ['prefrontal_cortex', 'anterior_cingulate_cortex', 'striatum'],
    species: 'rodent',
    significance: 'Maps cortico-striatal networks for impulsivity and inhibition control using multi-site LFP — bridges rodent circuit data with human fMRI inhibition networks',
    co_authors: ['L Fakhraei', 'M Francoeur', 'P Balasubramani', 'T Tang', 'DS Ramanathan'],
  },
  {
    title: 'Chronic, multi-site recordings supported by two low-cost probe designs for single unit or LFP activity in behaving rats',
    year: 2021,
    venue: 'Frontiers in Psychiatry',
    citations: 18,
    topics: ['neural_synchrony', 'spectral_dynamics'],
    regions: ['striatum', 'prefrontal_cortex'],
    species: 'rodent',
    significance: 'Open-source probe design enabling affordable large-scale chronic recordings in rodents — methods paper that underpins the entire BrainER translational platform',
    co_authors: ['MJ Francoeur', 'T Tang', 'L Fakhraei', 'X Wu', 'J Cramer', 'DS Ramanathan'],
  },
  {
    title: 'Open-source Raspberry Pi-based operant box for translational behavioral testing in rodents',
    year: 2020,
    venue: 'Journal of Neuroscience Methods 342',
    citations: 36,
    topics: ['reward_learning', 'decision_making'],
    regions: ['orbitofrontal_cortex', 'striatum'],
    species: 'rodent',
    significance: 'BrainER hardware platform — standardized, open-source operant box enabling cross-lab replication and direct comparison with human BrainE paradigms',
    co_authors: ['N Buscher', 'A Ojeda', 'M Francoeur', 'C Claros', 'T Tang', 'A Terry', 'DS Ramanathan'],
  },
] as const

export type SidPaper = typeof SID_PAPERS[number]

export type NeatLabsTopic = typeof NEATLABS.kg_topics[number]
export type NeatLabsCircuit = typeof NEATLABS.kg_circuits[number]
export type NeatLabsPaper = typeof NEATLABS.highlight_papers[number]
export type NeatLabsTech = typeof NEATLABS.technologies[number]
