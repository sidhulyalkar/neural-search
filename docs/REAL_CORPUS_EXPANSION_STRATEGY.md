# Real Corpus Expansion Strategy

This document provides actionable guidance for expanding Neural Search beyond curated demo data to real-world scientific datasets from DANDI, OpenNeuro, and linked literature via OpenAlex.

## Goals

1. **Breadth**: Cover the full ontology (reinforcement learning, motor control, sensory decision-making, BCI, speech/language, clinical neurophysiology, social behavior, naturalistic behavior).
2. **Quality**: Every ingested record should have provenance, confidence scores, and QA state.
3. **Linkage**: Datasets should be linked to associated papers where possible.
4. **Analysis Readiness**: Records should indicate which analyses they can support.

---

## 1. DANDI Archive Priorities

### Target Coverage by Domain

| Domain | Target Datasets | Priority Queries | Notes |
|--------|----------------|------------------|-------|
| Reinforcement Learning | 50 | reversal learning, bandit, temporal difference, reward prediction | Focus on reward-aligned ephys/fiber photometry |
| Decision Making | 40 | go nogo, 2AFC, perceptual decision | Prioritize trial-structured data |
| Motor Control | 40 | reaching, center-out, grasping, motor sequence | Kinematics required |
| Visual Processing | 30 | visual cortex, natural movie, receptive field | Allen Brain data high priority |
| BCI | 20 | motor imagery, cursor control, speech BCI | ECoG and array data |
| Clinical | 20 | epilepsy, seizure, interictal, sleep | Human iEEG priority |
| Social/Naturalistic | 20 | social interaction, free behavior, foraging | Pose tracking data |

### Query Strategy

```yaml
# Phase 1: Core Domains (HIGH priority)
dandi_queries_phase1:
  - query: "reversal learning"
    limit: 30
    filter: has_nwb=true
  - query: "go nogo"
    limit: 30
    filter: has_nwb=true
  - query: "reaching motor"
    limit: 25
    filter: has_neural=true
  - query: "visual cortex neuropixels"
    limit: 25
    filter: null
  - query: "decision making choice"
    limit: 25
    filter: has_behavior=true

# Phase 2: Extended Coverage (MEDIUM priority)
dandi_queries_phase2:
  - query: "reinforcement learning dopamine"
    limit: 20
  - query: "motor imagery BCI"
    limit: 15
  - query: "speech production ECoG"
    limit: 15
  - query: "social behavior"
    limit: 15
  - query: "epilepsy iEEG seizure"
    limit: 15

# Phase 3: Gap Filling (after coverage analysis)
dandi_queries_phase3:
  - query: "foraging exploration"
    limit: 10
  - query: "sleep staging"
    limit: 10
  - query: "temporal difference"
    limit: 10
  - query: "sequence learning"
    limit: 10
```

### NWB Inspection Fields

For each DANDI dataset, extract and validate these NWB fields:

```yaml
nwb_required_fields:
  # Identification
  - session_description
  - identifier
  - session_start_time
  - experimenter
  - institution
  - lab

  # Subject
  - subject.species
  - subject.age
  - subject.sex
  - subject.subject_id
  - subject.genotype  # if applicable

  # Neural Data
  - acquisition.ElectricalSeries  # ephys
  - acquisition.TwoPhotonSeries   # calcium imaging
  - acquisition.FiberPhotometry   # photometry
  - electrodes.location           # brain regions
  - electrodes.group              # probe info
  - electrodes.filtering

  # Behavioral Data
  - intervals.trials              # trial structure
  - intervals.trials.start_time
  - intervals.trials.stop_time
  - processing.behavior           # behavioral events
  - stimulus                      # stimulus info

nwb_helpful_fields:
  - devices                       # hardware info
  - electrode_groups.device
  - units                         # sorted spikes
  - units.spike_times
  - units.waveform_mean
  - processing.ecephys.LFP
  - stimulus.presentation         # stimulus timing

nwb_quality_indicators:
  - file_size_bytes               # data richness indicator
  - num_electrodes
  - num_units
  - num_trials
  - session_duration_seconds
```

### DANDI API Fields to Extract

```python
dandi_metadata_extraction = {
    # Core identifiers
    "dandiset_id": "identifier",
    "version": "version",
    "name": "name",
    "description": "description",
    "url": "url",

    # Contributors
    "contributors": "contributor[]",  # parse name, affiliation
    "contact_person": "contributor[role=ContactPerson]",

    # Scientific context
    "species": "assetsSummary.species[]",
    "approach": "assetsSummary.approach[]",
    "measurement_technique": "assetsSummary.measurementTechnique[]",
    "variables_measured": "assetsSummary.variablesMeasured[]",

    # Data standards
    "data_standard": ["NWB", "BIDS"],  # infer from files
    "num_files": "assetsSummary.numberOfFiles",
    "total_size": "assetsSummary.size",

    # Related resources
    "related_papers": "relatedResource[relation=IsDerivedFrom]",
    "related_code": "relatedResource[relation=IsSupplementTo]",

    # Access
    "access": "access[]",
    "license": "license",
    "embargo_until": "embargoedUntil",
}
```

---

## 2. OpenNeuro Priorities

### Target Coverage by Domain

| Domain | Target Datasets | Priority Queries | Notes |
|--------|----------------|------------------|-------|
| Motor Imagery BCI | 40 | motor imagery EEG, BCI | Standard BCI Competition format |
| Clinical EEG | 30 | epilepsy, seizure, sleep | BIDS EEG format |
| Cognitive EEG | 30 | decision making, working memory, attention | Event-related designs |
| Speech/Language | 20 | speech perception, language, reading | MEG/EEG language data |
| Stop Signal | 15 | stop signal, response inhibition | Cognitive control focus |

### Query Strategy

```yaml
# Phase 1: Core Domains
openneuro_queries_phase1:
  - query: "motor imagery"
    limit: 40
    filter: modality=eeg
  - query: "epilepsy"
    limit: 25
    filter: modality=ieeg OR modality=eeg
  - query: "BCI"
    limit: 20
  - query: "decision making"
    limit: 20
    filter: modality=eeg OR modality=meg

# Phase 2: Extended Coverage
openneuro_queries_phase2:
  - query: "sleep"
    limit: 15
    filter: has_psg=true OR modality=eeg
  - query: "stop signal"
    limit: 15
  - query: "language speech"
    limit: 15
  - query: "working memory"
    limit: 15
```

### BIDS Inspection Fields

For each OpenNeuro dataset, extract and validate these BIDS fields:

```yaml
bids_required_fields:
  # Dataset level (dataset_description.json)
  - Name
  - BIDSVersion
  - DatasetType
  - License

  # Participants (participants.tsv)
  - participant_id
  - age
  - sex
  - species  # if available

  # Sessions/Runs
  - session_id
  - run_id
  - task_name
  - task_description

  # Modality-specific
  # EEG
  - eeg.SamplingFrequency
  - eeg.EEGChannelCount
  - eeg.EEGReference
  - eeg.PowerLineFrequency

  # iEEG
  - ieeg.iEEGReference
  - ieeg.iEEGChannelCount
  - electrodes.tsv  # electrode locations

  # MEG
  - meg.SamplingFrequency
  - meg.MEGChannelCount

  # Events
  - events.tsv
  - events.onset
  - events.duration
  - events.trial_type

bids_helpful_fields:
  - README
  - CHANGES
  - participants.json  # column descriptions
  - task_description
  - coordsystem.json
  - channels.tsv

bids_quality_indicators:
  - num_subjects
  - num_sessions_per_subject
  - num_runs_per_session
  - unique_trial_types
  - has_derivatives
```

### OpenNeuro API Fields to Extract

```python
openneuro_metadata_extraction = {
    # Core identifiers
    "dataset_id": "id",
    "name": "draft.description.Name",
    "description": "draft.description.Description",
    "readme": "draft.readme",

    # Contributors
    "authors": "draft.description.Authors[]",
    "acknowledgements": "draft.description.Acknowledgements",

    # Data summary
    "num_subjects": "draft.summary.subjects",
    "num_sessions": "draft.summary.sessions",
    "num_files": "draft.summary.totalFiles",
    "size_bytes": "draft.summary.size",
    "modalities": "draft.summary.modalities[]",
    "tasks": "draft.summary.tasks[]",

    # Standards
    "bids_version": "draft.description.BIDSVersion",
    "data_type": "draft.description.DatasetType",

    # Related
    "related_papers": "draft.description.ReferencesAndLinks[]",
    "funding": "draft.description.Funding[]",
    "license": "draft.description.License",
}
```

---

## 3. OpenAlex Linking Strategy

### Purpose

Link papers to datasets bidirectionally:
1. **Paper → Dataset**: Find data availability statements, repository links, DOIs
2. **Dataset → Paper**: Find papers that cite or describe the dataset

### Query Strategy

```yaml
# Primary approach: Topic-based queries matching ontology
openalex_queries_by_domain:
  reinforcement_learning:
    - "reversal learning electrophysiology"
    - "temporal difference dopamine neuron"
    - "reward prediction error neural recording"
    - "multi-armed bandit prefrontal cortex"

  motor_control:
    - "motor cortex reaching neural population"
    - "brain computer interface motor decoding"
    - "movement trajectory neural recording"

  clinical:
    - "epilepsy intracranial EEG"
    - "seizure detection deep learning"
    - "sleep staging polysomnography"

# Secondary approach: Institution/lab-based queries
openalex_queries_by_source:
  - "Allen Institute visual cortex"
  - "International Brain Laboratory"
  - "Janelia motor cortex"

# Tertiary approach: Dataset-specific searches
openalex_dataset_linking:
  - query_template: '"{dandi_dataset_name}" DANDI'
  - query_template: '"{openneuro_id}" OpenNeuro'
  - query_template: 'DOI:"{related_doi}"'
```

### Linking Heuristics

```python
linking_rules = {
    # Strong links (confidence > 0.9)
    "strong_links": [
        "Paper mentions dataset DOI or accession number",
        "Paper lists dataset URL in data availability",
        "Dataset metadata includes paper DOI",
        "Same first author and matching topic",
    ],

    # Moderate links (confidence 0.6-0.9)
    "moderate_links": [
        "Paper mentions dataset name (fuzzy match)",
        "Same institution + similar topic + same year",
        "Paper methods describe same recording setup",
    ],

    # Weak links (confidence 0.3-0.6)
    "weak_links": [
        "Same task type and species",
        "Citation network suggests connection",
        "Co-author overlap",
    ],
}
```

### OpenAlex Fields to Extract

```python
openalex_extraction = {
    # Core identifiers
    "openalex_id": "id",
    "doi": "doi",
    "title": "title",
    "abstract": "abstract_inverted_index",  # reconstruct
    "publication_date": "publication_date",
    "publication_year": "publication_year",

    # Authors and institutions
    "authors": "authorships[].author",
    "institutions": "authorships[].institutions[]",
    "corresponding_authors": "authorships[].is_corresponding",

    # Classification
    "concepts": "concepts[]",  # topic tags
    "primary_topic": "primary_topic",
    "topics": "topics[]",

    # Metrics
    "cited_by_count": "cited_by_count",
    "counts_by_year": "counts_by_year[]",
    "is_oa": "open_access.is_oa",
    "oa_url": "open_access.oa_url",

    # Related works
    "referenced_works": "referenced_works[]",
    "related_works": "related_works[]",

    # Source
    "journal": "primary_location.source",
    "type": "type",  # journal-article, preprint, etc.
}
```

---

## 4. NWB/BIDS File Inspection Protocol

### Automated Inspection Pipeline

```python
def inspect_nwb_file(file_path: str) -> Dict:
    """
    Deep inspection of NWB file for corpus enrichment.

    Returns structured metadata with confidence scores.
    """
    import pynwb

    with pynwb.NWBHDF5IO(file_path, 'r') as io:
        nwb = io.read()

        inspection = {
            # Session metadata
            "session_id": nwb.identifier,
            "session_description": nwb.session_description,
            "session_start_time": str(nwb.session_start_time),
            "experimenter": nwb.experimenter,
            "institution": nwb.institution,
            "lab": nwb.lab,

            # Subject
            "subject": {
                "species": nwb.subject.species if nwb.subject else None,
                "age": nwb.subject.age if nwb.subject else None,
                "sex": nwb.subject.sex if nwb.subject else None,
                "genotype": getattr(nwb.subject, 'genotype', None),
            },

            # Neural data inventory
            "neural_data": {
                "has_ephys": len(nwb.acquisition) > 0,
                "num_electrodes": len(nwb.electrodes) if nwb.electrodes else 0,
                "brain_regions": extract_brain_regions(nwb),
                "has_units": hasattr(nwb, 'units') and len(nwb.units) > 0,
                "num_units": len(nwb.units) if hasattr(nwb, 'units') else 0,
            },

            # Trial structure
            "trial_structure": {
                "has_trials": hasattr(nwb, 'trials') and nwb.trials is not None,
                "num_trials": len(nwb.trials) if nwb.trials else 0,
                "trial_columns": list(nwb.trials.colnames) if nwb.trials else [],
            },

            # Behavioral data
            "behavior": {
                "has_behavior": 'behavior' in nwb.processing,
                "behavior_types": extract_behavior_types(nwb),
            },

            # Stimulus
            "stimulus": {
                "has_stimulus": len(nwb.stimulus) > 0,
                "stimulus_types": list(nwb.stimulus.keys()),
            },
        }

    return inspection


def extract_brain_regions(nwb) -> List[str]:
    """Extract unique brain region labels from electrode table."""
    if nwb.electrodes is None:
        return []

    regions = set()
    if 'location' in nwb.electrodes.colnames:
        for loc in nwb.electrodes['location'][:]:
            if loc:
                regions.add(str(loc))
    return sorted(regions)


def extract_behavior_types(nwb) -> List[str]:
    """Extract behavioral data types from processing modules."""
    types = []
    if 'behavior' in nwb.processing:
        behavior_module = nwb.processing['behavior']
        types = list(behavior_module.data_interfaces.keys())
    return types
```

### BIDS Inspection Pipeline

```python
def inspect_bids_dataset(dataset_path: str) -> Dict:
    """
    Deep inspection of BIDS dataset for corpus enrichment.
    """
    from bids import BIDSLayout

    layout = BIDSLayout(dataset_path)

    inspection = {
        # Dataset level
        "dataset_description": layout.description,
        "bids_version": layout.description.get('BIDSVersion'),
        "dataset_name": layout.description.get('Name'),

        # Subjects
        "subjects": layout.get_subjects(),
        "num_subjects": len(layout.get_subjects()),

        # Sessions
        "sessions": layout.get_sessions(),
        "num_sessions": len(layout.get_sessions()),

        # Tasks
        "tasks": layout.get_tasks(),
        "task_descriptions": extract_task_descriptions(layout),

        # Modalities
        "modalities": layout.get_datatypes(),
        "suffixes": layout.get_suffixes(),

        # Events
        "has_events": len(layout.get(suffix='events')) > 0,
        "event_columns": extract_event_columns(layout),
        "trial_types": extract_trial_types(layout),

        # Derivatives
        "has_derivatives": 'derivatives' in os.listdir(dataset_path),

        # Electrodes (for iEEG)
        "has_electrodes": len(layout.get(suffix='electrodes')) > 0,
        "electrode_locations": extract_electrode_locations(layout),
    }

    return inspection


def extract_task_descriptions(layout) -> Dict[str, str]:
    """Extract task descriptions from task JSON files."""
    descriptions = {}
    for task in layout.get_tasks():
        task_files = layout.get(task=task, extension='.json')
        for f in task_files:
            meta = f.get_metadata()
            if 'TaskDescription' in meta:
                descriptions[task] = meta['TaskDescription']
                break
    return descriptions


def extract_trial_types(layout) -> List[str]:
    """Extract unique trial types from events files."""
    import pandas as pd

    trial_types = set()
    events_files = layout.get(suffix='events', extension='.tsv')

    for ef in events_files[:5]:  # Sample up to 5 files
        df = pd.read_csv(ef.path, sep='\t')
        if 'trial_type' in df.columns:
            trial_types.update(df['trial_type'].dropna().unique())

    return sorted(trial_types)
```

---

## 5. Human QA Protocol

### QA States

```yaml
qa_states:
  - id: UNREVIEWED
    description: "Automated extraction only, no human validation"
    confidence_ceiling: 0.7

  - id: PENDING_REVIEW
    description: "Flagged for human review (low confidence or conflicts)"
    confidence_ceiling: 0.7

  - id: IN_REVIEW
    description: "Currently being reviewed by a human"
    confidence_ceiling: 0.7

  - id: NEEDS_REVISION
    description: "Review found issues requiring correction"
    confidence_ceiling: 0.5

  - id: REVIEWED
    description: "Human verified, may have minor issues"
    confidence_ceiling: 0.9

  - id: TRUSTED
    description: "High-quality, fully verified record"
    confidence_ceiling: 1.0

  - id: REJECTED
    description: "Dataset does not meet quality threshold"
    confidence_ceiling: 0.0
```

### Review Assignment Rules

```yaml
review_priority_scoring:
  # Higher score = higher priority for review
  factors:
    - name: "Low extraction confidence"
      condition: "confidence < 0.6"
      weight: +30

    - name: "High analysis readiness"
      condition: "readiness_score >= 8"
      weight: +25

    - name: "Core domain match"
      condition: "matches_core_ontology_domain"
      weight: +20

    - name: "Has linked papers"
      condition: "linked_papers_count >= 1"
      weight: +15

    - name: "Recent publication"
      condition: "publication_year >= 2023"
      weight: +10

    - name: "NWB/BIDS compliant"
      condition: "data_standard in ['NWB', 'BIDS']"
      weight: +10

    - name: "Missing required fields"
      condition: "missing_required_fields_count > 0"
      weight: +15  # Need review to assess

reviewer_assignment:
  domains:
    reinforcement_learning:
      required_expertise: ["RL modeling", "decision neuroscience"]
    motor_control:
      required_expertise: ["motor systems", "BCI"]
    clinical:
      required_expertise: ["clinical neurophysiology", "epilepsy/sleep"]
    speech_language:
      required_expertise: ["speech neuroscience", "language"]
```

### Review Checklist

```markdown
## Dataset Card Review Checklist

### Metadata Accuracy (Required)
- [ ] Title accurately describes the dataset
- [ ] Description matches actual content (checked against README/paper)
- [ ] Species labels are correct and specific
- [ ] Brain region labels use standard nomenclature (Allen CCF, MNI, etc.)
- [ ] Modality labels are complete (all recording types listed)
- [ ] Task labels match experimental paradigm
- [ ] Behavioral event labels match actual events in data

### Scientific Labels (Required)
- [ ] Task ontology mapping is correct
- [ ] Behavior labels capture key measured behaviors
- [ ] Analysis affordances are realistic (not over-promised)
- [ ] Missing fields are correctly identified

### Provenance (Required)
- [ ] Source URL is valid and accessible
- [ ] Dataset ID matches source archive
- [ ] License is correctly identified
- [ ] Linked papers are relevant (if any)

### Quality Assessment (Required)
- [ ] Readiness score (1-10) reflects actual usability
- [ ] Strengths accurately describe advantages
- [ ] Limitations honestly describe issues
- [ ] No obvious data quality problems noted in source

### Data Inspection (If accessible)
- [ ] Sample file opens without error
- [ ] Key data structures present as described
- [ ] Trial count/session count approximately matches metadata
- [ ] Event timestamps present and reasonable

### Decision
- [ ] **APPROVE** - Card is accurate and complete
- [ ] **APPROVE WITH NOTES** - Minor issues documented
- [ ] **NEEDS REVISION** - Corrections required (list below)
- [ ] **REJECT** - Does not meet quality threshold (reason required)
```

### Reviewer Workflow

```bash
# 1. Get next batch of datasets to review
python -m neural_search.review list \
  --status PENDING_REVIEW \
  --domain reinforcement_learning \
  --limit 10

# 2. Open review interface for specific dataset
python -m neural_search.review start \
  --dataset-id dandi:000123 \
  --reviewer "reviewer_name"

# 3. Mark review complete with decision
python -m neural_search.review complete \
  --dataset-id dandi:000123 \
  --decision APPROVE \
  --notes "Verified against paper DOI:10.1234/example"

# 4. For rejections or revisions
python -m neural_search.review complete \
  --dataset-id dandi:000456 \
  --decision NEEDS_REVISION \
  --corrections "task_labels: add 'reversal_learning'" \
  --corrections "brain_regions: change 'PFC' to 'mPFC'"

# 5. Generate review progress report
python -m neural_search.review progress \
  --output data/reports/review_progress.md
```

### Quality Metrics

```yaml
review_quality_metrics:
  # Reviewer performance
  - name: "Reviews per week"
    target: 20

  - name: "Average review time"
    target: "< 5 minutes per dataset"

  - name: "Inter-rater agreement"
    target: "> 0.85 Cohen's kappa"

  # Corpus quality
  - name: "Reviewed coverage"
    target: "> 50% of high-priority datasets"

  - name: "Rejection rate"
    target: "< 10% (indicates good automated filtering)"

  - name: "Revision rate"
    target: "< 20% (indicates good automated extraction)"
```

---

## 6. Implementation Sequence

### Phase 1: Infrastructure (Week 1)

```yaml
tasks:
  - name: "Set up ingestion pipeline"
    subtasks:
      - DANDI API client with rate limiting
      - OpenNeuro GraphQL client
      - OpenAlex API client
      - Raw response storage

  - name: "Set up NWB/BIDS inspection"
    subtasks:
      - NWB inspection module
      - BIDS inspection module
      - Inspection result storage

  - name: "Set up QA workflow"
    subtasks:
      - Review state machine
      - Reviewer assignment logic
      - Review CLI commands
```

### Phase 2: Ingestion (Weeks 2-3)

```yaml
tasks:
  - name: "DANDI Phase 1 ingestion"
    queries: [reversal learning, go nogo, reaching motor, visual cortex]
    target: 100 datasets

  - name: "OpenNeuro Phase 1 ingestion"
    queries: [motor imagery, epilepsy, BCI]
    target: 100 datasets

  - name: "OpenAlex linking"
    queries: [domain-based, institution-based]
    target: 500 papers linked
```

### Phase 3: QA (Weeks 3-4)

```yaml
tasks:
  - name: "Automated QA triage"
    subtasks:
      - Run confidence scoring
      - Flag low-confidence records
      - Prioritize for review

  - name: "Human review"
    target: 100 reviewed datasets

  - name: "Quality report"
    subtasks:
      - Coverage by domain
      - Quality distribution
      - Gaps and recommendations
```

### Phase 4: Iteration (Ongoing)

```yaml
tasks:
  - name: "Gap filling ingestion"
    based_on: coverage report gaps

  - name: "Re-review failed records"

  - name: "Paper-to-dataset linking refinement"

  - name: "Benchmark evaluation"
    using: benchmark_queries_v06.yaml
```

---

## Appendix: Field Mapping Reference

### Species Normalization

```yaml
species_mapping:
  # Mice
  mouse: "Mus musculus"
  "mus musculus": "Mus musculus"
  "m. musculus": "Mus musculus"

  # Rats
  rat: "Rattus norvegicus"
  "rattus norvegicus": "Rattus norvegicus"

  # Primates
  macaque: "Macaca mulatta"
  "rhesus macaque": "Macaca mulatta"
  "macaca mulatta": "Macaca mulatta"
  monkey: "Macaca mulatta"  # default to rhesus

  # Humans
  human: "Homo sapiens"
  "homo sapiens": "Homo sapiens"
  "homo sapien": "Homo sapiens"  # common typo
```

### Brain Region Normalization

```yaml
brain_region_mapping:
  # Prefrontal
  mPFC: "medial prefrontal cortex"
  PFC: "prefrontal cortex"
  dlPFC: "dorsolateral prefrontal cortex"
  vmPFC: "ventromedial prefrontal cortex"
  OFC: "orbitofrontal cortex"
  ACC: "anterior cingulate cortex"

  # Motor
  M1: "primary motor cortex"
  PMd: "dorsal premotor cortex"
  PMv: "ventral premotor cortex"
  SMA: "supplementary motor area"

  # Visual
  V1: "primary visual cortex"
  MT: "middle temporal area"
  LIP: "lateral intraparietal area"
  FEF: "frontal eye field"

  # Hippocampal
  CA1: "hippocampus CA1"
  CA3: "hippocampus CA3"
  DG: "dentate gyrus"
  EC: "entorhinal cortex"

  # Striatum
  NAc: "nucleus accumbens"
  dStr: "dorsal striatum"
  VS: "ventral striatum"
```

### Modality Normalization

```yaml
modality_mapping:
  # Electrophysiology
  neuropixels: "Neuropixels"
  utah_array: "Utah array"
  ephys: "extracellular electrophysiology"
  single_unit: "single-unit recording"

  # Imaging
  two_photon: "two-photon calcium imaging"
  calcium_imaging: "calcium imaging"
  fiber_photometry: "fiber photometry"
  miniscope: "miniscope imaging"

  # Human
  EEG: "electroencephalography"
  ECoG: "electrocorticography"
  iEEG: "intracranial EEG"
  sEEG: "stereo EEG"
  MEG: "magnetoencephalography"
  fMRI: "functional MRI"
```
