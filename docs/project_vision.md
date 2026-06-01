# Project Vision

Neural Search is a **provenance-backed research-object retrieval and experimental design engine** for neural and behavioral data.

## Mission

Enable researchers to discover, evaluate, and reuse neuroscience datasets through experiment-aware search that understands scientific intent, verifies claims with evidence, and accelerates the path from research question to analysis.

## Core Thesis

Generic semantic search and generic RAG are insufficient for scientific dataset reuse.

**RAG answers questions by retrieving text and generating prose.** Neural Search retrieves **research objects**. A good result is not a paragraph—it is a dataset with:

| Component | Description |
|-----------|-------------|
| **Experimental Labels** | Task type, behavioral events, brain regions, species, modality—grounded in a scientific ontology |
| **Metadata Constraints** | Structured filters that can be precisely satisfied or violated |
| **Provenance** | Source archive IDs, linked papers, extraction confidence, evidence chains |
| **Analysis Affordances** | What analyses this dataset can actually support, with required signals identified |
| **Match Evidence** | Explanation of why this result matches the query, with supporting quotes and labels |
| **Quality Assessment** | Readiness score, strengths, limitations, missing fields, QA state |
| **Reusable Artifacts** | Dataset card, starter notebook, design reference |

## Design Principles

### 1. Experiment-Aware

Search understands the structure of neuroscience experiments:

- **Tasks**: Go/no-go, reversal learning, reaching, motor imagery, speech production
- **Behaviors**: Choice, reward, reaction time, kinematics, pupil, seizure onset
- **Modalities**: Neuropixels, ECoG, calcium imaging, EEG, fiber photometry
- **Regions**: mPFC, VTA, motor cortex, hippocampus, STG
- **Species**: Mouse, rat, macaque, human
- **Analyses**: Q-learning modeling, choice decoding, seizure detection, sleep staging

The ontology is not just vocabulary—it powers matching, ranking, filtering, and explanation.

### 2. Provenance-First

Every claim needs evidence:

- **Source IDs**: DANDI accession, OpenNeuro dataset ID, paper DOI
- **Extraction Confidence**: How reliable is each extracted label?
- **Evidence Chains**: Where did this label come from? NWB field? Paper abstract? Manual annotation?
- **Linked Literature**: Papers that describe, cite, or reuse this dataset
- **QA State**: Unreviewed, pending, reviewed, trusted, rejected

Users should be able to trace any assertion back to its source.

### 3. Analysis-Ready

Results should be actionable:

- **Analysis Affordances**: Can this dataset support Q-learning modeling? Does it have the required signals?
- **Readiness Scoring**: Not all datasets are equally usable—score and explain quality
- **Starter Notebooks**: Generate code that loads the data and scaffolds the analysis
- **Design Reference**: What parameters did similar experiments use?

### 4. Evaluation-Visible

The system should know when it succeeds or fails:

- **Benchmark Queries**: 90+ queries across dataset search, adversarial constraints, paper linking, affordance matching, graph reasoning, and experimental design
- **Hard Negatives**: Queries designed to test precision under adversarial conditions
- **Coverage Reports**: Which domains are well-covered? Where are the gaps?
- **Quality Audits**: Systematic assessment of corpus metadata quality

### 5. Human-Reviewable

Automated extraction is fallible:

- **QA Workflow**: Unreviewed → Pending → In Review → Reviewed/Trusted/Rejected
- **Review Checklist**: Structured validation of metadata accuracy
- **Confidence Calibration**: System confidence should correlate with actual accuracy
- **Feedback Loop**: Human corrections improve extraction over time

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Neural Search System                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      Research Query Layer                        │    │
│  │  Natural Language → Structured Intent → Ontology Grounding       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                   │                                      │
│                                   ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Hybrid Retrieval Engine                       │    │
│  │  Text Embeddings + Ontology Match + Metadata Filters + Graph    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                   │                                      │
│                                   ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      Corpus Knowledge Base                       │    │
│  │                                                                  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │    │
│  │  │   Datasets   │  │    Papers    │  │   Ontology   │           │    │
│  │  │  (DANDI,     │──│  (OpenAlex)  │──│   (Tasks,    │           │    │
│  │  │  OpenNeuro)  │  │              │  │   Behaviors) │           │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │    │
│  │          │                 │                 │                   │    │
│  │          └─────────────────┴─────────────────┘                   │    │
│  │                            │                                     │    │
│  │                  ┌─────────▼─────────┐                           │    │
│  │                  │  Knowledge Graph  │                           │    │
│  │                  │  (Relationships)  │                           │    │
│  │                  └───────────────────┘                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                   │                                      │
│                                   ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                       AI Agent Layer                            │    │
│  │                                                                  │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │    │
│  │  │  Dataset   │ │  Paper-    │ │  Exper.    │ │  Notebook  │   │    │
│  │  │  Discovery │ │  Dataset   │ │  Design    │ │  Generation│   │    │
│  │  │  Agent     │ │  Linking   │ │  Agent     │ │  Agent     │   │    │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │    │
│  │                                                                  │    │
│  │                    ┌────────────────────┐                        │    │
│  │                    │  Benchmark/Audit   │                        │    │
│  │                    │  Agent             │                        │    │
│  │                    └────────────────────┘                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Domain Coverage (v0.6)

| Domain | Task Examples | Modality Focus |
|--------|---------------|----------------|
| **Reinforcement Learning** | Reversal learning, bandit, TD learning, foraging, effort-based decision | Fiber photometry, Neuropixels, calcium imaging |
| **Decision Making** | Go/no-go, 2AFC, set shifting, stop-signal | Extracellular ephys, EEG |
| **Sensory Decision** | Random dot motion, detection, categorization | Visual cortex recordings |
| **Motor Control** | Reaching, grasping, sequence learning, force tracking | Utah array, ECoG, EMG |
| **BCI** | Motor imagery, P300, SSVEP, speech BCI, cursor control | EEG, ECoG, Utah array |
| **Speech/Language** | Speech production, perception, reading, naming | ECoG, iEEG, MEG |
| **Clinical** | Seizure monitoring, sleep staging, interictal detection | iEEG, EEG, polysomnography |
| **Social/Naturalistic** | Social interaction, free behavior, foraging, prey capture | Calcium imaging, pose tracking |

## Analysis Affordance Framework

Neural Search doesn't just find datasets—it tells you what you can do with them:

| Affordance | Required Signals | Example Use Case |
|------------|------------------|------------------|
| `q_learning_model_fitting` | Choice, reward, trial structure | Fit RL models to behavioral data |
| `temporal_difference_modeling` | Reward timing, neural activity | Analyze dopamine RPE signals |
| `choice_decoding` | Neural data, choice labels | Decode decisions from population activity |
| `motor_decoding` | Neural data, kinematics | Build movement decoders |
| `speech_decoding` | Neural data, phoneme labels | Speech neuroprosthesis development |
| `seizure_detection` | Neural data, seizure annotations | Clinical seizure detector training |
| `sleep_stage_classification` | EEG, sleep annotations | Automated sleep staging |
| `population_dynamics` | Multi-unit data, trial structure | Latent dynamics analysis |
| `functional_connectivity` | Multi-region data | Network analysis |
| `pose_estimation` | Behavior video | Automated pose tracking |

## Provenance Model

Every dataset record maintains a complete provenance chain:

```yaml
dataset_record:
  # Source identification
  source_archive: "dandi"
  source_id: "000123"
  source_url: "https://dandiarchive.org/dandiset/000123"
  ingestion_timestamp: "2024-01-15T10:30:00Z"
  raw_response_path: "data/raw/dandi/000123.json"

  # Extraction provenance
  extraction:
    method: "rule_based_v0.3 + llm_assist_v0.2"
    confidence: 0.85
    evidence:
      tasks:
        - label: "reversal_learning"
          source: "description"
          evidence: "subjects performed a probabilistic reversal learning task"
          confidence: 0.92
      brain_regions:
        - label: "OFC"
          source: "nwb_electrodes_location"
          confidence: 0.98

  # Linked resources
  linked_papers:
    - doi: "10.1234/example"
      relation: "describes"
      link_confidence: 0.95
      link_evidence: "doi_in_data_availability"

  # QA state
  qa:
    state: "reviewed"
    reviewer: "reviewer_a"
    review_date: "2024-01-20"
    corrections_applied: 1
```

## Experimental Design Support

Neural Search helps researchers design experiments by providing:

1. **Reference Designs**: Find datasets using the same paradigm
2. **Parameter Recommendations**: Typical timing, trial counts, conditions
3. **Recording Guidance**: Target regions, probe configurations
4. **Analysis Pipelines**: Standard preprocessing and analysis workflows
5. **Design Validation**: Check completeness and identify potential confounds

## Future Directions

### Latent Neural-State Search

Beyond metadata and text, index learned representations of neural dynamics:

- **Neural Embeddings**: Encode population activity patterns
- **Behavioral Embeddings**: Encode movement, choice, state trajectories
- **Cross-Dataset Alignment**: Find datasets with similar neural dynamics
- **Interpretable Anchors**: Connect latent matches to ontology terms

### Multi-Modal Retrieval

Retrieve across data types:

- "Find datasets with neural activity similar to this firing pattern"
- "Find tasks with behavioral structure like this trial sequence"
- "Find papers with figures showing this type of result"

### Collaborative Annotation

Scale human curation:

- Community-contributed annotations
- Expert review pipelines
- Annotation quality scoring
- Incentive mechanisms

### Real-Time Corpus Expansion

Continuous integration of new data:

- Archive monitoring (DANDI, OpenNeuro releases)
- Paper monitoring (new publications)
- Automated ingestion with human-in-the-loop QA

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Search Precision@5** | > 0.7 (easy), > 0.5 (hard) | Benchmark evaluation |
| **Label Recall@10** | > 0.8 | Expected labels found |
| **Constraint Satisfaction** | 1.0 | Adversarial query compliance |
| **Hard Negative Rejection** | > 0.95 | Incorrect results excluded |
| **Corpus Coverage** | > 80% of ontology terms | Domain audit |
| **QA Throughput** | 20 datasets/week/reviewer | Review metrics |
| **User Task Completion** | > 70% | User study |

## Relationship to Existing Tools

| Tool | Focus | Neural Search Difference |
|------|-------|-------------------------|
| **DANDI/OpenNeuro** | Archive access | Adds semantic search, cross-archive, analysis affordances |
| **PubMed/OpenAlex** | Paper search | Links papers to data, extracts experimental details |
| **Google Dataset Search** | General datasets | Neuroscience-specific ontology and affordances |
| **ChatGPT/Claude** | General Q&A | Grounded in verified corpus, not hallucinated |
| **Semantic Scholar** | Paper understanding | Focus on datasets and experimental reuse |

## Design Philosophy

**Be useful, be honest, be verifiable.**

- **Useful**: Help researchers find data they can actually use
- **Honest**: Report confidence, flag limitations, show evidence
- **Verifiable**: Every claim traceable to source

Neural Search succeeds when a researcher can go from "I need data for X" to "Here's a notebook analyzing dataset Y for X" with confidence that the match is real and the analysis is appropriate.
