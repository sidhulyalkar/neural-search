# Adversarial Benchmark Scientific Rationale

**Version:** v0.3.0
**Created:** 2026-05-23
**Total Queries:** 35

---

## Purpose

The adversarial benchmark contains challenging queries designed to expose gaps in the neural search system. Unlike the regression benchmark (`demo_v02`), these queries are expected to have lower pass rates initially. They serve as:

1. **Scientific stress tests** - Queries that require deep domain understanding
2. **Exclusion tests** - Queries with explicit negative constraints
3. **Affordance tests** - Queries about what analyses a dataset supports
4. **Ambiguity tests** - Natural language queries with vague phrasing
5. **Missingness tests** - Queries about incomplete metadata
6. **Linking tests** - Queries about paper-dataset relationships

---

## Query Categories and Scientific Rationale

### Category 1: Hard Negative / Exclusion Queries (7 queries)

These test the system's ability to filter out datasets that match positive criteria but violate negative constraints.

| ID | Query Summary | Scientific Rationale |
|----|---------------|----------------------|
| `adv_001` | OFC reversal learning, exclude fMRI | OFC reversal is common in cognitive neuroscience. Users often need specific modalities (ephys, calcium) for spike-timing or optical analysis, not BOLD signals. |
| `adv_002` | Dopamine RPE, exclude seizure/sleep/auditory | Reward prediction error studies require reward-specific tasks. Seizure monitoring and sleep studies may have dopamine signals but not reward contexts. |
| `adv_003` | Mouse V1, exclude EEG/fMRI | Mouse visual cortex studies use invasive methods (calcium imaging, Neuropixels). EEG/fMRI are rare in mouse neuroscience. |
| `adv_004` | Primate motor cortex reaching, exclude Utah | Some analyses require specific electrode geometries. Excluding Utah arrays tests modality-level exclusion. |
| `adv_005` | Human speech iEEG, exclude fMRI | iEEG provides ms-resolution for speech decoding. fMRI language studies have different temporal resolution and analysis goals. |
| `adv_006` | Reversal learning ephys, exclude calcium/photometry | Some analyses require spike timing (e.g., credit assignment). Optical methods have different temporal resolution. |
| `adv_007` | Hippocampus spatial, exclude VR/head-fixed | Place cell properties differ between real-world and virtual navigation. Paradigm-level exclusion tests semantic understanding. |

### Category 2: Analysis Affordance Queries (8 queries)

These test whether the system understands what analyses a dataset can support based on its structure and content.

| ID | Query Summary | Scientific Rationale |
|----|---------------|----------------------|
| `adv_008` | Q-learning model fitting | RL model fitting requires explicit choice, reward, and outcome variables. Many datasets lack proper trial structure. |
| `adv_009` | Continuous kinematics for motor decoding | Motor decoding requires continuous hand/cursor position, not just trial events. Tests granularity awareness. |
| `adv_010` | Trial-aligned neural activity | PSTH/event-alignment requires precise event timestamps. Many datasets claim "trials" without proper intervals. |
| `adv_011` | Population dynamics with missing metadata | Tests awareness of both potential (calcium imaging exists) and gaps (session metadata missing). |
| `adv_012` | BCI decoding with neural-behavioral pairs | BCI training requires synchronized neural and behavioral streams. Tests cross-modal alignment awareness. |
| `adv_013` | Sleep stage classification | Sleep staging requires expert annotations (REM/NREM labels). Tests clinical affordance. |
| `adv_014` | Spike sorting + connectivity | Functional connectivity needs sorted units. Tests processing level awareness. |
| `adv_015` | RL modeling with possible missing reward timing | Tests partial affordance - dataset may have choices but lack precise reward timestamps. |

### Category 3: Ambiguous Natural Language Queries (6 queries)

These test the system's ability to expand vague or colloquial queries into structured searches.

| ID | Query Summary | Scientific Rationale |
|----|---------------|----------------------|
| `adv_016` | "reward in the brain" | Vague but meaningful. Should expand to reward-related regions and behaviors. |
| `adv_017` | "neural data from decision tasks" | "Decision" spans many task types. Should expand appropriately. |
| `adv_018` | "animals make choices" | Colloquial phrasing for choice behavior in non-human subjects. |
| `adv_019` | "brain recordings during behavior" | Extremely vague. Should return datasets with both neural and behavioral components. |
| `adv_020` | "how mice learn" | Natural language learning query. Should expand to learning-related tasks. |
| `adv_021` | "front of the brain" | Colloquial anatomical reference. Should map to prefrontal regions. |

### Category 4: Missing Metadata Awareness Queries (5 queries)

These test the system's ability to surface datasets with known gaps or infer missing information.

| ID | Query Summary | Scientific Rationale |
|----|---------------|----------------------|
| `adv_022` | Decoding potential but missing behavioral labels | Neural data may exist without behavioral annotation. Surface for potential curation. |
| `adv_023` | NWB without trial structure | NWB format doesn't guarantee trial tables. Tests format vs. content awareness. |
| `adv_024` | Calcium imaging without ROI segmentation | Raw calcium data requires processing. Tests processing level metadata. |
| `adv_025` | Spike sorting status unknown | Users need to know if they must sort spikes. Tests processing transparency. |
| `adv_026` | Probable behavior videos not in metadata | Many datasets have videos not reflected in structured metadata. Tests inference. |

### Category 5: Paper-Dataset Linking Queries (4 queries)

These test the system's ability to connect datasets to related literature.

| ID | Query Summary | Scientific Rationale |
|----|---------------|----------------------|
| `adv_027` | Datasets linked to dopamine RPE papers | Connect datasets to conceptually related literature. |
| `adv_028` | Papers using mouse reversal learning | Find methodology papers for specific paradigms. |
| `adv_029` | Datasets used in BCI publications | Published datasets may have stronger validation. |
| `adv_030` | Recent neuropixels decision-making papers | Neuropixels is recent technology; tests temporal awareness. |

### Category 6: Complex Multi-Constraint Queries (5 queries)

These test precision under multiple simultaneous constraints.

| ID | Query Summary | Scientific Rationale |
|----|---------------|----------------------|
| `adv_031` | Mouse mPFC calcium go/no-go lick NWB | Very specific. Tests precision with 6+ constraints. |
| `adv_032` | Human iEEG speech DANDI phoneme-level | Tests clinical + archive + annotation specificity. |
| `adv_033` | Primate M1 Utah reaching for population analysis | Classic motor neuroscience benchmark preparation. |
| `adv_034` | OpenNeuro BIDS EEG motor imagery left-right | Classic BCI benchmark with defined expected outputs. |
| `adv_035` | Rodent striatum fiber photometry probabilistic reward with omission | Omission trials critical for RPE studies. Tests paradigm detail. |

---

## Expected Failure Modes

### Exclusion Failures
- System returns fMRI when "exclude fMRI" specified
- System returns Utah array when alternatives requested
- System doesn't parse "NOT" or "but not" constructs

### Affordance Failures
- System ranks datasets without trial structure for trial-aligned analysis
- System doesn't surface missing required fields for analysis
- System over-claims analysis support without evidence

### Ambiguity Failures
- System doesn't expand colloquial terms
- System returns overly narrow results for broad queries
- System returns irrelevant results for vague queries

### Missingness Failures
- System doesn't surface datasets with known gaps
- System doesn't distinguish raw from processed data
- System doesn't track spike sorting status

### Linking Failures
- System can't find datasets from paper descriptions
- System can't identify papers relevant to datasets
- System doesn't use citation information

---

## Metrics Interpretation

| Metric | Target (v0.3) | Interpretation |
|--------|---------------|----------------|
| Mean P@5 | 50-70% | Adversarial queries are hard; 50%+ is good |
| Mean Label Recall@10 | 60-80% | Many expected labels may not be in corpus |
| Exclusion Correctness | 80%+ | Critical for user trust |
| Task Match Rate | 80%+ | Should understand tasks well |
| Modality Match Rate | 90%+ | Modalities are explicit |

---

## Iteration Process

1. **Initial run:** Document baseline performance
2. **Failure analysis:** Categorize why queries fail
3. **Targeted fixes:** Address specific failure modes
4. **Re-run:** Measure improvement
5. **Add new queries:** As gaps are found, add test cases

---

## Non-Goals

- Achieving 100% pass rate (these are designed to be hard)
- Tuning system specifically to pass adversarial queries
- Using adversarial benchmark for regression testing

---

## References

- [benchmark_queries_adversarial.yaml](../data/eval/benchmark_queries_adversarial.yaml)
