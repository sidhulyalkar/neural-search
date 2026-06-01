# Corpus Expansion Sprint Plan v0.1

**Goal**: Build a high-quality v0.1 corpus around decision-making, motor control, visual decision-making, BCI, and clinical neurophysiology.

## Targets

| Metric | Target | Priority |
|--------|--------|----------|
| DANDI datasets | 100 | High |
| OpenNeuro datasets | 100 | High |
| OpenAlex papers | 500 | Medium |
| Reviewed/trusted dataset cards | 50 | High |
| Benchmark queries | 50 | High |
| Generated notebooks | 10 | Medium |

---

## 1. Source Search Queries

### DANDI Archive Queries

#### Decision-Making Domain (40 datasets target)
```yaml
decision_making:
  - query: "go nogo"
    expected_yield: 15
    priority: HIGH
  - query: "reversal learning"
    expected_yield: 10
    priority: HIGH
  - query: "delay discounting"
    expected_yield: 8
    priority: HIGH
  - query: "two-alternative forced choice"
    expected_yield: 10
    priority: MEDIUM
  - query: "probabilistic learning"
    expected_yield: 5
    priority: MEDIUM
  - query: "reward prediction"
    expected_yield: 8
    priority: MEDIUM
  - query: "decision confidence"
    expected_yield: 5
    priority: LOW
```

#### Visual Decision-Making Domain (20 datasets target)
```yaml
visual_decision_making:
  - query: "visual discrimination"
    expected_yield: 10
    priority: HIGH
  - query: "visual decision making"
    expected_yield: 8
    priority: HIGH
  - query: "orientation discrimination"
    expected_yield: 5
    priority: MEDIUM
  - query: "motion discrimination"
    expected_yield: 5
    priority: MEDIUM
  - query: "perceptual decision"
    expected_yield: 8
    priority: HIGH
```

#### Motor Control Domain (20 datasets target)
```yaml
motor_control:
  - query: "reaching grasping"
    expected_yield: 10
    priority: HIGH
  - query: "motor cortex"
    expected_yield: 15
    priority: HIGH
  - query: "arm movement"
    expected_yield: 8
    priority: MEDIUM
  - query: "motor planning"
    expected_yield: 5
    priority: MEDIUM
  - query: "grip force"
    expected_yield: 3
    priority: LOW
```

#### BCI Domain (10 datasets target)
```yaml
bci:
  - query: "brain computer interface"
    expected_yield: 5
    priority: HIGH
  - query: "neural prosthesis"
    expected_yield: 3
    priority: MEDIUM
  - query: "cursor control"
    expected_yield: 3
    priority: MEDIUM
  - query: "motor imagery"
    expected_yield: 5
    priority: HIGH
```

#### Clinical Neurophysiology (10 datasets target)
```yaml
clinical:
  - query: "epilepsy ieeg"
    expected_yield: 8
    priority: HIGH
  - query: "seizure ecog"
    expected_yield: 5
    priority: HIGH
  - query: "intracranial eeg"
    expected_yield: 8
    priority: MEDIUM
  - query: "stereo eeg"
    expected_yield: 3
    priority: LOW
```

### OpenNeuro Queries

#### Motor & BCI Domain (40 datasets target)
```yaml
motor_bci:
  - query: "motor imagery eeg"
    expected_yield: 15
    priority: HIGH
  - query: "motor execution"
    expected_yield: 10
    priority: HIGH
  - query: "movement related"
    expected_yield: 8
    priority: MEDIUM
  - query: "BCI"
    expected_yield: 10
    priority: HIGH
  - query: "brain machine interface"
    expected_yield: 5
    priority: MEDIUM
```

#### Clinical Domain (30 datasets target)
```yaml
clinical:
  - query: "ieeg seizure"
    expected_yield: 10
    priority: HIGH
  - query: "epilepsy"
    expected_yield: 15
    priority: HIGH
  - query: "intracranial"
    expected_yield: 8
    priority: MEDIUM
  - query: "ecog"
    expected_yield: 10
    priority: HIGH
```

#### Decision-Making Domain (30 datasets target)
```yaml
decision_making:
  - query: "decision making fmri"
    expected_yield: 10
    priority: MEDIUM
  - query: "reward learning"
    expected_yield: 8
    priority: HIGH
  - query: "reinforcement learning"
    expected_yield: 10
    priority: HIGH
  - query: "gambling task"
    expected_yield: 5
    priority: MEDIUM
  - query: "stop signal"
    expected_yield: 8
    priority: HIGH
```

### OpenAlex Queries

#### Core Methodology Papers (200 papers target)
```yaml
methods:
  - query: "reversal learning electrophysiology"
    expected_yield: 50
    priority: HIGH
  - query: "go nogo task neural"
    expected_yield: 40
    priority: HIGH
  - query: "two-alternative forced choice neural recording"
    expected_yield: 30
    priority: MEDIUM
  - query: "delay discounting neuroimaging"
    expected_yield: 30
    priority: MEDIUM
  - query: "intertemporal choice electrophysiology"
    expected_yield: 25
    priority: MEDIUM
```

#### BCI & Motor Papers (150 papers target)
```yaml
bci_motor:
  - query: "brain computer interface motor cortex"
    expected_yield: 50
    priority: HIGH
  - query: "neural decoding movement"
    expected_yield: 40
    priority: HIGH
  - query: "motor imagery classification"
    expected_yield: 30
    priority: MEDIUM
  - query: "reach grasp neural population"
    expected_yield: 30
    priority: MEDIUM
```

#### Clinical Papers (150 papers target)
```yaml
clinical:
  - query: "intracranial eeg epilepsy"
    expected_yield: 50
    priority: HIGH
  - query: "ecog seizure prediction"
    expected_yield: 40
    priority: HIGH
  - query: "stereo eeg cognitive"
    expected_yield: 30
    priority: MEDIUM
  - query: "human single neuron"
    expected_yield: 30
    priority: MEDIUM
```

---

## 2. Ingestion Commands

### Phase 1: High Priority (Week 1)

```bash
# DANDI - Decision Making Core
python -m neural_search.ingestion.dandi --query "go nogo" --limit 25 --save-raw
python -m neural_search.ingestion.dandi --query "reversal learning" --limit 25 --save-raw
python -m neural_search.ingestion.dandi --query "visual decision making" --limit 25 --save-raw
python -m neural_search.ingestion.dandi --query "delay discounting" --limit 15 --save-raw

# OpenNeuro - Motor & Clinical Core
python -m neural_search.ingestion.openneuro --query "motor imagery eeg" --limit 25 --save-raw
python -m neural_search.ingestion.openneuro --query "ieeg seizure" --limit 25 --save-raw
python -m neural_search.ingestion.openneuro --query "epilepsy" --limit 25 --save-raw
python -m neural_search.ingestion.openneuro --query "BCI" --limit 15 --save-raw

# OpenAlex - Foundation Papers
python -m neural_search.ingestion.openalex --query "reversal learning electrophysiology" --limit 100 --save-raw
python -m neural_search.ingestion.openalex --query "go nogo task neural" --limit 100 --save-raw
python -m neural_search.ingestion.openalex --query "brain computer interface motor cortex" --limit 100 --save-raw
```

### Phase 2: Medium Priority (Week 2)

```bash
# DANDI - Extended Coverage
python -m neural_search.ingestion.dandi --query "motor cortex" --limit 20 --save-raw
python -m neural_search.ingestion.dandi --query "reaching grasping" --limit 15 --save-raw
python -m neural_search.ingestion.dandi --query "visual discrimination" --limit 15 --save-raw
python -m neural_search.ingestion.dandi --query "two-alternative forced choice" --limit 15 --save-raw
python -m neural_search.ingestion.dandi --query "brain computer interface" --limit 10 --save-raw

# OpenNeuro - Extended Coverage
python -m neural_search.ingestion.openneuro --query "motor execution" --limit 20 --save-raw
python -m neural_search.ingestion.openneuro --query "ecog" --limit 20 --save-raw
python -m neural_search.ingestion.openneuro --query "reinforcement learning" --limit 20 --save-raw
python -m neural_search.ingestion.openneuro --query "stop signal" --limit 15 --save-raw

# OpenAlex - Extended Papers
python -m neural_search.ingestion.openalex --query "intracranial eeg epilepsy" --limit 100 --save-raw
python -m neural_search.ingestion.openalex --query "neural decoding movement" --limit 100 --save-raw
```

### Phase 3: Fill Gaps (Week 3)

```bash
# DANDI - Gap Filling
python -m neural_search.ingestion.dandi --query "probabilistic learning" --limit 10 --save-raw
python -m neural_search.ingestion.dandi --query "reward prediction" --limit 10 --save-raw
python -m neural_search.ingestion.dandi --query "perceptual decision" --limit 10 --save-raw
python -m neural_search.ingestion.dandi --query "epilepsy ieeg" --limit 10 --save-raw

# OpenNeuro - Gap Filling
python -m neural_search.ingestion.openneuro --query "decision making fmri" --limit 15 --save-raw
python -m neural_search.ingestion.openneuro --query "reward learning" --limit 15 --save-raw
python -m neural_search.ingestion.openneuro --query "intracranial" --limit 10 --save-raw

# OpenAlex - Gap Filling
python -m neural_search.ingestion.openalex --query "delay discounting neuroimaging" --limit 50 --save-raw
python -m neural_search.ingestion.openalex --query "motor imagery classification" --limit 50 --save-raw
python -m neural_search.ingestion.openalex --query "ecog seizure prediction" --limit 50 --save-raw
```

### Batch Runner Script

```bash
#!/bin/bash
# scripts/run_corpus_expansion.sh

set -e

PHASE=${1:-1}
LOG_DIR="data/logs/ingestion"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_with_logging() {
    local source=$1
    local query=$2
    local limit=$3
    local log_file="$LOG_DIR/${source}_${TIMESTAMP}.log"

    echo "[$(date)] Running: $source query='$query' limit=$limit" | tee -a "$log_file"
    python -m neural_search.ingestion.$source --query "$query" --limit $limit --save-raw 2>&1 | tee -a "$log_file"
    echo "[$(date)] Completed: $source query='$query'" | tee -a "$log_file"
}

if [ "$PHASE" == "1" ]; then
    echo "=== Running Phase 1: High Priority ==="
    run_with_logging dandi "go nogo" 25
    run_with_logging dandi "reversal learning" 25
    run_with_logging dandi "visual decision making" 25
    run_with_logging openneuro "motor imagery eeg" 25
    run_with_logging openneuro "ieeg seizure" 25
    run_with_logging openalex "reversal learning electrophysiology" 100
fi

# Add phases 2 and 3 similarly...
```

---

## 3. Manual Review Workflow

### Review States

```
PENDING → IN_REVIEW → APPROVED | REJECTED | NEEDS_REVISION
```

### Dataset Card Review Process

#### Step 1: Automated Triage
```python
# Datasets are auto-assigned review priority based on:
# - Analysis readiness score (>= 7 = HIGH priority)
# - Completeness of metadata (all fields present = HIGH)
# - Domain relevance match (core domains = HIGH)
```

#### Step 2: Reviewer Assignment
```yaml
# data/review/assignments.yaml
reviewers:
  - name: "Reviewer A"
    domains: ["decision_making", "visual_decision"]
    capacity: 20  # cards per week
  - name: "Reviewer B"
    domains: ["motor_control", "bci"]
    capacity: 20
  - name: "Reviewer C"
    domains: ["clinical_neurophysiology"]
    capacity: 15
```

#### Step 3: Review Checklist (per card)

```markdown
## Dataset Card Review: {dataset_id}

### Metadata Accuracy
- [ ] Title accurately describes the dataset
- [ ] Description matches actual data content
- [ ] Species labels are correct
- [ ] Modality labels are complete and accurate
- [ ] Brain region labels are anatomically correct
- [ ] Task labels match the experimental paradigm
- [ ] Behavior labels capture measured behaviors

### Analysis Readiness Assessment
- [ ] Readiness score reflects actual usability (1-10)
- [ ] Strengths section is accurate
- [ ] Limitations section identifies real issues
- [ ] Missing fields are correctly identified

### Scientific Labels Verification
- [ ] Task classification is appropriate
- [ ] Behavioral measures are correctly identified
- [ ] Data standards (NWB/BIDS) are verified

### Quality Flags
- [ ] No broken URLs
- [ ] License is correctly identified
- [ ] Linked papers are relevant
- [ ] Suggested analyses are feasible

### Reviewer Notes
_Add any corrections or observations here_

### Decision
- [ ] APPROVE - Card is accurate and complete
- [ ] REJECT - Dataset doesn't meet quality threshold
- [ ] REVISE - Card needs corrections (list below)

Corrections needed:
1.
2.
```

#### Step 4: Batch Review Commands

```bash
# List pending reviews
python -m neural_search.cards.review --status pending --limit 20

# Open review interface for specific dataset
python -m neural_search.cards.review --dataset-id <id> --interactive

# Bulk approve reviewed cards
python -m neural_search.cards.review --approve-batch data/review/batch_001.yaml

# Export review progress
python -m neural_search.cards.review --export-progress data/reports/review_progress.json
```

### Review Session Template

```yaml
# data/review/session_template.yaml
session:
  reviewer: ""
  date: ""
  duration_minutes: 0

reviews:
  - dataset_id: ""
    card_version: ""
    decision: ""  # APPROVE | REJECT | REVISE
    corrections:
      - field: ""
        old_value: ""
        new_value: ""
    notes: ""
    time_spent_minutes: 0
```

---

## 4. QA Checklist

### Ingestion QA

```yaml
ingestion_qa:
  pre_ingestion:
    - [ ] Database backup completed
    - [ ] Raw response directory has space (>1GB)
    - [ ] API rate limits checked
    - [ ] Previous failed queries resolved

  during_ingestion:
    - [ ] Monitor for HTTP errors (429, 500, 503)
    - [ ] Check for empty responses
    - [ ] Verify record counts match expectations
    - [ ] Watch for duplicate detection

  post_ingestion:
    - [ ] Raw responses saved to data/raw/{source}/
    - [ ] Records inserted into database
    - [ ] No orphaned records (papers without datasets)
    - [ ] Deduplication run completed
    - [ ] Ingestion log reviewed for errors
```

### Dataset Quality QA

```yaml
dataset_qa:
  metadata_completeness:
    - [ ] Title present and meaningful (>10 chars)
    - [ ] Description present (>50 chars)
    - [ ] Source URL valid and accessible
    - [ ] At least one species identified
    - [ ] At least one modality identified
    - [ ] License identified

  label_accuracy:
    - [ ] Tasks match known ontology terms
    - [ ] Behaviors are measurable/observable
    - [ ] Brain regions use standard nomenclature
    - [ ] Modalities use standard terms

  analysis_readiness:
    - [ ] Score between 1-10
    - [ ] Score justification in strengths/limitations
    - [ ] Missing fields identified if score < 7
    - [ ] Suggested analyses are relevant
```

### Card Generation QA

```yaml
card_qa:
  content_quality:
    - [ ] Summary is concise (<200 words)
    - [ ] Why_relevant explains use cases
    - [ ] Strengths are specific and verifiable
    - [ ] Limitations are honest and helpful

  technical_accuracy:
    - [ ] NWB/BIDS compliance correctly detected
    - [ ] File formats accurately identified
    - [ ] Data size estimates reasonable
    - [ ] Subject/session counts accurate

  provenance:
    - [ ] Card version tracked
    - [ ] Generation timestamp recorded
    - [ ] Source extraction confidence logged
```

### Search Quality QA

```yaml
search_qa:
  query_parsing:
    - [ ] Natural language queries parsed correctly
    - [ ] Ontology terms matched appropriately
    - [ ] Structured filters applied correctly

  ranking_quality:
    - [ ] Top 5 results are relevant
    - [ ] Scores correlate with relevance
    - [ ] Evidence snippets are accurate
    - [ ] Match explanations are correct

  benchmark_performance:
    - [ ] Precision@5 >= 0.6 for benchmark queries
    - [ ] Label recall@10 >= 0.7
    - [ ] No regressions from baseline
```

### Notebook Generation QA

```yaml
notebook_qa:
  code_quality:
    - [ ] Notebook executes without errors
    - [ ] Imports are available in standard env
    - [ ] Data paths are correct
    - [ ] TODO comments mark customization points

  content_quality:
    - [ ] Dataset metadata section accurate
    - [ ] Analysis suggestions are relevant
    - [ ] Template matches dataset modality
    - [ ] Documentation cells are helpful
```

---

## 5. Weekly Milestones

### Week 1: Foundation

| Day | Task | Target | Owner |
|-----|------|--------|-------|
| Mon | Set up ingestion infrastructure | Scripts ready | - |
| Mon | Run Phase 1 DANDI ingestion | 90 raw datasets | - |
| Tue | Run Phase 1 OpenNeuro ingestion | 90 raw datasets | - |
| Tue | Run Phase 1 OpenAlex ingestion | 300 papers | - |
| Wed | Deduplication & normalization | Clean corpus | - |
| Wed | Generate initial cards | 180 cards | - |
| Thu | Begin manual review (batch 1) | 15 reviewed | - |
| Fri | Coverage report & gap analysis | Report v1 | - |

**Week 1 Targets:**
- 90 DANDI datasets ingested
- 90 OpenNeuro datasets ingested
- 300 OpenAlex papers ingested
- 15 cards reviewed and approved
- 10 benchmark queries validated

### Week 2: Expansion

| Day | Task | Target | Owner |
|-----|------|--------|-------|
| Mon | Run Phase 2 DANDI ingestion | +60 datasets | - |
| Mon | Run Phase 2 OpenNeuro ingestion | +75 datasets | - |
| Tue | Run Phase 2 OpenAlex ingestion | +200 papers | - |
| Wed | Link papers to datasets | 80% linked | - |
| Wed | Generate cards for new datasets | +135 cards | - |
| Thu | Manual review (batch 2) | +20 reviewed | - |
| Thu | Expand benchmark queries | 30 queries | - |
| Fri | Generate first notebooks | 5 notebooks | - |
| Fri | Coverage report v2 | Gap analysis | - |

**Week 2 Targets:**
- 150 total DANDI datasets
- 165 total OpenNeuro datasets
- 500 total OpenAlex papers
- 35 total cards reviewed
- 30 benchmark queries
- 5 generated notebooks

### Week 3: Refinement

| Day | Task | Target | Owner |
|-----|------|--------|-------|
| Mon | Run Phase 3 gap-filling ingestion | Fill gaps | - |
| Mon | Re-run failed queries | Recover data | - |
| Tue | Quality audit of corpus | Fix issues | - |
| Tue | Update low-quality cards | Improve scores | - |
| Wed | Manual review (batch 3) | +15 reviewed | - |
| Wed | Finalize benchmark suite | 50 queries | - |
| Thu | Generate remaining notebooks | +5 notebooks | - |
| Thu | Run full benchmark evaluation | Baseline set | - |
| Fri | Final coverage report | v0.1 complete | - |
| Fri | Sprint retrospective | Learnings doc | - |

**Week 3 Targets:**
- 100 DANDI datasets (deduplicated)
- 100 OpenNeuro datasets (deduplicated)
- 500 OpenAlex papers
- 50 reviewed/trusted cards
- 50 benchmark queries
- 10 generated notebooks

### Milestone Tracking

```bash
# Daily progress check
make corpus-status

# Weekly milestone report
python -m neural_search.reports.milestone --week 1
```

---

## 6. Corpus Coverage Report Format

### Report Template

```markdown
# Corpus Coverage Report

**Generated**: {timestamp}
**Sprint Week**: {week_number}
**Report Version**: {version}

## Executive Summary

| Metric | Target | Current | % Complete | Status |
|--------|--------|---------|------------|--------|
| DANDI datasets | 100 | {n} | {pct}% | {status_emoji} |
| OpenNeuro datasets | 100 | {n} | {pct}% | {status_emoji} |
| OpenAlex papers | 500 | {n} | {pct}% | {status_emoji} |
| Reviewed cards | 50 | {n} | {pct}% | {status_emoji} |
| Benchmark queries | 50 | {n} | {pct}% | {status_emoji} |
| Generated notebooks | 10 | {n} | {pct}% | {status_emoji} |

## Domain Coverage

### Decision-Making
| Sub-domain | DANDI | OpenNeuro | Papers | Cards |
|------------|-------|-----------|--------|-------|
| Go/No-Go | {n} | {n} | {n} | {n} |
| Reversal Learning | {n} | {n} | {n} | {n} |
| Delay Discounting | {n} | {n} | {n} | {n} |
| 2AFC | {n} | {n} | {n} | {n} |
| Probabilistic | {n} | {n} | {n} | {n} |

### Visual Decision-Making
| Sub-domain | DANDI | OpenNeuro | Papers | Cards |
|------------|-------|-----------|--------|-------|
| Visual Discrimination | {n} | {n} | {n} | {n} |
| Orientation | {n} | {n} | {n} | {n} |
| Motion | {n} | {n} | {n} | {n} |
| Perceptual | {n} | {n} | {n} | {n} |

### Motor Control
| Sub-domain | DANDI | OpenNeuro | Papers | Cards |
|------------|-------|-----------|--------|-------|
| Reaching/Grasping | {n} | {n} | {n} | {n} |
| Motor Planning | {n} | {n} | {n} | {n} |
| Motor Cortex | {n} | {n} | {n} | {n} |

### BCI
| Sub-domain | DANDI | OpenNeuro | Papers | Cards |
|------------|-------|-----------|--------|-------|
| Motor Imagery | {n} | {n} | {n} | {n} |
| Cursor Control | {n} | {n} | {n} | {n} |
| Neural Prosthesis | {n} | {n} | {n} | {n} |

### Clinical Neurophysiology
| Sub-domain | DANDI | OpenNeuro | Papers | Cards |
|------------|-------|-----------|--------|-------|
| Epilepsy/Seizure | {n} | {n} | {n} | {n} |
| iEEG | {n} | {n} | {n} | {n} |
| ECoG | {n} | {n} | {n} | {n} |
| sEEG | {n} | {n} | {n} | {n} |

## Quality Metrics

### Analysis Readiness Distribution
| Score Range | Count | Percentage |
|-------------|-------|------------|
| 8-10 (Excellent) | {n} | {pct}% |
| 6-7 (Good) | {n} | {pct}% |
| 4-5 (Fair) | {n} | {pct}% |
| 1-3 (Poor) | {n} | {pct}% |

### Metadata Completeness
| Field | % Present |
|-------|-----------|
| Description | {pct}% |
| Species | {pct}% |
| Modalities | {pct}% |
| Tasks | {pct}% |
| Brain Regions | {pct}% |
| License | {pct}% |

### Review Status
| Status | Count | Percentage |
|--------|-------|------------|
| Approved | {n} | {pct}% |
| Pending Review | {n} | {pct}% |
| In Review | {n} | {pct}% |
| Needs Revision | {n} | {pct}% |
| Rejected | {n} | {pct}% |

## Gaps & Blockers

### Coverage Gaps
{list of domains/sub-domains below target}

### Quality Issues
{list of systematic quality problems found}

### Ingestion Failures
{summary of failed queries and datasets}

## Next Actions
1. {action_item}
2. {action_item}
3. {action_item}
```

### Report Generation Command

```bash
# Generate coverage report
python -m neural_search.reports.coverage --output data/reports/coverage_{date}.md

# Generate JSON for dashboards
python -m neural_search.reports.coverage --format json --output data/reports/coverage_{date}.json
```

---

## 7. Priority Dataset Categories

### Tier 1: Must-Have (Critical)

```yaml
tier_1_critical:
  description: "Core datasets that define the v0.1 corpus"
  target: 80 datasets

  categories:
    - name: "Canonical Go/No-Go"
      criteria:
        - Has trial-level behavior data
        - Includes neural recordings (any modality)
        - Well-documented task structure
      target: 15

    - name: "Reversal Learning Reference"
      criteria:
        - Multiple reversal blocks
        - Reward/outcome data
        - Neural data aligned to behavior
      target: 15

    - name: "Motor BCI Benchmark"
      criteria:
        - Motor imagery or execution task
        - High-density neural recording
        - Movement kinematics available
      target: 15

    - name: "Clinical iEEG/ECoG"
      criteria:
        - Human intracranial recordings
        - Cognitive task during recording
        - Seizure-free periods identified
      target: 20

    - name: "Visual Decision Gold Standard"
      criteria:
        - Visual discrimination task
        - Visual cortex recordings
        - Trial-by-trial stimulus info
      target: 15
```

### Tier 2: Important (High Value)

```yaml
tier_2_important:
  description: "High-quality datasets that enrich the corpus"
  target: 80 datasets

  categories:
    - name: "Multi-Region Recordings"
      criteria:
        - Simultaneous multi-area recording
        - Identified brain regions
        - Good signal quality
      target: 20

    - name: "Large-Scale Neuropixels"
      criteria:
        - Neuropixels probe data
        - 100+ neurons
        - NWB formatted
      target: 15

    - name: "Calcium Imaging Decision"
      criteria:
        - 2-photon or fiber photometry
        - Decision-related task
        - Cell identification available
      target: 15

    - name: "Naturalistic Behavior"
      criteria:
        - Free behavior or semi-naturalistic
        - Video/tracking data
        - Neural recordings
      target: 15

    - name: "Human Cognitive EEG"
      criteria:
        - Human scalp EEG
        - Cognitive task (not just rest)
        - Event markers aligned
      target: 15
```

### Tier 3: Nice-to-Have (Additional)

```yaml
tier_3_additional:
  description: "Datasets that add breadth and diversity"
  target: 40 datasets

  categories:
    - name: "Cross-Species Comparison"
      criteria:
        - Non-human primate OR non-rodent
        - Comparable task to existing corpus
      target: 10

    - name: "Longitudinal/Learning"
      criteria:
        - Multiple sessions over days/weeks
        - Learning progression visible
      target: 10

    - name: "Novel Modalities"
      criteria:
        - Ultrasound, fNIRS, MEG, etc.
        - Decision or motor task
      target: 10

    - name: "Clinical Populations"
      criteria:
        - Patient populations
        - Matched controls
        - Behavioral data
      target: 10
```

### Priority Scoring Formula

```python
def calculate_priority_score(dataset):
    """
    Priority score for dataset ingestion/review order.
    Range: 0-100, higher = more urgent
    """
    score = 0

    # Domain relevance (0-30)
    if matches_core_domain(dataset):
        score += 30
    elif matches_secondary_domain(dataset):
        score += 15

    # Data quality signals (0-25)
    if dataset.data_standard in ['NWB', 'BIDS']:
        score += 10
    if dataset.has_behavior:
        score += 8
    if dataset.has_trials:
        score += 7

    # Completeness (0-20)
    completeness = count_present_fields(dataset) / total_fields
    score += int(completeness * 20)

    # Recency (0-15)
    if dataset.publication_year >= 2023:
        score += 15
    elif dataset.publication_year >= 2020:
        score += 10
    elif dataset.publication_year >= 2018:
        score += 5

    # Citation/usage signals (0-10)
    if dataset.linked_papers_count >= 3:
        score += 10
    elif dataset.linked_papers_count >= 1:
        score += 5

    return score
```

---

## 8. Failure Logging Format

### Ingestion Failure Log

```yaml
# data/logs/failures/ingestion_{timestamp}.yaml

failures:
  - id: "FAIL_001"
    timestamp: "2024-01-15T10:30:00Z"
    source: "dandi"
    operation: "fetch"
    query: "go nogo"

    error:
      type: "HTTPError"
      code: 429
      message: "Rate limit exceeded"
      traceback: |
        Traceback (most recent call last):
          File "neural_search/ingestion/dandi.py", line 45, in fetch_dandi
            response.raise_for_status()
        httpx.HTTPStatusError: 429 Too Many Requests

    context:
      limit_requested: 25
      records_fetched_before_error: 12
      retry_after_header: "60"

    resolution:
      status: "PENDING"  # PENDING | RESOLVED | WONT_FIX
      action: ""
      resolved_by: ""
      resolved_at: ""
      notes: ""
```

### Normalization Failure Log

```yaml
# data/logs/failures/normalization_{timestamp}.yaml

failures:
  - id: "NORM_001"
    timestamp: "2024-01-15T11:00:00Z"
    source: "openneuro"
    operation: "normalize"
    source_id: "ds004123"

    error:
      type: "ValidationError"
      field: "species"
      message: "Unknown species identifier: 'Homo sapien' (typo?)"
      raw_value: "Homo sapien"

    context:
      raw_record_path: "data/raw/openneuro/ds004123.json"
      attempted_mapping: "species"

    resolution:
      status: "RESOLVED"
      action: "Added synonym mapping 'Homo sapien' -> 'Homo sapiens'"
      resolved_by: "auto"
      resolved_at: "2024-01-15T11:05:00Z"
```

### Card Generation Failure Log

```yaml
# data/logs/failures/card_generation_{timestamp}.yaml

failures:
  - id: "CARD_001"
    timestamp: "2024-01-15T12:00:00Z"
    operation: "generate_card"
    dataset_id: "abc123"

    error:
      type: "ExtractionError"
      stage: "label_extraction"
      message: "Could not extract task labels from description"

    context:
      dataset_title: "Neural recordings during behavior"
      description_length: 45
      description_preview: "Neural recordings during behavior..."

    analysis:
      root_cause: "Description too vague/short"
      confidence: 0.8

    resolution:
      status: "PENDING"
      recommended_action: "Manual review required - add to low-quality queue"
```

### Review Failure Log

```yaml
# data/logs/failures/review_{timestamp}.yaml

failures:
  - id: "REV_001"
    timestamp: "2024-01-15T14:00:00Z"
    operation: "review"
    dataset_id: "def456"
    card_id: "card_789"
    reviewer: "reviewer_a"

    issue:
      type: "AccuracyError"
      severity: "HIGH"  # LOW | MEDIUM | HIGH | CRITICAL
      field: "tasks"

      expected: ["reversal_learning", "probabilistic_reversal"]
      actual: ["go_nogo"]
      evidence: "Paper clearly states 'probabilistic reversal paradigm'"

    resolution:
      status: "RESOLVED"
      action: "Updated task labels, regenerated card"
      resolved_by: "reviewer_a"
      resolved_at: "2024-01-15T14:30:00Z"
```

### Failure Summary Report

```markdown
# Failure Summary Report

**Period**: {start_date} to {end_date}
**Generated**: {timestamp}

## Overview

| Category | Total | Resolved | Pending | Won't Fix |
|----------|-------|----------|---------|-----------|
| Ingestion | {n} | {n} | {n} | {n} |
| Normalization | {n} | {n} | {n} | {n} |
| Card Generation | {n} | {n} | {n} | {n} |
| Review | {n} | {n} | {n} | {n} |

## Top Failure Patterns

1. **{pattern_name}** ({count} occurrences)
   - Root cause: {cause}
   - Recommended fix: {fix}

2. **{pattern_name}** ({count} occurrences)
   - Root cause: {cause}
   - Recommended fix: {fix}

## Pending Actions

| ID | Category | Dataset | Issue | Recommended Action |
|----|----------|---------|-------|-------------------|
| {id} | {cat} | {dataset} | {issue} | {action} |

## Resolution Statistics

- Mean time to resolution: {hours} hours
- Auto-resolved: {pct}%
- Manual intervention required: {pct}%
```

### Failure Logging Commands

```bash
# Log a new failure
python -m neural_search.logging.failure log \
  --category ingestion \
  --source dandi \
  --operation fetch \
  --error-type HTTPError \
  --message "Rate limit exceeded" \
  --query "go nogo"

# View pending failures
python -m neural_search.logging.failure list --status pending

# Mark failure as resolved
python -m neural_search.logging.failure resolve \
  --id FAIL_001 \
  --action "Implemented exponential backoff" \
  --resolved-by "developer_name"

# Generate failure summary
python -m neural_search.logging.failure summary --period week
```

---

## Appendix: Quick Reference

### Make Commands

```bash
make corpus-expand-phase1    # Run Phase 1 ingestion
make corpus-expand-phase2    # Run Phase 2 ingestion
make corpus-expand-phase3    # Run Phase 3 ingestion
make corpus-status           # Show current corpus stats
make corpus-coverage         # Generate coverage report
make corpus-review           # Start review session
make corpus-benchmark        # Run benchmark evaluation
```

### Key File Locations

| Purpose | Path |
|---------|------|
| Raw API responses | `data/raw/{source}/` |
| Curated sources | `data/seed/curated_sources.yaml` |
| Benchmark queries | `data/eval/benchmark_queries.yaml` |
| Coverage reports | `data/reports/coverage_*.md` |
| Failure logs | `data/logs/failures/` |
| Generated notebooks | `data/notebooks/generated/` |
| Review sessions | `data/review/sessions/` |

### Daily Standup Template

```markdown
## Corpus Expansion Standup - {date}

### Yesterday
- Ingested: {n} DANDI, {n} OpenNeuro, {n} papers
- Reviewed: {n} cards
- Resolved: {n} failures

### Today
- [ ] {task_1}
- [ ] {task_2}
- [ ] {task_3}

### Blockers
- {blocker_if_any}

### Metrics
- Total corpus: {n} datasets, {n} papers
- Review progress: {n}/{target} ({pct}%)
- Quality score (avg readiness): {score}
```
