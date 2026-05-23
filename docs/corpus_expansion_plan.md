# Neural Search Corpus Expansion Plan

## Overview

This document outlines the plan to expand the Neural Search corpus from 5 demo datasets to:
- **50-100 DANDI datasets**
- **50-100 OpenNeuro datasets**
- **200-500 OpenAlex papers**
- **25-50 high-quality dataset cards**

## Priority Scientific Areas

1. **Go/NoGo and response inhibition**
2. **Reversal learning and value updating**
3. **Delay discounting and value-based decision-making**
4. **Reaching, motor control, and BCI**
5. **Visual decision-making and naturalistic vision**
6. **Motor imagery EEG**
7. **Seizure monitoring iEEG/ECoG**

---

## 1. Search Queries by Source

### DANDI Archive Queries

```yaml
# Priority 1: Go/NoGo and Response Inhibition
dandi_queries:
  gonogo_response_inhibition:
    - query: "go no-go"
      expected_count: 5-10
    - query: "response inhibition"
      expected_count: 3-5
    - query: "stop signal"
      expected_count: 3-5
    - query: "impulsivity"
      expected_count: 2-4

  # Priority 2: Reversal Learning
  reversal_learning:
    - query: "reversal learning"
      expected_count: 5-8
    - query: "probabilistic reversal"
      expected_count: 2-4
    - query: "contingency reversal"
      expected_count: 2-3
    - query: "rule switching"
      expected_count: 3-5

  # Priority 3: Delay Discounting
  delay_discounting:
    - query: "delay discounting"
      expected_count: 3-5
    - query: "intertemporal choice"
      expected_count: 2-4
    - query: "temporal discounting"
      expected_count: 2-3
    - query: "impulsive choice"
      expected_count: 2-3

  # Priority 4: Reaching and Motor Control
  motor_bci:
    - query: "reaching"
      expected_count: 10-15
    - query: "motor cortex"
      expected_count: 15-20
    - query: "brain computer interface"
      expected_count: 5-10
    - query: "BCI"
      expected_count: 5-8
    - query: "neuropixels motor"
      expected_count: 3-5

  # Priority 5: Visual Decision-Making
  visual_decision:
    - query: "visual decision"
      expected_count: 8-12
    - query: "visual discrimination"
      expected_count: 5-8
    - query: "visual cortex"
      expected_count: 15-20
    - query: "naturalistic vision"
      expected_count: 3-5
    - query: "natural images"
      expected_count: 5-8

  # Priority 6: Motor Imagery (less likely on DANDI)
  motor_imagery:
    - query: "motor imagery"
      expected_count: 1-3
    - query: "imagined movement"
      expected_count: 1-2

  # Priority 7: Seizure/Epilepsy
  clinical_ieeg:
    - query: "epilepsy"
      expected_count: 5-10
    - query: "seizure"
      expected_count: 5-8
    - query: "intracranial EEG"
      expected_count: 5-10
    - query: "ECoG"
      expected_count: 8-12
    - query: "depth electrode"
      expected_count: 3-5
```

### OpenNeuro Queries

```yaml
openneuro_queries:
  # Priority 1: Go/NoGo
  gonogo:
    - query: "go nogo"
      modality: ["eeg", "fmri"]
      expected_count: 5-10
    - query: "stop signal task"
      expected_count: 5-8
    - query: "response inhibition"
      expected_count: 3-5

  # Priority 2: Reversal Learning
  reversal:
    - query: "reversal learning"
      expected_count: 3-5
    - query: "probabilistic learning"
      expected_count: 5-8
    - query: "reward learning"
      expected_count: 5-8

  # Priority 3: Decision Making
  decision_making:
    - query: "delay discounting"
      expected_count: 3-5
    - query: "decision making"
      expected_count: 10-15
    - query: "gambling task"
      expected_count: 5-8
    - query: "Iowa gambling"
      expected_count: 2-4

  # Priority 4: Motor
  motor:
    - query: "motor task"
      expected_count: 8-12
    - query: "finger tapping"
      expected_count: 5-8
    - query: "motor imagery"
      expected_count: 8-12
    - query: "movement"
      expected_count: 10-15

  # Priority 5: Visual
  visual:
    - query: "visual perception"
      expected_count: 8-12
    - query: "visual discrimination"
      expected_count: 3-5
    - query: "object recognition"
      expected_count: 5-8
    - query: "naturalistic"
      expected_count: 3-5

  # Priority 6: Motor Imagery (strong on OpenNeuro)
  motor_imagery_eeg:
    - query: "motor imagery EEG"
      expected_count: 10-15
    - query: "BCI motor imagery"
      expected_count: 5-8
    - query: "imagined movement EEG"
      expected_count: 3-5

  # Priority 7: Epilepsy/iEEG
  clinical:
    - query: "epilepsy"
      expected_count: 5-10
    - query: "seizure EEG"
      expected_count: 3-5
    - query: "intracranial"
      expected_count: 3-5
```

### OpenAlex Paper Queries

```yaml
openalex_queries:
  # Query format: concept + method + species (where applicable)

  # Priority 1: Go/NoGo
  gonogo_papers:
    - query: "go/no-go task neural recording"
      filters: {publication_year: ">2018", cited_by_count: ">10"}
      expected_count: 30-50
    - query: "response inhibition electrophysiology"
      expected_count: 20-30
    - query: "stop signal task neuroscience"
      expected_count: 20-30

  # Priority 2: Reversal Learning
  reversal_papers:
    - query: "reversal learning neural"
      filters: {publication_year: ">2018"}
      expected_count: 40-60
    - query: "orbitofrontal cortex reversal"
      expected_count: 20-30
    - query: "value updating neural"
      expected_count: 15-25

  # Priority 3: Delay Discounting
  discounting_papers:
    - query: "delay discounting neural"
      expected_count: 30-40
    - query: "intertemporal choice fMRI"
      expected_count: 20-30
    - query: "impulsive decision neuroscience"
      expected_count: 20-30

  # Priority 4: Motor/BCI
  motor_papers:
    - query: "motor cortex reaching neural"
      expected_count: 40-60
    - query: "brain computer interface motor"
      expected_count: 30-50
    - query: "neuropixels motor cortex"
      expected_count: 10-20

  # Priority 5: Visual Decision
  visual_papers:
    - query: "visual decision making neural"
      expected_count: 40-60
    - query: "visual cortex decision"
      expected_count: 30-50
    - query: "naturalistic vision neural"
      expected_count: 20-30

  # Priority 6: Motor Imagery
  imagery_papers:
    - query: "motor imagery EEG BCI"
      expected_count: 40-60
    - query: "imagined movement brain"
      expected_count: 20-30

  # Priority 7: Epilepsy/iEEG
  clinical_papers:
    - query: "intracranial EEG cognition"
      expected_count: 30-50
    - query: "ECoG language motor"
      expected_count: 20-30
    - query: "seizure prediction neural"
      expected_count: 20-30
```

---

## 2. Ingestion Commands

### Phase 1: DANDI Ingestion

```bash
# Create output directory
mkdir -p data/ingested/dandi

# Run DANDI ingestion for each priority area
# Priority 1: Go/NoGo
python -m neural_search.ingestion.dandi \
  --query "go no-go" \
  --output data/ingested/dandi/gonogo.json \
  --limit 20

python -m neural_search.ingestion.dandi \
  --query "response inhibition" \
  --output data/ingested/dandi/response_inhibition.json \
  --limit 10

# Priority 2: Reversal Learning
python -m neural_search.ingestion.dandi \
  --query "reversal learning" \
  --output data/ingested/dandi/reversal.json \
  --limit 15

# Priority 3: Delay Discounting
python -m neural_search.ingestion.dandi \
  --query "delay discounting" \
  --output data/ingested/dandi/discounting.json \
  --limit 10

# Priority 4: Motor/BCI
python -m neural_search.ingestion.dandi \
  --query "reaching motor" \
  --output data/ingested/dandi/motor.json \
  --limit 25

python -m neural_search.ingestion.dandi \
  --query "brain computer interface" \
  --output data/ingested/dandi/bci.json \
  --limit 15

# Priority 5: Visual
python -m neural_search.ingestion.dandi \
  --query "visual decision" \
  --output data/ingested/dandi/visual.json \
  --limit 20

# Priority 7: Clinical iEEG
python -m neural_search.ingestion.dandi \
  --query "epilepsy ECoG" \
  --output data/ingested/dandi/clinical.json \
  --limit 20

# Merge all DANDI results
python -m neural_search.ingestion.merge \
  --input-dir data/ingested/dandi/ \
  --output data/ingested/dandi_merged.json
```

### Phase 2: OpenNeuro Ingestion

```bash
mkdir -p data/ingested/openneuro

# Priority 1: Go/NoGo
python -m neural_search.ingestion.openneuro \
  --query "go nogo" \
  --modality eeg \
  --output data/ingested/openneuro/gonogo_eeg.json \
  --limit 15

# Priority 4: Motor
python -m neural_search.ingestion.openneuro \
  --query "motor task" \
  --output data/ingested/openneuro/motor.json \
  --limit 20

# Priority 6: Motor Imagery (focus area for OpenNeuro)
python -m neural_search.ingestion.openneuro \
  --query "motor imagery" \
  --modality eeg \
  --output data/ingested/openneuro/motor_imagery.json \
  --limit 25

# Priority 7: Epilepsy
python -m neural_search.ingestion.openneuro \
  --query "epilepsy" \
  --output data/ingested/openneuro/epilepsy.json \
  --limit 15

# Merge
python -m neural_search.ingestion.merge \
  --input-dir data/ingested/openneuro/ \
  --output data/ingested/openneuro_merged.json
```

### Phase 3: OpenAlex Paper Ingestion

```bash
mkdir -p data/ingested/openalex

# Batch paper ingestion
python -m neural_search.ingestion.openalex \
  --query "go/no-go task neural recording" \
  --min-citations 10 \
  --year-from 2018 \
  --output data/ingested/openalex/gonogo.json \
  --limit 50

python -m neural_search.ingestion.openalex \
  --query "reversal learning neural" \
  --year-from 2018 \
  --output data/ingested/openalex/reversal.json \
  --limit 60

python -m neural_search.ingestion.openalex \
  --query "motor imagery EEG BCI" \
  --output data/ingested/openalex/motor_imagery.json \
  --limit 60

python -m neural_search.ingestion.openalex \
  --query "intracranial EEG cognition" \
  --output data/ingested/openalex/ieeg.json \
  --limit 50

# Merge
python -m neural_search.ingestion.merge \
  --input-dir data/ingested/openalex/ \
  --output data/ingested/openalex_merged.json
```

### Phase 4: Link Papers to Datasets

```bash
# Link papers to datasets based on DOI and author matching
python -m neural_search.ingestion.link_papers \
  --datasets data/ingested/dandi_merged.json \
  --papers data/ingested/openalex_merged.json \
  --output data/ingested/linked_corpus.json
```

### Phase 5: Generate Dataset Cards

```bash
# Generate cards for top datasets
python -m neural_search.cards.generate_all \
  --input data/ingested/linked_corpus.json \
  --output data/cards/ \
  --top-n 50
```

---

## 3. Manual Review Checklist

### Dataset Review Checklist

For each ingested dataset, verify:

```markdown
## Dataset Review: [DATASET_ID]

### Basic Metadata
- [ ] Title is descriptive and accurate
- [ ] Description explains the experimental paradigm
- [ ] Species is correctly identified
- [ ] License allows reuse

### Scientific Content
- [ ] Task/paradigm is correctly labeled
- [ ] Modalities match actual data files
- [ ] Brain regions are anatomically correct
- [ ] Behavioral events are clearly defined

### Data Quality Indicators
- [ ] Has associated publication (DOI/paper)
- [ ] Has trial/event structure
- [ ] Has behavioral variables
- [ ] Data format is standard (NWB/BIDS)

### Flags
- [ ] No obvious data quality issues
- [ ] Metadata is complete (>80% fields filled)
- [ ] No duplicate of existing dataset

### Review Decision
- [ ] ACCEPT - Include in corpus
- [ ] REVISE - Needs metadata correction
- [ ] REJECT - Does not meet criteria

Reviewer: _______________
Date: _______________
Notes: _______________
```

### Paper Review Checklist

```markdown
## Paper Review: [PAPER_ID]

### Relevance
- [ ] Paper describes neural data collection
- [ ] Task matches priority scientific areas
- [ ] Has reusable dataset or clear methods

### Linkage
- [ ] Can be linked to DANDI/OpenNeuro dataset
- [ ] DOI is valid
- [ ] Authors match dataset contributors

### Quality
- [ ] Published in peer-reviewed venue
- [ ] Has citations (>5 for recent papers)
- [ ] Methods are reproducible

### Review Decision
- [ ] ACCEPT - Include in corpus
- [ ] LINK - Link to existing dataset
- [ ] REJECT - Not relevant

Reviewer: _______________
Date: _______________
```

---

## 4. Dataset Card QA Checklist

### Automated Checks

```python
# Run automated QA
python -m neural_search.cards.qa \
  --input data/cards/ \
  --report data/qa/card_qa_report.json
```

### QA Criteria

```markdown
## Dataset Card QA: [CARD_ID]

### Summary Quality
- [ ] Summary is 1-3 sentences
- [ ] Does not invent claims beyond metadata
- [ ] Mentions task and modality
- [ ] Score: ___/5

### Label Accuracy
- [ ] Task labels match ontology
- [ ] All labels have evidence strings
- [ ] Confidence scores are reasonable (>0.7)
- [ ] No hallucinated labels
- [ ] Score: ___/5

### Analysis Readiness
- [ ] Score reflects actual data quality
- [ ] Strengths are accurate
- [ ] Limitations are honest
- [ ] Suggested analyses are appropriate
- [ ] Score: ___/5

### Provenance
- [ ] Source is correctly attributed
- [ ] Linked papers are accurate
- [ ] Claim policy is stated
- [ ] Score: ___/5

### Overall
- Total Score: ___/20
- Status: PASS (>15) / REVIEW (10-15) / FAIL (<10)

QA Reviewer: _______________
Date: _______________
```

### QA Acceptance Criteria

| Criterion | Minimum Threshold |
|-----------|------------------|
| Summary quality | 3/5 |
| Label accuracy | 4/5 |
| Analysis readiness | 3/5 |
| Provenance | 4/5 |
| **Total score** | **15/20** |

---

## 5. Final Corpus Report Format

### Report Structure

```markdown
# Neural Search Corpus Report v1.0

Generated: YYYY-MM-DD

## Executive Summary

- Total datasets indexed: XX
- Total papers linked: XX
- High-quality cards generated: XX
- Coverage across priority areas: XX%

## Corpus Statistics

### By Source
| Source | Count | % of Total |
|--------|-------|------------|
| DANDI | XX | XX% |
| OpenNeuro | XX | XX% |
| Manual | XX | XX% |

### By Priority Area
| Scientific Area | Datasets | Papers | Cards |
|-----------------|----------|--------|-------|
| Go/NoGo & Response Inhibition | XX | XX | XX |
| Reversal Learning | XX | XX | XX |
| Delay Discounting | XX | XX | XX |
| Motor Control & BCI | XX | XX | XX |
| Visual Decision-Making | XX | XX | XX |
| Motor Imagery EEG | XX | XX | XX |
| Seizure/iEEG | XX | XX | XX |

### By Modality
| Modality | Count |
|----------|-------|
| Extracellular ephys | XX |
| Calcium imaging | XX |
| EEG | XX |
| ECoG/iEEG | XX |
| fMRI | XX |
| Behavior video | XX |

### By Species
| Species | Count |
|---------|-------|
| Mouse | XX |
| Rat | XX |
| Human | XX |
| Non-human primate | XX |

## Quality Metrics

### Dataset Cards
- Cards generated: XX
- QA pass rate: XX%
- Average readiness score: XX/100
- Cards with linked papers: XX%

### Search Quality (Benchmark)
- Mean Precision@5: XX%
- Mean Label Recall@10: XX%
- Queries passing: XX/XX

## Coverage Gaps

### Missing Priority Areas
- [List areas with <5 datasets]

### Underrepresented Modalities
- [List modalities with <3 datasets]

### Recommended Next Steps
1. [Specific ingestion recommendations]
2. [Manual curation needs]
3. [Ontology expansion needs]

## Appendices

### A. Full Dataset List
[Table with all dataset IDs, titles, sources]

### B. Paper Citations
[Table with all paper DOIs, titles, dataset links]

### C. QA Failures
[List of datasets/cards that failed QA with reasons]
```

### Report Generation Command

```bash
# Generate final corpus report
python -m neural_search.reports.corpus_report \
  --datasets data/ingested/linked_corpus.json \
  --cards data/cards/ \
  --benchmark data/eval/results/latest_eval_report.json \
  --output data/reports/corpus_report_v1.md
```

---

## 6. Execution Timeline

### Week 1: DANDI Ingestion
- Day 1-2: Run all DANDI queries
- Day 3-4: Manual review of top 50 datasets
- Day 5: Merge and validate

### Week 2: OpenNeuro Ingestion
- Day 1-2: Run all OpenNeuro queries
- Day 3-4: Manual review focusing on motor imagery & clinical
- Day 5: Merge and validate

### Week 3: Paper Linking
- Day 1-2: Run OpenAlex queries
- Day 3-4: Link papers to datasets
- Day 5: Manual review of linkages

### Week 4: Card Generation & QA
- Day 1-2: Generate all dataset cards
- Day 3-4: QA review of top 50 cards
- Day 5: Generate final corpus report

---

## 7. Success Criteria

| Metric | Target | Minimum |
|--------|--------|---------|
| DANDI datasets | 75 | 50 |
| OpenNeuro datasets | 75 | 50 |
| Linked papers | 300 | 200 |
| High-quality cards | 40 | 25 |
| Priority area coverage | 100% | 85% |
| Benchmark precision@5 | >70% | >60% |
| Card QA pass rate | >80% | >70% |

---

## Appendix: Curated Source Additions

Add these to `data/seed/curated_sources.yaml`:

```yaml
# High-priority datasets to manually curate
curated_sources:
  # Go/NoGo
  - source_type: dandi
    source_id: "000003"  # Allen Institute Visual Behavior
    priority: high
    expected_tasks: [go_nogo, visual_discrimination]

  # Reversal Learning
  - source_type: dandi
    source_id: "000017"  # IBL Brain Wide Map
    priority: high
    expected_tasks: [reversal_learning, decision_making]

  # Motor/BCI
  - source_type: dandi
    source_id: "000128"  # Motor cortex reaching
    priority: high
    expected_tasks: [reaching, motor_control]

  # Motor Imagery
  - source_type: openneuro
    source_id: "ds003490"  # Motor imagery EEG
    priority: high
    expected_tasks: [motor_imagery]
    expected_modalities: [eeg]

  # Epilepsy
  - source_type: dandi
    source_id: "000055"  # Human iEEG
    priority: high
    expected_tasks: [clinical_monitoring]
    expected_modalities: [ieeg]
```
