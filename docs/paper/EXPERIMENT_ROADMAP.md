# Neural Search Experiment Roadmap

This document outlines the planned validation experiments for Neural Search, including implementation status, required infrastructure, and expected outputs.

## Overview

| Experiment | Priority | Implementation Status | Data Requirements | Expected Output |
|------------|----------|----------------------|-------------------|-----------------|
| Baseline Ladder | P0 | Implemented | Benchmark queries, corpus | Ladder report |
| Hard-Negative Benchmark | P0 | Partial | Adversarial queries | Violation report |
| Affordance Validation | P1 | Infrastructure ready | Manual labels | Validation rubric |
| Cross-Dataset Pairing | P1 | Partial | Pairing ground truth | Compatibility report |
| Metadata Robustness | P2 | Not implemented | Perturbation configs | Degradation curves |
| Embedding Model Bakeoff | P2 | Not implemented | Model checkpoints | Comparison report |
| Graph Link Prediction | P3 | Not implemented | Edge holdout splits | Prediction metrics |
| Latent Neural Signature | P3 | Not implemented | NWB feature extraction | Prototype demo |

---

## 1. Baseline Ladder

**Objective**: Compare retrieval quality across incrementally complex system configurations.

### Configurations

1. **keyword**: Exact match on title/description
2. **BM25**: Standard BM25 on concatenated text
3. **field-weighted BM25**: BM25 with domain-specific field weights
4. **dense-only**: Sentence transformer embeddings only
5. **BM25 + dense RRF**: Reciprocal Rank Fusion of BM25 and dense
6. **+ ontology**: Add ontology matching scores
7. **+ graph**: Add graph-based features (PathSim, edge traversal)
8. **full system**: All scoring signals enabled

### Metrics

- Precision@5, Recall@10, MRR, NDCG@10
- Hard-negative violation count
- Latency (ms per query)
- Candidate set size

### Output

- `reports/baseline_ladder_results.json`
- `reports/baseline_ladder_results.md`

### Run Command

```bash
python -m neural_search.evaluation.run_baseline_ladder --suite demo_v02
```

---

## 2. Hard-Negative Adversarial Benchmark

**Objective**: Test constraint satisfaction on queries with explicit exclusions.

### Query Types

- Species exclusions: "mouse decision-making NOT human"
- Modality exclusions: "electrophysiology NOT fMRI NOT EEG"
- Brain-region exclusions: "motor cortex NOT visual cortex"
- Task exclusions: "reward learning NOT Pavlovian"
- Data-format exclusions: "NWB datasets NOT BIDS"
- Affordance exclusions: "supports choice decoding NOT spontaneous"

### Violation Detection

Each violation includes:
- Query ID
- Result ID and rank
- Excluded field (modality, species, region, etc.)
- Offending value
- Evidence source
- Explanation

### Output

- `reports/hard_negative_report.json`
- `reports/hard_negative_report.md`
- `benchmarks/hard_negative_queries.yaml`

### Run Command

```bash
python -m neural_search.evaluation.run_hard_negative_benchmark --config benchmarks/hard_negative_queries.yaml
```

---

## 3. Affordance Validation Rubric

**Objective**: Verify that predicted analysis affordances match actual dataset capabilities.

### Affordances to Validate

| Affordance | Required Fields | Optional Fields | Confidence Threshold |
|------------|-----------------|-----------------|---------------------|
| choice_decoding | neural_activity, trial_labels, choice_labels | reaction_time, stimulus | 0.80 |
| q_learning | choices, rewards, trial_order, subject_id | block_structure, state_labels | 0.80 |
| state_space_modeling | time_series, timestamps, sampling_rate | trial_boundaries | 0.75 |
| cross_session_analysis | subject_id, session_dates, unit_metadata | drift_metrics | 0.70 |
| causal_perturbation | intervention_labels, timing, control_condition | dose, stim_params | 0.85 |
| rsa | aligned_stimuli, population_responses | model_features | 0.75 |
| neural_behavior_alignment | neural_ts, behavior_ts, timestamps | event_labels | 0.70 |

### Validation Protocol

1. Select 20 datasets per affordance with high predicted support
2. Manual inspection: does dataset actually contain required fields?
3. Record: true positive, false positive, uncertain
4. Compute precision, recall (if ground truth available)

### Output

- `config/affordance_rubric.yaml`
- `reports/affordance_validation_report.json`
- `reports/affordance_validation_report.md`

---

## 4. Cross-Dataset Pairing Scorer

**Objective**: Rank scientifically compatible dataset pairs.

### Pairing Types

1. **Same task, different region**: Compare neural representations
2. **Same task, different species**: Cross-species generalization
3. **Same modality, different task**: Modality-specific analysis
4. **Same analysis affordance**: Method validation across datasets
5. **Dataset to paper**: Find supporting/related publications
6. **Dataset to model**: Match data with computational models

### Scoring Features

- Task similarity (ontology distance)
- Modality compatibility (same signal type?)
- Species relationship (taxon tree distance)
- Region overlap (shared targets?)
- Shared behavioral events
- Graph path confidence
- Novelty/diversity bonus
- Hard incompatibility penalties

### Output

- `reports/cross_dataset_pairing_report.json`
- `reports/cross_dataset_pairing_report.md`

---

## 5. Metadata Robustness Experiments

**Objective**: Measure retrieval degradation under metadata perturbations.

### Perturbation Types

| Perturbation | Description | Expected Impact |
|--------------|-------------|-----------------|
| drop_description | Remove description field | High (primary text source) |
| drop_task_labels | Remove task annotations | Medium |
| drop_modality_labels | Remove modality annotations | Medium |
| drop_brain_regions | Remove region annotations | Medium |
| remove_paper_links | Remove DOI/paper associations | Low-Medium |
| corrupt_synonyms | Replace terms with synonyms | Low |
| remove_species | Remove species information | Medium |
| inject_ambiguity | Add ambiguous terms | Variable |

### Protocol

1. Create perturbed corpus copies
2. Run benchmark on original and perturbed
3. Compute degradation: (original_metric - perturbed_metric) / original_metric
4. Identify most critical fields

### Output

- `reports/metadata_robustness_report.json`
- `reports/metadata_robustness_report.md`

---

## 6. Embedding Model Bakeoff

**Objective**: Compare embedding models for neuroscience retrieval.

### Candidate Models

- `sentence-transformers/all-MiniLM-L6-v2` (baseline)
- `sentence-transformers/all-mpnet-base-v2` (stronger baseline)
- `allenai/scibert_scivocab_uncased` (scientific)
- `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` (biomedical)
- `allenai/specter2` (scientific papers)
- Domain-specific fine-tuned models (if available)

### Metrics

- Benchmark NDCG@10 per model
- Query latency
- Embedding dimension/memory footprint
- Domain vocabulary coverage

---

## 7. Graph Link Prediction

**Objective**: Evaluate whether graph structure predicts missing relationships.

### Setup

1. Hold out 20% of edges
2. Train link predictor (TransE, DistMult, or metapath-based)
3. Predict held-out edges
4. Measure hits@10, MRR for link prediction

### Edge Types to Predict

- Dataset → Task
- Dataset → Modality
- Dataset → Paper
- Paper → Paper (citations)

---

## 8. Latent Neural Signature Search

**Objective**: Prototype search over extracted neural population features.

### Feature Extraction (from NWB)

- Firing rate statistics (mean, CV, Fano factor)
- Population dimensionality (PCA explained variance)
- Event-aligned modulation (PSTH peak, latency)
- Temporal dynamics (autocorrelation, spectral power)

### Search Modes

- "Find datasets with high-dimensional population activity"
- "Find datasets with strong stimulus modulation"
- "Find datasets with similar firing statistics to [reference]"

### Status

Not implemented. Requires NWB parsing infrastructure and feature embedding pipeline.

---

## Timeline and Dependencies

```
Phase 1 (Immediate): Baseline Ladder, Hard-Negative Benchmark
Phase 2 (Near-term): Affordance Validation, Cross-Dataset Pairing
Phase 3 (Medium-term): Metadata Robustness, Embedding Bakeoff
Phase 4 (Future): Graph Link Prediction, Latent Neural Signature
```

---

*Last updated: 2026-05-26*
