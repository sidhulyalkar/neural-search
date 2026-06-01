# Claude Results for Manuscript

This document summarizes the implementation work completed during the paper and experiment upgrade session, providing guidance for updating the manuscript with new infrastructure.

## Implementation Summary

### What Was Implemented

| Component | Status | Location | Description |
|-----------|--------|----------|-------------|
| Baseline Ladder | Implemented | `neural_search/evaluation/run_baseline_ladder.py` | 8-mode retrieval comparison |
| Hard-Negative Benchmark | Implemented | `neural_search/evaluation/run_hard_negative_benchmark.py` | Adversarial query testing |
| Affordance Validation | Implemented | `neural_search/evaluation/affordance_validation.py` | Rubric-based validation |
| Cross-Dataset Pairing | Implemented | `neural_search/evaluation/cross_dataset_pairing.py` | Compatibility scoring |
| Metadata Robustness | Implemented | `neural_search/evaluation/metadata_robustness.py` | Perturbation testing |
| Unified Report | Implemented | `neural_search/evaluation/unified_report.py` | Combined reporting |

### Configuration Files Created

- `config/affordance_rubric.yaml` - Affordance validation requirements
- `benchmarks/hard_negative_queries.yaml` - Adversarial benchmark queries

### Documentation Created

- `docs/paper/CLAIM_STATUS_AND_EVIDENCE.md` - Claim verification tracking
- `docs/paper/EXPERIMENT_ROADMAP.md` - Validation experiment plan
- `docs/paper/RELATED_WORK_CHECKLIST.md` - Literature coverage tracking
- `docs/paper/REPUTATION_CHECKLIST.md` - Submission readiness
- `docs/paper/MANUSCRIPT_TODO.md` - Remaining tasks

### Tests Added

- `tests/test_baseline_ladder.py` (15 tests)
- `tests/test_hard_negative_benchmark.py` (17 tests)
- `tests/test_affordance_validation.py` (13 tests)
- `tests/test_cross_dataset_pairing.py` (17 tests)

All tests pass.

---

## How to Run Each Experiment

### Baseline Ladder

```bash
python -m neural_search.evaluation.run_baseline_ladder --suite demo_v02

# With specific modes
python -m neural_search.evaluation.run_baseline_ladder --suite demo_v02 --modes bm25,full_system
```

**Output**: `reports/baseline_ladder_results.{json,md}`

### Hard-Negative Benchmark

```bash
python -m neural_search.evaluation.run_hard_negative_benchmark

# With custom config
python -m neural_search.evaluation.run_hard_negative_benchmark --config benchmarks/hard_negative_queries.yaml
```

**Output**: `reports/hard_negative_report.{json,md}`

### Affordance Validation

```bash
python -m neural_search.evaluation.affordance_validation

# With custom rubric
python -m neural_search.evaluation.affordance_validation --rubric config/affordance_rubric.yaml
```

**Output**: `reports/affordance_validation_report.{json,md}`

### Cross-Dataset Pairing

```bash
python -m neural_search.evaluation.cross_dataset_pairing --top-k 20
```

**Output**: `reports/cross_dataset_pairing_report.{json,md}`

### Metadata Robustness

```bash
python -m neural_search.evaluation.metadata_robustness --suite demo_v02 --seed 42
```

**Output**: `reports/metadata_robustness_report.{json,md}`

### Unified Report

```bash
python -m neural_search.evaluation.unified_report
```

**Output**: `reports/neural_search_experiment_report.{json,md}`

---

## Manuscript Sections Updated

### Abstract
- Softened "state-of-the-art" claims
- Added caveat about ongoing validation

### Contributions (Section 1)
- Changed "state-of-the-art" to "strong performance"
- Added "roadmap for expanded validation"

### Claim Status Table (Table 1)
- Added after contributions section
- Classifies claims by implementation/validation status

### Related Work (Section 2)
- Added PROV-O, RO-Crate, LinkML, Bioschemas
- Added EBRAINS Knowledge Graph, openMINDS

### Graph Schema (Section 4)
- Added compact node type table (Table 2)
- Added edge type table (Table 3)

### Analysis Affordances (Section 5)
- Added affordance requirements table (Table 4)

### Experiments (Section 9)
- Expanded benchmark protocol with:
  - Corpus statistics
  - Query categories
  - Relevance labeling
  - Error taxonomy
  - Reproducibility statement

### Limitations (Section 10)
- Expanded with specific weaknesses:
  - Corpus coverage
  - Benchmark scale
  - Label quality
  - Embedding generalization
  - Affordance validation
  - Graph edge quality
  - Hard-negative heuristics

### Future Experiments (Section 10)
- Added 8-experiment validation roadmap
- Distinguished implemented from pending

---

## What Remains Unresolved

### High Priority (Before Submission)

1. **Full experiment execution**: Run all 5 evaluation harnesses on full corpus
2. **Inter-annotator agreement**: Need second annotator for benchmark queries
3. **Bootstrap significance**: Add p-values to benchmark comparisons
4. **Complete bibliography**: Some citation entries need URLs/DOIs

### Medium Priority (Revision Round)

1. **External baselines**: Compare to Elasticsearch, basic embedding search
2. **Corpus expansion**: Increase to 500+ datasets
3. **User study**: Task completion time measurement
4. **Embedding bakeoff**: Compare model variants

### Lower Priority (Future Work)

1. **Latent neural signature search**: Prototype NWB feature extraction
2. **Cross-species alignment benchmark**: Create ground truth
3. **Computational model integration**: Connect to ModelDB

---

## What Codex Should Do Later

When messages return, Codex should:

1. **Run full experiment suite**:
   ```bash
   python -m neural_search.evaluation.run_baseline_ladder --suite real_v07
   python -m neural_search.evaluation.run_hard_negative_benchmark
   python -m neural_search.evaluation.affordance_validation
   python -m neural_search.evaluation.cross_dataset_pairing
   python -m neural_search.evaluation.metadata_robustness --suite real_v07
   python -m neural_search.evaluation.unified_report
   ```

2. **Update manuscript with real numbers**:
   - Replace TODOs in experiments section
   - Add actual baseline ladder comparison table
   - Add hard-negative violation counts (expected: 0)

3. **Add statistical significance**:
   - Bootstrap confidence intervals for all metrics
   - Paired t-tests for mode comparisons

4. **Expand hard-negative benchmark**:
   - Add 20 more adversarial queries
   - Test edge cases in constraint parsing

5. **Complete bibliography**:
   - Verify all DOIs resolve
   - Add missing URLs

---

## File Locations Summary

### Paper Files
- Main manuscript: `docs/whitepaper/neural_search_iclr_whitepaper.tex`
- Technical appendix: `docs/whitepaper/neural_search_whitepaper.tex`
- Paper support docs: `docs/paper/*.md`

### Evaluation Code
- All evaluation modules: `neural_search/evaluation/`
- Test files: `tests/test_*.py`

### Config Files
- Retrieval config: `data/config/retrieval.yaml`
- Affordance rubric: `config/affordance_rubric.yaml`
- Hard-negative queries: `benchmarks/hard_negative_queries.yaml`
- Benchmark queries: `data/eval/benchmark_queries*.yaml`

### Report Outputs
- All reports go to: `reports/`
- Suite-specific results: `data/eval/results/{suite}/`

---

*Last updated: 2026-05-26*
*Implementation by: Claude Opus 4.5*
