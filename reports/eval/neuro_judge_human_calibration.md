# Neuro-Judge Audit Report

> PRELIMINARY NEURO-JUDGE EVALUATION — RAG-GROUNDED LLM LABELS, NOT PURE HUMAN GOLD
These labels are produced by an LLM judge with retrieval-augmented evidence.
They have NOT been reviewed by domain experts and must not be reported as final results.

**Judgment file**: `artifacts/field_state/neuro_qrels_consensus_mock.jsonl`  
**Human labels**: `artifacts/field_state/adjudicated_qrels.jsonl`  

## Summary

| Metric | Value |
|--------|-------|
| Total pairs | 675 |
| Conflicts | 0 |
| Hard-negative detected | 659 |

## Label Distribution

| Label | Count | % |
|-------|-------|---|
| 0 | 659 | 97% |
| 1 | 0 | 0% |
| 2 | 16 | 2% |
| 3 | 0 | 0% |

## Confidence Distribution

| Range | Count | % |
|-------|-------|---|
| <0.5 | 0 | 0% |
| 0.5–0.7 | 0 | 0% |
| 0.7–0.9 | 675 | 100% |
| >=0.9 | 0 | 0% |

## Top Failure Modes

None detected.

### Label 0 Examples

**q_0001 / neurovault:4778**
- Label: 0 | Confidence: 0.85
- Rationale: mock: hard-negative signal detected

**q_0001 / neurovault:1323**
- Label: 0 | Confidence: 0.85
- Rationale: mock: hard-negative signal detected

**q_0001 / openneuro:ds005230**
- Label: 0 | Confidence: 0.85
- Rationale: mock: hard-negative signal detected

### Label 1 Examples

None.

### Label 2 Examples

**q_0002 / figshare:10380131**
- Label: 2 | Confidence: 0.7
- Rationale: mock: 1 dimension(s) matched

**q_0002 / figshare:14150336**
- Label: 2 | Confidence: 0.7
- Rationale: mock: 1 dimension(s) matched

**q_0002 / zenodo:19958578**
- Label: 2 | Confidence: 0.7
- Rationale: mock: 1 dimension(s) matched

### Label 3 Examples

None.

### High-Confidence Examples (≥0.9)

None.

### Low-Confidence Examples (<0.5)

None.

### High Label but Severe Missing Information

None.

## Human Calibration

| Metric | Value |
|--------|-------|
| Pairs evaluated | 3 |
| Exact agreement | 0.333 |
| Agreement within 1 | 0.667 |
| QWK | 0.000 |

### Confusion Matrix

```
         | pred 0 | pred 1 | pred 2 | pred 3 |
---------|--------|--------|--------|--------|
  true 0 |      1 |      0 |      0 |      0 |
  true 1 |      1 |      0 |      0 |      0 |
  true 2 |      1 |      0 |      0 |      0 |
  true 3 |      0 |      0 |      0 |      0 |
```

### False-Low Examples (judge < human)

- q_0003/ibl:session_efb1ae99-bf9f-43: judge=0 human=2
- q_0015/crcns:mt-1: judge=0 human=1
