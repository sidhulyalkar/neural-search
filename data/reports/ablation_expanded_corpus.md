# Neural Search Ablation Report

Generated: 2026-05-25T19:25:38.547499+00:00
Suite: demo_v02
Modes: hybrid_default, no_ontology, no_semantic, no_metadata, no_hard_negative

## Metric Comparison

| Metric | hybrid_default | no_ontology | no_semantic | no_metadata | no_hard_negative |
| --- | --- | --- | --- | --- | --- |
| mean_precision_at_5 | 78.7% | 77.3% | 78.7% | 73.3% | 78.7% |
| mean_label_recall_at_10 | 88.5% | 86.0% | 88.5% | 87.9% | 87.7% |
| mean_mrr | 93.3% | 90.0% | 95.0% | 93.3% | 94.2% |
| mean_ndcg_at_10 | 93.6% | 89.5% | 94.5% | 92.6% | 93.9% |

## Best Mode by Metric

- **mean_precision_at_5**: hybrid_default (78.7%)
- **mean_label_recall_at_10**: hybrid_default (88.5%)
- **mean_mrr**: no_semantic (95.0%)
- **mean_ndcg_at_10**: no_semantic (94.5%)

## Failure Analysis

| Mode | Failed Queries | Examples |
| --- | --- | --- |
| hybrid_default | 0 |  |
| no_ontology | 0 |  |
| no_semantic | 0 |  |
| no_metadata | 0 |  |
| no_hard_negative | 0 |  |

## Per-Mode Details

### hybrid_default

- Precision@5: 78.7%
- Label Recall@10: 88.5%
- MRR: 0.933
- NDCG@10: 0.936

### no_ontology

- Precision@5: 77.3%
- Label Recall@10: 86.0%
- MRR: 0.900
- NDCG@10: 0.895

### no_semantic

- Precision@5: 78.7%
- Label Recall@10: 88.5%
- MRR: 0.950
- NDCG@10: 0.945

### no_metadata

- Precision@5: 73.3%
- Label Recall@10: 87.9%
- MRR: 0.933
- NDCG@10: 0.926

### no_hard_negative

- Precision@5: 78.7%
- Label Recall@10: 87.7%
- MRR: 0.942
- NDCG@10: 0.939
