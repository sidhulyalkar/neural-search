# Neural Search Ablation Report

Generated: 2026-05-25T06:36:47.562040+00:00
Suite: demo_v02
Modes: hybrid_default, no_ontology, no_behavior, no_modality, no_semantic, no_affordance, no_metadata, no_readiness, no_paper_confidence, no_graph_context, no_hard_negative

## Component Impact Analysis

Impact of ablating each scoring component (compared to baseline):

| Component | P@5 Delta | MRR Delta | Critical | Recommendation |
| --- | --- | --- | --- | --- |
| metadata | -4.7% | +0.000 |  | OPTIONAL - marginal impact |
| ontology | -1.3% | -0.034 |  | OPTIONAL - marginal impact |
| behavior | -0.7% | -0.050 |  | OPTIONAL - marginal impact |
| modality | -0.7% | -0.009 |  | OPTIONAL - marginal impact |
| hard_negative | +0.7% | +0.008 |  | OPTIONAL - marginal impact |
| semantic | +0.0% | +0.008 |  | OPTIONAL - marginal impact |
| affordance | +0.0% | +0.000 |  | OPTIONAL - marginal impact |
| readiness | +0.0% | +0.000 |  | OPTIONAL - marginal impact |
| paper_confidence | +0.0% | +0.000 |  | OPTIONAL - marginal impact |
| graph_context | +0.0% | +0.000 |  | OPTIONAL - marginal impact |

### Marginal Components (<5% impact)

- metadata: -4.7% precision
- ontology: -1.3% precision
- behavior: -0.7% precision
- modality: -0.7% precision
- hard_negative: +0.7% precision
- semantic: +0.0% precision
- affordance: +0.0% precision
- readiness: +0.0% precision
- paper_confidence: +0.0% precision
- graph_context: +0.0% precision

### Potentially Harmful Components (negative impact)

- hard_negative: +0.7% (consider removing)

## Metric Comparison

| Metric | hybrid_default | no_ontology | no_behavior | no_modality | no_semantic | no_affordance | no_metadata | no_readiness | no_paper_confidence | no_graph_context | no_hard_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean_precision_at_5 | 78.0% | 76.7% | 77.3% | 77.3% | 78.0% | 78.0% | 73.3% | 78.0% | 78.0% | 78.0% | 78.7% |
| mean_label_recall_at_10 | 88.5% | 86.0% | 87.4% | 87.4% | 87.9% | 88.5% | 87.9% | 87.9% | 88.5% | 88.5% | 87.7% |
| mean_mrr | 94.2% | 90.8% | 89.2% | 93.3% | 95.0% | 94.2% | 94.2% | 94.2% | 94.2% | 94.2% | 95.0% |
| mean_ndcg_at_10 | 94.1% | 90.1% | 92.3% | 93.2% | 94.3% | 94.1% | 93.1% | 93.9% | 94.0% | 94.2% | 94.4% |

## Best Mode by Metric

- **mean_precision_at_5**: no_hard_negative (78.7%)
- **mean_label_recall_at_10**: hybrid_default (88.5%)
- **mean_mrr**: no_semantic (95.0%)
- **mean_ndcg_at_10**: no_hard_negative (94.4%)

## Failure Analysis

| Mode | Failed Queries | Examples |
| --- | --- | --- |
| hybrid_default | 0 |  |
| no_ontology | 0 |  |
| no_behavior | 0 |  |
| no_modality | 0 |  |
| no_semantic | 0 |  |
| no_affordance | 0 |  |
| no_metadata | 0 |  |
| no_readiness | 0 |  |
| no_paper_confidence | 0 |  |
| no_graph_context | 0 |  |
| no_hard_negative | 0 |  |

## Per-Mode Details

### hybrid_default

- Precision@5: 78.0%
- Label Recall@10: 88.5%
- MRR: 0.942
- NDCG@10: 0.941

### no_ontology

- Precision@5: 76.7%
- Label Recall@10: 86.0%
- MRR: 0.908
- NDCG@10: 0.901

### no_behavior

- Precision@5: 77.3%
- Label Recall@10: 87.4%
- MRR: 0.892
- NDCG@10: 0.923

### no_modality

- Precision@5: 77.3%
- Label Recall@10: 87.4%
- MRR: 0.933
- NDCG@10: 0.932

### no_semantic

- Precision@5: 78.0%
- Label Recall@10: 87.9%
- MRR: 0.950
- NDCG@10: 0.943

### no_affordance

- Precision@5: 78.0%
- Label Recall@10: 88.5%
- MRR: 0.942
- NDCG@10: 0.941

### no_metadata

- Precision@5: 73.3%
- Label Recall@10: 87.9%
- MRR: 0.942
- NDCG@10: 0.931

### no_readiness

- Precision@5: 78.0%
- Label Recall@10: 87.9%
- MRR: 0.942
- NDCG@10: 0.939

### no_paper_confidence

- Precision@5: 78.0%
- Label Recall@10: 88.5%
- MRR: 0.942
- NDCG@10: 0.940

### no_graph_context

- Precision@5: 78.0%
- Label Recall@10: 88.5%
- MRR: 0.942
- NDCG@10: 0.942

### no_hard_negative

- Precision@5: 78.7%
- Label Recall@10: 87.7%
- MRR: 0.950
- NDCG@10: 0.944
