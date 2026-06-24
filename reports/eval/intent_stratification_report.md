# Intent-Stratified Retrieval Report

**Qrels:** 13654 pairs across 317 queries
**Queries:** `data/eval/benchmark_queries_canonical.yaml`

| System | Intent | Queries | NDCG@10 | MRR | Recall@50 |
|--------|--------|---------|---------|-----|-----------|
| bm25 | CROSS_DATASET_COMPARISON | 1 | 0.8217 | 1.0000 | 0.4242 |
| bm25 | EXPLORATION | 289 | 0.6542 | 0.8805 | 0.6438 |
| bm25 | PIPELINE_REUSE | 3 | 0.8939 | 1.0000 | 0.6351 |
| bm25 | REANALYSIS_FEASIBILITY | 19 | 0.6148 | 0.8421 | 0.6512 |
| bm25 | REPLICATION | 1 | 0.7767 | 1.0000 | 0.6875 |
| bm25 | STRICT_LOOKUP | 4 | 0.7817 | 0.8333 | 0.6798 |
| bm25_structured | CROSS_DATASET_COMPARISON | 1 | 0.8721 | 1.0000 | 0.4545 |
| bm25_structured | EXPLORATION | 289 | 0.6411 | 0.8627 | 0.6168 |
| bm25_structured | PIPELINE_REUSE | 3 | 0.8828 | 1.0000 | 0.6048 |
| bm25_structured | REANALYSIS_FEASIBILITY | 19 | 0.4562 | 0.7302 | 0.5867 |
| bm25_structured | REPLICATION | 1 | 0.7767 | 1.0000 | 0.6875 |
| bm25_structured | STRICT_LOOKUP | 4 | 0.8493 | 1.0000 | 0.5528 |
| dense_bge | CROSS_DATASET_COMPARISON | 1 | 0.8338 | 1.0000 | 0.6061 |
| dense_bge | EXPLORATION | 289 | 0.5671 | 0.8814 | 0.5363 |
| dense_bge | PIPELINE_REUSE | 3 | 0.8736 | 1.0000 | 0.5131 |
| dense_bge | REANALYSIS_FEASIBILITY | 19 | 0.5422 | 0.9123 | 0.5837 |
| dense_bge | REPLICATION | 1 | 0.7174 | 1.0000 | 0.6875 |
| dense_bge | STRICT_LOOKUP | 4 | 0.6426 | 0.7083 | 0.4387 |
| hybrid_rrf | CROSS_DATASET_COMPARISON | 1 | 0.9261 | 1.0000 | 0.8182 |
| hybrid_rrf | EXPLORATION | 289 | 0.6638 | 0.9214 | 0.7465 |
| hybrid_rrf | PIPELINE_REUSE | 3 | 0.8865 | 1.0000 | 0.6256 |
| hybrid_rrf | REANALYSIS_FEASIBILITY | 19 | 0.6288 | 0.8759 | 0.7353 |
| hybrid_rrf | REPLICATION | 1 | 0.7536 | 1.0000 | 0.8438 |
| hybrid_rrf | STRICT_LOOKUP | 4 | 0.8051 | 1.0000 | 0.7683 |
| hybrid_graph | CROSS_DATASET_COMPARISON | 1 | 0.8415 | 1.0000 | 0.6970 |
| hybrid_graph | EXPLORATION | 289 | 0.6342 | 0.8807 | 0.6963 |
| hybrid_graph | PIPELINE_REUSE | 3 | 0.8042 | 1.0000 | 0.6192 |
| hybrid_graph | REANALYSIS_FEASIBILITY | 19 | 0.6229 | 0.8524 | 0.6948 |
| hybrid_graph | REPLICATION | 1 | 0.8143 | 1.0000 | 0.7188 |
| hybrid_graph | STRICT_LOOKUP | 4 | 0.7995 | 1.0000 | 0.7117 |
| full | CROSS_DATASET_COMPARISON | 1 | 0.8415 | 1.0000 | 0.6667 |
| full | EXPLORATION | 289 | 0.6342 | 0.8807 | 0.6921 |
| full | PIPELINE_REUSE | 3 | 0.8042 | 1.0000 | 0.6723 |
| full | REANALYSIS_FEASIBILITY | 19 | 0.6229 | 0.8524 | 0.6910 |
| full | REPLICATION | 1 | 0.8143 | 1.0000 | 0.7188 |
| full | STRICT_LOOKUP | 4 | 0.7995 | 1.0000 | 0.6082 |
