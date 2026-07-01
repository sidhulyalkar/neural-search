# Ablation Ladder Report

| Rung | Queries | NDCG@10 |
|------|---------|---------|
| bm25 (ok) | 317 | 0.8023 |
| bm25_structured (skipped) | ? | N/A |
| dense_bge (ok) | 317 | 0.6898 |
| hybrid_rrf (ok) | 317 | 0.8124 |
| hybrid_graph (ok) | 317 | 0.8564 |
| typed_kg (skipped) | ? | N/A |
| typed_kg_qualified (skipped) | ? | N/A |
| full (ok) | 317 | 0.8564 |
