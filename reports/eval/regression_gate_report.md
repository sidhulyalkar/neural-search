# Eval Regression Gate

Status: `PASS`

| Check | Status | Detail |
|-------|--------|--------|
| qrels_labeled_pairs | PASS | 13654 labeled pairs >= 10000 |
| bm25_query_coverage | PASS | 317 judged queries >= 300 |
| bm25_structured_query_coverage | PASS | 317 judged queries >= 300 |
| dense_bge_query_coverage | PASS | 317 judged queries >= 300 |
| hybrid_rrf_query_coverage | PASS | 317 judged queries >= 300 |
| hybrid_rrf_ge_bm25_ndcg@10 | PASS | hybrid_rrf ndcg@10=0.6667; bm25 ndcg@10=0.6566 |
| hybrid_rrf_ge_bm25_mrr | PASS | hybrid_rrf mrr=0.9209; bm25 mrr=0.8795 |
| hybrid_rrf_ge_bm25_recall@50 | PASS | hybrid_rrf recall@50=0.7455; bm25 recall@50=0.6440 |

## Warnings

- Dual-judge QWK is not estimable because no pair has two non-error judge labels.
