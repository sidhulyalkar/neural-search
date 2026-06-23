# Ablation Ladder — NDCG Report (LLM Qrels)

**Qrels:** 13654 pairs across 317 queries


| Rung | Queries | NDCG@10 | MRR | Recall@50 |
|------|---------|---------|-----|-----------|
| bm25 | 317 | 0.6566 | 0.8795 | 0.6440 |
| bm25_structured | 317 | 0.6361 | 0.8587 | 0.6138 |
| dense_bge | 317 | 0.5708 | 0.8829 | 0.5384 |
| hybrid_rrf | 317 | 0.6667 | 0.9209 | 0.7455 |
| hybrid_graph | 0 | 0.0000 | 0.0000 | 0.0000 |
| full | 0 | 0.0000 | 0.0000 | 0.0000 |
