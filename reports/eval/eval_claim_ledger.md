# Eval Claim Ledger

| Claim | Evidence Level | Key Evidence |
|-------|----------------|--------------|
| `claim_eval_canonical_qrels` | `supported` | queries=317; labeled_pairs=13654 |
| `claim_hybrid_rrf_beats_bm25` | `partially_supported` | ndcg@10_delta=0.0101; mrr_delta=0.0414; recall@50_delta=0.1015; caveat=NDCG@10 is directionally higher but not significant versus BM25 by the current sign test. |
| `claim_hybrid_rrf_beats_dense_bge` | `supported` | ndcg@10_delta=0.0959; mrr_delta=0.038; recall@50_delta=0.2071 |
| `claim_calibrated_graph_beats_hybrid_rrf` | `supported` | ndcg@10_delta=0.0029; mrr_delta=0.0046; recall@50_delta=0.0011; caveat=Small effect on LLM-judged qrels; calibration and edge quality should be rechecked on independently adjudicated labels. |
| `claim_intent_stratification_available` | `supported` | caveat=Several non-exploration intent buckets have very small query counts. |
| `claim_dual_judge_qwk` | `not_estimable` | pairs_with_two_or_more_judges=0; caveat=Current labels have no non-error pair judged by two models. |
