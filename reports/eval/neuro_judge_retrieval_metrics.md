# Neural Search Sprint 1 Gold Benchmark Report

Generated: 2026-06-11T21:01:13.468047+00:00
Qrels: `artifacts/field_state/adjudicated_qrels.jsonl`
Candidates: `qrels_candidates_pooled.jsonl`
Annotated pairs: **675**
Queries covered: **15**

---

## Per-System Metrics (macro-avg over labeled queries)

| System | NDCG@10 | MRR | P@5 | Recall@10 |
|--------|---------|-----|-----|-----------|
| bm25 | 0.0506 | 0.0222 | 0.0% | 13.3% |
| dense_prf | 0.0000 | 0.0000 | 0.0% | 0.0% |
| hybrid_rrf | 0.0421 | 0.0167 | 0.0% | 13.3% |
| usefulness | 0.0258 | 0.0133 | 1.3% | 6.7% |

HN violation rate: **97.6%**  (659/675 pairs)

---

## By Intent (system: `hybrid_rrf`)

| Intent | Queries | NDCG@10 | MRR | P@5 |
|--------|---------|---------|-----|-----|
| CROSS_DATASET_COMPARISON | 2 | 0.0000 | 0.0000 | 0.0% |
| EXPLORATION | 1 | 0.0000 | 0.0000 | 0.0% |
| META_ANALYSIS | 2 | 0.0000 | 0.0000 | 0.0% |
| METHOD_TRANSFER | 2 | 0.0000 | 0.0000 | 0.0% |
| MODEL_VALIDATION | 2 | 0.1577 | 0.0625 | 0.0% |
| PIPELINE_REUSE | 2 | 0.0000 | 0.0000 | 0.0% |
| REANALYSIS_FEASIBILITY | 2 | 0.0000 | 0.0000 | 0.0% |
| REPLICATION | 2 | 0.1577 | 0.0625 | 0.0% |

---

## Per-Query Results

| Query | Intent | N | NDCG@10 (bm25) | NDCG@10 (dense_prf) | NDCG@10 (hybrid_rrf) | NDCG@10 (usefulness) |
|---|---|---|---|---|---|---|
| q_0001 | META_ANALYSIS | 41 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_0002 | MODEL_VALIDATION | 37 | 0.4030 | 0.0000 | 0.3155 | 0.0000 |
| q_0003 | PIPELINE_REUSE | 56 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_0004 | CROSS_DATASET_COMPARISON | 44 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_0005 | REANALYSIS_FEASIBILITY | 43 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_0006 | METHOD_TRANSFER | 49 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_0007 | REPLICATION | 49 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_0008 | EXPLORATION | 43 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_0009 | META_ANALYSIS | 46 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_0010 | PIPELINE_REUSE | 37 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_0011 | CROSS_DATASET_COMPARISON | 46 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_0012 | MODEL_VALIDATION | 49 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_0013 | REANALYSIS_FEASIBILITY | 44 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_0014 | METHOD_TRANSFER | 48 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_0015 | REPLICATION | 43 | 0.3562 | 0.0000 | 0.3155 | 0.3869 |

---

## Confidence Notes

- 675 pairs labeled across 15 queries — results are approaching minimum credibility.
- Dual annotation (20% of pairs) needed for inter-annotator agreement.
- Per-query coverage is uneven — queries with N<5 labels are unreliable.

## Reproducibility

```bash
# Re-generate candidates
python scripts/eval/build_pooled_qrels_candidates.py

# Annotate (--limit 30 for a quick session)
python scripts/annotate_qrels_fast.py --resume --limit 30

# Re-run this report
python scripts/eval/report_gold_qrels_metrics.py
```