# Ablation Ladder — NDCG Report (LLM Qrels), Full-Corpus Variant

**Generated:** 2026-06-23. **Corpus:** `data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl`
(7,171 records) + `data/graph/neural_search_graph.real_corpus.json` (7,593 nodes) —
**not** the canonical 2,821-record packet-aligned corpus used for `reports/eval/ndcg_report.md`.

**Why this exists:** `reports/eval/ndcg_report.md` (committed from `claude/latent-usefulness-v08`)
is the canonical benchmark, but its underlying run files and the matching embeddings/graph were
generated on a different machine and are gitignored — not reproducible here. This run regenerates
all 8 rungs together, on this machine, against the full corpus, so that `typed_kg` and
`typed_kg_qualified` (added 2026-06-23 to isolate the relationship-KG layer's retrieval
contribution from the aggregate `hybrid_graph` signal) are at least internally comparable to each
other and to a freshly-computed `bm25`/`hybrid_rrf`/`hybrid_graph` on identical data. Treat the
absolute numbers below as a separate, self-consistent experiment, not a replication of the
canonical packet-corpus benchmark — the two use different corpus sizes and the canonical run used
a corpus-scoped 3,180-node graph rather than the full 7,593-node one.

**Result:** at the full-corpus scale, with the existing default weights (`graph_score_weight` and
`typed_kg_score_weight` both 0.005), `hybrid_graph`, `typed_kg`, and `full` are **numerically
identical to `hybrid_rrf`** to 4 decimal places (0.7142/0.9133) — neither the existing aggregate
graph signal nor the new isolated typed-relationship signal moved the top-10 ranking at all on this
corpus at this weight. Only `typed_kg_qualified` (the qualified-consensus region bonus, which is
not gated on paper-linkage and so reaches more candidates) showed any movement, and it was tiny
(+0.0002 NDCG@10, +0.0003 Recall@50). This is a different outcome from the canonical packet-corpus
run, where `hybrid_graph` showed a small but real gain over `hybrid_rrf` (+0.0029 NDCG@10). The
likely explanation is that the calibrated weight was tuned for the smaller, denser 2,821-record
corpus and is proportionally too small to matter once RRF scores are spread across 7,171 records —
not that the signal is inherently useless, but that it needs separate recalibration per corpus
scale. Combined with `relationship_edge_quality.md`'s ~50% (coin-flip) helpful rate for the
relationship edges that feed `hybrid_graph`, this is a second, independent line of evidence that
the typed/relationship KG layer's retrieval contribution is real but small, and not yet worth
further investment ahead of human-adjudicated qrels (see `reports/strategy/2026-06-23_qrels_branch_merge_and_kg_rigor_plan.md`).

**Qrels:** 13654 pairs across 317 queries


| Rung | Queries | NDCG@10 | MRR | Recall@50 |
|------|---------|---------|-----|-----------|
| bm25 | 317 | 0.6424 | 0.8743 | 0.6395 |
| bm25_structured | 317 | 0.6597 | 0.8799 | 0.6300 |
| dense_bge | 317 | 0.6777 | 0.9204 | 0.6401 |
| hybrid_rrf | 317 | 0.7142 | 0.9133 | 0.8921 |
| hybrid_graph | 317 | 0.7142 | 0.9133 | 0.8921 |
| typed_kg | 317 | 0.7142 | 0.9133 | 0.8921 |
| typed_kg_qualified | 317 | 0.7144 | 0.9133 | 0.8924 |
| full | 317 | 0.7142 | 0.9133 | 0.8946 |
