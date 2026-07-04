# Spec 1 — IR-Evaluation Rigor

**Date:** 2026-06-23
**Status:** Approved design, pending implementation plan
**Branch context:** `claude/latent-usefulness-v08`
**Predecessor:** completion of the parallel LLM qrels run (13,654 labeled pairs, 317 queries × 371 datasets)
**Successor spec:** Spec 2 — Search-as-Science-Instrument (outward-facing; not in scope here)

---

## 1. Purpose

The LLM qrels run is complete, which unblocks an evaluation pipeline that is mostly
*built but never run end-to-end at scale*. This spec turns that latent pipeline into a
reproducible, statistically honest evaluation of the Neural Search retrieval system, and
records exactly which claims the result can and cannot support.

Per the project north star ("a verifiable approach for doing science," scoped as
*both, sequenced*), this is **Phase 1: internal IR-evaluation rigor**. It establishes the
measured, CI-bounded foundation that the later science-instrument layer (Spec 2) will
build on.

**Core principle:** build the *measurement apparatus* around retrieval; do **not** modify
the retrieval algorithms in this spec. The qrels are the fixed measuring stick; every
retrieval variant is measured against the same stick with stated uncertainty.

## 2. Current state (verified 2026-06-23)

| Fact | Value | Source |
|------|-------|--------|
| Labeled pairs | 13,654 | `data/qrels/qrels.trec` |
| Judgment records | 13,673 | `data/qrels/llm_judgments.jsonl` |
| Distinct queries | 317 | `cut -d' ' -f1 qrels.trec \| sort -u` |
| Corpus size | 371 datasets | CURRENT_SYSTEM_MAP.md |
| `judge_error` rows | ~14 | error-rate < 0.1% post JSON-guard fix |
| `reports/eval/runs/` | **empty** | ladder never run at scale |
| BGE field embeddings | **missing** | `data/embeddings/real_all.dense.field_embeddings.jsonl` |
| `sentence-transformers`/`torch` in env | **absent** | `neural-search` conda env |
| qrels format mismatch | NDCG reads `.trec`, bootstrap-CI reads JSONL | scripts diverge |

Notable: **317 queries already satisfies the CLAIM_LEDGER's "expand to 200+ queries"
upgrade** for the core retrieval claims.

## 3. Architecture & data flow

```
corpus (371 datasets) ──embed──> BGE field embeddings ─┐
                                                        ├─> ablation ladder ──> runs/*.jsonl (6 rungs)
benchmark queries (317) ────────────────────────────────┘                            │
                                                                                      ▼
qrels (13,654 pairs) ──validate──> canonical qrels ──────────────> metrics + bootstrap CI
        │                                                                             │
        └──> dual-judge reliability (QWK) ──> reliability bound ──┐                   ▼
                                                                  └──> CLAIM_LEDGER update + `make eval` gate
```

The flow is acyclic. Only two new artifacts are introduced; everything else reuses
existing scripts:

1. **Canonical qrels adapter** — one source of truth derived from `llm_judgments.jsonl`
   (drop `judge_error` rows, dedup `(query_id, dataset_id)`), emitting both the `.trec`
   form (for `compute_ndcg_from_qrels.py`) and the JSONL form with a `label` key (for
   `compute_bootstrap_ci.py`). Removes the format divergence.
2. **Dual-judge consensus script** — runs a second, different LLM over a stratified
   sample and reports judge-vs-judge agreement.

## 4. Work tracks

### Track 0 — Prerequisites (unblock dense rungs)
- Install `sentence-transformers` + `torch` into the `neural-search` env.
- Reconcile the hardcoded `/home/sid21/anaconda3` interpreter path in eval scripts
  (stale Linux path; resolve to the active env interpreter).
- Generate BGE field embeddings → `data/embeddings/real_all.dense.field_embeddings.jsonl`
  via the existing `scripts/embed_flat_corpus.py` / `scripts/recompute_embeddings.py`
  (model `BAAI/bge-large-en-v1.5`, already wrapped by `DenseEmbeddingProvider`).
- Build the **canonical qrels adapter** (§3, item 1).

### Track 1 — Run + metrics
`validate_qrels.py` → `run_ablation_ladder.py` (all 6 rungs: bm25 → bm25_structured →
dense_bge → hybrid_rrf → hybrid_graph → full) → `compute_ndcg_from_qrels.py`
(NDCG@10, MRR, Recall@50) → `compute_bootstrap_ci.py` (per-rung 95% bootstrap CI +
pairwise significance on each additive delta).
**Headline output form:** "rung N adds +X.X NDCG (95% CI [a, b], p=…)".

### Track 2 — Per-intent stratification
Break every metric down by the 7 query intents (`EXACT_LOOKUP`, `REPLICATION`,
`PIPELINE_REUSE`, `CROSS_DATASET_COMPARISON`, `META_ANALYSIS`, `METHOD_TRANSFER`,
`EXPLORATION`). Reveals which capability helps which question type.

### Track 3 — Dual-judge reliability
Stratified ~250-pair sample via `select_neuro_judge_validation_set.py` → re-judge with a
*different* LLM model → judge-vs-judge QWK / within-1 / confusion matrix via an adapted
`audit_neuro_qrels.py`. Produces a reliability bound that becomes a stated caveat on
downstream claims. (Human gold labels are explicitly **out of scope** — decision: dual-judge
consensus only.)

### Track 4 — Claim ledger + reproducibility gate
- Update `docs/CLAIM_LEDGER.md` statuses with CI'd numbers (e.g.
  `prototype_validated` → `statistically_validated` where a CI excludes zero).
- Add a deterministic `make eval` target chaining Tracks 1–2.
- Add a regression gate: CI fails if a frozen rung's NDCG drops below its CI band.

## 5. Verifiability boundary (what this can / cannot claim)

**Can support after this spec:**
- "Capability X changes NDCG@10 by Δ (95% CI […]) over 317 queries" — per additive rung.
- "The improvement is / is not statistically distinguishable from zero" — via bootstrap CI.
- "Capability X helps intent class Y but not Z" — via per-intent stratification.
- "The qrels have inter-judge reliability QWK = κ on a stratified sample" — reliability bound.

**Cannot support (deferred / out of scope):**
- Human-validated relevance (no human gold labels this phase — reliability is judge-vs-judge only).
- Generalization beyond this 371-dataset corpus / neuroscience domain.
- Causal claims about *why* a capability helps (correlational ablation only).
- Embedding-model superiority claims beyond the single BGE model run here (SPECTER2/ColBERT
  comparison remains `not_started` — candidate for a follow-up).
- Any outward "the system produces correct science" claim — that is Spec 2.

**Statistical-power caveat:** 317 queries supports headline NDCG deltas with reasonable
CIs, but fine-grained per-intent significance (some intents may have <30 queries) will be
under-powered and must be reported as directional, not conclusive.

## 6. Out of scope (this spec)
- Modifying retrieval algorithms or ranking weights.
- Corpus or query-set expansion beyond what is already labeled.
- Human relevance labeling.
- SPECTER2 / ColBERT / multi-embedding comparison.
- The Spec 2 science-instrument layer.

## 7. Success criteria
1. `make eval` reproduces all 6 rungs' metrics from a clean checkout (given embeddings cached).
2. Every headline number carries a 95% bootstrap CI.
3. Per-intent breakdown exists for NDCG@10 and MRR.
4. A dual-judge reliability figure (QWK) is reported with its sample size.
5. CLAIM_LEDGER core-retrieval rows are updated to cite CI'd numbers.
6. A regression gate is wired into CI on at least one frozen rung.
