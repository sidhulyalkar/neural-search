# Qrels validation status & go-forward plan — 2026-06-23

Companion to [2026-06-23-ir-evaluation-rigor.md](2026-06-23-ir-evaluation-rigor.md)
(the master eval-rigor plan). This doc records the current **validated state of
the qrels artifacts**, the whitespace-safety fix shipped today, and the concrete
next steps — repo hygiene, validator/gate hardening, external-tool
cross-validation, and the remaining rigor track.

## 1. Where the qrels stand (verified 2026-06-23)

| Artifact | Rows | Status |
|----------|------|--------|
| `data/qrels/qrels.canonical.jsonl` | 13,654 | ✅ canonical, 100% valid JSON, original ids |
| `data/qrels/qrels.trec` | 13,654 | ✅ 4-column clean (regenerated) |
| `data/qrels/qrels.canonical.trec` | 13,654 | ✅ 4-column clean (regenerated) |
| `data/qrels/llm_judgments.jsonl` | 13,673 | ✅ raw judge audit trail (19 judge_error rows dropped on build) |

- **317 queries**, graded relevance **0–3**, distribution `{0: 3311, 1: 5671, 2: 4652, 3: 20}`.
- NDCG@10 / MRR / Recall@50 across all six ablation rungs (bm25, bm25_structured,
  dense_bge, hybrid_rrf, hybrid_graph, full) are stable; hybrid_graph and full
  lead at **NDCG@10 = 0.6696**.

**Conclusion: the qrels are complete and trustworthy as a validation measure.**
The canonical JSONL is the source of truth; the `.trec` files are derived
exports for TREC-format tooling.

## 2. Fix shipped today (commit `4172d8c`)

Four rows carried neuromorpho dataset ids containing internal spaces
(`neuromorpho:Physio Lab - Medical Faculty - UoI`, `Munoz et al.`,
`Allen Cell Types`). TREC qrels is positional and whitespace-delimited with no
escaping, so those rows had 5+ tokens and were not `trec_eval`-parseable.

- New shared boundary normalizer `scripts/eval/docid.py::normalize_docid()`
  (collapse internal whitespace → `_`, idempotent).
- Applied in **both** `.trec` writers and at **both** sides of the qrels↔run
  join in `compute_ndcg_from_qrels.py`.
- Canonical JSONL keeps original ids (provenance); only `.trec` exports change.
- Proven **metric-preserving**: post-fix NDCG JSON is byte-identical to the
  pre-fix baseline. Unit tests in `tests/eval/test_docid_normalize.py`. 88 tests pass.

## 3. Repo hygiene before pushing — DO THIS FIRST

The qrels themselves are small and safe to push (400 KB–12 MB; GitHub limits are
50 MB warn / 100 MB hard-reject per file). **No USB / no Git LFS needed for
qrels.** Validation qrels belong in-repo (reproducibility; standard for
TREC/BEIR-style benchmarks).

The real risk is an unrelated large artifact getting swept in:

- [ ] `data/embeddings/real_all.dense.field_embeddings.jsonl` is **112 MB** and
      currently **untracked** — it *exceeds GitHub's 100 MB hard limit* and would
      bounce a push. Add a `.gitignore` entry (`data/embeddings/`) so a stray
      `git add -A` can't stage it.
- [ ] `data/graph/neural_search_graph.real_corpus.json` (36 MB) — decide:
      gitignore, or Git LFS if it must be versioned.
- [ ] Audit other large *tracked* files (coverage jsonl ~35 MB each, ledger.duckdb
      17 MB): fine for now (<50 MB) but consider LFS if they keep growing.
- [ ] `git push` the current branch `claude/latent-usefulness-v08`.

## 4. Validator / gate hardening (prevent silent regression)

`scripts/eval/validate_qrels.py` validates only the **JSONL**. The `.trec`
malformation slipped through because nothing checked the export.

- [ ] Add a `.trec` format check (every row splits into exactly 4 tokens; grade
      ∈ {0,1,2,3}; ids whitespace-free) — either to `validate_qrels.py` or as a
      cheap assertion inside `build_canonical_qrels.py` after write.
- [ ] Wire that check into `scripts/eval/check_eval_regression_gate.py` so CI
      fails on a malformed export, not just on metric drift.

## 5. External-tool cross-validation (now unblocked)

With clean `.trec`, we can sanity-check our hand-rolled NDCG against the
reference implementation:

- [ ] Export run files to TREC run format using the **same `normalize_docid`**
      on record ids (so qrels and runs join identically).
- [ ] Run `trec_eval` / `pytrec_eval` and confirm NDCG@10 / MRR / Recall match
      `compute_ndcg_from_qrels.py` within float tolerance. This independently
      validates the metric code that the whole eval rests on.

## 6. Remaining eval-rigor track (per master plan)

Most reports already exist in `reports/eval/` (bootstrap CI, dual-judge
reliability, intent stratification, ablation ladder, claim ledger). Outstanding:

- [ ] **Dual-judge reliability**: confirm QWK / agreement on the second judge
      pass meets the threshold in the master plan; document inter-judge κ.
- [ ] **Bootstrap CIs on rung deltas**: report per-rung NDCG with 95% CIs and
      flag which rung improvements are significant (hybrid_graph vs hybrid_rrf
      is only +0.003 — likely within noise; the CI will say).
- [ ] **Per-intent stratification**: ensure every query-intent stratum has
      adequate judged coverage; surface thin strata.
- [ ] **Claim ledger**: keep `eval_claim_ledger` entries tied to the regenerated
      metrics so whitepaper claims stay traceable.

## 7. Silver → gold qrels (human validation)

These are LLM-generated ("silver") qrels. To harden the headline numbers:

- [ ] Sample a stratified subset for human adjudication
      (`triage_qrels_for_human_review.py` already exists).
- [ ] Compute LLM-vs-human agreement; report it alongside the metrics as the
      qrels' credibility measure.
- [ ] Promote the human-adjudicated subset to a small **gold** set and re-report
      the lead rungs against gold.

---

### Immediate next action
Push hygiene (§3) → then external `trec_eval` cross-check (§5), since it
independently validates the metric pipeline now that the export is clean.
