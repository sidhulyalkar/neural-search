# Post-Pull Development Plan: Merging the Qrels Benchmark Branch + KG Scientific Rigor

**Date:** 2026-06-23
**Author:** Claude Sonnet 4.6, full review of `claude/neuronpedia-foundation` (this checkout) and `claude/latent-usefulness-v08` (Mac eval branch)

---

## 0. Correcting the starting picture

Initial review of `claude/neuronpedia-foundation` alone (commits `1aceae1`..`20b96aa`: claims KG, vault/Obsidian prompts, relationship/typed-field KG Phases 0-6b, aperiodic spectral phenotype layer) showed `artifacts/qrels_gold.jsonl` at 0 rows and all four human-audit CSVs blank — looked like the qrels labelling session hadn't landed anywhere. It hadn't landed **here**. It landed on a sibling branch.

`claude/latent-usefulness-v08` diverged from this branch 90 commits ago at a shared ancestor (`4a67e8e`) and has since produced, independently:
- `data/qrels/qrels.canonical.jsonl` — **13,654 pairs, 317 queries**, graded 0–3 (`{0:3311, 1:5671, 2:4652, 3:20}`)
- A 6-rung ablation ladder (`bm25`, `bm25_structured`, `dense_bge`, `hybrid_rrf`, `hybrid_graph`, `full`) with **real, differentiated** NDCG@10/MRR/Recall@50 — the first non-degenerate ablation this project has ever produced (the 2026-06-11 roadmap's "ablation is a non-result" finding is now obsolete for this branch)
- A whitespace-safe TREC export fix (`4172d8c`, `normalize_docid()`), bootstrap CIs (2000 resamples), and a passing regression gate
- A go-forward doc (`5e3d5b7`) that is itself honest about what's still missing

This is real progress. It is also, by the branch's own admission, **LLM-silver, not human-gold** — "these are LLM-generated ('silver') qrels," per §7 of the go-forward doc. The two branches need to be reconciled before either one's claims can be trusted in isolation.

---

## 1. The actual numbers (verified by direct read of `origin/claude/latent-usefulness-v08`)

`reports/eval/ndcg_report.md`:

| Rung | NDCG@10 | MRR | Recall@50 |
|---|---|---|---|
| bm25 | 0.6566 | 0.8795 | 0.6440 |
| bm25_structured | 0.6361 | 0.8587 | 0.6138 |
| dense_bge | 0.5708 | 0.8829 | 0.5384 |
| hybrid_rrf | 0.6667 | 0.9209 | 0.7455 |
| hybrid_graph | 0.6696 | 0.9256 | 0.7465 |
| full | 0.6696 | 0.9256 | 0.7450 |

**With bootstrap 95% CIs** (`reports/eval/bootstrap_ci_report.json`, n=317 queries, 2000 resamples), the picture is more nuanced than the leaderboard order suggests:

| Rung | NDCG@10 CI |
|---|---|
| bm25 | [0.6322, 0.6807] |
| dense_bge | [0.5464, 0.5939] |
| hybrid_rrf | [0.6421, 0.6893] |
| hybrid_graph / full | [0.6453, 0.6921] |

- **The hybrid_graph/full "lead" over plain BM25 (+0.013 NDCG@10) is inside bootstrap noise** — CIs overlap substantially. This is directionally consistent with graph features helping, but not yet a statistically defensible claim at n=317 queries.
- **dense_bge underperforming BM25 is real** — its CI `[0.5464, 0.5939]` does not overlap BM25's `[0.6322, 0.6807]`. This is a genuine, citable negative result, not noise.
- The branch's own regression gate already flags the one thing nobody should skip: *"Dual-judge QWK is not estimable because no pair has two non-error judge labels"* — there is currently exactly one LLM judge pass, no inter-judge agreement measurement yet.

Any plan or whitepaper update built on this benchmark needs to carry these caveats forward, not just the leaderboard.

---

## 2. Merge plan (this session, before any new feature work)

The two branches touch almost entirely disjoint subsystems — eval/qrels infra (`latent-usefulness-v08`) vs. KG/claims/vault/frontend (`neuronpedia-foundation`) — but both touch shared docs (`docs/whitepaper/neural_search_whitepaper.tex`, `docs/CLAIM_LEDGER.md`, `docs/WHITEPAPER_IMPLEMENTATION_ALIGNMENT.md`).

1. **Checkpoint current WIP first.** This checkout has uncommitted frontend/whitepaper trust-hardening work (`SearchPage.tsx` evidence-tier strip + dynamic corpus count, `DatasetCard.tsx`, `GraphControls.tsx`, `DatasetPage.tsx`, `search.ts`, `main.py`) — ready, low-risk, isolated from the merge. Commit it before merging so it can't get tangled into conflict resolution.
2. **Merge `claude/latent-usefulness-v08` into `claude/neuronpedia-foundation`** (not the reverse) — the foundation branch has 90 commits of newer architecture the eval branch doesn't know about; resolving doc conflicts is cheaper than resolving KG-code conflicts the other direction.
3. **Do not bring over `data/embeddings/` or `data/graph/` blobs** from the Mac session — confirmed gitignored/absent there, and this machine's own 1GB `data/embeddings/real_all.dense.field_embeddings.jsonl` is separately gitignored already. The Mac branch's own go-forward doc still has an open TODO to gitignore its 112MB `real_all.dense.field_embeddings.jsonl` before push — that's an action item for that branch/session, not this one.
4. **Do not touch `artifacts/literature/findings_tier1_ollama.jsonl`** or its checkpoint — the paper-extraction job is running live on this machine (190,078 findings as of this check, up from 122,544 on 6/20) and both are untracked/gitignored, so a branch merge shouldn't disturb them, but worth confirming after merge that nothing got swept in.
5. **After merging**, regenerate (don't hand-edit) `reports/eval/current_artifact_manifest.json`, `reports/eval/qrels_progress_report.{json,md}`, and `reports/eval/benchmark_safety_gate_report.json` against the merged state. Expect: gold still 0 (correct — this is silver), silver now 13,654/317 instead of the stale 175/319/13-pair snapshot from 6/20.

---

## 3. Reporting the science honestly (whitepaper / claim ledger)

- Every rung-vs-rung claim in the whitepaper or claim ledger must carry its bootstrap CI, not just the point estimate. Adopt the format `NDCG@10 = 0.6696 [0.6453, 0.6921]`, not bare numbers.
- Describe the hybrid_graph/full vs. BM25 gap as **"directionally consistent, not yet statistically significant at n=317 queries"** — not as a win. This is consistent with the project's existing "don't overclaim" culture (`REPRODUCIBILITY_CAPSULE.md`, `PEER_VALIDATION_PROTOCOL.md` tiering).
- Report the **dense_bge underperformance** prominently as a real negative result, with a concrete follow-up hypothesis (field-embedding granularity vs. short scientific queries favoring lexical overlap; worth one quick ablation — e.g. swap BGE-large for a different backbone or retune field weighting — before it goes in a paper as a flat "dense embeddings underperform" claim).
- Classify the whole 13,654-pair ladder as **Tier C (LLM-silver diagnostic)** in the existing tier system, explicitly noting the unresolved dual-judge step as the blocker to Tier B, and a small human-adjudicated subset as the blocker to Tier A — i.e., make the silver→gold path (go-forward doc §7) visible in the whitepaper itself, not just in an internal planning doc.

---

## 4. Where this plan answers "highlight scientific complexity efficiently within the KGs"

This is the part that actually changes what gets built next, not just how results are reported.

**a. Test the typed/relationship KG layer on the ablation ladder itself — don't just test "graph vs. no graph."**
`hybrid_graph`/`full` currently use the whole graph (similarity edges, etc.) as one undifferentiated signal. The relationship/typed-field work already shipped on this branch (Phases 0–6b: `relationship_kg_builder.py`, `claim_kg_builder.py`, qualified consensus tiers, `contradiction_subtype`) has never been tested for retrieval impact in isolation. Add two new ablation rungs to `run_ablation_ladder.py`:
- `typed_kg` — typed finding/claim edges only (frequency_band, negation-aware contradictions, region/task typed matches)
- `typed_kg_qualified` — + Phase 6b qualified consensus tier

This is the literal BrainKnow-plan Milestone 4 test ("does typed relation semantics retrieve datasets a co-occurrence/dense baseline misses") and the only thing missing is plugging existing infrastructure into the existing ladder — no new KG schema needed.

**b. Don't grow the graph schema further until (a) has an answer.** The next-gen KG plan already gated Phase 5 ("prove it matters") on qrels existing — they now do (silver-tier). Phase 5 should unblock now, scoped specifically to the typed_kg rungs above. Continuing typed-field lexicon expansion or hidden-relationship mining (Tasks 3–6 of the typed-field-coverage plan) before that checkpoint repeats the exact "build the rocket before the runway" pattern the 2026-06-11 roadmap already flagged once for concept memory.

**c. Where mechanistic complexity should surface: ranking features and dataset-card evidence, not more graph nodes.** The cheapest high-leverage move is feeding `contradiction_subtype`, `specificity_tier`/`facet_fields` (Phase 6b), and select typed fields into the *existing* usefulness/affordance scorer as named, inspectable features — a handful of columns, not new infrastructure — then checking via (a) whether they move NDCG. This is also exactly what the in-progress frontend work (DatasetCard "Can test this?" panel, Workstream C Phase 2 of the execution prompt) needs to display; build the feature once, surface it in both places.

**d. Park the spectral phenotype layer (`20b96aa`, most recent commit) until it has a use.** It's a scientifically careful, well-caveated, but currently fully disconnected workstream — not wired into search ranking, KG, or any benchmark query. Recommend leaving it as documented, available infrastructure and not wiring it in until a concrete benchmark query needs it (e.g., an "aperiodic exponent flattening under anesthesia" query exists and is judged). Otherwise it's the same overbuilding pattern one layer down.

**e. When paper extraction finishes today (190K+ findings and climbing):** before any further relationship mining at the new scale, run a fresh precision audit sample. The existing `finding_audit_llm_judge_summary.md` (100-sample LLM-judge pass, 88% strict precision) found its dominant errors in exactly the fields the new contradiction logic depends on — `result_direction` (6/12 errors) and `species` (4/12). Those error modes propagate directly into `contradiction_subtype` quality. Audit before scaling, not after.

---

## 5. Ordered task list for the next sprint

1. Commit the uncommitted frontend/whitepaper WIP on this branch (isolated, ready).
2. Merge `claude/latent-usefulness-v08` → `claude/neuronpedia-foundation`; resolve doc conflicts; leave data blobs alone.
3. Regenerate artifact manifest / qrels progress report / safety gate against the merged state.
4. Add CI-aware reporting (bm25-vs-hybrid_graph overlap, dense_bge negative result) to whitepaper + claim ledger.
5. Add `typed_kg` / `typed_kg_qualified` ablation rungs — first real test of whether Phases 0–6b move NDCG at all.
6. Run the dual-judge consensus pass (`run_dual_judge_consensus.py`, already scoped in the ir-evaluation-rigor plan) on a stratified sample — Tier C → Tier B.
7. Human-adjudicate a small stratified gold subset (`triage_qrels_for_human_review.py` already exists) — even 75–100 pairs gets a first Tier A number, the single most-repeated blocker across every strategy doc since 2026-06-11.
8. Once extraction finishes: rerun typed-field coverage profiling at the new scale; fill in at least one of the four still-blank human-audit CSVs (findings / paper-links / affordances / typed-fields).
9. Explicitly defer until step 5 has an answer: typed-field lexicon expansion, hidden-relationship mining, spectral-phenotype KG wiring, further graph schema growth.

---

## 6. Flags for the user

- The hybrid_graph/full "lead" over BM25 is within bootstrap noise — don't headline it as a win externally until step 5/7 sharpen it.
- dense_bge underperforming BM25 is a real, CI-separated negative result — worth a quick look at the embedding setup before it's cited as-is.
- Dual-judge reliability is currently inestimable (zero second-judge passes) — the branch's own regression gate already says so.
- All four human-audit-template CSVs (findings, paper-dataset links, affordances, typed-fields) are still completely blank, same as 6/20 — cheap, code-free, and would unblock several "pending" evidence tiers at once if there's any spare review bandwidth.
