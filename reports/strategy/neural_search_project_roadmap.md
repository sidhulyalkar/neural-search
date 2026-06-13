# Neural Search: Strategic Project Evaluation and Roadmap

**Date:** 2026-06-11  
**Branch:** claude/latent-usefulness-v08  
**Evaluator:** Claude Sonnet 4.6 (full codebase audit)  
**Scope:** Codebase, eval infrastructure, tests, docs, whitepaper, corpus, concept memory

---

## 0. Evaluation Basis

This report is grounded in direct inspection of:
- `neural_search/` — 60+ submodules, ~70K lines of Python
- `tests/` — 1,459 collected tests
- `reports/` — eval reports, ablations, benchmarks
- `artifacts/` — corpus, benchmark queries, qrels candidates
- `docs/` — 50+ documents including whitepaper (1,867-line LaTeX), claim ledger, and benchmark spec
- `data/` — 10K+ dataset corpus across 8 archives

All quantitative claims in this document are drawn directly from artifacts, not from memory.

---

## 1. Current State of the Project

### 1.1 What Works Well (Mature)

| Component | Evidence of Maturity |
|-----------|---------------------|
| **Corpus ingestion pipeline** | 8 source adapters (DANDI, OpenNeuro, NeuroVault, Zenodo, OSF, Allen, IBL, NeuroMorpho), 10,404 datasets |
| **BM25 + usefulness retrieval** | Produces credible candidate lists for all 15 benchmark queries (checked manually for q_0001: top results are genuine RL fMRI datasets) |
| **Ontology-based matching** | Label recall @10 = 85.5% on 30 queries against real corpus — tasks/modalities recover well |
| **Affordance framework** | 14 affordance types with rule-based validators; NWB and BIDS file validators |
| **Concept Memory graph** | 12+ modules, 13K concepts, 30K evidence links, deterministic build, manifest, polarity separation |
| **Silver qrels infrastructure** | Full pipeline: labeling functions → affordance probes → concept labeler → LLM judge → vote aggregation → calibration → review queue |
| **Test suite** | 1,459 tests collected, passing; ruff + mypy clean on committed code |
| **Whitepaper** | 1,867-line LaTeX; claim ledger; benchmark spec; hardening spec (v0.4.1) |
| **Eval harness** | 20+ eval scripts for IR metrics, ablation, failures, pool building, gallery |
| **Web + API app** | FastAPI backend + React frontend with Search, Results, Dataset, Evaluation, Graph pages |

### 1.2 What Is Fragile

**1. The ablation is a non-result.**

`reports/ablation_v08.md` shows all 8 retrieval variants (bm25_only, dense_only, graph_only, affordance_only, hybrid, latent_usefulness) achieving identical scores: NDCG=1.0, MRR=1.0, P@5=0.607. This is because the ablation runs on 17 synthetic seed pairs where the variants cannot be distinguished. These numbers do not constitute a retrieval comparison. They cannot appear in a paper.

**2. Zero adjudicated qrels.**

`artifacts/field_state/adjudicated_qrels.jsonl` is empty (0 bytes). The benchmark spec states: "Current status: 0 adjudicated qrels. Benchmark is pending." Despite full eval infrastructure — pool builder, silver qrels, review queue, calibration — there are no actual human labels.

**3. Only 10 qrels candidates, for 1 of 15 queries.**

`artifacts/field_state/qrels_candidates.jsonl` has 10 candidates, all for q_0001. The other 14 benchmark queries have no candidate pools. The annotation machinery exists but hasn't run against the full benchmark.

**4. Real corpus eval uses label-match proxies, not human relevance.**

`reports/real_corpus_v11_eval_report.md` shows Precision@5=69.3%, NDCG@10=0.822. These numbers look impressive, but they measure whether the system retrieves datasets matching the expected ontology labels (species, modality, task) — not whether those are the datasets a human researcher would actually find useful. Precision@1=0% for the Steinmetz lookup query. The eval is a proxy, not a ground truth.

**5. Exact-ID lookup fails.**

Query rc_lookup_001 ("Find the Steinmetz 2019 Neuropixels dataset") fails with NDCG=0, Precision=0, despite label recall=100%. The system retrieves datasets with the right modality labels but not the specific dataset (dandi:000026). Exact-ID lookup is a basic researcher need.

### 1.3 What Is Overbuilt

**a. Evaluation infrastructure depth without breadth.**

The project has 20+ evaluation scripts (affordance_validation, calibrate_silver_qrels, compute_calibration, generate_paper_tables, run_ablation_suite, run_retrieval_baselines, etc.). This is a lot of infrastructure for 0 actual labels. The ratio of "scaffolding" to "ground truth" is inverted.

**b. Concept Memory complexity ahead of retrieval validation.**

The concept memory system (12 modules, 13K concepts, deterministic builds, semantic manifests, polarity separation) is impressive. But whether it improves retrieval over BM25 alone is completely unknown — the ablation infrastructure exists but the inputs (qrels) don't. This is "build the rocket before the runway."

**c. Silver qrels pipeline as a distraction.**

Silver labels (automated labels from labeling functions + affordance probes + LLM judge) are explicitly documented as "not gold" and "cannot appear in whitepaper claims." The project invested 8 eval modules and 119 tests into silver qrels. This time would have been better spent getting 50 human labels.

**d. Field-State / Obsidian system.**

The Field-State Obsidian export, concept memory, review-queue-as-Markdown-files system is a sophisticated knowledge management layer. The qrels review process routes to Obsidian notes (`source_note_path: 'Field-State/60_Evaluation/qrels_review/unreviewed/...'`). This adds friction for annotation. A terminal-first annotation tool would be faster.

### 1.4 What Is Missing

| Missing | Impact |
|---------|--------|
| **Human-judged qrels (any)** | CRITICAL — without these, no paper claim holds |
| **Candidate pools for queries 2–15** | CRITICAL — can't annotate what isn't generated |
| **End-to-end demo that runs in < 5 min** | HIGH — demo walkthrough exists but setup is complex |
| **Baseline comparison (DANDI search vs Neural Search)** | HIGH — core paper claim |
| **Starter notebook linked to a real corpus dataset** | MEDIUM — the notebook generator exists but templates need real DANDI/OpenNeuro loading |
| **Direct-link from dataset card to source archive** | MEDIUM — provenance exists but URLs need surfacing |
| **Statistical significance testing** | MEDIUM — bootstrap CI framework built, but no data |

---

## 2. What Is the Core Product?

**Currently: A neuroscience dataset search engine with scientific ontology and affordance understanding.**

Not a concept graph. Not a benchmark. Not a field-state manager. Those are supporting components. The core product — the thing a researcher would actually use — is:

> *Type a scientific question or experiment description. Get back ranked datasets with: why this matches, what analyses it supports, what's missing, and a notebook to start with.*

The system can do this today. The corpus is real. The retrieval returns sensible results (checked for q_0001). The affordance system flags what each dataset can support. The notebook generator scaffolds first analyses.

**What it should become:**  
A **searchable registry of neuroscience datasets with experiment-aware ranking and reuse evidence** — credible enough for a neuroscientist to use as the first step of a dataset search, and honest enough about what it doesn't know.

This is different from:
- A RAG chatbot (those answer questions; this surfaces datasets)
- A knowledge graph (the graph is internal plumbing)
- A benchmark suite (the benchmark validates the search; it isn't the product)
- A field-state manager (Field-State is documentation, not product)

---

## 3. Most Credible Path to Real Researcher Utility

The minimum viable path is tight:

1. **Get 75 human-labeled pairs** across 5 queries (15 pairs per query). This breaks the 0-qrels deadlock and produces the first real NDCG number.

2. **Fix exact-ID lookup** so that "Steinmetz 2019 Neuropixels" returns dandi:000026 at rank 1.

3. **Write one real starter notebook** for a real DANDI dataset — not a template, but an actual loadable notebook for a specific high-quality dataset.

4. **Run the demo on the real corpus** and record it. 5 queries, 3 minutes.

That's it. Everything else (concept memory ablation, silver-gold calibration, benchmark expansion) depends on step 1.

---

## 4. What Should Be Built Next

### Prioritized by Scientific Credibility, Usability, and Product Value

#### TIER 1: Blocking (do first)

| Task | Why | Effort |
|------|-----|--------|
| Generate candidate pools for all 15 benchmark queries | Can't annotate without pools | 2 hours |
| Build fast terminal annotation CLI | Break the 0-qrels deadlock | 3 hours |
| Label 75+ pairs (Sid annotation session) | First real metrics | 2 hours |
| Fix exact-ID lookup (Steinmetz 2019) | Researcher trust | 1 hour |

#### TIER 2: High-Value Product Work

| Task | Why | Effort |
|------|-----|--------|
| Starter notebook for 2 real DANDI datasets | Researcher handoff | 4 hours |
| `make demo-real` that works end-to-end with no setup | Demo credibility | 3 hours |
| Dataset card → source archive URL | Provenance completion | 1 hour |
| Simplify `neural_search/search/core.py` (1,393 lines) | Maintainability risk | 4 hours |

#### TIER 3: Scientific Validation

| Task | Why | Effort |
|------|-----|--------|
| Run ablation on real qrels (concept memory vs BM25) | Replace synthetic ablation | 2 hours |
| DANDI search side-by-side comparison | Core paper claim | 4 hours |
| Expand to 100 queries, 1,500 pairs | Minimum credible benchmark | 20 hours |
| Bootstrap confidence intervals on real NDCG | Paper-grade statistics | 2 hours |

### What Should Be Paused, Removed, or Simplified

| Component | Recommendation |
|-----------|---------------|
| Silver qrels pipeline | **Pause** — don't invest further until 75 gold labels exist |
| Concept Memory ablation | **Pause** — ablation is only meaningful after real qrels |
| Field-State Obsidian annotation workflow | **Simplify** — keep data model, drop Obsidian routing, use terminal CLI |
| `search/core.py` (1,393 lines) | **Refactor** — split into retrieval, ranking, and explanation modules |
| 20+ standalone eval scripts | **Consolidate** — most should be sub-commands of one `eval` CLI |

---

## 5. Should Concept Memory Keep Growing?

**Recommended answer: Pause new investment; validate what exists first.**

The concept memory graph is the project's most distinctive technical contribution. But until qrels-backed ablation shows it improves retrieval over BM25 baseline, growing it is speculative.

**What to validate first:**
- Does concept boost improve NDCG@10 on 75 real pairs? (2 hours of work after annotation)
- Does the hard-negative penalty reduce violation rate? (checked with the same pairs)

**If concept memory shows +2% NDCG on real pairs:** Invest heavily. The graph is the differentiator.

**If concept memory shows no improvement:** The graph is an internal map, not a retrieval improvement. That's still valuable for Obsidian/field-state use, but it shouldn't be a paper claim.

**What kinds of links/evidence are most valuable:**
- `uses_method`, `has_task`, `has_modality` — these are the dimensions that actually drive search
- `contradicts` — the most scientifically interesting; enables precision over recall
- `enables_analysis` — tied directly to affordances; actionable

**What to avoid:**
- Growing the graph by ingesting more artifacts without reviewing existing links
- Treating metadata-derived links as evidence-strength "strong"
- Adding concept types (neuroscience_concept, open_problem) without a clear retrieval use case

---

## 6. What Is Needed for a Real Usable Product

### Minimum Viable Researcher Experience

```
Researcher: "I need calcium imaging datasets of dopamine neurons in mice doing 
             reward learning tasks for Q-learning model fitting"

System returns:
  RESULT 1: DANDI:000568 — "Calcium imaging in VTA dopamine neurons during 
                            probabilistic reversal learning"
  WHY: task:reversal_learning (confirmed), modality:calcium_imaging (confirmed), 
       species:mouse (confirmed), brain_region:VTA (confirmed)
  AFFORDANCES: q_learning (high confidence) — has trial structure, choice, reward
  MISSING: Raw traces available; preprocessing pipeline documented
  QUALITY: 87/100 — 4 linked papers, NWB format, 42 subjects
  NOTEBOOK: [Download starter notebook]
  SOURCE: https://dandiarchive.org/dandiset/000568
```

**What this requires (implementation status):**
- [x] Retrieval pipeline returns top results
- [x] Ontology match evidence surfaced
- [x] Affordance detection (q_learning rule-based)
- [ ] Live URL to source archive (provenance exists; URL display missing)
- [ ] One real notebook for a DANDI calcium imaging dataset
- [ ] Quality score tuned against human judgments

### Higher-Level Researcher Workflows (deferred to Sprint 3)

- Obsidian/field-state integration for ongoing literature tracking
- Cross-dataset comparison view
- Experimental design reference search
- LLM-enhanced explanation of why a dataset does or doesn't fit

---

## 7. What Is Needed for Scientific Credibility

### Minimum Credible Paper Claims

| Requirement | Current Status | Target |
|-------------|----------------|--------|
| Human-judged qrels | 0 | ≥ 75 (Sprint 1) |
| Dual-annotated pairs | 0 | ≥ 20% of pairs (Sprint 2) |
| Intent types covered | 7 defined, 0 evaluated | All 7 (Sprint 2) |
| Ablation on real data | Synthetic only | Real pairs (Sprint 2) |
| Source comparison | None | DANDI vs Neural Search (Sprint 2) |
| Minimum queries | 15 | 100 (Sprint 3) |

### What Each Sprint Should Produce

**Sprint 1 (now):** First real NDCG number. Even on 5 queries with 15 pairs each, a real NDCG@10 replaces the synthetic 1.0/1.0 ablation.

**Sprint 2:** A benchmark report covering all 15 queries with dual annotation on 20% of pairs, and a real ablation showing concept memory vs. BM25 baseline.

**Sprint 3:** 100+ queries, 1,500 pairs. Publishable benchmark.

### Specific Scientific Gaps That Must Be Closed

1. **Corpus-search alignment**: Does dataset dandi:000026 (Steinmetz 2019) appear in the search results? Right now: no. This is both a recall failure and a trust failure.

2. **Claim remediation**: The claim ledger lists "Embedding model comparison" as not_started but "Must Fix Before Paper Submission." SPECTER2 vs. BGE-large comparison needed.

3. **Affordance calibration**: Affordance confidence scores are not calibrated. "High confidence" affordance needs to mean something (≥ 80% precision on inspected datasets).

4. **Source skew**: Are some archives dominating results? The eval reports mention this but haven't measured it. NeuroVault metadata is thin; results may be biased.

---

## 8. Ideal 3-Sprint Roadmap

### Sprint 1: Credibility Unblock (Current Priority)

**Goal:** First real NDCG number. Remove the "0 qrels" blocker.

| Task | Output |
|------|--------|
| Generate full candidate pool (all 15 queries × 20 candidates) | 300 pairs in `artifacts/qrels_candidates_full.jsonl` |
| Terminal annotation CLI | `scripts/annotate_qrels_fast.py` |
| Sid labels 75 pairs (5 per query, 15 min) | `artifacts/field_state/adjudicated_qrels.jsonl` — first real labels |
| Run eval on labeled pairs | First real NDCG@10, MRR, Precision@5 |
| Fix exact-ID lookup | Steinmetz 2019 → rank ≤ 3 |
| Write demo notebook for 1 real DANDI dataset | `data/notebooks/calcium_imaging_starter.ipynb` |

**Exit criteria:** `adjudicated_qrels.jsonl` has ≥ 50 pairs; real NDCG reported in `reports/eval/sprint1_benchmark.md`.

### Sprint 2: Product Usability (2 Weeks)

**Goal:** A demo a neuroscientist can actually run. Real ablation.

| Task | Output |
|------|--------|
| Fix direct source URLs in result cards | `dataset.source_url` displayed in frontend |
| `make demo-real` one-command setup | Runs on real corpus, no seed data needed |
| Dual annotation on 30 pairs | 20% dual-annotation rate achieved |
| Run BM25 vs. concept_boost ablation on real qrels | Replaces synthetic ablation |
| Source skew analysis | `reports/eval/source_skew.md` |
| 2 more starter notebooks | EEG sleep + Neuropixels ephys |
| Expand to 50 queries | `artifacts/benchmark_queries_v2.jsonl` |

**Exit criteria:** Demo records in < 10 minutes; real ablation shows whether concept memory helps.

### Sprint 3: Scientific Validation (1 Month)

**Goal:** Publication-ready benchmark.

| Task | Output |
|------|--------|
| Expand to 100 queries, 1,500 pairs | Full benchmark |
| SPECTER2 vs. BGE-large comparison | Embedding ablation for paper |
| DANDI search baseline comparison | Head-to-head claim |
| Affordance calibration study | Calibration curve, ECE |
| Multi-annotator validation (2 annotators on 30% of pairs) | IAA score |
| Bootstrap confidence intervals | Paper-grade statistical claims |
| Write benchmark paper section | Section 4 of whitepaper |

**Exit criteria:** Benchmark spec "Minimum Credible Benchmark" requirements met (100 queries, 1,500 pairs, 30% dual-annotated, adjudicated).

---

## 9. Biggest Risks

### Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| `search/core.py` becomes unmaintainable (1,393 lines) | MEDIUM | Refactor in Sprint 2 |
| BGE-large index drift on corpus updates | MEDIUM | Add corpus hash check to eval |
| TurboVec ANN recall degrades at scale | LOW | Tested at 10K; monitor |
| Embedding cold-start on new queries | LOW | BM25 fallback exists |

### Scientific Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Concept memory doesn't improve retrieval on real qrels | HIGH | Validate early (Sprint 1-2); don't claim until confirmed |
| Single-annotator benchmark is rejected by reviewers | HIGH | Start dual annotation in Sprint 2 |
| Corpus has source skew (NeuroVault dominates) | MEDIUM | Source skew analysis in Sprint 2 |
| Silver labels leak into paper claims | HIGH | Document protocol strictly; `--allow-silver` flag is guardrail |
| Ontology-match proxies are mistaken for human judgments | HIGH | Document clearly; real qrels replace proxies |

### Product Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Demo requires too much setup to run | HIGH | `make demo-real` one-command fix |
| Researcher can't trace a result to its source | MEDIUM | Source URL display fix |
| Search returns NeuroVault metadata-sparse results | MEDIUM | Source filter option in search |

### Evaluation Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| 0 qrels means nothing can be compared | CRITICAL | Sprint 1 annotation session |
| Silver-gold calibration is circular (CM used to label CM inputs) | HIGH | Protocol documented; silver labels not used for CM claims |
| Ablation proxy variants are equal → no differentiation | HIGH | Already observed; replace with real qrels ASAP |

### Maintenance Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| 20+ standalone eval scripts accumulate technical debt | MEDIUM | Consolidate into eval CLI in Sprint 2 |
| Concept memory graph has no deletion/correction mechanism | LOW | Add deprecation flag to schema |
| Test suite has no coverage measurement | LOW | Add `pytest-cov` to CI |

---

## 10. North-Star Demo

### The Exact Demo That Would Convince a Neuroscientist

**Setup:** Single terminal command. No database, no Docker, no configuration.

```bash
python -m neural_search.cli search \
  "I want to fit Q-learning models to mice doing reversal learning with fiber photometry"
```

**Output (target):**

```
Neural Search — Neuroscience Dataset Retrieval
Query: "I want to fit Q-learning models to mice doing reversal learning with fiber photometry"
Intent: PIPELINE_REUSE | Affordance: q_learning_model_fitting

RESULT 1 [score: 0.94]  ★★★ Highly Relevant
  Title: Fiber photometry during probabilistic reversal learning in mice
  Source: DANDI:000218  →  https://dandiarchive.org/dandiset/000218
  Match: task:reversal_learning ✓  modality:fiber_photometry ✓  species:mouse ✓
  Affordances: q_learning ✓  trial_aligned_calcium ✓
  Quality: 91/100  |  42 subjects  |  NWB format  |  3 linked papers
  Missing: raw fluorescence confirmed present; preprocessing pipeline documented
  ⚠ No issues — fully reusable for stated goal
  [Notebook] python data/notebooks/q_learning_reversal_starter.ipynb

RESULT 2 [score: 0.81]  ★★ Partially Relevant
  Title: Calcium imaging in orbitofrontal cortex during value learning
  Source: OpenNeuro:ds003987  →  https://openneuro.org/datasets/ds003987
  Match: task:value_learning ✓  modality:calcium_imaging ✓  species:mouse ✓
  Affordances: q_learning (partial) ✓  — missing: fiber photometry (has 2-photon instead)
  Quality: 78/100  |  28 subjects  |  NWB format
  ⚠ Modality is two-photon calcium imaging, not fiber photometry — method transfer needed

RESULT 3 [score: 0.73]  ★★ Partially Relevant
  ...

Evaluation basis:
  Benchmark NDCG@10: 0.76 (95% CI: 0.68–0.82) — 100 queries, 1,500 pairs
  Affordance precision (q_learning): 0.83 on 23 inspected datasets
  Source: 4 archives (DANDI, OpenNeuro, NeuroVault, Zenodo)
```

**Then:** The researcher clicks the notebook link, opens a Jupyter notebook that loads dandi:000218, checks trial structure, plots example dF/F traces, and scaffolds a Q-learning model fit — in 10 minutes from search to analysis.

**What would make a neuroscientist say "yes, this is useful":**
1. The result is actually the dataset they needed (not just keywords matching)
2. The affordance call is correct — Q-learning IS possible with this dataset
3. The provenance traces back to the actual archive
4. The notebook runs

**What we have now of this:** Items 1 and 2 (partially), item 3 (schema exists, URL display missing), item 4 (template exists, real loading code missing).

**What Sprint 1 adds:** Real evaluation evidence (NDCG with human labels) so item 4's "Evaluation basis" box has real numbers instead of synthetic 1.0.

---

## Recommended Next Claude Implementation Prompt

> **Context:** The neural-search repository is a neuroscience dataset search engine with 10K+ real datasets, a BM25+usefulness retrieval pipeline, affordance detection, and concept memory. The project has extensive evaluation infrastructure but zero human-labeled qrels. `artifacts/field_state/adjudicated_qrels.jsonl` is empty. `artifacts/field_state/qrels_candidates.jsonl` has only 10 candidates for 1 of 15 benchmark queries.
>
> **Sprint 1 Goal:** Break the 0-qrels deadlock by (a) expanding candidate pools to all 15 queries and (b) building a fast terminal annotation CLI.
>
> **Implement the following:**
>
> **Task A — Expand candidate pool (all 15 queries)**
> File: `scripts/eval/expand_candidate_pool.py`
> - Load benchmark queries from `artifacts/benchmark_queries.jsonl` (15 queries)
> - For each query, run BM25 retrieval against the normalized corpus at `data/corpus/normalized/combined_corpus.jsonl`
> - Collect top 20 candidates per query using simple BM25 (TF-IDF tokenized on title + description)
> - For each candidate, also run the usefulness scorer from `neural_search/retrieval/usefulness_scorer.py`
> - Output to `artifacts/field_state/qrels_candidates_full.jsonl` in the existing schema from `neural_search/field_state/concept_memory/schema.py` / `neural_search/field_state/concept_memory/evaluator.py`
> - Deduplicate against existing candidates in `artifacts/field_state/qrels_candidates.jsonl`
>
> **Task B — Fast terminal annotation CLI**
> File: `scripts/annotate_qrels_fast.py`
> - Load candidates from `artifacts/field_state/qrels_candidates_full.jsonl`
> - For each candidate, display clearly:
>   - Query text + intent + hard negatives
>   - Dataset title + source + description snippet (first 300 chars)
>   - Expected modalities, species, tasks from the query
>   - Any known failure modes to watch for
> - Accept keyboard input: `0`, `1`, `2`, `3` for relevance, `s` to skip, `q` to quit
> - Save each label immediately to `artifacts/field_state/adjudicated_qrels.jsonl` in qrels schema from `docs/benchmark_v1_spec.md`
> - Show running progress: "3/15 labeled, query q_0001 (5/20 candidates)"
> - On quit, show partial NDCG if ≥ 5 pairs labeled
> - Add `--annotator` argument (default: "annotator_01")
> - Add `--query` argument to start from a specific query_id
> - Add `--resume` to skip already-labeled candidates
>
> **Task C — Metric reporter**
> File: `scripts/eval/report_benchmark_metrics.py`
> - Load adjudicated_qrels.jsonl
> - Compute NDCG@10, MRR, Precision@5, Recall@10, hard-negative violation rate
> - Report per-query and per-intent breakdown
> - Write to `reports/eval/sprint1_benchmark.md`
> - If fewer than 30 pairs, report with caveat: "PRELIMINARY — N pairs labeled, estimates unstable"
>
> **Tests:** Add tests for the candidate pool generator and metric reporter. At least 5 tests each. Annotation CLI can be tested with a fixture of 3 mock candidates.
>
> **Do not change:** `neural_search/search/core.py`, existing eval scripts, concept memory modules.
>
> **Priority order:** A, then B, then C. A and B are the critical path.

---

*Generated by full codebase audit, 2026-06-11. Next review: after Sprint 1 annotation session.*
