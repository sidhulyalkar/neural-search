# Next-Phase Plan: Growing, Validating, and Accumulating Neuroscience Knowledge

**Date:** 2026-07-01
**Supersedes:** `2026-06-23_qrels_branch_merge_and_kg_rigor_plan.md`, `brainknow_comparison_publishable_plan.md`, `claude_whitepaper_frontend_execution_prompt.md`, `neural_search_project_roadmap.md`, `next_gen_kg_development_plan.md` (all pre-2026-07-01, see `reports/repository_cleanup_proposal_2026-07-01.md`)
**Builds on:** `2026-07-01_information_completeness_and_technical_depth_plan.md`, `2026-07-01_platform_evaluation_and_adoption_plan.md`, `reports/architecture_connectivity_audit_2026-07-01.md`

## Where we actually are

This session's work (Methodology Registry, reanalysis candidates, the connectivity audit) surfaced the real shape of the problem: **this project has more built knowledge than it has connected knowledge.** Concretely:

- The production search graph grew from 7,438/81,518 to 7,680/141,218 nodes/edges today purely by *reconnecting* already-built, already-tested code (`methods_builder.py`, a new Methodology Registry bridge, and reanalysis-candidate detection) — no new ingestion, no new data source, just wiring.
- The architecture audit found **6 more fully-built KG layers sitting orphaned** the same way: disorder ontology, Allen structural connectivity, paradigm, oscillation, species-homology, and HCP-connectivity builders all write to `artifacts/kg/composed_kg.jsonl`, which nothing merges into the graph search actually reads.
- Meanwhile, the platform evaluation plan's core finding still holds: **gold qrels remain at 0 rows.** Every retrieval number in this repo, old or new, is measured against LLM-silver labels.
- The information-completeness plan's 8-layer model is the right long-term shape, but too large to execute as a single program. This plan sequences a subset of it against what's actually tractable now.

The strategic implication: **the next phase is dominated by reconnection and validation, not new construction.** Building more KG layers before reconnecting the ones that exist would repeat the exact mistake this session found and fixed three times over (methods_builder, the Methodology Registry, the orphaned-builder pattern).

## Phase 1: Finish the reconnection sweep — COMPLETE (2026-07-01)

Applied the exact pattern used for the Methodology Registry to all 6 remaining orphaned builders (`disorder_builder.py`, `allen_connectivity_builder.py`, `paradigm_builder.py`, `oscillation_builder.py`, `species_homology_builder.py`, `hcp_connectivity.py`). What was actually found and fixed, in order:

1. **Every builder was far more broken standalone than expected.** Each references node types (`ontology_region:*`, `circuit:*`, `topic:*`, `oscillation:*`, `species:*`) that no single builder creates, using at least 3 mutually-incompatible hand-rolled id conventions across the 6 files (e.g. `ontology_region:{region}` vs `ontology_region:{species}:{region}`). Rather than hand-reconciling each vocabulary, added a general `neural_search.graph.schema.resolve_dangling_edges()` utility that creates minimal, clearly-marked (`stub: true`) placeholder nodes for any edge endpoint missing after all layers merge. Applied once in `scripts/build_real_corpus_graph.py`, it produced 183 stub nodes and brought the fully-merged graph to zero dangling edges.
2. **`hcp_connectivity.py` had a real logic bug**, not just an id mismatch: a "circuit annotation" edge set its `source_node_id` to another edge's `edge_id` string (a meta-edge the schema doesn't support), guaranteeing it could never resolve. Fixed to link both endpoint regions to the circuit directly.
3. **Critical: ran the ablation ladder as this plan already required, and it caught a real regression.** Reconnecting these 6 layers alone was NDCG-neutral (proven by isolated A/B testing with/without them: identical 0.8494 either way). But the *combination* including the Methodology Registry's reanalysis-candidate edges regressed `hybrid_graph`/`full` NDCG@10 from 0.8594 to 0.8494 — bisected to two causes in `neural_search/graph/search_features.py`: the 59,126 `dataset_old_dataset_new_method_candidate` edges dominated the generic `graph_degree` feature, and a `reanalysis_edge` score weight (0.018) had been configured speculatively before this edge type ever existed. Fixed both (excluded from degree like `dataset_similar_to_dataset` already is; weight set to 0.0 pending gold-qrels validation) — NDCG@10 confirmed restored to exactly 0.8594. Full writeup in `reports/reanalysis_candidates_report.md`.

**Production graph, final state:** 7,946 nodes / 141,911 edges, zero dangling, NDCG@10 unchanged from pre-session baseline. 44 new tests added across `tests/test_resolve_dangling_edges.py`, `tests/test_orphaned_kg_layers_reconnection.py`, and extensions to `tests/test_build_real_corpus_graph.py` / `tests/test_graph_search_features.py`.

**Lesson for future reconnection work:** the ablation-check step in this plan is not optional ceremony — it caught a real, silent regression that would have shipped unnoticed if "zero dangling edges" had been treated as the finish line instead of "zero dangling edges AND unchanged-or-improved ranking quality."

For each: after merging, run the ablation ladder (`scripts/eval/run_ablation_ladder.py`) to check whether the new edges move NDCG@10 at all before claiming any retrieval benefit — reconnecting a layer is not the same as proving it helps. Silence (no movement) is an acceptable, honest outcome to report.

Also in this phase: resolve `neural_search/graph/similarity.py` — either delete it (the inline duplicate in `build_real_corpus_graph.py` is what's actually live and was deliberately kept cheap/capped) or explicitly repurpose it as a documented offline research utility. Don't leave it as ambiguous dead code.

## Phase 2: Gold qrels campaign (2-3 weeks, can run in parallel with Phase 1)

Unchanged from the platform evaluation plan's Phase 1 — this remains the single highest-leverage scientific-credibility action and nothing this session found changes that:
- Stratified 100-query set (25 each: strict lookup, pipeline reuse, reanalysis feasibility, cross-dataset comparison).
- 300 human-labeled pairs, 75 dual-annotated for agreement stats.
- Re-run the full ablation ladder against the gold subset once available, reporting gold vs. silver separately.

The reanalysis-candidate edges built today are a natural source of *stratified reanalysis-feasibility queries* for this campaign — e.g. "does this dataset actually support Granger causality analysis" pairs drawn directly from the 59,126 candidate edges, giving the annotation campaign a concrete, high-value slice to start with rather than a purely synthetic query set.

## Phase 3: Reanalysis-bridge completion — `dataset_reanalysis_bridge_dataset` DONE (2026-07-01)

Turned out simpler than scoped: the original plan assumed this edge type needed the NER `METHOD_SURFACE_TO_ID` vocabulary reconciled against `methods_taxonomy.yaml` first. Reading the actual edge shape (`search_features.py::_relationship_summaries`) showed `dataset_reanalysis_bridge_dataset` connects two **dataset** nodes with method as a string property, not a graph edge to a method node — so the vocabulary mismatch, while still real, was never a blocker for this specific edge type.

Built by joining `artifacts/literature/paper_dataset_links.jsonl` (dataset → OpenAlex paper, 393/7,171 real matches) with `artifacts/ner/ner_kg.jsonl` (21,185 `paper_uses_method` edges) to get 82 datasets with real, evidence-backed method-usage facts, then traversing existing `dataset_similar_to_dataset` edges from those precedents: 2,517 bridge edges across 818 candidate datasets. New module: `neural_search/graph/reanalysis_bridge_builder.py`.

**Same regression pattern found again, at smaller scale**: populating these edges regressed NDCG@10 from 0.8594 to 0.8545 (graph-degree inflation, same mechanism as the candidate-edge regression, proportionally smaller because the edge count is ~23x smaller). Fixed the same way (excluded from `graph_degree`); confirmed NDCG@10 restored to exactly 0.8594. This edge type also feeds the already-tuned `relationship_edge` weight (0.012, nonzero) — that weight was deliberately left unchanged since the degree fix alone was sufficient, verified empirically before committing.

**Remaining, still explicitly deferred:**
- `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate` — still no defensible data source identified; don't force these before a real signal exists.
- Full NER-`methods_taxonomy.yaml` vocabulary reconciliation (18-id `analysis_affordances.py` vs. 21-id `affordances/registry.py` vs. 17-id ontology vs. 27-string `analysis_families` vs. NER's own ~48-term surface vocabulary) — confirmed not required for anything built so far, but real technical debt if a future edge type ever needs to cite NER-extracted methods against the taxonomy directly. Its own multi-week project touching four+ independently-shipped systems.
- Expanding `paper_dataset_links.jsonl` beyond 393/7,171 matched datasets — a literature-linking improvement (would directly grow the reanalysis-bridge precedent pool from 82), not a KG-wiring one.

## Phase 3.5: SSOT manifest, evidence tiers, file validation, reinterpretation candidates (2026-07-02)

- **SSOT manifest**: `scripts/build_artifact_manifest.py` (computes everything from files) + `scripts/generate_whitepaper_stats.py` (generates whitepaper LaTeX macros from it) replace the hand-maintained, twice-drifted-stale `reports/eval/current_artifact_manifest.json`. All whitepaper numbers now regenerate instead of rotting.
- **Evidence tiers**: the whitepaper's 6-tier framework is now a real `evidence_tier` graph-edge property (`neural_search/kg/schemas/evidence_tier.py`), not just prose.
- **Live file validation, zero new dependencies**: DANDI (header-only NWB reads via HTTP range requests) and OpenNeuro (GraphQL summary field) validators, both real and tested. Found two real bugs in the process (pre-existing `dandi` package dependency silently failing; a pagination bug in the new validator caught by a test before shipping). Top-50 suggestions validated for real: 10/20 DANDI confirmed, 60 edges upgraded to `file_validated`. NDCG confirmed unchanged.
- **`dataset_reinterpretation_candidate` DONE**: real data source found (`finding_edges.jsonl`, 117K+ literature contradictions), builder implemented and wired. Current yield is 0 edges — honest result of low paper-link coverage (5.6%), not a bug; will self-populate as link coverage grows.
- **`dataset_reprocessing_candidate`**: data source found (NWB `nwb_version` attribute) but not yet built — corpus-scale live-check work, scoped not started.
- **Paper-link expansion hit a real wall**: this environment's OpenAlex request budget was exhausted after ~70 records during a full-corpus live-relinking attempt. Fixed a real silent-failure bug this exposed (budget/rate-limit errors were being recorded as false "not found" — see `TransientLookupError` in `neural_search/literature/linking.py`). Salvaged 10 genuine new matches (393 -> 403/7,171 real matches) before the wall hit; a full re-run needs the budget to reset (external, not code-blocked).

**Lesson reinforced a third time**: every new edge type or property addition gets an ablation-ladder check before being called done — evidence tiers and the file-validation upgrade were both confirmed NDCG-neutral empirically, not assumed safe just because "it's just metadata."

## Phase 3.6: Scholarpedia reconnection + full orphaned-builder audit close-out (2026-07-02)

- **`scholarpedia_builder.py` reconnected**: the last of the 7 originally-identified orphaned KG-merge builders. Unlike the other 6, it is fully self-contained (198 nodes, 333 edges, zero dangling edges standalone before merging) — its concept/domain/alias vocabulary doesn't reference any node type another builder owns. Production graph: 8,144 nodes / 144,761 edges (up from 7,946/141,911). NDCG@10 confirmed unchanged at exactly 0.8594 after regeneration — verified via the ablation ladder, not assumed, per the now-3x-proven standing rule.
- It now reaches production through two independent paths: the graph-merge above, and the pre-existing `neural_search/search/concept_authority.py` query-expansion path (`SCHOLARPEDIA_CONCEPTS` imported directly, live at `neural_search/search/core.py:1307`).
- **Follow-up audit completed** for the remaining 4 KG-producing modules not merged into `build_real_corpus_graph.py` plus `neural_search/graph/similarity.py`. Outcomes:
  - `concept_builder.py` — confirmed fully disconnected and valuable, reconnected the same way as scholarpedia (21 nodes / 217 edges standalone, resolved via `resolve_dangling_edges`). Production graph now 8,234 nodes / 144,978 edges. NDCG@10 confirmed unchanged (0.8594).
  - `ner_builder.py`, `citation_builder.py` — confirmed genuinely connected via side-channel artifacts read directly by `search_features.py` / `reanalysis_bridge_builder.py`; `citation_builder.py`'s own `paper_cites_paper`-edge logic remains deliberately deferred (needs paper nodes, which don't exist in production yet — a real prerequisite, not pure wiring).
  - `neurosynth_builder.py` — confirmed connected via `composed_kg.jsonl`, but that artifact had **no regeneration pipeline** (gitignored, manually built) — a real operational fragility. Fixed: `_load_neurosynth_index()` now falls back to building directly from source when the artifact is missing, instead of silently scoring zero.
  - `neural_search/graph/similarity.py` — reconfirmed zero production callers; deleted (module + test + `__init__.py` re-exports) rather than left as ambiguous dead code.
  - See `reports/architecture_connectivity_audit_2026-07-01.md` for full per-module detail. This closes out the KG-builder connectivity sweep: every ingestion module producing graph-like output is now either merged, genuinely side-channel-connected, removed, or explicitly deferred with a stated reason.

## Phase 3.7: Literature-source expansion, Phase 1 — DataCite (2026-07-02)

Approved plan: `C:\Users\sidso\.claude\plans\cheerful-cuddling-knuth.md` (Crossref/DataCite/Semantic Scholar/PubMed literature sources + a Bluesky discourse layer, phased). Phase 1 (DataCite) shipped:

- New shared infra reused by every future source: `neural_search/http_utils.py`, `neural_search/literature/api_client.py`, `api_config.py`, `corpus_io.py` (factored from `linking.py`), `title_match.py` (factored from `linking.py`).
- `neural_search/literature/datacite.py`: DataCite's `relatedIdentifiers` are publisher-declared, not fuzzy-matched — a structurally stronger signal than OpenAlex's doi/title matching. Corpus-scale run: 91 real matches from 2,387 DOI-bearing records, 90 of which are genuinely new datasets beyond the existing 403 OpenAlex matches (only 1 overlap). Combined real coverage: 493/7,171 (6.9%), up from 403/7,171 (5.6%).
- Found (not caused) a real gap: DANDI/OpenNeuro (1,147/7,171 records) carry no `doi` field in the corpus despite having DataCite-registered DOIs — honestly recorded as `not_applicable_no_dataset_doi`, distinct from a genuine miss. Capturing these is separate corpus-normalization future work.
- `neural_search/graph/paper_node_builder.py`: real `paper` nodes reached the production graph for the first time (8,687 nodes / 145,472 edges, zero dangling). Found and fixed a real fixture-pollution bug during testing (this builder must scope itself to datasets already in the graph being built, matching `reanalysis_bridge_builder.py`'s pattern, or it silently injects real-world edges into any fixture-scale build).
- **Third instance this session of "populate first, discover the regression after"**: the pre-existing `linked_paper` score weight (0.04) had been configured before paper nodes ever existed, was permanently inert, and measurably regressed NDCG@10 (0.8594 → 0.8583) once real data existed. Fixed by zeroing pending gold-qrels validation (same treatment as `reanalysis_edge`), confirmed by isolation testing before committing. `paper_mentions_dataset`/`paper_uses_dataset` pre-emptively excluded from `graph_degree`.
- Remaining phases (Crossref/Semantic Scholar/PubMed, `citation_builder.py` reconnection, Bluesky discourse layer) are scoped in the plan file but not yet executed.

## Phase 3.8: Literature-source expansion, Phase 2 — Crossref, PubMed, bioRxiv (2026-07-03)

- **Crossref, PubMed/bioRxiv, Semantic Scholar added**: PubMed/bioRxiv processed the full 7,171-record corpus cleanly (1,783 real matches). Crossref also completed the full corpus (2,398 real matches) after two real infrastructure fixes (see below). Semantic Scholar's unauthenticated tier hit HTTP 429 after 1 request — blocked without an API key, a genuine external constraint documented like the OpenAlex budget wall.
- **Combined real paper-link coverage: 2,510/7,171 (35%)**, up from 403/7,171 (5.6%) at session start and 493/7,171 (6.9%) after Phase 1 — a ~6x improvement overall, driven by title-fuzzy matching against records with no DOI at all (DANDI/OpenNeuro-shaped data).
- **Two real bugs found and fixed during the Crossref corpus-scale run**: (1) uncaught `httpx.ReadTimeout` crashing the whole process after retries exhausted — fixed with a new `get_or_raise_transient()` helper unifying transport-level and HTTP-status-level transience into one `TransientLookupError`; (2) Crossref's HTTP 301 redirects (case-normalized/superseded DOIs) crashing via an uncaught `raise_for_status()` — fixed with `follow_redirects=True` in `http_get_with_retry`. Both covered by regression tests before being trusted at corpus scale.
- **A live, transient Crossref-side reliability incident** (intermittent 500s/timeouts on the bibliographic search endpoint) was observed and outlasted, not routed around — 3 aborted attempts (891, 80, 1 records) before a 4th succeeded fully.
- **Retraction/correction status — a real new paper-validation signal**: `scripts/check_paper_retraction_status.py` checked all 2,931 unique linked DOIs via Crossref's `update-to` field: 0 retracted, 15 corrected. Attached as a `retraction_status` property on `paper` nodes (not an edge/tier).
- **Production graph**: 12,748 nodes / 149,654 edges (up from 8,687/145,472), zero dangling edges, 4,514 unique paper nodes across 5 sources. **NDCG@10 confirmed unchanged at exactly 0.8594** despite `paper_mentions_dataset` growing from 403 to 4,585 edges — Phase 1's pre-emptive degree exclusions and weight zeroing already covered this.
- Remaining: `citation_builder.py` reconnection (Phase 3) and the Bluesky discourse layer (Phase 4), still scoped in the approved plan.

## Phase 4: Whitepaper and release-checker reconciliation (ongoing hygiene)

- Whitepaper (`docs/whitepaper/neural_search_whitepaper.tex`) updated today with the new KG counts and Methodology Registry section — keep this in the loop every time a reconnection lands (Phase 1), not as a one-time catch-up.
- `neural_search/release/` (old `demo_v05`/`real_v07` release checker) should be updated to understand the current artifact regime (`full_corpus_v09`, current manifest) rather than removed outright — removing it without a replacement leaves no release-readiness gate at all. This was already flagged in the platform evaluation plan's Phase 0 and remains open.

## What "done" looks like for this phase

- All 12 KG builder modules identified in the connectivity audit are either live in production or explicitly, deliberately archived with a documented reason — no more silent orphans.
- At least one gold-qrels-backed NDCG/MRR report exists, clearly separated from silver diagnostics.
- The reanalysis-candidate signal has been used to seed real annotation work, not just sit as an unvalidated heuristic.
- The whitepaper's artifact truth table and non-claims section stay synchronized with each reconnection, so "current state" never drifts from "what the paper says" the way `demo_v05`/`real_v07` drifted from `full_corpus_v09`.

## Bottom line

The system does not need a bigger knowledge graph. It needs the knowledge graph it already has switched on, and it needs human judgment on a representative slice of its output before making any more retrieval-quality claims. Everything else — new sources, new edge types, broader vocabularies — should wait behind those two things.
