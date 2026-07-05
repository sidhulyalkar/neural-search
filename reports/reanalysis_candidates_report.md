# Reanalysis Candidates Report

- Corpus records scanned: 7171
- Candidate edges (`dataset_old_dataset_new_method_candidate`): 59126
- Datasets with >=1 candidate: 5720/7171 (79.8%)
- Candidate edges on datasets with existing linked papers (weak signal only, not a prior-usage proof): 283/59126

**Caveat:** every edge is a heuristic ('this dataset's profile matches a data form/analysis family this technique supports'), not a verified claim that the dataset hasn't already been analyzed this way. All edges carry `requires_human_review=True`. See the Methodology Registry work (`reports/methodology_coverage_report.md`) for why 10/27 analysis_families have no technique mapping yet.

**Ranking impact (measured, 2026-07-01):** populating these 59,126 edges initially *regressed* `hybrid_graph`/`full` NDCG@10 on the 317-query canonical benchmark from 0.8594 to 0.8494 — the edges dominated the generic `graph_degree` connectivity feature and a speculative `reanalysis_edge` score weight (0.018) that had been configured before this edge type was ever populated. Both were fixed in `neural_search/graph/search_features.py`: `dataset_old_dataset_new_method_candidate` is now excluded from `graph_degree` (same treatment as `dataset_similar_to_dataset`, for the same reason — too densely populated to be a useful degree signal), and `reanalysis_edge` weight is set to 0.0 pending gold-qrels validation. NDCG@10 is confirmed restored to exactly 0.8594 with both fixes applied. **The data remains fully present in the graph for its intended purpose (surfacing candidates for review/reporting) — it is excluded from ranking, not deleted.** This is a concrete instance of the principle in `reports/strategy/2026-07-01_next_phase_growth_validation_plan.md`: reconnecting a layer is not the same as proving it helps, and an ablation check caught a real regression before it shipped unnoticed.

## Reanalysis bridge edges (`dataset_reanalysis_bridge_dataset`, evidence-backed)

A stronger, evidence-backed sibling to the heuristic candidates above: instead of "this dataset's profile matches a technique's requirements," this edge answers "a similar dataset was actually analyzed with method X (per a real OpenAlex paper); this dataset has no such evidence."

Built by joining two existing artifacts for the first time for this purpose:
- `artifacts/literature/paper_dataset_links.jsonl` — only 393/7,171 datasets have a real OpenAlex paper match (`doi_exact` or `title_fuzzy_local`; the rest are `not_found`).
- `artifacts/ner/ner_kg.jsonl` — 21,185 `paper_uses_method` edges extracted from paper text via `neural_search/ingestion/ner_builder.py`.

Joining these: **82 datasets have real, evidence-backed "analyzed with method X" data.** From those 82 precedents, traversing the existing `dataset_similar_to_dataset` edges produced **2,517 bridge edges across 818 candidate datasets**, confidence ~0.34 on average (a transparent three-hop decay: paper-match confidence × NER extraction confidence × similarity-edge confidence).

The small base (82/7,171 datasets) is an honest reflection of current dataset-paper linkage coverage, not a shortcut taken here — expanding `paper_dataset_links.jsonl` coverage is separate, valuable future work.

**Ranking impact (measured, 2026-07-01):** populating these 2,517 edges also regressed `hybrid_graph`/`full` NDCG@10, from 0.8594 to 0.8545 — smaller in magnitude than the candidate-edge regression above (proportional to the smaller edge count) but the same mechanism: `graph_degree` inflation. Fixed the same way — `dataset_reanalysis_bridge_dataset` added to the `graph_degree` exclusion list in `neural_search/graph/search_features.py`. Unlike the heuristic candidates, this edge type also feeds the already-tuned, nonzero `relationship_edge` score weight (0.012) via `RELATIONSHIP_EDGE_TYPES` — that weight was **not** changed, since excluding the edge type from `graph_degree` alone was sufficient to restore NDCG@10 to exactly 0.8594 (confirmed empirically, not assumed).

## Evidence tiers (built 2026-07-02)

`neural_search/kg/schemas/evidence_tier.py` formalizes the whitepaper's 6-tier framework (`heuristic_candidate` < `evidence_backed_bridge` < `source_declared` < `file_validated` < `human_validated` < `computed`) as a real `evidence_tier` edge property. `dataset_old_dataset_new_method_candidate` -> `heuristic_candidate`; `dataset_reanalysis_bridge_dataset` and `dataset_reinterpretation_candidate` -> `evidence_backed_bridge`.

## Live file validation against the top 50 suggestions (2026-07-02)

Built two live validators with **zero new dependencies** (`httpx` + `h5py`/`pynwb`, all already installed):
- `neural_search/graph/dandi_nwb_validator.py` — reads only the NWB/HDF5 **header** of a remote DANDI asset (units/trials/electrodes/imaging presence and counts) via a custom HTTP-range file-like object, without downloading the file. Verified on a real 8.4GB asset: ~50 requests, ~20KB fetched, <2s. Along the way, found that the pre-existing `neural_search/data/dandi_streaming.py` and `fetch_dandiset_rich_metadata` both depend on the uninstalled `dandi` package and have been silently failing (broad except-Exception) unnoticed.
- `neural_search/graph/openneuro_bids_validator.py` — OpenNeuro's GraphQL `summary` field (modalities/tasks/subjects) is server-computed from real content, no downloads needed.

Ran `scripts/validate_top_reanalysis_suggestions.py` against the 50 highest-confidence suggestions (all tied at 0.9, all `time_frequency`; sources: dandi 20, figshare 14, gin 11, crcns 5). **10/20 DANDI datasets confirmed live** via real electrode-table presence; the other 30 honestly recorded as `validator: "none"` (no live validator for that source), not silently skipped. 60 edges (multiple techniques per confirmed dataset) upgraded to `file_validated`. Full results: `artifacts/validation/top_suggestions_file_validation.jsonl`, `reports/top_suggestions_validation_report.md`. Applied via `neural_search/graph/evidence_tier_upgrader.py`, wired into `scripts/build_real_corpus_graph.py` as a durable step. **NDCG@10 confirmed unchanged (0.8594)** after wiring — verified, not assumed.

## Reinterpretation candidates (`dataset_reinterpretation_candidate`, built 2026-07-02)

A real data source was found for this previously-unbuilt edge type: `artifacts/literature/relationships/finding_edges.jsonl` already contains 117,475+ real `contradicts` relationships between typed findings from different papers. Joined with `paper_dataset_links.jsonl`, this answers "this dataset's linked paper's finding is directly contradicted by another paper linked to a different dataset." Built `neural_search/graph/reinterpretation_candidate_builder.py`, wired into production.

**Current yield: 0 edges.** Honest result, not a bug: both sides of a contradiction must resolve to a matched corpus dataset, and at 403/7,171 (5.6%) paper-link coverage, the expected intersection over 63,352 unique contradiction pairs is well under 1. The builder is implemented, tested, and will surface real edges automatically as paper-link coverage grows — no code changes needed, only more linked papers.

**`dataset_reprocessing_candidate` — data source found, not yet built.** NWB files carry a root-level `nwb_version` attribute, readable via the same header-only technique (confirmed: `"2.2.5"` on a real asset). An old schema version is a genuine, objective reprocessing signal. Corpus-scale population is scoped but not implemented this session (would need a live check per DANDI dataset, similar cost to the top-50 validator).

## Paper-link coverage expansion attempt (2026-07-02) — a budget wall, not a code failure

Discovered 2,387/7,171 corpus records have a `doi` field directly, but the existing local-index linker (`link_corpus_to_local_literature`) is limited to the ~255K-paper tier-1 (≥100 citation) local OpenAlex index, explaining the low 393/7,171 baseline match rate. The **live** linker (`link_corpus_to_literature`, hitting the real OpenAlex API) found matches on 11/20 in a spot check — a dramatically higher hit rate.

A full-corpus live run was launched and ran for ~87 minutes before this environment's OpenAlex request budget hit **$0 remaining** (HTTP 429, "Insufficient budget... resets at midnight UTC") — a genuine external constraint, not something fixable from this session. Critically, `neural_search/literature/linking.py`'s pre-existing `except Exception: return None` silently converted every subsequent budget-exhausted request into a false "not_found" for the remaining ~7,100 records — this was **fixed** (new `TransientLookupError`, raised on HTTP 429/5xx and distinguished from a genuine 404; `link_corpus_to_literature` now stops immediately and reports how far it got, rather than completing "successfully" with corrupted data). The 26 matches found before the budget ran out were audited; 10 were genuinely new (not already in the local-index match set) and were merged in — bringing real matches from 393 to **403/7,171**. Re-running the full live pass is future work, blocked on budget availability, not on missing code.

## Literature-source expansion, Phase 1: DataCite (2026-07-02)

Rather than depend on a single budget-limited API (OpenAlex), added DataCite as a second, independently-metered literature-linking source — and, unlike OpenAlex's fuzzy/DOI-string matching, DataCite's `relatedIdentifiers` are **structurally declared by the data publisher** (`IsCitedBy`/`IsSupplementTo`/`IsDescribedBy` relations to a paper DOI), no fuzzy matching involved. Built `neural_search/literature/datacite.py`, plus new shared infrastructure reused by every future source: `neural_search/http_utils.py` (generic retry/backoff on 429/5xx), `neural_search/literature/api_client.py` (generalized `TransientLookupError`), `neural_search/literature/api_config.py` (typed env-var config), `corpus_io.py`/`title_match.py` (factored out of `linking.py`).

**Real, honest gap found in corpus normalization** (not a DataCite limitation): DANDI and OpenNeuro records — 1,147/7,171 combined, among the corpus's largest sources — carry no `doi` field at all, even though DANDI datasets are DataCite-registered (`10.48324/dandi.*`). Only neuromorpho/zenodo/figshare/osf/harvard_dataverse records have a `doi` field (2,387/7,171). DataCite lookups for datasets without one are recorded as `match_method="not_applicable_no_dataset_doi"`, distinct from a genuine lookup miss (`"not_found"`) — an honest accounting, not a silently-skipped gap. Capturing DANDI/OpenNeuro DOIs is real, separate future work in the corpus normalization pipeline, not this literature-linking layer.

**Corpus-scale run results (2026-07-02):** of 2,387 DOI-bearing records, **91 real DataCite-declared paper links found** (62 unique papers, including at least one bioRxiv preprint DOI). Only 1 dataset overlaps with the existing 403 OpenAlex matches — DataCite adds **90 genuinely new linked datasets**. Combined real paper-link coverage: **493/7,171 (6.9%)**, up from 403/7,171 (5.6%) — a ~22% relative increase from one additional source.

**Real paper nodes reached production for the first time.** `neural_search/graph/paper_node_builder.py` builds `paper` nodes + `paper_mentions_dataset` (OpenAlex, weaker DOI/fuzzy claim) / `paper_uses_dataset` (DataCite, stronger publisher-declared claim) edges, wired into `scripts/build_real_corpus_graph.py`'s `orphaned_layers`. Production graph: 8,687 nodes / 145,472 edges (up from 8,234/144,978), zero dangling edges. DataCite-sourced edges are tagged `evidence_tier=source_declared` — the first real occupant of that tier (previously defined but unused).

**A real regression was found and fixed, the same pattern as twice before this session.** `linked_paper` (score weight 0.04, capped at 3 papers/dataset) had been configured speculatively before any paper nodes ever existed in production — `find_papers_for_dataset()` always returned `[]`, so the weight was permanently inert. Once real paper nodes existed, NDCG@10 measurably regressed (0.8594 → 0.8583). Isolated by testing the weight alone (zeroed → exactly 0.8594 restored) before committing a fix. Set `linked_paper` to 0.0 pending gold-qrels validation, same treatment as `reanalysis_edge` — the paper nodes/edges remain fully in the graph (used by `find_papers_for_dataset`/the API layer), only excluded from ranking until validated. `paper_mentions_dataset`/`paper_uses_dataset` were also pre-emptively added to `_DEGREE_EXCLUDED_EDGE_TYPES` before this count grows further in later phases.

Remaining phases (Crossref/Semantic Scholar/PubMed coverage expansion, `citation_builder.py` reconnection, Bluesky discourse layer) are scoped in the approved implementation plan; not yet executed.

## Literature-source expansion, Phase 2: Crossref, PubMed, bioRxiv (2026-07-03)

Added three more sources: `neural_search/literature/crossref.py`, `pubmed.py` (bioRxiv folded in, distinct `10.1101/` DOI prefix), and `semantic_scholar.py`. Each does full-corpus DOI-exact and title-fuzzy matching (unlike DataCite, which is dataset-DOI-only) — this matters because title-fuzzy matching works even for DANDI/OpenNeuro-shaped records with no `doi` field at all, the exact gap DataCite couldn't close.

**Results, measured exactly, not smoothed over:**
- **PubMed/bioRxiv**: full 7,171-record corpus processed cleanly in one run. **1,783 real matches** (973 `pubmed_doi_exact`, 782 `pubmed_title_fuzzy`, 28 `biorxiv_doi_exact`).
- **Crossref**: full corpus processed after two infrastructure fixes (below). **2,398 real matches** (1,080 `crossref_doi_exact`, 1,318 `crossref_title_fuzzy`).
- **Semantic Scholar**: unauthenticated tier returned HTTP 429 after a single request in this environment — not usable without an API key (`SEMANTIC_SCHOLAR_API_KEY`, none configured). Documented as a real external constraint, same category as the OpenAlex budget wall, not a code defect.

**Combined real paper-link coverage across all 5 sources (OpenAlex, DataCite, Crossref, PubMed, bioRxiv): 2,510/7,171 (35%)**, up from 403/7,171 (5.6%) before this session and 493/7,171 (6.9%) after Phase 1 — driven mostly by title-fuzzy matches Crossref/PubMed found for records with no DOI at all.

**Two real infrastructure bugs found and fixed during the Crossref corpus-scale run** (both caught by the run itself, not by review):
1. An uncaught `httpx.ReadTimeout` crashed the whole process after `http_get_with_retry`'s retries were exhausted — the retry helper's transport-level failures were never converted into the shared `TransientLookupError` type the calling loops check for. Fixed with a new `neural_search.literature.api_client.get_or_raise_transient()` that unifies HTTP-status-level transience (429/5xx) and transport-level transience (timeouts, connection errors) into one exception, so a persistent network failure aborts a run gracefully (saving partial progress) instead of crashing uncaught. All 4 source modules (`datacite.py`, `crossref.py`, `semantic_scholar.py`, `pubmed.py`) updated to use it.
2. Crossref returns HTTP 301 for DOIs it has case-normalized or superseded — `httpx.get()`'s default is `follow_redirects=False`, so `resp.raise_for_status()` raised an uncaught `HTTPStatusError` on an entirely valid, resolvable request. Fixed by enabling `follow_redirects=True` in `neural_search/http_utils.py::http_get_with_retry`.

Both fixes are covered by regression tests (`tests/test_literature_api_client.py::TestGetOrRaiseTransient`, `tests/test_http_utils.py::test_follows_redirects`) before being trusted at corpus scale.

**A live, transient Crossref-side reliability incident** was also observed and worked around by patience, not code changes: three consecutive corpus-scale attempts hit intermittent HTTP 500s and timeouts on the bibliographic (title) search endpoint specifically, aborting at 891, then 80, then 1 records — each abort was clean (no crash, partial progress saved, per design). A fourth attempt (after the two fixes above and some time passing) completed the full corpus. Documented as an external service condition observed on 2026-07-03, not a defect in this codebase.

**Retraction/correction status — a genuine new paper-validation signal.** Crossref's `update-to` field (confirmed via a real retraction notice during development, `10.1016/j.micpro.2020.103768`) enables checking whether a linked paper was later retracted or corrected. `scripts/check_paper_retraction_status.py` checked all 2,931 unique DOIs across every real link file: **0 retracted, 15 corrected**. Attached via `neural_search.graph.paper_node_builder.attach_retraction_status()` as `properties["retraction_status"]` on the corresponding `paper` nodes — a plain property, not a new edge type or evidence tier, since a retraction is a fact about the paper itself, independent of graph topology.

**Production graph after Phase 2:** 12,748 nodes / 149,654 edges (up from 8,687/145,472 after Phase 1), zero dangling edges, 4,514 unique paper nodes across 5 sources (2,308 crossref, 1,724 pubmed, 391 openalex, 62 datacite, 28 biorxiv, 1 semantic_scholar). **NDCG@10 confirmed unchanged at exactly 0.8594** — the Phase 1 pre-emptive `_DEGREE_EXCLUDED_EDGE_TYPES` exclusions and `linked_paper` weight zeroing already covered this much larger edge volume; no new regression despite `paper_mentions_dataset` growing from 403 to 4,585 edges.

Remaining: `citation_builder.py` reconnection (Phase 3) and the Bluesky discourse layer (Phase 4) are still scoped in the approved plan, not yet executed.

## Candidates by data form

- mri: 27214
- extracellular_ephys: 11490
- optical_imaging: 7641
- connectomics: 6048
- eeg_meg: 4125
- behavior: 1610
- clinical: 690
- intracranial_human_ephys: 258
- fiber_photometry: 50

## Candidates by analysis family

- connectivity: 18785
- encoding_modeling: 9896
- decoding: 7992
- clinical_prediction: 6274
- spike_train_analysis: 4596
- event_aligned_analysis: 4046
- population_dynamics: 2547
- time_frequency: 1650
- circuit_mapping: 1008
- behavioral_modeling: 966
- reinforcement_learning: 644
- bci_decoding: 550
- speech_decoding: 86
- memory_analysis: 86

## Candidates by technique

- method:bayesian_inference: 5779
- method:linear_mixed_effects: 5482
- method:dti_tractography: 4765
- method:plv: 3800
- method:granger_causality: 3757
- method:transfer_entropy: 3757
- method:dcm: 3757
- method:information_theory: 3623
- method:pca: 3165
- method:ica: 3165
- method:umap: 2847
- method:cramers_rao_bound: 2474
- method:efficient_coding: 2474
- method:predictive_coding: 2474
- method:cluster_permutation: 2023
- method:spike_lfp_coupling: 1192
- method:population_vector_coding: 1149
- method:burst_analysis: 1149
- method:drift_diffusion_model: 644
- method:fft: 275
- method:stft: 275
- method:wavelet_transform: 275
- method:multitaper: 275
- method:hilbert_transform: 275
- method:fooof_specparam: 275

## Data forms with zero candidate-eligible analysis families (open gap)

- intracellular_ephys
- molecular
