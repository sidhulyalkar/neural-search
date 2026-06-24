# Whitepaper Implementation Alignment

Updated: 2026-06-17

This document tracks which whitepaper claims are currently supported by repository artifacts and which claims still require validation before publication.

## Claim Status Legend

- Supported: implemented and backed by a current artifact.
- Partial: implemented or scaffolded, but validation or runtime integration is incomplete.
- Historical: supported only by an older corpus/report and must not be presented as current.
- Not supported: should be removed or framed as future work.

## Current Evidence Snapshot

| Artifact | Current value | Evidence |
|---|---:|---|
| Dataset corpus | 7,171 rows / 7,121 unique ids | `data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl` |
| OpenAlex literature corpus | 255,940 tier1 papers (≥100 citations) | `data/corpus/normalized/openalex_neuro/` (26 JSONL shards) |
| Paper-dataset links | 7,171 / 7,171 corpus datasets linked | `artifacts/literature/paper_dataset_links.jsonl` |
| Finding extraction | Running — 550 papers, 262 findings @ 47% yield | `artifacts/literature/findings_tier1_ollama.jsonl` |
| Dense field embeddings | 2,840 rows over 625 records | `data/embeddings/real_all.dense.field_embeddings.jsonl` |
| Embedding model | BAAI/bge-large-en-v1.5 | embedding rows and provider code |
| Embedding dimension | 1024 | `data/index/turbovec_dense_1024.index/meta.json` |
| Indexed ids | 625 | `data/index/turbovec_dense_1024.index/meta.json` |
| Knowledge graph | 7,593 nodes / 31,920 edges | `artifacts/field_state/current_manifest.json` |
| Weak supervision eval | 13 labeling functions, Obsidian vault, metric tier support | `neural_search/eval/`, `scripts/eval/` |

## Core Claims

### Claim 1: Structured multi-signal retrieval combines metadata, ontology, embeddings, and graph signals

Status: Partial

Evidence:

- `neural_search/search/core.py`
- `neural_search/core/retrieval.py`
- `data/config/retrieval.yaml`
- `neural_search/search/field_semantic.py`
- `neural_search/graph/`

Limitations:

- The retrieval benchmark has not yet been rerun on a frozen expanded-corpus snapshot.
- Graph artifacts and graph ablation should be regenerated against the current snapshot.
- The corpus, embedding, index, source-distribution, and recall artifacts need reconciliation before final benchmark reporting.

Publication framing:

Use "implemented retrieval architecture" and "expanded corpus plus current indexed/evaluated slice available"; do not claim final expanded-corpus retrieval performance until a reconciled benchmark report exists.

### Claim 2: Corpus contains an expanded normalized neuroscience artifact

Status: Supported with artifact-reconciliation caveat

Evidence:

- `data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl`: 7,171 rows / 7,121 unique ids
- `data/index/turbovec_dense_1024.index/meta.json`: 625 ids
- `data/index/turbovec_dense_1024.index/fallback_vecs.npy`: 625 x 1024 vectors

Current dataset source counts:

| Source | Records |
|---|---:|
| NeuroVault | 2,000 |
| NeuroMorpho | 1,000 |
| DANDI | 848 |
| Zenodo | 500 |
| Harvard Dataverse | 500 |
| Allen | 500 |
| GIN | 408 |
| OpenNeuro | 299 |
| Figshare | 200 |
| OSF | 200 |
| Brain Image Library | 200 |
| IBL | 198 |
| CRCNS | 153 |
| BlueBrain | 100 |
| Buzsaki Lab | 35 |
| SPARK | 20 |
| NEMO | 10 |
| **Total** | **7,171** |

Publication framing:

This is the current live corpus-scale claim. Older 10,404-record claims should remain archived or marked as stale until the manifest/table artifacts are regenerated from a frozen bundle.

### Claim 3: Corpus expansion includes new and refreshed sources

Status: Supported

Evidence:

- `neural_search/ingestion/zenodo.py`
- `neural_search/ingestion/figshare.py`
- `neural_search/ingestion/neuromorpho.py`
- `neural_search/ingestion/osf.py`
- `scripts/corpus/fetch_ibl.py`
- `scripts/corpus/fetch_crcns.py`
- `scripts/corpus/fetch_neurovault.py`
- `scripts/corpus/enrich_dandi_metadata.py`
- `scripts/corpus/enrich_study_targets.py`
- `scripts/corpus/extract_nwb_electrode_regions.py`
- `scripts/corpus/extract_nwb_surgery_regions.py`
- `scripts/corpus/fetch_paper_abstracts.py`
- `data/reports/regional_map/regional_map.md`
- `data/corpus/enrichment/regional_signals/regional_signal_report.md`

Supported changes:

- Zenodo expanded to 3,000 records and uses page size 100.
- Figshare source added with open-license filtering.
- NeuroMorpho source added with archive-level morphology records.
- OSF expanded to 39 neuroscience tags, embedded license handling, and persistent identifier support.
- DANDI, GIN, and OpenNeuro refreshed after modality synonym improvements.
- IBL, CRCNS, and NeuroVault append scripts add source-specific expansion without a full corpus rebuild.
- DANDI enrichment scripts refresh rich metadata, study targets, NWB electrode locations, and NWB surgery/experiment-description text.
- Paper abstract mining adds DOI/CrossRef-derived regional evidence as silver provenance, not human gold.
- Regional coverage reports distinguish verified regions from candidate-only mentions and generate a regionless review queue.

Limitations:

- Zenodo, Figshare, and OSF remain high-risk heterogeneous sources and need stricter off-topic QA.
- NeuroMorpho records are archive-level, not individual-neuron records.
- Regional extraction is still evidence-tiered and partly silver; it needs manual precision audit before strong anatomical recall claims.

### Claim 4: Dense BGE-large field embeddings and compressed index exist

Status: Supported with artifact-reconciliation caveat

Evidence:

- `neural_search/embeddings/dense_provider.py`
- `neural_search/embeddings/turbovec_index.py`
- `data/embeddings/real_all.dense.field_embeddings.jsonl`: 2,840 rows over 625 records
- `data/index/turbovec_dense_1024.index/meta.json`: 625 ids, 1024 dimension, 4-bit metadata
- `data/index/turbovec_dense_1024.index/fallback_vecs.npy`: 625 x 1024 vectors
- `reports/turbovec_recall.json`: recall@50 = 1.0, p50 = 7.24 ms, p95 = 24.47 ms

Limitations:

- The paper should disclose fallback/exact behavior when the turbovec runtime is not available.
- Query-time provider and corpus embedding provider must remain locked to the same vector space.
- Embedding, index, and recall artifacts need to be regenerated together before expanded-corpus retrieval claims.
- Weight calibration is still preliminary.

### Claim 5: Current expanded-corpus retrieval performance is publication-grade

Status: Preliminary / not publication-grade

Evidence:

- A qrels-backed retrieval snapshot now exists: 317 canonical queries and 13,654 non-error LLM-judged query--dataset pairs.
- Current reports: `reports/eval/ndcg_report.md`, `reports/eval/bootstrap_ci_report.json`, `reports/eval/intent_stratification_report.md`, `reports/eval/eval_claim_ledger.md`, and `reports/eval/regression_gate_report.md`.
- `hybrid_rrf` is strong on aggregate NDCG@10 (0.6667), MRR (0.9209), and Recall@50 (0.7455). The MRR gain over BM25 is significant; the NDCG@10 gain over BM25 is directional but not significant by the current sign test.
- Graph-backed rungs are now populated from an eval-corpus graph built over 2,821 records. After graph-weight calibration, `hybrid_graph` is currently the best qrels-backed preview rung (NDCG@10 0.6696, MRR 0.9256, Recall@50 0.7465), with small but significant sign-test gains over `hybrid_rrf` on NDCG@10 and MRR.
- Reinterpretation/reprocessing discovery reports now exist: `reports/eval/reanalysis_affordance_report.md`, `reports/eval/new_method_dataset_matches.md`, and `reports/eval/metadata_enrichment_priorities.md`.

Limitations:

- The labels are LLM-judged, not independently human-adjudicated.
- Dual-judge QWK is not estimable because no non-error pair has labels from two models.
- Graph/full gains are small and calibrated on LLM-judged qrels; they need duplicate/human adjudication and source-aware edge QA before publication-grade claims.
- Reanalysis affordance and new-method matching reports are metadata-derived prioritization tools, not file-validated compatibility judgments.
- Exact lookup must be validated on the frozen expanded snapshot.

Publication framing:

It is reasonable to describe this as a meaningful LLM-judged ablation preview and regression gate. Do not describe it as publication-grade human relevance evidence until duplicate/human adjudication, exact lookup validation, and graph-weight calibration are confirmed on independently reviewed labels.

### Claim 6: Latent usefulness scoring is implemented

Status: Prototype-validated on LLM qrels / not publication-grade

Evidence:

- `neural_search/retrieval/usefulness_scorer.py`
- `reports/usefulness_correlation_v09.json`: Spearman r = 0.3999 over 270 pairs
- `reports/optimized_weights_v11.json`
- `docs/LATENT_USEFULNESS_OPTIMIZATION.md`

Limitations:

- Usefulness labels are limited.
- The current r = 0.3999 supports discriminative signal, not final downstream utility.
- Human validation and content-derived labels are needed.

Publication framing:

Claim that Neural Search operationalizes latent usefulness through a multi-dimensional scorer and preliminary correlation results. Do not claim validated downstream usefulness yet.

### Claim 7: Exact identifier lookup is robust

Status: Partial

Evidence:

- Identifier fields exist in the live corpus artifact, with 7,121 unique ids across 7,171 rows.
- Constraint and query parsing infrastructure exists.

Limitations:

- `reports/real_corpus_v11_eval_report.md` shows missed direct lookup queries for DANDI and OpenNeuro.
- A deterministic pinned exact-match lane must be implemented and tested against the frozen expanded snapshot.

Publication framing:

List exact lookup robustness as required validation, not as a completed feature.

### Claim 8: Analysis affordance detection supports scientific reuse search

Status: Partial

Evidence:

- `neural_search/analysis_affordances.py`
- `neural_search/affordances/registry.py`
- `neural_search/affordances/validators/`
- affordance tests under `tests/`

Limitations:

- Metadata-based predictions are not yet enough for publication-grade reuse claims.
- File/content validation against NWB and BIDS datasets is still required.
- Precision/recall against human or file-inspection labels is not yet reported for the expanded corpus.

Publication framing:

Use "affordance representation and prediction framework" rather than "validated reuse-readiness engine."

### Claim 9: Knowledge graph enhances retrieval and connects datasets to literature

Status: Partial — materially improved

Evidence:

- `neural_search/literature/kg_builder.py` — adds paper/finding/venue nodes and cross-edges
- Knowledge graph at 7,593 nodes / 31,920 edges (rebuilt 2026-06-14 from full corpus)
- New node types: `paper`, `finding`, `venue`
- New edge types: `paper_reports_finding`, `finding_involves_region/task/modality/species`, `dataset_linked_to_paper`
- `artifacts/literature/paper_dataset_links.jsonl`: 7,171 corpus datasets linked to OpenAlex papers
- `reports/eval/ndcg_report.md`: calibrated `hybrid_graph` NDCG@10 0.6696 vs `hybrid_rrf` 0.6667 vs `bm25` 0.6566; MRR 0.9256 vs 0.9209 vs 0.8795 (first non-degenerate ablation ladder this project has produced, 13,654 LLM-silver qrels / 317 queries)
- `reports/eval/bootstrap_ci_report.json`: `hybrid_graph` NDCG@10 95% CI [0.6453, 0.6921] overlaps `bm25`'s CI [0.6322, 0.6807] substantially — the lead over the plain BM25 baseline is directionally consistent but not statistically established at n=317 queries. The narrower, paired comparison against `hybrid_rrf` (same corpus, same qrels) shows a small but more consistent sign-test gain.
- `reports/eval/graph_weight_calibration.md`: balanced calibrated graph setting is default profile at global weight 0.005
- `reports/eval/relationship_edge_quality.md`: relationship-edge promotions are mixed — `same_region_same_task`/`dataset_reprocessing_candidate` and `same_region_cross_modality`/`dataset_reanalysis_bridge_dataset` sit at 0.498 and 0.522 helpful rate respectively among judged top-10 promotions, i.e. close to a coin flip. This report predates the 2026-06-23 typed-field Phase 6b additions (qualified consensus tiers, `contradiction_subtype`); re-running it against those edge types is open work, not yet measured.

Limitations:

- The current evidence is LLM-judged, not human-adjudicated.
- The graph gain vs. the plain BM25 baseline is within bootstrap noise; only the narrower vs.-`hybrid_rrf` comparison shows a more defensible (still small) gain, and that comparison was calibrated on the same qrels snapshot it's evaluated against.
- Relationship-edge quality is mixed; some edge classes help roughly as often as they hurt, and the newest typed/qualified-consensus edge types (Phase 6b) haven't been run through this quality analysis at all yet.
- KG rebuild incorporating the still-running tier1 extraction (190K+ findings as of 2026-06-23, growing) will substantially change node/edge counts; current counts predate that run.
- Paper links need better confidence/evidence surfacing.

Publication framing:

Claim calibrated graph reranking shows a small LLM-qrels-backed improvement relative to the hybrid RRF baseline and supports relational context; do not claim a demonstrated improvement over plain BM25, since that comparison's confidence intervals overlap. Claim paper-dataset and finding-dataset bidirectional linking as an architectural capability. Do not describe any of this as publication-grade or human-validated until adjudicated labels confirm it.

### Claim 10: Query intent classification and routing exist

Status: Partial

Evidence:

- `neural_search/core/query.py`
- `neural_search/intelligence/planner.py`
- `data/config/intent_profiles.yaml`
- `reports/optimized_weights_v11.json`

Limitations:

- Intent classification is still largely heuristic.
- Per-intent weighting needs expanded-corpus evaluation.
- Planner/runtime defaults should be verified.

### Claim 11: Paper-dataset linking is provenance-aware and fully implemented

Status: Supported — upgraded from Partial

Evidence:

- `neural_search/literature/linking.py`: DOI exact match (confidence=1.0) + title fuzzy match (confidence 0.75–0.90)
- `artifacts/literature/paper_dataset_links.jsonl`: all 7,171 corpus datasets linked to OpenAlex (DOI or title fuzzy)
- `neural_search/ingestion/openalex_bulk.py`: 255,940 tier1 papers ingested via cursor-based pagination with checkpoint/resume
- Tests: `tests/test_paper_dataset_linking.py` (19 tests), `tests/test_openalex_bulk.py` (37 tests)

Limitations:

- Title fuzzy match at threshold ≥0.75 may produce false positives for short or generic titles; a human spot-check is recommended before publication claims.
- DOI match is exact and high-confidence; fuzzy match confidence is silver evidence.

### Claim 12: The system is deployable as a real-corpus product

Status: Partial

Evidence:

- FastAPI app exists under `apps/api/`
- React frontend exists under `apps/web/`
- Docker and infra files exist

Limitations:

- Runtime must be verified to load the real corpus snapshot by default.
- Demo mode and real mode must be impossible to confuse.
- Health endpoints should expose snapshot id, record count, embedding model, and index status.

### Claim 13: Field-State memory management preserves provenance and human review state

Status: Supported for engineering/artifact integrity; partial for scientific validation

Evidence:

- `neural_search/field_state/memory_graph.py`
- `neural_search/field_state/graph_store.py`
- `neural_search/field_state/memory/index.py`
- `neural_search/field_state/memory/diff.py`
- `neural_search/field_state/memory/review_overlay.py`
- `scripts/field_state/update_field_state.py`
- `scripts/field_state/compare_snapshots.py`
- `artifacts/field_state/memory_graph_manifest.json`: 7,593 nodes / 31,920 edges (rebuilt 2026-06-14)
- `artifacts/field_state/current_manifest.json`: latest snapshot pointer and record hashes
- `reports/field_state/memory_graph_validation.md`
- `docs/OBSIDIAN_EVAL_MEMORY.md`, `docs/WEAK_SUPERVISION_LABELING.md`, `docs/HUMAN_AUDIT_PROTOCOL.md`

Supported changes:

- Content-hash change detection over title, description, source id, and source.
- Versioned snapshot directories containing corpus, memory graph, and index manifests.
- Obsidian generated/human block separation with review overlays imported into separate JSONL files.
- Memory index and diff reports for human edits, duplicate ids, schema mismatches, missing markers, and missing notes.
- Provenance guardrails separating neuro-judge silver labels, downstream user feedback, silver enrichment labels, and human gold labels.
- Weak supervision pipeline: 13 labeling functions, evidence ensembling, LLM judge fallback.
- Metric tier support (`--qrels-tier gold/silver/bronze`) on all IR metric scripts.
- Human audit protocol with HUMAN_OWNED_FIELDS write protection.

Limitations:

- Current memory graph covers the field-state/evidence-management slice, not every record in the expanded corpus artifact.
- Memory integrity tests do not prove retrieval relevance or scientific usefulness.

### Claim 14: Literature-scale ingestion enables evidence-backed neuroscience discovery [NEW]

Status: Partial — tier1 ingestion AND extraction now complete; KG/relationship layer built on it, quality audited

Evidence:

- `neural_search/ingestion/openalex_bulk.py`: BulkIngester with cursor-based pagination, checkpoint/resume, three tiers
  - Tier 1: 255,940 neuroscience papers (≥100 citations) — **fully ingested**
  - Tier 2: ~1.39M open-access papers — staged, not started (`bulk_ingest_openalex.py --tier tier2` is ready to run)
  - Tier 3: ~4.36M total papers — future
- `neural_search/literature/finding_extractor.py`: LLM-powered structured finding extraction
  - `FindingRecord` schema: finding_text, result_direction, regions, species, modalities, tasks, cell_types, molecules, confidence
  - Ollama local inference backend (no cloud cost): `extract_batch_ollama()` with `_repair_json()` for markdown fence handling
  - Multi-provider fallback: Ollama → Anthropic → OpenRouter
  - **Tier1 extraction is complete as of 2026-06-23 19:09 PDT**: 255,940/255,940 papers checkpointed, 190,279 findings (74.4% yield) in `artifacts/literature/findings_tier1_ollama.jsonl` (144MB+, gitignored — do not commit)
- `neural_search/literature/typed_finding_extractor.py`: 27 rule-based typed fields (frequency band, temporal pattern, negation, spatial frame, injury model, molecular marker, etc.), regex-only, no model calls. **Audited 2026-06-24** (`reports/eval/typed_field_audit_summary_2026.md`): found and fixed `negation` false positives (bare `inhibit`/`suppress`/`block` patterns matching affirmative findings) and `frequency_band`/`temporal_pattern` collisions with molecule names (`amyloid-beta`, `alpha-lipoic acid`, `cyclic AMP`). Precision on the audited 32-row sample went from 46.9%/53.1% to 90.6%/93.75% after the fix (16 new regression tests).
- `neural_search/literature/relationship_builder.py`: cross-finding supports/contradicts edges, region co-occurrence, base + qualified (per-facet) consensus summaries. **Rebuilt 2026-06-24 at full scale** after discovering `findings_tier1_normalized.jsonl` had never actually carried typed fields (it predated the extractor's integration into the normalizer, so `direct_refutation` had structurally always been 0 regardless of the negation bug — a missing pipeline connection, not a corrupted artifact). Re-ran `normalize_findings.py` against the complete 190,279-finding extraction, then `build_finding_relationships.py` against the result: 200,000 finding edges (capped; 82,525 supports / 117,475 contradicts, of which **9,833 are `direct_refutation`** — nonzero for the first time), 11,869 region co-occurrence edges, 8,520 base + 9,823 qualified consensus records (274 "strong", >=0.8 strength and >=3 papers, citing real named regions like left inferior frontal gyrus at 18 papers). The 2026-06-23 run of this same pipeline (194,903 edges, 0 `direct_refutation`) is superseded — see `reports/eval/typed_field_audit_summary_2026.md` for the before/after.
- `scripts/literature/ingest_relationships_to_kg.py`: materializes the above into `data/graph/relationships_kg.jsonl` (9,104 finding nodes, 4,780 region nodes) — rebuilt 2026-06-24. Also fixed a crash at this larger scale: one region pair with non-Latin-script names (`枕部`/`顶叶`) normalized to an empty graph-node identifier; `relationship_kg_builder.py::add_region_cooccurrence_to_graph` now skips non-normalizable pairs instead of crashing the whole batch (regression test added).
- `neural_search/literature/kg_builder.py`: KG integration — adds paper/finding/venue nodes and cross-edges (predates the full extraction; not yet rebuilt at the new finding count)
- **Quality audit** (`reports/eval/finding_audit_llm_judge_summary_2026.md`, 100-row LLM-judge sample post-completion): 82% strict / 90% weighted precision — holds above the 80% whitepaper-citation threshold, down modestly from the 88%/92% pre-completion snapshot (consistent with judging a fresh sample, not a quality regression — the FALSE rate actually halved). Dominant failure mode is now region-field errors (inferred-not-stated regions, and a new malformed-value subtype: genomic loci/peptide names/comparative phrases landing in the region field). A mechanical normalizer fix for the malformed-value subtype shipped same day (`neural_search/literature/normalizer.py::_is_malformed_region`). A v3 extraction prompt (`configs/literature/finding_extraction_v3.yaml`) addresses all four named failure modes (domain contamination, methods-statements-as-findings, malformed regions, direction-field misuse) but has not yet been used for a production extraction run.
- `artifacts/claims/` 5-stage claim synthesis pipeline (cluster → synthesize via Claude Haiku → detect contradictions → ingest to KG) exists but is **not yet run on the full extraction**: `findings_normalized.jsonl` (123,331 rows) and `finding_clusters.jsonl` (4,345 clusters) are from an earlier, smaller snapshot; `claims_raw.jsonl` (the Claude Haiku synthesis output) does not exist yet. This step costs real API calls and was deliberately not auto-triggered.

Limitations:

- Extraction quality is LLM-judge audited (Claude Sonnet 4.6), not human-audited — see the dual caveat in both 2026-06-22 and 2026-06-23 audit summaries.
- Corpus/domain contamination is now a *confirmed recurring* issue (non-neuroscience papers entering the findings corpus), not a one-off — an upstream topic filter remains unbuilt.
- Literature search not yet integrated into main search path — operates as a separate index.
- KG rebuild with full findings (`kg_builder.py`'s paper/finding/venue layer) is still pending — only the relationship layer (`relationships_kg.jsonl`) has been rebuilt at full scale so far.
- Claim synthesis (Claude Haiku) has not been run at all on real data yet — claims_router.py is wired into the API but has no real claims to serve.
- The typed/relationship KG layer's effect on retrieval, when isolated from the aggregate hybrid_graph signal, was measured in the 2026-06-23 `typed_kg`/`typed_kg_qualified` ablation rungs and found null/negligible at the full-corpus scale tested (see Claim 9 above) — that ablation predates the 2026-06-24 negation/frequency_band fixes and the full-scale relationship rebuild, so it should be re-run against the corrected data before treating the null result as final.
- `findings_tier1_normalized.jsonl` is now 271MB (over GitHub's 100MB limit) and gitignored; only the small pre-fix 12,609-record/7.7MB snapshot remains in git history (`e9b0c21`). Regenerate locally via `scripts/literature/normalize_findings.py` rather than expecting it to be present after a fresh clone.
- The new 9,833 `direct_refutation` edges and 274 strong consensus records are unaudited at this specific layer — the finding-extraction audits validate the underlying `result_direction`/`negation` fields in isolation (82%/90% and 90.6%/93.75% respectively), not the contradiction-detection logic built on top of them end-to-end.

Publication framing:

Claim literature-scale ingestion (255,940 OpenAlex neuroscience papers, **complete**) and LLM-structured finding extraction (190,279 findings, **complete**) as implemented and now-finished capabilities, with a fresh 100-row quality audit at full scale. Claim the relationship layer, rebuilt 2026-06-24 at full scale with corrected typed-field extraction (200,000 finding edges, 9,833 `direct_refutation`, 8,520+9,823 consensus records), as built and KG-integrated. Do **not** yet claim the claims/synthesis layer is populated with real data, that finding extraction precision has been human-validated, or that the contradiction-detection layer specifically has been audited end-to-end.

## Publication Required Before Strong Claims

1. ~~Complete tier1 finding extraction~~ — **done 2026-06-23** (190,279 findings, 255,940/255,940 papers).
2. Freeze a reconciled expanded-corpus snapshot manifest with checksums — `reports/eval/current_artifact_manifest.json` exists but is still hand-maintained (no `build_artifact_manifest.py` script despite being referenced by `docs/REPRODUCIBILITY_CAPSULE.md`).
3. Rerun retrieval benchmarks on that frozen corpus — in progress; the canonical 317-query/13,654-pair ladder now includes `typed_kg`/`typed_kg_qualified` rungs (2026-06-23) to isolate the relationship layer's contribution.
4. Implement and test exact identifier pinning.
5. Generate source-by-source metadata completeness and false-positive audits.
6. Validate affordances against file inspection or human labels — `reports/eval/affordance_audit_template.csv` exists, still blank.
7. Audit memory graph coverage so the paper distinguishes the field-state graph from the literature graph.
8. Generate manuscript metric tables directly from JSON reports.
9. ~~Conduct a precision spot-check on 100 random extracted findings against source abstracts~~ — **done twice** (2026-06-22 pre-completion, 2026-06-23 post-completion; both LLM-judge, not human).
10. Run the claim-synthesis pipeline (cluster → Claude Haiku synthesis → contradiction detection → KG ingest) on the full 190K-finding extraction — not started; requires real API spend, needs explicit go-ahead.
11. Rebuild `kg_builder.py`'s paper/finding/venue KG layer at the full 190K-finding scale (only the relationship layer has been rebuilt so far).
12. Sample and fill at least one of the four still-blank human-audit CSVs (findings, paper-dataset links, affordances, typed-fields) with a genuine human pass, not an LLM-judge pass — every audit run so far, including both 2026 finding-extraction audits, has been LLM-judged.
13. Human-adjudicate a stratified subset of the canonical 13,654-pair qrels (still 0 gold rows) — the single most-repeated blocker across every strategy doc in this repo since 2026-06-11.
10. Run literature search quality evaluation against a curated query set.

## Current Bottom Line

Neural Search now supports a corpus-scale, literature-linked, and evidence-management claim:

> **7,171 normalized neuroscience datasets** linked via DOI/title to **255,940 OpenAlex tier1 papers** (≥100 citations), with **LLM-structured finding extraction** running on local inference (Ollama, qwen2.5:7b), a **7,593-node / 31,920-edge knowledge graph** connecting datasets, papers, findings, brain regions, species, modalities, tasks, molecules, and cell types, and a **provenance-preserving evaluation infrastructure** with weak supervision, human audit protocols, and metric tier gating.

The defensible publication claim is:

> Neural Search is an evidence-aware research infrastructure for neuroscience dataset and literature discovery, with multi-source dataset normalization, OpenAlex-scale literature ingestion, LLM-powered structured finding extraction, bidirectional paper-dataset linking, knowledge graph traversal across experimental dimensions, dense field embeddings, graph/reuse scoring, LLM-assisted weak supervision, and provenance-preserving memory management. The next phase is completing tier1 finding extraction, rebuilding the knowledge graph, running frozen-snapshot retrieval benchmarks, and launching spatial ontology integration for full brain-region traversal.
