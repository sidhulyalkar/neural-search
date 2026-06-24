# Whitepaper Implementation Alignment

Updated: 2026-06-13

This document tracks which whitepaper claims are currently supported by repository artifacts and which claims still require validation before publication.

## Claim Status Legend

- Supported: implemented and backed by a current artifact.
- Partial: implemented or scaffolded, but validation or runtime integration is incomplete.
- Historical: supported only by an older corpus/report and must not be presented as current.
- Not supported: should be removed or framed as future work.

## Current Evidence Snapshot

| Artifact | Current value | Evidence |
|---|---:|---|
| Live normalized corpus artifact | 7,171 rows / 7,121 unique ids | `data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl` |
| Dense field embeddings | 2,840 rows over 625 records | `data/embeddings/real_all.dense.field_embeddings.jsonl` |
| Embedding model | BAAI/bge-large-en-v1.5 | embedding rows and provider code |
| Embedding dimension | 1024 | `data/index/turbovec_dense_1024.index/meta.json` |
| Indexed ids | 625 | `data/index/turbovec_dense_1024.index/meta.json` |
| Turbovec metadata bit width | 4-bit | `data/index/turbovec_dense_1024.index/meta.json` |
| Recall report | recall@50 = 1.0, but report/index sizes need reconciliation | `reports/turbovec_recall.json` |
| Corpus quality | stale/failing local report; regenerate before citation | `reports/corpus_quality.md` |
| Tier-2 rejection summary | 24,160 in CSV report; 762 rows in local JSONL | `reports/eval/rejection_summary.csv`, `data/corpus/rejected/tier2_rejected.jsonl` |
| Field-State memory graph | 2,200 nodes / 3,788 edges | `artifacts/field_state/memory_graph_manifest.json` |
| Latest field-state update | 0 new / 2 changed / 0 removed records | `artifacts/field_state/snapshots/20260613T053153Z/update_report.md` |
| Regional coverage map | 223/625 curated-depth records with verified regions | `data/reports/regional_map/regional_map.md` |

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
- `reports/corpus_quality.md`: stale/failing local report; regenerate before publication citation

Current source counts:

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
| Total | 7,171 |

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
- refreshed normalized source files under `data/corpus/normalized/`

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
- `reports/turbovec_recall.json`: recall@50 = 1.0, p50 = 7.24 ms, p95 = 24.47 ms; reported index size must be regenerated

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
- `hybrid_rrf` is the best populated rung on aggregate NDCG@10 (0.6667), MRR (0.9209), and Recall@50 (0.7455). The MRR gain over BM25 is significant; the NDCG@10 gain over BM25 is directional but not significant by the current sign test.
- Graph-backed rungs are now populated from an eval-corpus graph built over 2,821 records. `hybrid_graph` and `full` currently trail `hybrid_rrf` (NDCG@10 0.6385, MRR 0.8824), so the graph signal should be described as implemented and evaluable, not yet as a retrieval-quality improvement.
- Reinterpretation/reprocessing discovery reports now exist: `reports/eval/reanalysis_affordance_report.md`, `reports/eval/new_method_dataset_matches.md`, and `reports/eval/metadata_enrichment_priorities.md`.

Limitations:

- The labels are LLM-judged, not independently human-adjudicated.
- Dual-judge QWK is not estimable because no non-error pair has labels from two models.
- Graph/full rungs need calibration and source-aware graph weighting before improvement claims.
- Reanalysis affordance and new-method matching reports are metadata-derived prioritization tools, not file-validated compatibility judgments.
- Exact lookup must be validated on the frozen expanded snapshot.

Publication framing:

It is reasonable to describe this as a meaningful LLM-judged ablation preview and regression gate. Do not describe it as publication-grade human relevance evidence until duplicate/human adjudication, exact lookup validation, and graph-weight calibration are complete.

### Claim 6: Latent usefulness scoring is implemented

Status: Partial

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

### Claim 9: Knowledge graph enhances retrieval

Status: Partial

Evidence:

- `neural_search/graph/`
- `reports/graph_ablation.json`: 39% of pairs changed rank with graph signal

Limitations:

- The current ablation report does not show NDCG improvement.
- Graph artifacts should be rebuilt and validated against the frozen expanded corpus.
- Paper links need better confidence/evidence surfacing.

Publication framing:

Claim graph signals affect rankings and support relational context. Do not claim demonstrated metric gains until rerun.

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

### Claim 11: Paper-dataset linking is provenance-aware

Status: Partial

Evidence:

- `neural_search/core/linking.py`
- `neural_search/graph/paper_linking.py`
- graph/report infrastructure

Limitations:

- Links should be regenerated for the frozen expanded snapshot.
- Dataset cards should surface link confidence and evidence.
- Human review is still needed for high-impact claims.

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
- `artifacts/field_state/memory_graph_manifest.json`: 2,200 nodes / 3,788 edges
- `artifacts/field_state/current_manifest.json`: latest snapshot pointer and record hashes
- `reports/field_state/memory_graph_validation.md`

Supported changes:

- Content-hash change detection over title, description, source id, and source.
- Versioned snapshot directories containing corpus, memory graph, and index manifests.
- Obsidian generated/human block separation with review overlays imported into separate JSONL files.
- Memory index and diff reports for human edits, duplicate ids, schema mismatches, missing markers, and missing notes.
- Provenance guardrails separating neuro-judge silver labels, downstream user feedback, silver enrichment labels, and human gold labels.

Limitations:

- Current memory graph covers the field-state/evidence-management slice, not every record in the expanded corpus artifact.
- Memory integrity tests do not prove retrieval relevance or scientific usefulness.

## Publication Required Before Strong Claims

1. Freeze a reconciled expanded-corpus snapshot manifest with checksums.
2. Rerun retrieval benchmarks on that frozen corpus.
3. Implement and test exact identifier pinning.
4. Generate source-by-source metadata completeness and false-positive audits.
5. Validate affordances against file inspection or human labels.
6. Rebuild graph and paper-link artifacts for the frozen expanded snapshot.
7. Audit memory graph coverage so the paper distinguishes the field-state graph from any future full-corpus graph.
8. Generate manuscript metric tables directly from JSON reports.

## Current Bottom Line

Neural Search now supports a corpus-scale and artifact-management claim with an explicit reconciliation caveat: **a 7,171-row live normalized neuroscience corpus artifact, a 625-record BGE-large indexed/evaluated slice, a 1024-dimensional turbovec vector file, a 675-packet neuro-judge evidence pool, and a versioned Field-State memory graph currently at 2,200 nodes and 3,788 edges.**

The publication claim should not yet be "we have solved reusable neuroscience dataset search." The defensible claim is:

> Neural Search is an evidence-aware research prototype for experiment-aware neuroscience dataset retrieval, with expanded corpus normalization, dense field embeddings, graph/reuse scoring infrastructure, LLM-assisted silver-label triage, provenance-preserving memory management, and preliminary retrieval/usefulness validation. The next phase is artifact reconciliation, frozen-snapshot evaluation, exact-lookup hardening, source-specific extraction QA, and content-validated analysis affordances.
