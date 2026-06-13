# Whitepaper Implementation Alignment

Updated: 2026-06-03

This document tracks which whitepaper claims are currently supported by repository artifacts and which claims still require validation before publication.

## Claim Status Legend

- Supported: implemented and backed by a current artifact.
- Partial: implemented or scaffolded, but validation or runtime integration is incomplete.
- Historical: supported only by an older corpus/report and must not be presented as current.
- Not supported: should be removed or framed as future work.

## Current Evidence Snapshot

| Artifact | Current value | Evidence |
|---|---:|---|
| Canonical corpus | 10,404 unique records | `data/corpus/normalized/combined_corpus.jsonl` |
| Dense field embeddings | 60,175 rows | `data/embeddings/real_all.dense.field_embeddings.jsonl` |
| Embedding model | BAAI/bge-large-en-v1.5 | embedding rows and provider code |
| Embedding dimension | 1024 | `data/index/turbovec_dense_1024.index/meta.json` |
| Indexed ids | 10,404 | `data/index/turbovec_dense_1024.index/meta.json` |
| Turbovec metadata bit width | 4-bit | `data/index/turbovec_dense_1024.index/meta.json` |
| Recall report | recall@50 = 1.0 | `reports/turbovec_recall.json` |
| Corpus quality | PASS | `reports/corpus_quality.md` |
| Tier-2 rejections | 24,160 | `data/corpus/rejected/tier2_rejected.jsonl` |

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

- The retrieval benchmark has not yet been rerun on the 10,404-record corpus.
- Graph artifacts and graph ablation should be regenerated against the current snapshot.
- The scoring stack needs calibrated BGE weights for the expanded corpus.

Publication framing:

Use "implemented retrieval architecture" and "current 10K index available"; do not claim final 10K retrieval performance until `reports/real_corpus_10k_eval_report.md` exists.

### Claim 2: Corpus contains 10K+ normalized neuroscience records

Status: Supported

Evidence:

- `data/corpus/normalized/combined_corpus.jsonl`: 10,404 lines
- `data/index/turbovec_dense_1024.index/meta.json`: 10,404 ids
- `reports/corpus_quality.md`: corpus checks PASS

Current source counts:

| Source | Records |
|---|---:|
| Zenodo | 3,000 |
| OpenNeuro | 1,749 |
| NeuroVault | 819 |
| DANDI | 842 |
| NeuroMorpho | 1,000 |
| Figshare | 800 |
| Allen | 500 |
| GIN | 380 |
| Brain Image Library | 300 |
| BlueBrain | 300 |
| IBL | 198 |
| CRCNS | 153 |
| OSF | 321 |
| Others | 42 |
| Total | 10,404 |

Publication framing:

This is the current corpus-scale claim. Older corpus-count claims should remain archived and should not appear in the main whitepaper.

### Claim 3: Corpus expansion includes new and refreshed sources

Status: Supported

Evidence:

- `neural_search/ingestion/zenodo.py`
- `neural_search/ingestion/figshare.py`
- `neural_search/ingestion/neuromorpho.py`
- `neural_search/ingestion/osf.py`
- refreshed normalized source files under `data/corpus/normalized/`

Supported changes:

- Zenodo expanded to 3,000 records and uses page size 100.
- Figshare source added with open-license filtering.
- NeuroMorpho source added with archive-level morphology records.
- OSF expanded to 39 neuroscience tags, embedded license handling, and persistent identifier support.
- DANDI, GIN, and OpenNeuro refreshed after modality synonym improvements.

Limitations:

- Zenodo, Figshare, and OSF remain high-risk heterogeneous sources and need stricter off-topic QA.
- NeuroMorpho records are archive-level, not individual-neuron records.

### Claim 4: Dense BGE-large field embeddings and compressed index exist

Status: Supported

Evidence:

- `neural_search/embeddings/dense_provider.py`
- `neural_search/embeddings/turbovec_index.py`
- `data/embeddings/real_all.dense.field_embeddings.jsonl`: 60,175 rows
- `data/index/turbovec_dense_1024.index/meta.json`: 10,404 ids, 1024 dimension, 4-bit metadata
- `reports/turbovec_recall.json`: recall@50 = 1.0, p50 = 7.24 ms, p95 = 24.47 ms

Limitations:

- The paper should disclose fallback/exact behavior when the turbovec runtime is not available.
- Query-time provider and corpus embedding provider must remain locked to the same vector space.
- Weight calibration is still preliminary.

### Claim 5: Current 10K retrieval performance is publication-grade

Status: Partial

Evidence:

- No current 10K retrieval benchmark report exists yet.
- Current supporting artifacts cover corpus scale, corpus quality, embedding/index validation, usefulness correlation, and graph-rank perturbation.

Limitations:

- Older retrieval reports predate the 10,404-record corpus and should remain archived.
- Exact lookup must be validated on the 10K snapshot.
- Existing labels are too small and not independently multi-annotated.

Publication framing:

Do not make current ranking-performance claims until `reports/real_corpus_10k_eval_report.md` exists.

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

- Identifier fields exist and no records lack identifiers in the current quality report.
- Constraint and query parsing infrastructure exists.

Limitations:

- `reports/real_corpus_v11_eval_report.md` shows missed direct lookup queries for DANDI and OpenNeuro.
- A deterministic pinned exact-match lane must be implemented and tested against the 10K snapshot.

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
- Precision/recall against human or file-inspection labels is not yet reported for the 10K corpus.

Publication framing:

Use "affordance representation and prediction framework" rather than "validated reuse-readiness engine."

### Claim 9: Knowledge graph enhances retrieval

Status: Partial

Evidence:

- `neural_search/graph/`
- `reports/graph_ablation.json`: 39% of pairs changed rank with graph signal

Limitations:

- The current ablation report does not show NDCG improvement.
- Graph artifacts should be rebuilt and validated against the 10K corpus.
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
- Per-intent weighting needs 10K evaluation.
- Planner/runtime defaults should be verified.

### Claim 11: Paper-dataset linking is provenance-aware

Status: Partial

Evidence:

- `neural_search/core/linking.py`
- `neural_search/graph/paper_linking.py`
- graph/report infrastructure

Limitations:

- Links should be regenerated for the 10K snapshot.
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

## Publication Required Before Strong Claims

1. Freeze a 10K snapshot manifest with checksums.
2. Rerun retrieval benchmarks on the 10,404-record corpus.
3. Implement and test exact identifier pinning.
4. Generate source-by-source metadata completeness and false-positive audits.
5. Validate affordances against file inspection or human labels.
6. Rebuild graph and paper-link artifacts for the 10K snapshot.
7. Generate manuscript metric tables directly from JSON reports.

## Current Bottom Line

Neural Search now supports a strong corpus-scale claim: **10,404 unique normalized neuroscience records with 60,175 BGE-large field embeddings and a 10,404-record 1024-dimensional turbovec index.**

The publication claim should not yet be "we have solved reusable neuroscience dataset search." The defensible claim is:

> Neural Search is a 10K-scale, evidence-aware research prototype for experiment-aware neuroscience dataset retrieval, with structured corpus normalization, dense field embeddings, graph/reuse scoring infrastructure, and preliminary retrieval/usefulness validation. The next phase is rigorous 10K-snapshot evaluation, exact-lookup hardening, source-specific extraction QA, and content-validated analysis affordances.
