# Neural Search Whitepaper + Repository Audit

Date: 2026-06-03

## Executive Diagnosis

Neural Search has moved from a small proof-of-concept into a 10K-record neuroscience dataset discovery system. The central research idea is still strong and differentiated: retrieve datasets by **latent future usefulness** rather than surface textual similarity. The system now has the scale, ingestion breadth, dense embedding infrastructure, and quality-gate scaffolding needed for a credible research prototype.

The next development cycle should stop treating corpus growth as the primary milestone. The 10K target is achieved. The publication-grade question is now:

> Can Neural Search prove that its top results are scientifically reusable, explain why, expose uncertainty, and reproduce every claim from versioned evidence artifacts?

The answer is: partially, but not yet at publication standard. The corpus and embedding stack are now strong enough to justify a serious validation sprint. The remaining work is evaluation rigor, extraction QA, exact lookup guarantees, source-aware provenance, content-level affordance validation, and a clean claim ledger that separates demonstrated results from intended capabilities.

## Current System Snapshot

### Corpus snapshot

The canonical expanded corpus now contains **10,404 unique records**, meeting the 10K target. This total is verified by `data/corpus/normalized/combined_corpus.jsonl` and by the turbovec index metadata.

| Source | Records | Status |
|---|---:|---|
| Zenodo | 3,000 | Expanded from 2K to 3K; future fetches use page size 100 |
| OpenNeuro | 1,749 | Refreshed with improved modality extraction |
| NeuroVault | 819 | Active source |
| DANDI | 842 | Refreshed with improved modality extraction |
| NeuroMorpho | 1,000 | New source; archive-level records with morphology labels |
| Figshare | 800 | New source; open-license dataset search |
| Allen | 500 | Active source |
| GIN | 380 | Refreshed with improved modality extraction |
| Brain Image Library | 300 | Active source |
| BlueBrain | 300 | Active source |
| IBL | 198 | Active source |
| CRCNS | 153 | Active source |
| OSF | 321 | Expanded with 39 neuroscience tags, embedded license handling, and DOI/persistent-id fix |
| Others | 42 | PhysioNet 32, NeMO 10 |
| **Total** | **10,404** | **10K target achieved** |

Important accounting note: `reports/corpus_quality.md` reports **10,542 usable lines** because it includes backups, paper/demo files, and auxiliary normalized files. The publication-facing corpus number should be **10,404 unique records** from `combined_corpus.jsonl`.

### Repository scale

The codebase remains a substantial research system rather than a single script:

| Area | Current state |
|---|---|
| Python modules | Ingestion adapters, schemas, graph construction, embeddings, retrieval, evaluation, QA, reports |
| Frontend | FastAPI backend plus Vite/React web app |
| Test surface | Hundreds of tests across ingestion, retrieval, graph, embeddings, evaluation, affordances, and API |
| Reports | Corpus quality, benchmark, turbovec recall, usefulness correlation, graph ablation, source distribution |
| Data artifacts | Normalized corpus, dense field embeddings, compressed index metadata, rejection logs |

### Embedding and index snapshot

The dense retrieval stack has been rebuilt on the expanded corpus:

| Artifact | Current value |
|---|---:|
| Dense field embeddings | 60,175 rows in `data/embeddings/real_all.dense.field_embeddings.jsonl` |
| Model | `BAAI/bge-large-en-v1.5` via `DenseEmbeddingProvider` |
| Embedding dimension | 1024 |
| Index records | 10,404 ids in `data/index/turbovec_dense_1024.index/meta.json` |
| Quantization metadata | 4-bit |
| Recall report | recall@50 = 1.0 on 50 queries |
| Latency report | p50 = 7.24 ms, p95 = 24.47 ms in `reports/turbovec_recall.json` |

The whitepaper should describe this as a reproducible dense retrieval artifact, while still being precise about fallback/exact behavior if turbovec is unavailable in a local environment.

### Corpus quality snapshot

`reports/corpus_quality.md` passes all current checks:

| Check | Status |
|---|---|
| Total usable above 4,000 | PASS |
| Tier-2 rejection log exists | PASS |
| Tier-2 rejection log is non-empty | PASS |
| No records without identifier | PASS |
| Tier-2 rejections logged | 24,160 |

Field completeness on the canonical 10,404-record corpus:

| Field | Non-empty | Coverage |
|---|---:|---:|
| title | 10,400 | 100.0% |
| url | 10,404 | 100.0% |
| description | 7,779 | 74.8% |
| license | 7,813 | 75.1% |
| modalities | 6,559 | 63.0% |
| behaviors | 6,090 | 58.5% |
| data_standards | 5,727 | 55.0% |
| species | 5,715 | 54.9% |
| doi | 5,108 | 49.1% |
| brain_regions | 3,992 | 38.4% |
| tasks | 3,510 | 33.7% |

This is a strong current corpus snapshot for scale and indexing. The publication risk is now concentrated in task, brain-region, DOI, and source-specific provenance completeness.

## What Changed This Session

### Corpus expansion

- Zenodo scaled from 2,000 to 3,000 records.
- Zenodo fetch pagination now uses `size=100`, improving future run efficiency.
- Figshare was added as a new Tier-2 source with open-license filtering.
- NeuroMorpho was added as a new source with one archive-level record per publication/archive, including morphology modality and SWC data-standard labels.
- OSF expanded to 39 neuroscience search tags.
- OSF now requests embedded license metadata.
- OSF now provides a persistent DOI-equivalent field using the OSF URL, preventing otherwise-valid OSF records from failing the persistent-identifier gate.
- DANDI, GIN, and OpenNeuro were refreshed after improved modality synonym extraction.
- BGE-large embeddings were rebuilt across the expanded corpus.
- A 10,404-record turbovec index was rebuilt.
- Tier-2 rejection logging now records 24,160 rejected records.

### Source-specific observations

Zenodo and Figshare are useful for scale, but they are also the highest-risk sources for off-topic or weakly structured records. NeuroMorpho is high-value because its records are structured around neuron morphology, but archive-level aggregation means it should be described carefully: the unit of retrieval is an archive/publication set, not an individual neuron reconstruction.

OSF now has better inclusion mechanics, but OSF remains heterogeneous. The quality story for OSF should emphasize filtering and rejection logs rather than raw count alone.

DANDI and OpenNeuro remain the most important sources for scientific validation because they expose standards-backed metadata and can support file/content inspection through NWB and BIDS.

## Evidence Ledger for Current Claims

| Claim | Current support | Evidence artifact | Publication status |
|---|---|---|---|
| 10K-record corpus achieved | 10,404 unique records | `data/corpus/normalized/combined_corpus.jsonl` | Supported |
| Multi-source neuroscience corpus | 14 source buckets including Others | normalized source files | Supported |
| Quality gates pass | All corpus checks pass | `reports/corpus_quality.md` | Supported |
| Tier-2 filtering is active | 24,160 rejections logged | `data/corpus/rejected/tier2_rejected.jsonl` | Supported |
| Dense BGE field embeddings rebuilt | 60,175 field embeddings | `data/embeddings/real_all.dense.field_embeddings.jsonl` | Supported |
| 10,404-record turbovec index exists | 10,404 ids, 1024d, 4-bit metadata | `data/index/turbovec_dense_1024.index/meta.json` | Supported |
| ANN/exact recall validated | recall@50 = 1.0, p50 7.24 ms | `reports/turbovec_recall.json` | Supported with fallback caveat |
| 10K retrieval benchmark exists | Not yet run | planned `reports/real_corpus_10k_eval_report.md` | Required before ranking claims |
| Usefulness score correlates with labels | Spearman r = 0.3999 over 270 pairs | `reports/usefulness_correlation_v09.json` | Preliminary |
| Graph signal affects ranking | 39% pair rank changes | `reports/graph_ablation.json` | Supported for rank perturbation; outcome improvement not yet shown |
| Analysis affordance search is validated | Framework exists | affordance code/tests | Not publication-grade until content/file labels are collected |

## Evaluation Status

### 10K retrieval benchmark

The 10,404-record corpus and index are current, but the repository does not yet contain a retrieval benchmark run against this exact snapshot. Older benchmark reports remain useful engineering history, but their numbers have been removed from the main whitepaper and should not be used as current paper results.

The next retrieval report should include Precision@1/5/10, MRR, NDCG@10, label recall, exact-lookup success, hard-negative violations, and source-skew analysis on the 10,404-record corpus.

### Usefulness correlation

`reports/usefulness_correlation_v09.json` reports:

| Metric | Value |
|---|---:|
| Query count | 30 |
| Pair count | 270 |
| Spearman r | 0.3999 |
| Mean score, relevant | 0.2244 |
| Mean score, irrelevant | 0.1678 |

This supports the claim that the usefulness scorer has discriminative signal. It does not yet support strong claims about downstream scientific usefulness because the labels are limited and not multi-annotator.

## Publication-Grade Whitepaper Revisions Needed

### 1. Rewrite the abstract around the 10K snapshot

The abstract should lead with the 10K snapshot. It should say:

- Neural Search indexes 10,404 unique normalized neuroscience dataset records.
- The corpus spans DANDI, OpenNeuro, NeuroVault, Zenodo, Figshare, NeuroMorpho, Allen, GIN, IBL, CRCNS, Brain Image Library, BlueBrain, OSF, and smaller sources.
- Records are represented through structured fields, source provenance, ontology-derived labels, dense field embeddings, and a compressed 1024d index.
- Current evaluation shows promising retrieval and usefulness signals, but publication-grade validation requires rerunning the benchmark on the 10K snapshot with independent labels.

### 2. Keep only current results in the main paper

The manuscript should report the current corpus/index/quality/usefulness artifacts and mark 10K retrieval benchmarking as not yet run. Older metrics should stay in archived reports, not in the main whitepaper.

### 3. Add a corpus construction section

The paper needs a rigorous corpus section with:

- Source registry and source-specific fetch strategy.
- Tier-1 versus Tier-2 inclusion logic.
- Four-gate inclusion classifier: neuroscience signal, dataset-like artifact, open/reusable access, persistent identifier.
- Deduplication policy.
- Rejection logging and auditability.
- Source limitations and off-topic risk.

### 4. Add a field-completeness table

Publication reviewers will ask whether the system has enough structured metadata to justify structured search claims. Include the completeness table above, and explicitly say task and brain-region coverage remain active targets.

### 5. Add an embedding/index manifest table

The BGE/turbovec section should include:

- Model name.
- Dimension.
- Normalization.
- Field count.
- Corpus snapshot size.
- Index bit width.
- Recall/latency report.
- Fallback behavior.

### 6. Tighten claims about analysis affordances

The system has an affordance framework, but real scientific validation remains incomplete. The publication-grade claim should be:

> Neural Search represents and predicts analysis affordances from metadata, and the next validation phase will estimate precision/recall against file-inspection and human labels.

Avoid saying the system can reliably determine reuse readiness until NWB/BIDS content validation is complete.

### 7. Promote limitations from afterthought to core scientific honesty

The paper should explicitly state:

- Metadata quality varies sharply by source.
- Zenodo/Figshare/OSF require stronger off-topic filtering than standards-backed archives.
- Task and brain-region coverage remain below publication comfort.
- Existing labels are too small for final claims.
- Exact identifier lookup must be deterministic.
- Dense embeddings improve semantic coverage but need calibration and per-intent weight tuning.

## Most Important Remaining Gaps

### 1. 10K retrieval benchmark has not been rerun

The corpus and index are current, but the strongest retrieval metrics are historical. The next benchmark must use the 10,404-record snapshot.

Acceptance criteria:

- 100+ query benchmark runs against `combined_corpus.jsonl`.
- Metrics include Precision@1/5/10, MRR, NDCG@10, label recall, hard-negative violations, exact-lookup success, and source distribution.
- Per-query failures are written to an actionable report.

### 2. Exact identifier lookup must be fail-safe

The prior v11 report missed direct identifier queries:

- `dataset:dandi:000026`
- `dataset:dandi:000020`
- `dataset:openneuro:ds003505`

No serious dataset search product can miss a direct accession query. Exact ID parsing and pinned ranking must be guaranteed before any public demo or paper claim.

Acceptance criteria:

- DANDI IDs, OpenNeuro IDs, NeuroVault IDs, Zenodo IDs, Figshare IDs, and source-prefixed canonical IDs are parsed.
- Exact matches are pinned to rank 1 unless excluded by explicit hard constraints.
- Result cards explain why a record was pinned.
- Exact lookup tests run in CI.

### 3. Metadata extraction needs source-specific QA

Scale is solved; extraction precision is not. The biggest publication risk is weak structured evidence for tasks, regions, DOI, and provenance.

Acceptance criteria:

- Source-by-source completeness report for all 10,404 records.
- DANDI and OpenNeuro use official metadata structures where available.
- Zenodo, Figshare, and OSF receive stricter off-topic and artifact-type checks.
- Every extracted label has confidence and evidence where feasible.

### 4. Analysis affordances need content validation

The affordance framework is one of the most original parts of Neural Search, but it must be validated against actual files.

Acceptance criteria:

- At least 50 NWB/BIDS datasets receive file-inspection labels.
- Affordance precision, recall, and calibration are reported.
- Metadata-only predictions are separated from content-validated affordances.
- Result cards show support level and missing requirements.

### 5. Graph and paper links need to be rebuilt for the 10K corpus

The graph work is valuable, but publication claims need graph artifacts aligned with the current corpus snapshot.

Acceptance criteria:

- 10K snapshot graph is rebuilt.
- Paper-dataset links are surfaced on dataset cards.
- Link confidence and evidence are stored.
- Graph ablation is rerun with meaningful labels.

### 6. Product runtime must use the real snapshot

The API and UI must not drift into demo mode while the paper claims real-corpus scale.

Acceptance criteria:

- FastAPI loads a versioned `CorpusSnapshot` by default.
- Demo mode is explicit.
- `/healthz` reports corpus id, record count, embedding model, and index status.
- Search traces include snapshot id.

## Robust Next-Step Plan

### Phase 0: Freeze the 10K snapshot

Deliverable: `snapshot_manifest.json`

Tasks:

1. Generate a manifest for `combined_corpus.jsonl`, dense embeddings, turbovec index, rejection log, and source files.
2. Store counts, checksums, model name, embedding dimension, bit width, created timestamp, and source-file list.
3. Add a validation command that refuses to run search if corpus, embeddings, and index counts disagree.
4. Update every whitepaper metric table to cite a manifest id.

Success criteria:

- One command can print the active corpus snapshot.
- Corpus count, embedding id count, and index id count all equal 10,404.
- The paper can cite the manifest instead of informal run notes.

### Phase 1: Rerun core validation

Deliverable: `reports/real_corpus_10k_eval_report.md`

Tasks:

1. Rerun corpus quality checks.
2. Rerun exact lookup benchmark.
3. Rerun hard-negative benchmark.
4. Rerun 30-query real-corpus benchmark on the 10K snapshot.
5. Expand to 100+ queries after the 30-query harness is stable.
6. Report source skew in top-10 results so Zenodo/Figshare do not dominate by volume.

Success criteria:

- Exact lookup success = 100%.
- Hard-negative violations = 0.
- Metrics are reported overall and by query type.
- Failures include actionable explanations.

### Phase 2: Build a publication-grade benchmark

Deliverable: benchmark v1 with independent labels

Tasks:

1. Curate 100-150 queries across exact lookup, task, modality, species, brain region, standards, affordance, cross-dataset comparison, and hard negatives.
2. Label top candidates with at least two annotators for a representative subset.
3. Compute inter-annotator agreement.
4. Include baseline ladder: keyword, BM25, dense-only, BM25+dense, ontology, graph, affordance, full system.
5. Bootstrap confidence intervals for key metrics.

Success criteria:

- Metrics have confidence intervals.
- Baseline improvements are statistically interpretable.
- The paper can claim retrieval performance without relying on single-annotator or demo labels.

### Phase 3: Improve source-aware extraction

Deliverable: `reports/source_extraction_qa_10k.md`

Tasks:

1. Add source-specific extraction QA for DANDI, OpenNeuro, Zenodo, Figshare, OSF, NeuroMorpho, NeuroVault, and Allen.
2. Track precision-like review outcomes for high-risk labels.
3. Add DOI/license/provenance normalization checks.
4. Add source-specific exclusion rules for tables, supplements, pure papers, and non-dataset artifacts.
5. Prioritize fields below 60% coverage: tasks, brain regions, DOI, species, data standards.

Success criteria:

- No source silently drops to zero modality coverage.
- Task coverage improves from 33.7% toward 50% without obvious false positives.
- Brain-region coverage improves from 38.4% toward 55% on standards-backed sources.

### Phase 4: Optimize retrieval for the BGE corpus

Deliverable: tuned retrieval config for 10K

Tasks:

1. Recalibrate field weights for BGE-large score distributions.
2. Add per-intent weight profiles for strict lookup, pipeline reuse, replication, meta-analysis, exploration, method transfer, and cross-dataset comparison.
3. Add source diversity and source reliability features.
4. Measure lexical/dense/ontology/graph/affordance contribution through ablations.
5. Add score calibration so result cards can show meaningful confidence bands.

Success criteria:

- BGE relevance separation improves over the current r = 0.3999 preliminary result.
- Exact lookup remains deterministic after tuning.
- Top results are not dominated by high-volume generic repositories.

### Phase 5: Validate analysis affordances and latent usefulness

Deliverable: affordance validation report

Tasks:

1. Select a balanced 50-100 dataset subset across NWB, BIDS, morphology, imaging, and electrophysiology sources.
2. Inspect actual file manifests and content where feasible.
3. Label support for affordances such as event-aligned PSTH, choice decoding, Q-learning, stimulus-response modeling, pose-neural correlation, trial-aligned calcium analysis, and cross-session generalization.
4. Compare metadata-only predictions to content-validated labels.
5. Update usefulness scoring to downweight unsupported affordance claims.

Success criteria:

- Affordance predictions have precision/recall numbers.
- Result cards distinguish predicted, weakly supported, and content-validated affordances.
- The paper can defend latent usefulness as an evaluated signal rather than a slogan.

### Phase 6: Runtime and UI hardening

Deliverable: evidence-first product demo

Tasks:

1. Make real corpus snapshot the default runtime mode.
2. Add search trace IDs and snapshot IDs to API responses.
3. Add result-card sections for matched fields, missing metadata, source provenance, exact-match reason, score breakdown, and linked evidence.
4. Add warnings for low-confidence labels and missing affordance requirements.
5. Add a source/filter panel so users can inspect cross-repository behavior.

Success criteria:

- A user can tell why each result appeared.
- The UI does not overstate weak metadata.
- Runtime mode cannot be confused with demo mode.

### Phase 7: Publication package

Deliverable: reproducible paper bundle

Tasks:

1. Update the LaTeX whitepaper abstract, corpus section, methods, results, limitations, and roadmap.
2. Generate paper tables from JSON reports rather than hand-copying values.
3. Add a claim ledger with claim, artifact, command, status, and limitation.
4. Archive stale reports or mark them historical.
5. Add a clean export script that excludes `.git`, caches, `node_modules`, and transient artifacts.

Success criteria:

- Every numeric claim in the paper maps to an artifact.
- Historical/demo results are clearly labeled.
- The paper is honest about pending validation.

## Recommended Next Sprint

The next sprint should be narrow and evidence-heavy:

1. Freeze the 10,404-record snapshot and generate checksums.
2. Build the exact-lookup lane and rerun the three known failed accession queries.
3. Rerun the 30-query benchmark on the 10K snapshot.
4. Generate a source-by-source completeness and top-result skew report.
5. Rebuild or validate graph artifacts against the 10K corpus.
6. Update the LaTeX manuscript using this audit as the evidence source.

This is the shortest path from "10K records indexed" to "publication-grade claim."
