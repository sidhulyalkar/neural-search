# Neural Search Whitepaper + Repository Audit

Date: 2026-06-02

## Executive Diagnosis

Neural Search has crossed the line from concept into a real research prototype. The codebase contains substantial systems work: ingestion adapters, normalized schemas, graph construction, dense embedding infrastructure, field-semantic scoring, benchmark harnesses, a FastAPI backend, a Vite/React frontend, dataset-card generation, notebooks, and many reports. The strongest idea remains highly differentiated: searching for datasets by **latent future usefulness** rather than surface similarity.

The next development cycle should stop expanding horizontally and should instead harden the system into a **trustworthy dataset discovery product**. The limiting factor is no longer whether the architecture sounds interesting. It is whether the system can reliably answer: “Why is this dataset actually reusable for this analysis, and how do we know?”

## Snapshot Findings

### Repository scale

- Python files: 359
- Python LOC: ~83,097
- Markdown docs: 167 files, ~51,866 LOC
- Frontend TS/TSX files: 17 files, ~4,764 LOC
- Tests collected by pytest before environment failure: 497
- Zip includes generated/should-not-ship artifacts such as `.git`, `__pycache__`, `.pytest_cache`, and `apps/web/node_modules`.

### Test/dev environment

Running `python -m pytest -q` in the sandbox collected 497 tests but stopped during collection with 47 import errors because dependencies such as SQLAlchemy were not installed in the sandbox. This does not prove the project tests fail in a proper environment, but it does show that the current artifact is not self-contained or reproducible from the uploaded zip alone.

The frontend build could not run from the uploaded artifact because `vite` in `apps/web/node_modules/.bin` had permission issues. This reinforces that `node_modules` should not be shipped in repo/package artifacts. Use `npm ci` from a lockfile instead.

### Whitepaper/repo alignment

The whitepaper abstract reports 371 records and benchmark metrics of 76.7% Precision@5, 87.8% Recall@10, 0.950 MRR, and 0.937 NDCG@10 on an initial 30-query benchmark. The current repository also contains newer artifacts, including:

- `reports/real_corpus_v11_eval_report.md`: 30 real-corpus queries, Mean Precision@5 = 69.3%, Mean MRR = 0.839, Mean NDCG@10 = 0.822.
- `reports/baseline_v09.json`: 876 total corpus records, 738 dataset records implied by current normalized files, 97 paper records, 186 seed pairs, hashing baseline Spearman r = 0.5044.
- `reports/usefulness_correlation_v09.json`: Spearman r = 0.3999 over 270 pairs.
- `reports/turbovec_recall.json`: recall@50 = 1.0 in the available index path, with fallback/exact behavior noted elsewhere.

This means the paper needs a clear “versioned evidence ledger”: which corpus, which index, which benchmark, which commit, which metric table, and which claims are still demo-only.

## Current Normalized Corpus Quality

I parsed the four main normalized real dataset files:

- `data/corpus/normalized/real_dandi.jsonl`: 358 records
- `data/corpus/normalized/real_openneuro.jsonl`: 362 records
- `data/corpus/normalized/real_allen.jsonl`: 8 records
- `data/corpus/normalized/real_nemo.jsonl`: 10 records

Total: 738 dataset records.

Field completeness across these 738 records:

| Field | Non-empty | Coverage |
|---|---:|---:|
| title | 738 | 100.0% |
| description | 370 | 50.1% |
| species | 374 | 50.7% |
| modalities | 357 | 48.4% |
| brain_regions | 180 | 24.4% |
| tasks | 271 | 36.7% |
| behavioral_events | 310 | 42.0% |
| analysis_goals | 0 | 0.0% |
| data_standards | 385 | 52.2% |
| file_formats | 0 | 0.0% |
| linked_papers | 0 | 0.0% |
| analysis_affordances | 0 | 0.0% |

This is the central product problem. The system’s value proposition depends on structured reuse evidence, but the current normalized real corpus is too sparse in the fields that matter most for reuse: brain region, task, file format, linked papers, analysis affordances, assets, and actual data-content signatures.

## Most Important Gaps

### 1. Product/runtime path still defaults to demo mode

`apps/api/main.py` loads `build_demo_seed()` into an in-memory `_demo_data` store. The search endpoint calls `search_datasets(... datasets=records_with_qa ...)` using this demo seed. Meanwhile, the UI copy says `v2.0 · 835 datasets`, and the repo contains real corpus indexes/reports. This creates a product mismatch: the impressive real-corpus work is not the default product surface.

Required next step: introduce a canonical runtime corpus loader that can load the real normalized corpus, graph, embeddings, QA state, and paper links through one versioned `CorpusSnapshot` object.

### 2. Exact identifier lookup is not fail-safe

The real-corpus v11 report shows direct lookup failures:

- `Find the Steinmetz 2019 Neuropixels visual coding dataset` missed `dataset:dandi:000026`.
- `DANDI dataset 000020 mouse hippocampus` missed `dataset:dandi:000020`.
- `OpenNeuro motor imagery EEG BCI dataset ds003505` missed `dataset:openneuro:ds003505`.

A serious dataset search tool must never miss a direct accession/id query. Exact ID matching should bypass the scoring soup and enter a “pinned exact match lane.”

### 3. Metadata extraction is broad but shallow

The repo has many ingestion adapters, but real normalized records remain sparse. This indicates the bottleneck is not connector count. The bottleneck is extraction quality, schema fill, provenance, and validation.

Next step: build a source-specific extraction QA loop for DANDI and OpenNeuro first, not nine more adapters.

### 4. Analysis affordances are architecturally strong but not populated in real records

The affordance registry and validators are promising. However, real normalized records currently have 0 non-empty `analysis_affordances`. Until these are populated and validated against actual NWB/BIDS contents, “analysis affordance search” remains more of a claim than a trustworthy product feature.

### 5. Paper-dataset linking exists in the graph but not in dataset records

The graph has 97 paper nodes and `paper_related_to_dataset` edges, but `linked_papers` is empty in the parsed real dataset records. This split makes it difficult for the result UI and dataset cards to show traceable provenance directly.

Next step: canonicalize links into both graph and dataset-card surfaces, with confidence and evidence.

### 6. Embeddings are moving in the right direction, but the retrieval stack is not yet cleanly versioned

The repo contains `DenseEmbeddingProvider` for BGE-large-en-v1.5, dense field embeddings, and a turbovec wrapper. However, the current docs and reports still mix hashing, dense, fallback exact search, and real/demonstration corpora. The product needs an explicit embedding registry and index manifest:

- model name
- model version
- dimension
- normalization
- corpus snapshot id
- embedding file checksum
- ANN backend
- recall/latency report
- fallback behavior

### 7. Evaluation is promising but not yet publication-grade

The repo has many evaluation modules, but the actual human/label surface is still small:

- `data/eval/relevance_labels_v01.jsonl`: 50 labels, reviewer `claude_auto`
- `data/eval/human_judgments_search_intelligence_task23.jsonl`: 9 fixture judgments
- `data/labels/linkage_labels_sid.jsonl`: 6 labels
- `data/eval/usefulness_seed_pairs.jsonl`: 186 seed pairs, but mostly synthetic/curated usefulness labels

The next evaluation step should be a multi-annotator benchmark, not another benchmark report generated from weak labels.

### 8. Backend/database/worker architecture is not productized

`docker-compose.yml` includes Postgres, Redis, API, web, and worker services, but the API still uses in-memory demo data. `infra/Dockerfile.api` currently copies `packages`, but this repo has no `packages/` directory; it also attempts editable install before copying the `neural_search/` package. This Dockerfile needs repair before any deployment claim.

### 9. Repository hygiene and release packaging need a reset

The uploaded zip includes `.git`, pyc caches, pytest cache, node_modules, and many stale reports/plans. The codebase is being dragged around with a barnacle coat. A serious tool needs a clean release profile:

- source package
- generated artifacts package
- benchmark reports package
- demo package
- paper package

Do not ship all of them as one zip.

## Recommended Next Development Direction

### North Star

Build Neural Search into a **reusability evidence engine** for neuroscience datasets.

Not just: “find datasets similar to this query.”

Instead: “show me datasets that can support this scientific analysis, explain the evidence, expose uncertainty, and let me launch a reproducible first-pass analysis.”

## Development Track A: Trustworthy Corpus Runtime

Deliverable: `CorpusSnapshotV1`

Required features:

- Load normalized datasets, papers, graph, embeddings, QA state, and benchmark metadata together.
- Validate all IDs across files.
- Refuse to run if graph/index/corpus snapshots are inconsistent.
- Generate a `snapshot_manifest.json` with counts, checksums, model versions, created_at, and source files.
- Make the FastAPI app use this snapshot instead of `build_demo_seed()` by default.
- Keep demo seed as an explicit `--demo` mode only.

Acceptance tests:

- API `/healthz` returns active snapshot id and corpus counts.
- `/api/search` searches the real corpus in production mode.
- Demo and real modes are impossible to confuse.

## Development Track B: Exact Lookup and Constraint Gate

Deliverable: deterministic pre-ranking lane.

Required features:

- Parse dataset identifiers: `DANDI:000026`, `dandi 000026`, `dataset:dandi:000026`, `OpenNeuro ds003505`, raw `ds003505`.
- Pin exact matches to rank 1 unless hard-excluded by explicit query constraints.
- Add source-aware alias tables.
- Add strict negative constraints before scoring.
- Add “why pinned” explanation.

Acceptance tests:

- The three failed v11 lookup queries return the expected dataset at rank 1.
- Queries with `NOT fMRI`, `NOT visual cortex`, etc. do not leak hard negatives.

## Development Track C: Metadata Completeness Sprint

Deliverable: 90%+ fill rate for core fields on DANDI/OpenNeuro subsets.

Prioritize 200 high-value datasets first.

Core fields:

- species
- modality
- brain region
- task
- behavioral events
- data standard
- file formats
- assets
- linked papers
- analysis affordance candidates

Approach:

1. Implement source-specific extractors for DANDI/OpenNeuro using official metadata structures.
2. Add confidence and evidence text to every extracted label.
3. Add a field-level QA report by source.
4. Add active-learning review queue for low-confidence or high-impact records.
5. Regenerate graph and embeddings only after corpus QA passes.

Acceptance tests:

- DANDI/OpenNeuro source-specific completeness report generated.
- No core field silently drops to 0% in CI.
- Missing fields appear in user-facing result cards.

## Development Track D: Analysis Affordance Validation

Deliverable: `analysis_affordances` populated and validated.

Required features:

- Run metadata-based affordance detector on all normalized records.
- For a smaller subset, inspect actual NWB/BIDS contents.
- Store `support_level`, `confidence`, `required_fields_present`, `missing_fields`, and evidence.
- Show affordance matrix in the UI.

Acceptance tests:

- `analysis_affordances` non-empty for datasets where sufficient metadata exists.
- At least 50 NWB/BIDS datasets content-validated.
- Precision/recall of affordance predictions reported against human/file-inspection labels.

## Development Track E: Embedding/Index Registry

Deliverable: reproducible dense retrieval stack.

Required features:

- Add `EmbeddingIndexManifest`.
- Support hashing for CI, BGE for local/GPU, optional API embeddings if explicitly configured.
- Ensure query embeddings are generated in the same vector space as corpus embeddings.
- Track fallback behavior explicitly.
- Add benchmark comparison: lexical, BM25, hashing, BGE, BGE+graph, BGE+graph+affordance.

Acceptance tests:

- Loading a BGE index without BGE installed fails loudly or falls back with explicit UI/API warnings.
- CI can run small deterministic hashing tests.
- Dense benchmark report includes per-intent breakdown and confidence intervals.

## Development Track F: Evidence-First UI

Deliverable: elegant “scientific result card” interface.

Each result card should show:

- match score with score breakdown
- direct match badges: task, modality, region, species, data standard
- source badge and accession id
- reuse readiness score
- missing metadata warnings
- analysis affordance matrix
- linked papers with confidence/evidence
- graph neighborhood mini-view
- suggested first analyses
- one-click starter notebook
- “why this result” and “why not fully trusted” panels

The UI should make uncertainty beautiful instead of burying it.

## Development Track G: Evaluation and Paper Claim Ledger

Deliverable: versioned evaluation suite + paper table generator.

Required features:

- Create benchmark v1 with 100+ queries.
- Use at least two human annotators for a subset.
- Measure inter-annotator agreement.
- Add hard negatives and exact lookup tests.
- Generate paper-ready metric tables directly from `reports/*.json`.
- Maintain `docs/CLAIM_LEDGER.md` with claim, evidence file, status, and limitations.

Acceptance tests:

- Paper metrics regenerate from current repo.
- No paper claim exists without a linked evidence artifact.
- Demo metrics and real-corpus metrics are clearly separated.

## Development Track H: Deployment and Repo Hygiene

Deliverable: clean, deployable repo.

Tasks:

- Fix `infra/Dockerfile.api` to copy/install the actual package.
- Wire API to real corpus snapshot and persistent storage.
- Add auth for QA/review endpoints.
- Add structured logging and search trace IDs.
- Add a clean export script that excludes `.git`, `node_modules`, pycache, raw caches, and stale generated outputs.
- Split docs into `active/`, `archive/`, and `paper/`.
- Add `make doctor` to verify Python, Node, dependencies, data files, indexes, and Docker.

## Best Next Sprint

The next sprint should be narrow and ruthless:

1. Fix runtime mode: real corpus snapshot powers the API.
2. Fix exact lookup: direct DANDI/OpenNeuro IDs always work.
3. Generate corpus completeness report and start filling missing fields.
4. Populate analysis affordances from metadata.
5. Add a result-card UI that exposes evidence and missingness.
6. Update the whitepaper to separate demo metrics, real-corpus metrics, and future claims.

This is the shortest path from “cool research prototype” to “tool scientists might actually trust.”
