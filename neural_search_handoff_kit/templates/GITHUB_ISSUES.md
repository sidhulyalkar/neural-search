# GitHub Issues

## Issue 1: Fix frontend build and repo hygiene

**Goal:** Make the repo buildable from a clean clone.

Tasks:

- [ ] Fix unused `datasets` variable in `ComparisonDrawer.tsx`.
- [ ] Remove committed `node_modules`, `dist`, `__pycache__`, `.pytest_cache`.
- [ ] Update `.gitignore`.
- [ ] Confirm `pytest -q` passes.
- [ ] Confirm `cd apps/web && npm ci && npm run build` passes.

## Issue 2: Add working CLI entry point

**Goal:** Resolve broken `neural-search` console script.

Tasks:

- [ ] Add `neural_search/cli.py`.
- [ ] Support `demo`, `search`, `benchmark`, `report`.
- [ ] Support `ingest dandi/openneuro/openalex`.
- [ ] Add CLI tests.

## Issue 3: Replace ingestion API stubs with service calls

**Goal:** Make `/api/ingest/*` endpoints actually ingest or normalize source records.

Tasks:

- [ ] Add `neural_search/ingestion/services.py`.
- [ ] Create `IngestionRunResult` model.
- [ ] Wire DANDI/OpenNeuro/OpenAlex endpoints.
- [ ] Save raw payload paths.
- [ ] Add API tests.

## Issue 4: Expand benchmark queries with hard negatives

**Goal:** Make retrieval quality measurable.

Tasks:

- [ ] Add at least 30 benchmark queries.
- [ ] Add expected labels and hard negatives.
- [ ] Report precision/recall/MRR/NDCG.
- [ ] Add regression comparison report.

## Issue 5: Add embedding provider abstraction

**Goal:** Add semantic retrieval without making CI depend on external model downloads.

Tasks:

- [ ] Add `EmbeddingProvider` protocol.
- [ ] Add deterministic hashing provider.
- [ ] Add optional sentence-transformers provider.
- [ ] Store/recompute dataset embeddings.
- [ ] Add ablation metrics.

## Issue 6: Corpus dashboard and evaluation UI polish

**Goal:** Make demo pages communicate coverage, trust, and failure modes.

Tasks:

- [ ] Add corpus stats endpoint if needed.
- [ ] Improve reports page.
- [ ] Improve evaluation page.
- [ ] Improve result score breakdown.
- [ ] Add user-friendly missing metadata display.

## Issue 7: Latent search scaffold

**Goal:** Prepare future learned neural-state search without pretending it exists yet.

Tasks:

- [ ] Add `neural_search/latent/` namespace.
- [ ] Define session/trial summary schema.
- [ ] Add simple feature summary functions.
- [ ] Add docs connecting ontology search to future latent-state search.
