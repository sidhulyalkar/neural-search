# Claude Code Prompt - Neural Search Next-Level Product Build

You are working in the `neural-search` repository. Your role is full-stack product architect and senior research software engineer. Take the current MVP to the next level without breaking existing tests.

## Project thesis

Neural Search is experiment-aware search for reusable neural and behavioral datasets. It should retrieve datasets by scientific meaning: task, behavior, modality, brain region, data standard, species, provenance, analysis readiness, and eventual latent neural-state summaries. This is not generic RAG. The core artifact is a ranked, explainable, reusable dataset result with cards, notebooks, provenance, QA, and benchmark evidence.

## Immediate context

Current backend tests pass after dependency installation: `56 passed`.

Known blockers:

1. Frontend build fails because `apps/web/src/components/ComparisonDrawer.tsx` destructures unused `datasets` in `SummaryView`.
2. Ruff reports many fixable lint issues and should be scoped away from generated notebooks.
3. `pyproject.toml` declares `neural-search = "neural_search.cli:main"`, but `neural_search/cli.py` is missing.
4. Ingestion API endpoints are stubs even though ingestion modules exist.
5. Repo hygiene needs cleanup: generated/cache/vendor files should not be committed.

## Your mission

Deliver a clean, demoable, next-level version of the repo with:

- Passing backend tests.
- Passing frontend TypeScript build.
- A sane lint/format configuration.
- A basic CLI entry point or removed broken entry point.
- Real ingestion service wiring for DANDI/OpenNeuro/OpenAlex at least through deterministic CLI/service paths, preferably also API endpoints.
- A stronger evaluation/reporting surface.
- Frontend polish that emphasizes scientific explainability and reuse.

## Work plan

### Step 1: Repo hygiene and quality gates

- Fix the frontend build blocker.
- Remove generated/cache/vendor files from version control.
- Update `.gitignore`.
- Modernize Ruff config to `[tool.ruff.lint]`.
- Exclude generated notebooks from default Ruff if needed.
- Add or fix `neural_search/cli.py`.
- Add `scripts/quality_gate.sh` if missing.
- Add GitHub Actions CI if missing.

Run:

```bash
pytest -q
ruff check neural_search apps/api scripts tests
cd apps/web && npm run build
```

### Step 2: Real ingestion orchestration

Create service-level functions that the API and CLI can share:

```text
neural_search/ingestion/services.py
```

Suggested interface:

```python
def ingest_dandi_query(query: str, limit: int, database_url: str, force: bool = False) -> IngestionRunResult: ...
def ingest_openneuro_query(query: str, limit: int, database_url: str, force: bool = False) -> IngestionRunResult: ...
def ingest_openalex_query(query: str, limit: int, database_url: str, force: bool = False) -> IngestionRunResult: ...
```

Return counts, skipped records, raw response paths, warnings, source, query, and normalized IDs. Preserve raw payloads under `data/raw/<source>/`.

Wire API endpoints in `apps/api/main.py` to these services. Do not leave TODO stubs.

### Step 3: Benchmark-first retrieval improvements

Before changing ranking weights, expand the benchmark set with positive and hard-negative expectations.

Add benchmark cases for:

- Go/NoGo, reversal learning, delay discounting, PRL, reaching, visual decision-making.
- Calcium imaging, Neuropixels, extracellular ephys, fiber photometry, EEG, ECoG/iEEG, fMRI, behavior video, pose tracking.
- OFC, mPFC, striatum, M1, visual cortex, hippocampus, basal ganglia.
- NWB, BIDS, DANDI, OpenNeuro.
- Analysis goals: choice decoding, reward prediction, event alignment, closed-loop BCI, trial outcome prediction, QC-ready reuse.

Then add retrieval metrics and failure examples to reports.

### Step 4: Frontend demo polish

Focus on scientific clarity rather than decoration.

Improve:

- Search cards: show why matched, readiness, missing metadata, provenance, linked papers, and suggested next actions.
- Dataset details: card JSON, generated notebook link, source, license, data standard, warnings, QA status.
- Comparison drawer: common/missing metadata matrix, best-use recommendations, analysis compatibility.
- Evaluation page: benchmark pass/fail, regression status, top failure modes.
- Corpus report page: source/task/modality/region coverage and QA state.

### Step 5: Prepare the latent-search bridge

Add an experimental scaffold only, not a fake trained model.

```text
neural_search/latent/
  __init__.py
  schema.py
  feature_summary.py
  indexing.py
  search.py
```

Purpose: define how future neural/behavioral latent summaries will plug into the existing ontology/provenance search stack.

## Guardrails

- Do not replace the current explainable retrieval with opaque RAG.
- Do not require paid APIs for core tests.
- Do not remove existing features unless replacing them with better tested versions.
- Preserve deterministic demo mode.
- All live ingestion must save raw payloads and normalized records separately.
- Keep the backend usable without Postgres; SQLite/demo mode should still work.
- Prefer small, reviewed PR-sized commits.

## Definition of done

At the end, provide a summary containing:

- Files changed.
- Commands run and outputs.
- New capabilities added.
- Remaining limitations.
- How to demo the product in 5 minutes.

Required final checks:

```bash
pytest -q
ruff check neural_search apps/api scripts tests
cd apps/web && npm run build
python -m neural_search.evaluation.run_benchmark
python -m neural_search.reports
```
