# 01 - Next Steps Roadmap

## Phase 0: Stabilize the repo

Goal: make every developer action reproducible.

1. Fix frontend TypeScript build blocker in `ComparisonDrawer.tsx`.
2. Remove committed/generated junk: `.pytest_cache`, `__pycache__`, `apps/web/node_modules`, `apps/web/dist`.
3. Add or fix `.gitignore` entries.
4. Add missing `neural_search/cli.py` or remove the console script.
5. Update Ruff config to modern `[tool.ruff.lint]` format.
6. Exclude generated notebooks from lint or lint them with a separate notebook-specific command.
7. Add a single quality gate script and CI job.

Definition of done:

```bash
pytest -q
ruff check neural_search apps/api scripts tests
cd apps/web && npm run build
python -m neural_search.evaluation.run_benchmark
```

## Phase 1: Turn ingestion from stubs into working pipelines

Goal: compile a real, searchable neural/behavioral dataset corpus.

Prioritize these connectors:

1. DANDI: NWB-first source, most aligned with electrophysiology/calcium behavior datasets.
2. OpenNeuro: BIDS/iEEG/EEG/fMRI source, good for noninvasive and human BCI search.
3. OpenAlex: paper evidence and literature linking, not primary dataset storage.

Implement ingestion at three levels:

- CLI: deterministic commands that save raw payloads and normalized records.
- Backend service functions: reusable ingestion orchestration.
- API endpoints: call the service functions, return counts, warnings, raw file paths, and normalized IDs.

Definition of done:

```bash
python -m neural_search.ingestion.dandi --query "go no-go calcium imaging" --limit 25 --save
python -m neural_search.ingestion.openneuro --query "iEEG motor BCI" --limit 25 --save
python -m neural_search.ingestion.openalex --query "reversal learning electrophysiology" --limit 25 --save
python -m neural_search.reports
```

## Phase 2: Make retrieval scientifically convincing

Goal: ranking should feel explainable, measurable, and improvable.

Add:

- A hybrid retriever with BM25/keyword + ontology + embeddings + metadata filters.
- A pluggable embedding provider interface with local default.
- Dataset-level embeddings for title, abstract/description, task labels, modalities, regions, analysis intents, and provenance snippets.
- A `why_matched` explanation that decomposes score contributions.
- Negative tests for false-positive retrieval.
- Benchmark expansion across tasks, species, modalities, brain regions, analysis goals, data standards, and behavioral events.

Definition of done:

- Every benchmark query has expected positives and hard negatives.
- Evaluation reports include precision@k, recall@k, label recall, MRR/NDCG, and failure examples.
- Retrieval weights can be changed in `data/config/retrieval.yaml` and compared against a baseline report.

## Phase 3: Make the frontend demo irresistible

Goal: convert “search results” into “scientific reuse cockpit.”

Add/polish:

- Search result cards with score decomposition.
- Dataset detail page with provenance, missing metadata, confidence, linked papers, generated notebook, and readiness score.
- Comparison drawer with dataset-by-dataset matrix and “best for X” recommendations.
- Evaluation page with benchmark pass/fail cards and regressions.
- Corpus dashboard: source coverage, modality coverage, task coverage, missing metadata, QA state.

Definition of done:

A user can run the app locally, search a scientific question, compare two datasets, inspect why they matched, generate a notebook, and see benchmark evidence.

## Phase 4: Build the “latent state search” bridge

Goal: prepare the system for neural embeddings without overpromising.

Add a new experimental namespace:

```text
neural_search/latent/
  tokenization.py
  summary_features.py
  embedding_schema.py
  indexing.py
  search.py
```

Start with feature summaries rather than training a full neural foundation model:

- Trial-aligned event histograms.
- Neural summary statistics.
- Behavior event transition summaries.
- Task-state labels.
- Session-level QC vectors.

The product story becomes: ontology/provenance search today, latent neural-state search tomorrow.

## Highest-leverage work order

1. Fix build/lint/package hygiene.
2. Wire real ingestion through CLI and API.
3. Expand benchmark queries before tuning retrieval.
4. Implement embeddings only after the benchmark harness is ready.
5. Polish frontend around explainability, comparison, and readiness.
6. Add the latent-search namespace as an experimental future-facing module.
