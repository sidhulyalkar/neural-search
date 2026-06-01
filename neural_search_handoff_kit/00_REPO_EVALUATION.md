# 00 - Repository Evaluation

## Snapshot

The uploaded `neural-search.zip` contains a functioning MVP for experiment-aware dataset discovery. It has a Python package, FastAPI backend, React/Vite frontend, ontology-driven retrieval, generated dataset cards, notebook generation, QA state, evaluation reports, and a seed/demo corpus.

Approximate repo inventory after excluding `.git`, `node_modules`, `dist`, `__pycache__`, and `.pytest_cache`:

- 153 relevant files
- 72 Python files, about 10.6k lines
- 15 TypeScript/TSX files, about 4.2k lines
- 20 Markdown docs
- 11 YAML files
- Current source surfaces: `neural_search/`, `apps/api/`, `apps/web/`, `data/`, `docs/`, `tests/`

## What is strong

### Product direction

The repo has a coherent product thesis: search for neural and behavioral datasets by experimental meaning, not by filename or generic document chunks. The README communicates this well and frames the system as dataset retrieval plus explanations, provenance, cards, notebooks, and evaluation.

### Backend architecture

The backend is organized into clear modules:

- `neural_search/ontology`: task/modality/region/behavior vocabulary and matching.
- `neural_search/search`: query parsing, scoring, filtering, retrieval config.
- `neural_search/cards`: dataset card generation.
- `neural_search/notebooks`: starter notebook generation and validation.
- `neural_search/ingestion`: demo fixtures and initial source connector structure.
- `neural_search/qa`: review state and trust/rejection workflows.
- `neural_search/evaluation`: benchmark runner and reports.
- `neural_search/reports`: dataset compilation report.

This is the right skeleton for turning the MVP into a serious research infrastructure product.

### Testing foundation

After installing dependencies, the backend test suite passes:

```text
56 passed in 15.81s
```

Coverage areas include analysis recipes, QA, comparison, query builder, extraction, live ingestion utilities, notebook templates, ontology, readiness/cards/notebooks/search, retrieval parsing, and retrieval ranking.

### Demo artifacts

The repo already includes generated benchmark reports, generated notebooks, curated/demo datasets, and docs such as architecture, retrieval, evaluation, demo walkthrough, and known limitations. This gives Claude/Codex enough surface area to improve rather than invent from vapor.

## Current blockers

### 1. Frontend build fails

Command:

```bash
cd apps/web && npm run build
```

Failure:

```text
src/components/ComparisonDrawer.tsx(162,20): error TS6133: 'datasets' is declared but its value is never read.
```

Minimal fix: remove `datasets` from the destructure in `SummaryView`.

```ts
function SummaryView({ comparison }: { comparison: ComparisonResult }) {
  const { summary } = comparison
```

A patch is included at `patches/known_frontend_build_fix.diff`.

### 2. Ruff lint currently fails

Command:

```bash
ruff check .
```

Observed: 124 lint errors, mostly auto-fixable. Main categories:

- Import ordering (`I001`)
- Notebook lint noise in generated notebooks
- Module-level imports after `sys.path.insert` in `scripts/demo.py`
- Unused imports/variables
- Minor modernization warnings

Recommendation: exclude generated notebooks and frontend build artifacts from Ruff, then run `ruff check --fix .` and manually handle the remaining script/import issues.

### 3. Packaging has a broken console entry point

`pyproject.toml` declares:

```toml
[project.scripts]
neural-search = "neural_search.cli:main"
```

But `neural_search/cli.py` does not exist. Either add a real CLI or remove the entry point. This will bite packaging/deployment later.

### 4. Uploaded repo contains generated/cache/vendor material

The zip includes `.git`, `.pytest_cache`, `__pycache__`, frontend `node_modules`, and `apps/web/dist`. These should not be committed or shipped in future handoffs. Keep the repo lean and reproducible.

Recommended cleanup:

```bash
rm -rf .pytest_cache **/__pycache__ apps/web/node_modules apps/web/dist
```

Then ensure `.gitignore` covers those patterns.

### 5. API ingestion endpoints are still stubs

`apps/api/main.py` exposes:

- `/api/ingest/dandi`
- `/api/ingest/openneuro`
- `/api/ingest/openalex`

but they return pending messages. The lower-level ingestion utility modules exist, so the next step is to wire real ingestion flows into API endpoints and CLI commands.

## Current maturity score

| Area | Score | Notes |
| --- | ---: | --- |
| Product concept | 9/10 | Strong, differentiated, easy to narrate. |
| Backend structure | 8/10 | Good modularity, needs live ingestion and stronger persistence. |
| Retrieval quality | 6/10 | Good rule/ontology scaffold, needs embeddings and richer eval. |
| Dataset corpus | 5/10 | Demo corpus plus seed outputs, needs real curated expansion. |
| Frontend | 6/10 | Product surface exists, build blocker and polish needed. |
| Testing | 7/10 | Backend suite passes; frontend/CI/lint quality gates need tightening. |
| Demo readiness | 7/10 | Good enough for controlled walkthrough after build fix. |
| Research-grade readiness | 4/10 | Needs source coverage, metadata confidence, provenance, benchmark rigor. |

## Strategic diagnosis

The project has crossed the first river: it is no longer just an idea. The next river is reliability. The highest-leverage next work is not adding 50 flashy features. It is making the system ingest real sources, score them transparently, evaluate retrieval quality against benchmark queries, and present the results beautifully enough that a lab, neuro-AI team, or funder can understand the magic in one demo.
