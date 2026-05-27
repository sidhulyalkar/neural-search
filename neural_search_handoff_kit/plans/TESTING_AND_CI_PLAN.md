# Testing and CI Plan

## Local test tiers

### Tier 1: fast sanity

```bash
pytest -q tests/test_ontology.py tests/test_retrieval_query_parsing.py tests/test_retrieval_ranking.py
```

Use this while editing ontology/search.

### Tier 2: backend full

```bash
pytest -q
```

Expected current baseline after dependency installation: `56 passed`.

### Tier 3: quality gate

```bash
ruff check neural_search apps/api scripts tests
cd apps/web && npm run build
python -m neural_search.evaluation.run_benchmark
python -m neural_search.reports
```

### Tier 4: live ingestion smoke

Run manually because it may hit external services:

```bash
python -m neural_search.ingestion.dandi --query "go no-go calcium imaging" --limit 5 --save
python -m neural_search.ingestion.openneuro --query "iEEG BCI motor" --limit 5 --save
python -m neural_search.ingestion.openalex --query "reversal learning electrophysiology" --limit 5 --save
```

## CI workflow

Create `.github/workflows/ci.yml` with:

```yaml
name: ci
on:
  push:
  pull_request:

jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e '.[dev]'
      - run: pytest -q
      - run: ruff check neural_search apps/api scripts tests

  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: npm
          cache-dependency-path: apps/web/package-lock.json
      - run: cd apps/web && npm ci
      - run: cd apps/web && npm run build

  demo-artifacts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e '.[dev]'
      - run: python -m neural_search.evaluation.run_benchmark
      - run: python -m neural_search.reports
```

## Tests to add next

- FastAPI endpoint smoke tests for `/healthz`, `/api/search`, dataset details, comparison, evaluation, reports, and ingestion endpoints.
- API error tests for empty search and invalid structured filters.
- Ingestion normalization tests using frozen JSON fixtures from DANDI/OpenNeuro/OpenAlex.
- Retrieval hard-negative tests.
- Benchmark regression test comparing current eval report to stored baseline.
- Notebook generation validation test for each template.
- QA persistence test for reviewed/trusted/rejected state transitions.

## Quality principles

- External API calls should be mocked in unit tests.
- Live ingestion tests should be opt-in, not part of default CI.
- Generated notebooks should be validated, not linted by default.
- Frontend build should be required before demo.
- Benchmark regressions should be visible, not silently overwritten.
