# Contributing to Neural Search

Thanks for helping improve Neural Search. The project sits at the intersection of scientific search, neuroscience data infrastructure, knowledge graphs, and evaluation, so changes should preserve both software quality and scientific traceability.

## Development setup

Prerequisites:

- Python 3.11+
- Node.js 20+
- Git

Create a virtual environment, install the development dependencies, and install the frontend:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cd apps/web
npm ci
cd ../..
```

Run the built-in environment check:

```bash
neural-search doctor
```

## Before opening a pull request

Run the repository quality gate:

```bash
bash scripts/quality_gate.sh
```

At minimum, changes should keep these checks green:

```bash
ruff check neural_search apps/api scripts tests
pytest -q
cd apps/web && npm run build && npm run lint
```

If your change affects retrieval, ranking, normalization, knowledge-graph construction, literature linking, or benchmark artifacts, also run the relevant evaluation command and include the resulting metric change in the pull request description.

## Scientific-change expectations

Neural Search distinguishes engineering behavior from scientific claims. Please follow these rules:

1. Do not turn weak/silver/heuristic evidence into gold or expert-validated evidence through naming alone.
2. Do not weaken benchmark assertions simply to make CI pass. If an artifact is genuinely unavailable in a clean checkout, skip or mark that condition explicitly and document why.
3. Keep provenance. New derived fields should record enough source information to be auditable.
4. Prefer deterministic transformations for ontology normalization and metadata cleanup when possible.
5. If a metric changes, state whether the change comes from code, corpus state, qrels, model configuration, or regenerated artifacts.
6. Avoid committing secrets, private datasets, large generated caches, or machine-specific paths.

## Pull request scope

Keep PRs reviewable. A good PR explains:

- the user or scientific problem being solved;
- the implementation approach;
- tests and evaluation performed;
- any data or artifact regeneration required;
- known limitations or follow-up work.

For changes that alter public APIs, CLI behavior, configuration, or contributor workflows, update the README or relevant documentation in the same PR.

## Data and generated artifacts

The repository contains a mix of committed fixtures, reports, and references to locally generated large artifacts. Do not assume every large corpus, embedding cache, graph export, or DuckDB file is present in a fresh clone.

Code and tests should fail with actionable messages or skip explicitly when optional local artifacts are absent. They should not silently create an empty production-looking artifact as a side effect of a read operation.

## Security

Do not open a public issue for a vulnerability that could expose credentials, private data, or remote-code execution. Follow the instructions in `SECURITY.md`.
