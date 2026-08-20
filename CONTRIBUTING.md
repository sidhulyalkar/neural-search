# Contributing to Neural Search

Thanks for helping improve Neural Search. The project sits at the intersection of scientific search, neuroscience data infrastructure, knowledge graphs, and evaluation, so changes should preserve both software quality and scientific traceability.

## Development setup

Prerequisites:

- Python 3.11+
- Node.js 20+
- Git

Create a virtual environment and follow the same portable setup path CI uses:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
make setup
neural-search profile check demo
```

For specialized work, install the matching profile instead of every optional dependency:

```bash
make install-researcher
make install-corpus-builder
make install-evaluator
make install-full-stack
```

See `docs/execution_profiles.md` before adding a dependency to the base package or changing what a profile requires.

## Before opening a pull request

Run the repository quality gate:

```bash
bash scripts/quality_gate.sh
```

At minimum, changes should keep these checks green:

```bash
neural-search profile check demo
neural-search profile check evaluator
ruff check neural_search apps/api scripts tests
pytest -q
cd apps/web && npm run build && npm run lint
```

If your change affects retrieval, ranking, normalization, knowledge-graph construction, literature linking, or benchmark artifacts, also run the relevant evaluation command and include the resulting metric change in the pull request description.

## Scientific-change expectations

Neural Search distinguishes engineering behavior from scientific claims. Please follow these rules:

1. Do not turn weak/silver/heuristic evidence into gold or expert-validated evidence through naming alone.
2. Do not weaken benchmark assertions simply to make CI pass. If an artifact is genuinely unavailable in a clean checkout, classify that dependency explicitly and document why.
3. Keep provenance. New derived fields should record enough source information to be auditable.
4. Prefer deterministic transformations for ontology normalization and metadata cleanup when possible.
5. If a metric changes, state whether the change comes from code, corpus state, qrels, model configuration, or regenerated artifacts.
6. Avoid committing secrets, private datasets, large generated caches, or machine-specific paths.
7. A green engineering gate is not evidence that a scientific claim improved. Retrieval/scientific changes require the appropriate benchmark, ablation, audit, or calibration evidence.

## Execution-profile rules

The supported operating modes are `demo`, `researcher`, `corpus-builder`, `evaluator`, and `full-stack`.

When adding a dependency or runtime assumption:

- put a package in the base dependency set only if portable/core functionality imports it directly;
- prefer a named profile extra for specialized infrastructure;
- keep `demo` runnable without production corpora, GPUs, Postgres, Redis, or network access;
- keep the portable evaluator contract limited to committed evaluation inputs;
- do not make `researcher` or `full-stack` silently fall back to scientifically mismatched artifact revisions;
- update `neural_search/runtime/catalog.py`, tests, and `docs/execution_profiles.md` when profile semantics change.

## Data and artifact rules

The repository contains committed fixtures, frozen evaluation inputs, generated local research assets, and derived reports. `neural_search/runtime/catalog.py` is the first-class capability contract for these assets.

If a new durable artifact affects a user-facing or scientific capability:

1. register it with a stable ID, expected path, lifecycle kind, and purpose;
2. identify the profile for which it is required, recommended, or produced;
3. provide a repair command only if one canonical command really reconstructs the expected artifact;
4. add tests showing that portable profiles remain portable;
5. include the artifact in reproducibility/reporting surfaces when it can alter scientific results.

Do not assume every corpus, embedding cache, graph export, literature file, or DuckDB database is present in a fresh clone. Read operations must not silently create empty production-looking assets just to satisfy existence checks.

See `docs/artifact_registry.md` for the lifecycle model.

## API and service boundaries

New API work should use `apps/api/application.py` as the composition boundary. Avoid adding unrelated orchestration to the already-large `apps/api/main.py`.

As legacy endpoints are modified, prefer extracting a user-goal-oriented application service that can be called independently of FastAPI. Keep retrieval, graph, literature, and evaluation logic inside their domain modules rather than HTTP routers.

See `docs/architecture/service_boundaries.md` for migration rules.

## Pull request scope

Keep PRs reviewable. A good PR explains:

- the user or scientific problem being solved;
- the execution profile(s) affected;
- the implementation approach;
- tests and evaluation performed;
- any data or artifact regeneration required;
- scientific/evidence-tier impact;
- known limitations or follow-up work.

For changes that alter public APIs, CLI behavior, configuration, artifact contracts, or contributor workflows, update the README or relevant documentation in the same PR.

## Reproducibility

For changes that depend on nontrivial artifact state, include or reference a profile manifest:

```bash
neural-search profile manifest evaluator \
  --output reports/reproducibility/evaluator.json
```

A manifest can describe an incomplete environment too. The important property is that absence and version state are visible rather than implicit.

## Security

Do not open a public issue for a vulnerability that could expose credentials, private data, or remote-code execution. Follow the instructions in `SECURITY.md`.
