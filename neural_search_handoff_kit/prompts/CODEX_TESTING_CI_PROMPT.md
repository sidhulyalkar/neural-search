# Codex Prompt - Testing, CI, and Quality Gate Hardening

You are responsible for making the repo reliable. Do not redesign the product. Build guardrails.

## Current observed state

- Backend tests pass after dependencies are installed: 56 passed.
- Frontend build fails on an unused variable in `ComparisonDrawer.tsx`.
- Ruff reports many fixable errors, partly because generated notebooks are included.
- The package script entry points to missing `neural_search.cli`.

## Tasks

1. Fix the frontend TypeScript build.
2. Fix or scope linting so default lint checks source code, not generated notebooks or build artifacts.
3. Add `scripts/quality_gate.sh` that runs backend tests, lint, frontend build, benchmark, and report generation.
4. Add GitHub Actions CI with Python 3.11 and Node 20.
5. Add a smoke test for FastAPI endpoints using TestClient.
6. Add frontend API contract tests or at least TypeScript type checks.
7. Add documentation to README: setup, quality gate, common failures.
8. Add `.gitignore` entries for caches and build artifacts.

## Quality gate target

```bash
python -m pip install -e ".[dev]"
pytest -q
ruff check neural_search apps/api scripts tests
cd apps/web && npm ci && npm run build
python -m neural_search.evaluation.run_benchmark
python -m neural_search.reports
```

## CI notes

Use separate jobs if needed:

- `backend`: Python install, pytest, ruff.
- `frontend`: Node install, TypeScript/Vite build.
- `demo-artifacts`: benchmark/report generation.

Cache pip and npm if easy, but correctness matters more than speed.
