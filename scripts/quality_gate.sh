#!/usr/bin/env bash
set -euo pipefail

echo "[1/7] Dependency graph"
python -m pip check

echo "[2/7] CLI environment"
neural-search doctor

echo "[3/7] Backend tests"
pytest -q

echo "[4/7] Python lint"
ruff check neural_search apps/api scripts tests

echo "[5/7] Frontend build"
(cd apps/web && npm run build)

echo "[6/7] Benchmark report"
python -m neural_search.evaluation.run_benchmark

echo "[7/7] Dataset compilation report"
python -m neural_search.reports

echo "Quality gate passed."
