#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] Backend tests"
pytest -q

echo "[2/5] Python lint"
ruff check neural_search apps/api scripts tests

echo "[3/5] Frontend build"
(cd apps/web && npm run build)

echo "[4/5] Benchmark report"
python -m neural_search.evaluation.run_benchmark

echo "[5/5] Dataset compilation report"
python -m neural_search.reports

echo "Quality gate passed. The search beast is wearing its lab coat correctly."
