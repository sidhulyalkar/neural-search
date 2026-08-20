#!/usr/bin/env bash
set -euo pipefail

echo "[1/9] Dependency graph"
python -m pip check

echo "[2/9] Demo profile"
neural-search doctor --profile demo
neural-search profile check demo

echo "[3/9] Portable evaluator profile"
neural-search profile check evaluator

echo "[4/9] Reproducibility manifest"
neural-search profile manifest demo --output /tmp/neural-search-demo-manifest.json

echo "[5/9] Backend tests"
pytest -q

echo "[6/9] Python lint"
ruff check neural_search apps/api scripts tests

echo "[7/9] Frontend build"
(cd apps/web && npm run build)

echo "[8/9] Benchmark report"
python -m neural_search.evaluation.run_benchmark

echo "[9/9] Dataset compilation report"
python -m neural_search.reports

echo "Quality gate passed."
