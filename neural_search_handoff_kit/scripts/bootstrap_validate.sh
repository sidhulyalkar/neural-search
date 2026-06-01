#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

pytest -q
ruff check neural_search apps/api scripts tests

if [ -d apps/web ]; then
  (cd apps/web && npm ci && npm run build)
fi

python -m neural_search.evaluation.run_benchmark
python -m neural_search.reports
