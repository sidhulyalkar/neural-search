# Pull Request Checklist

## Required

- [ ] `pytest -q` passes.
- [ ] `ruff check neural_search apps/api scripts tests` passes.
- [ ] `cd apps/web && npm run build` passes.
- [ ] Benchmark report generated.
- [ ] Dataset report generated.
- [ ] README/docs updated for any user-facing behavior.
- [ ] No `node_modules`, `dist`, `.pytest_cache`, `__pycache__`, or raw giant files committed accidentally.

## Ingestion changes

- [ ] Raw payloads are saved with timestamped paths.
- [ ] Normalization is deterministic.
- [ ] Existing records are skipped unless `force=True`.
- [ ] Unit tests use frozen fixtures or injected clients, not live network calls.
- [ ] Source IDs are stable.

## Retrieval changes

- [ ] Score changes are benchmarked against previous report.
- [ ] Hard negatives do not regress.
- [ ] `why_matched` remains clear to a scientist.
- [ ] Weights are configurable.

## Frontend changes

- [ ] Loading, error, empty states handled.
- [ ] API type contracts updated.
- [ ] Page works in a clean local run.
- [ ] Demo query flow still works.
