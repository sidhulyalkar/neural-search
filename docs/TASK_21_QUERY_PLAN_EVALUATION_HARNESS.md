# Task 21: Query Plan Evaluation Harness

Status: first implementation slice complete

## Goal

Evaluate whether planner-selected modes and weights improve retrieval quality before wiring them into the default search path.

## Scope

- Run benchmark suites with baseline search, awareness search, and intelligence search.
- Compare precision, label recall, MRR, NDCG, hard-negative violations, and coverage warnings.
- Group results by planner intent and data form.
- Generate Markdown and JSON decision reports.

## Acceptance Criteria

- The harness can run without network access or paid embedding providers.
- Planner changes are never promoted if hard-negative violations increase.
- Reports identify which intents benefit from planner blending and which should stay baseline.

## First Slice

- [x] Add a deterministic evaluation module comparing baseline, awareness, and intelligence wrappers.
- [x] Emit per-query hit@5, MRR, result IDs, hard-negative violation deltas, and promotion blockers.
- [x] Group summaries by planner intent.
- [x] Add JSON and Markdown report writing.
- [x] Add CLI: `python -m neural_search.intelligence.evaluation`.
- [x] Add `make query-plan-eval`.

## Next Slice

- Add suite-aware dataset loading for `real_v07` instead of the demo seed fallback.
- Integrate human-labeled metrics from Task 20.
- Add promotion recommendations per intent once enough judged queries exist.
