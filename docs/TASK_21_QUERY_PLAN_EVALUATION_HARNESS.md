# Task 21: Query Plan Evaluation Harness

Status: planned

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

- Add a small evaluation module that runs a fixed list of queries through baseline and intelligence wrappers.
- Emit per-query deltas and grouped intent summaries.
