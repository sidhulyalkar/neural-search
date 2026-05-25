# Task 17: Search Intelligence Planner

Status: completed first implementation slice

## Goal

Turn neuroscience query awareness into an actionable retrieval plan before ranking. The planner should understand data forms, analysis intent, hard negatives, required signals, and cross-modal opportunities across neuroscience data types.

## Completed Slice

- Added `neural_search.intelligence.plan_search_intelligence`.
- Classifies query mode as constraint-first, cross-modal, analysis-readiness, exact-lookup, recall-first, or balanced.
- Emits deterministic retrieval weight recommendations including `awareness`, `field_semantic`, `graph`, and `negative_constraint` heads.
- Carries required/excluded data forms, required signals, complementary data forms, quality checks, and benchmark tags.
- Adds corpus-coverage warnings when a requested data form is thin or absent.

## Next Integration

- Feed the planner into the primary retrieval config once Task 12 semantic-core work settles.
- Persist the plan in search traces and benchmark debug reports.
- Compare planner-selected weights against ablation results from Task 16.
