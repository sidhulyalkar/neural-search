# Task 18: Coverage-Driven Search Expansion

Status: completed first implementation slice

## Goal

Use normalized corpus records and benchmark queries to identify which neuroscience data forms are underrepresented, then produce concrete ingestion and benchmark-query priorities.

## Completed Slice

- Added `neural_search.intelligence.build_search_coverage_plan`.
- Counts data-form coverage in normalized dataset records.
- Counts data-form coverage in benchmark queries.
- Produces prioritized gaps with recommended source families and query seeds.
- Adds JSON and Markdown report output through `write_search_coverage_plan`.
- Adds a CLI: `python -m neural_search.intelligence.coverage`.
- Writes `benchmark_query_seeds.yaml` for human-reviewed benchmark expansion.
- Adds `make search-intelligence-report`.

## Next Integration

- Run this report after corpus builds and benchmark updates.
- Use generated query seeds to expand real-corpus benchmarks for fMRI, MEG, connectomics, molecular, clinical, computational-model, and cross-modal searches.
- Feed high-priority gaps into Task 15 corpus expansion intake.
