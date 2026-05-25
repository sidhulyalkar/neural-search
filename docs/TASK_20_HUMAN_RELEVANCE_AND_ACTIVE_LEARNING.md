# Task 20: Human Relevance and Active Learning Loop

Status: first implementation slice complete

## Goal

Turn search failures and coverage gaps into a human-review loop that improves benchmarks, corpus priorities, and ranking behavior across neuroscience data forms.

## Scope

- Define a JSONL relevance judgment schema for query-result pairs.
- Add deterministic utilities for precision, recall, and disagreement metrics from human labels.
- Use Task 18 coverage gaps to prioritize query labels.
- Use benchmark failures to generate review queues.
- Produce reviewer-facing Markdown/JSON reports with top uncertain results.

## Acceptance Criteria

- Relevance labels can be loaded, validated, and summarized without network or model dependencies.
- Review queues identify under-covered data forms and ambiguous result pairs.
- Human-labeled metrics can be reported separately from synthetic benchmark expectations.
- No search-ranking changes depend on unlabeled human feedback until labels are explicitly promoted.

## First Implementation Slice

- [x] Add deterministic relevance judgment loading and summaries.
- [x] Generate review queues from coverage reports and benchmark seed YAML.
- [x] Write JSON and Markdown review queue reports.
- [x] Add CLI: `python -m neural_search.intelligence.review`.
- [x] Add `make human-review-queue`.
- [x] Add fixture tests for positive and hard-negative labels plus queue generation.

## Next Implementation

- Merge this review queue with benchmark failure output once Task 16 ablation work settles.
- Add reviewer assignment, timestamps, and promotion rules for validated labels.
- Keep human labels separate from benchmark expectations until explicitly promoted.
