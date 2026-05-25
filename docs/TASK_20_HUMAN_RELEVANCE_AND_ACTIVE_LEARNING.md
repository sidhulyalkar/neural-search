# Task 20: Human Relevance and Active Learning Loop

Status: prepared

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

- Add `neural_search.evaluation.human_labels` or extend the current relevance module if the concurrent Task 15/16 files settle.
- Generate a review queue from `search_coverage_plan.json` and benchmark failure outputs.
- Add fixture tests for exact, relevant, partial, hard-negative, and unknown judgments.
