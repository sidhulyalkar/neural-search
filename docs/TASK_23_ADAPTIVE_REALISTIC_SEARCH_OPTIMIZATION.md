# Task 23: Adaptive Realistic Search Optimization

Status: planned

## Goal

Make Neural Search more realistic, smarter, and more adaptable by grounding ranking changes in real corpus coverage, human relevance judgments, query-plan evaluation, and intent-specific promotion gates.

## Current Audit Finding

- The search intelligence wrapper is hard-negative safe on the tiny `real_v07` suite.
- The current `real_v07` evaluation uses only 3 normalized dataset records and 3 benchmark queries.
- Intelligence deltas are neutral, not yet evidence for default promotion.
- Promotion is correctly blocked by insufficient query count and disabled manifest gates.

## Optimization Strategy

1. Expand real evaluated coverage before tuning weights.
2. Build human-labeled review sets for critical gaps: computational models, connectomics, intracellular ephys, and molecular/single-cell data.
3. Calibrate retrieval by intent instead of using one global ranking policy.
4. Keep hard-negative filtering as a non-negotiable safety gate.
5. Measure baseline, awareness, and intelligence wrappers before any default promotion.
6. Promote intent by intent, with rollback notes and release checks.

## First Implementation Slices

- Add per-intent human-label gates after the first reviewed judgment files exist.
- Extend query-plan evaluation with modality/data-form group summaries.
- Add calibration reports comparing score distributions against human judgments.
- Add a realistic search fixture set covering fMRI, MEG, connectomics, molecular, intracellular ephys, computational models, and clinical data.
- Add source-aware ranking diagnostics for DANDI, OpenNeuro, OpenAlex, ModelDB, cellxgene, MICrONS, and curated landmark records.

## Acceptance Criteria

- Ranking changes are backed by query-plan evaluation and human relevance summaries.
- No promotion can increase hard-negative violations.
- Each promoted intent has enough evaluated queries and reviewed judgments.
- Search reports make corpus gaps visible instead of hiding them behind generic semantic scores.
