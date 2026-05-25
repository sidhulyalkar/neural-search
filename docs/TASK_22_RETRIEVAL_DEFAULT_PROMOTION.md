# Task 22: Retrieval Default Promotion

Status: first implementation slice complete

## Goal

Promote intelligence and awareness features from wrappers into the default retrieval path only after evaluation and human-review gates show benefit.

## Scope

- Define promotion gates for each planner intent.
- Add config presets for CI, local real corpus, and exploratory research.
- Move planner metadata into standard search traces.
- Decide which score heads become visible in the default UI payload.

## Acceptance Criteria

- Default search remains backward compatible.
- CI uses hashing embeddings and deterministic fixtures.
- Promotion is intent-specific, not all-or-nothing.
- Release checks fail on hard-negative regressions.

## First Slice

- [x] Add `data/config/search_intelligence_promotion.yaml`.
- [x] Keep default promotion disabled.
- [x] Add deterministic promotion gate evaluation from Task 21 reports.
- [x] Emit JSON and Markdown promotion reports.
- [x] Add CLI: `python -m neural_search.intelligence.promotion`.
- [x] Add `make promotion-check`.

## Next Slice

- Add human-label minimums per intent once Task 20 has reviewed judgments.
- Add rollback notes and owner metadata to each intent gate.
- Wire promotion checks into release reporting without failing CI until gates mature.
