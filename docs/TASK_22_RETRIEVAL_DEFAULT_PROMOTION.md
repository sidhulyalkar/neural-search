# Task 22: Retrieval Default Promotion

Status: planned

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

- Add a promotion manifest describing enabled intents, weights, gates, and rollback notes.
- Keep all new behavior disabled by default until Task 21 reports are reviewed.
