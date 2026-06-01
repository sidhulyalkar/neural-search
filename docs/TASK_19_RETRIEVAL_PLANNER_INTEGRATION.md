# Task 19: Retrieval Planner Integration

Status: bridge implemented

## Goal

Promote the Task 17 planner from an inspection tool into the main retrieval path without breaking `search_datasets` compatibility.

## Scope

- Add an optional `intelligence` retrieval config section.
- When enabled, compute `SearchIntelligencePlan` inside search parsing.
- Blend planner weights with existing intent-router weights conservatively.
- Add `search_intelligence_plan` to search traces and benchmark debug outputs.
- Keep hashing embeddings as the default CI provider.
- Preserve hard-negative filtering as a required post-parse constraint.

## Acceptance Criteria

- Existing `search_datasets` calls return the same schema unless intelligence config is enabled.
- Main search responses include planner metadata in `parsed_query` when enabled.
- Planner-selected weights improve or preserve benchmark metrics on `demo_v02`, `adversarial`, and `real_v07`.
- Hard-negative violations remain zero in release benchmarks.

## First Implementation Slice

- [x] Add config-only planner application without editing the active search core.
- [x] Add `search_datasets_with_intelligence` as a bridge wrapper.
- [x] Blend planner weights conservatively with supplied retrieval config.
- [x] Preserve hard-negative filtering for constraint-first plans.
- [x] Expose planner metadata in `parsed_query`.
- [x] Add focused tests for config application and wrapper response metadata.

## Remaining Main-Core Integration

- Add disabled-by-default `intelligence.enabled` plumbing to `search_datasets`.
- Persist planner metadata in search traces and benchmark debug outputs.
- Compare benchmark metrics before enabling planner blending by default.
