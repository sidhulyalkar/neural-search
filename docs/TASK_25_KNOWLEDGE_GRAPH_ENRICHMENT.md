# Task 25: Knowledge Graph Enrichment

Status: in progress

## Goal

Make the knowledge graph useful for scientific retrieval, not just artifact inspection, by expanding explicit relationships between datasets, papers, modalities, tasks, species, analysis affordances, standards, and required signals.

## Scope

- Add graph coverage gates for required node and edge types.
- Expand dataset-paper linking with stronger provenance and link-strength labels.
- Add analysis requirement edges from data forms to modalities, events, standards, and signals.
- Add source-specific graph summaries for DANDI, OpenNeuro, OpenAlex, ModelDB, cellxgene, MICrONS, and curated records.
- Keep placeholder nodes visible until normalized records resolve them.

## First Implementation Slices

- [ ] Add graph coverage thresholds to reports and release checks.
- [x] Add `analysis_requires_modality`, `analysis_requires_behavioral_event`, and `analysis_requires_task_structure` edges from the deterministic awareness taxonomy.
- [x] Add a graph requirement report for analysis-to-requirement edges.
- [x] Add requirement-aware graph context to retrieval explanations when graph config is enabled.
- [ ] Add graph report sections for unresolved placeholder nodes and weak dataset-paper links.
- [x] Add tests that verify required graph edges exist for representative records.

## Retrieval Integration Notes

Requirement-aware graph context is opt-in through existing graph retrieval config.
When enabled, search results can include matched analysis requirements inside
`graph_context.requirement_matches` without changing the public `graph_score`
field or existing graph payload keys.

## Acceptance Criteria

- Graph reports show node and edge coverage by relationship type.
- Search explanations can cite graph edges with evidence and provenance.
- Missing graph relationships become explicit tasks instead of silent ranking failures.
- Existing graph build APIs remain backward compatible.
