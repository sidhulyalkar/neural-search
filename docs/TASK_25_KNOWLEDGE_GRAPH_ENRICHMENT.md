# Task 25: Knowledge Graph Enrichment

Status: planned

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
- [ ] Add `analysis_requires_modality`, `analysis_requires_behavioral_event`, and `analysis_requires_task_structure` edges from the deterministic awareness taxonomy.
- [ ] Add graph report sections for unresolved placeholder nodes and weak dataset-paper links.
- [ ] Add tests that verify required graph edges exist for representative records.

## Acceptance Criteria

- Graph reports show node and edge coverage by relationship type.
- Search explanations can cite graph edges with evidence and provenance.
- Missing graph relationships become explicit tasks instead of silent ranking failures.
- Existing graph build APIs remain backward compatible.
