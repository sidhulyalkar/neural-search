# Brain Region Knowledge Map Gameplan

## Goal

Expand the knowledge map so it understands brain regions as structured anatomy, not just keyword labels. The map should differentiate regions by hierarchy, aliases, species context, evidence strength, modality compatibility, and functional neighborhood.

## Design Principles

- Treat regions as ontology nodes with stable IDs, parent-child links, synonyms, and external atlas crosswalks.
- Separate verified anatomical evidence from weak text mentions.
- Preserve ambiguity instead of forcing a single label when a term can mean a task, stimulus, region, or broad anatomical family.
- Prefer structured metadata and file-level evidence over title/abstract mentions.
- Make region reasoning queryable: users should be able to ask for hippocampus broadly, CA1 specifically, or hippocampal formation excluding entorhinal cortex.

## Region Model

Each brain region should carry:

- `id`: stable snake_case ID used in records and graph edges.
- `label`: display name.
- `aliases`: common names, abbreviations, spelling variants, and shorthand.
- `parents`: broader anatomical regions.
- `children`: finer subdivisions when represented locally.
- `system`: cortical, hippocampal, thalamic, basal ganglia, cerebellar, brainstem, hypothalamic, olfactory, visual-periphery, spinal-cord, whole-brain, or peripheral.
- `species_scope`: human, mouse, rat, macaque, generic mammal, or cross-species.
- `atlas_refs`: Allen, UBERON, BRAIN Initiative, NeuroNames, or BIDS-style mappings when known.
- `disambiguation_notes`: terms that need context before becoming verified labels.

## Logical Differentiation Rules

1. Hierarchical expansion:
   A query for `hippocampus` should include CA1, CA2, CA3, dentate gyrus, subiculum, and hippocampal formation unless the user asks for exact-match only.

2. Sibling distinction:
   A query for `CA1` should not silently match `CA3` unless the query asks for hippocampal subfields broadly.

3. Broad-region caution:
   Terms like `cortex`, `visual cortex`, `forebrain`, or `brainstem` should be broad labels until a child region is present or a trusted source supplies a specific structured region.

4. Ambiguous-term handling:
   Terms like `V1`, `M1`, `PFC`, `ACC`, `S1`, `VTA`, and `NAcc` should resolve through species, modality, nearby words, and known aliases. Ambiguous matches should enter candidate evidence, not verified evidence.

5. Species-aware mapping:
   Mouse `barrel cortex`, human `primary somatosensory cortex`, and macaque `area V4` need species-aware alias rules so the map does not flatten distinct anatomy into generic cortex.

6. Evidence precedence:
   Use this order for confidence: manual review, structured metadata, file/electrode/location metadata, curated overlay, title/description exact alias, inferred task or modality context.

## Implementation Slices

### Slice 1: Ontology Hardening

- Expand `data/ontology/brain_regions.yaml` with parent-child links and system labels for the high-yield regions already appearing in DANDI and OpenNeuro.
- Add alias collision notes for abbreviations that need context.
- Add tests for exact, alias, parent, child, and ambiguous matches.

### Slice 2: Evidence-Tiered Extraction

- Teach DANDI and OpenNeuro extractors to emit region evidence with `source_field`, `evidence_tier`, and `confidence`.
- Keep candidate-only mentions separate from verified `dataset_records_region` edges.
- Use NWB electrode/location metadata as a high-confidence source when available.

### Slice 3: Graph Semantics

- Add explicit graph edges for `region_parent_of`, `region_alias_of`, and `region_part_of_system`.
- Keep dataset-to-region edges annotated by evidence tier.
- Support query expansion from parent to child regions while preserving exact-match filters.

### Slice 4: Evaluation Set

- Build a small gold set of region-sensitive queries:
  `CA1 electrophysiology`, `visual cortex calcium imaging`, `basal ganglia behavior`, `thalamic relay recordings`, `spinal cord motor datasets`, and `prefrontal fMRI`.
- Include hard negatives where nearby but wrong regions should not pass, such as CA1 vs CA3, V1 vs V4, striatum vs nucleus accumbens, and motor cortex vs motor task.

### Slice 5: Product Surface

- Emit a frontend-friendly regional map artifact with region, system, verified count, candidate count, modalities, species, sources, and examples.
- Show candidate-only regions as review-needed rather than as hard filters.
- Add an "exact region" toggle for users who do not want parent-child expansion.

## First Expansion Targets

- Hippocampal formation: hippocampus, CA1, CA2, CA3, dentate gyrus, subiculum, entorhinal cortex.
- Visual system: V1, V2, V4, visual cortex, retina, lateral geniculate nucleus, superior colliculus.
- Sensorimotor cortex: M1, S1, barrel cortex, premotor cortex, somatosensory cortex.
- Basal ganglia: striatum, dorsal striatum, ventral striatum, nucleus accumbens, globus pallidus, substantia nigra.
- Thalamus: thalamus, VPM, LGN, mediodorsal thalamus.
- Brainstem and neuromodulatory nuclei: pons, locus coeruleus, VTA, raphe nuclei, periaqueductal gray.
- Cerebellum: cerebellar cortex, deep cerebellar nuclei, vermis.
- Spinal cord: cervical, thoracic, lumbar, dorsal horn, ventral horn.

## Definition of Done

- Region ontology has parent-child-system structure for the first expansion targets.
- Extraction outputs evidence-tiered region annotations.
- Regional map reports verified vs candidate coverage.
- Tests cover exact match, alias match, hierarchy expansion, ambiguity handling, and hard-negative separation.
- Search can distinguish broad-region, subregion, and exact-region intent.

## Progress: 2026-06-13

- Expanded the brain-region ontology with explicit motor, somatosensory, thalamic, basal ganglia, basal forebrain, brainstem, cerebellar, hypothalamic, and spinal-cord subregions.
- Added optional ontology metadata for region `system`, `species_scope`, `species_aliases`, `atlas_refs`, `children`, and `disambiguation_notes`.
- Moved child-specific aliases such as `M1`, `S1`, `barrel cortex`, `pons`, `medulla`, `cerebellar cortex`, `dorsal horn`, and `ventral horn` toward explicit child nodes instead of broad parent nodes.
- Added species-aware matching for species-scoped aliases, so shorthand such as `M1` or `S1` is interpreted in species context and does not become a global false-positive synonym.
- Added explicit region hierarchy expansion helpers for exact-vs-descendant query behavior.
- Expanded regional reporting targets and regenerated the regional map and regional signal overlay.
- Added tests for species-aware aliases, parent-child inheritance, explicit hippocampal descendant expansion, spinal sibling separation, and regional map product fields.
