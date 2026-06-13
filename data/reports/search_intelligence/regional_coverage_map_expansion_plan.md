# Regional Coverage Map Expansion Plan

## Current Baseline

- Generated artifact: `data/reports/regional_map/regional_map.md`
- Machine artifact: `data/reports/regional_map/regional_map.json`
- Review queue: `data/reports/regional_map/regionless_review_queue.json`
- Current local corpus baseline: 625 dataset records
- Verified region coverage: 206 / 625 records, 33.0%
- Regionless records with atlas candidate mentions: 18
- Atlas targets tracked: 69
- Atlas targets verified: 42
- Atlas targets candidate-only: 11
- Atlas targets uncovered: 16
- Regional signal overlay: `data/corpus/enrichment/regional_signals/regional_signal_overlay.jsonl`
- Regional acquisition backlog: `data/corpus/enrichment/regional_signals/regional_acquisition_backlog.json`
- Signal overlay candidates: 42 regionless records, 19 high-confidence records

This layer is intentionally orthogonal to extractor work. Claude's ontology,
DANDI, OpenNeuro, and NWB extraction phases should flow into the same regional
map without changing the reporting contract.

## Slice A: Candidate Review Loop

Goal: turn high-confidence candidate-only mentions into verified labels.

- Use `regionless_review_queue.json` as the reviewer worklist.
- Use `regional_signal_overlay.jsonl` as the higher-yield worklist for
  regionless records with exact aliases, named-collection signals, and compact
  neuroscience shorthand.
- Prioritize records where candidate regions are specific rather than broad:
  `barrel_cortex`, `v2`, `v4`, `dentate_gyrus`, `subiculum`, `vpm`,
  `ventral_tegmental_area`.
- Prioritize signal-overlay groups by acquisition backlog priority:
  `visual_cortex`, `pons`, `basal_forebrain`, `hippocampus`, and
  `spinal_cord` currently have the best confidence-to-effort ratio.
- Treat broad `cortex` hits as weak signals until a child region is also found.
- Add accepted mappings through the extraction pathway Claude is editing, not by
  patching the report output.
- Re-run `make regional-map-build` after every extraction slice and compare:
  `records_with_verified_regions`, `atlas_targets_verified`,
  `atlas_targets_candidate_only`, and `atlas_targets_uncovered`.

## Slice B: Atlas Completeness Targets

Goal: make the regional map complete enough for a first UI heatmap.

- Expand `data/config/regional_map_targets.yaml` only as a reporting target map.
- Keep extraction synonyms in `data/ontology/brain_regions.yaml`.
- Add missing child targets when they become common in the corpus:
  `claustrum`, `septum`, `bed_nucleus_stria_terminalis`, `habenula`,
  `zona_incerta`, `periaqueductal_gray`, `red_nucleus`, and laminar cortical
  terms.
- Add a stable `system` for every target so the UI can group regions into
  cortical, hippocampal, thalamic, basal ganglia, cerebellar, brainstem,
  hypothalamic, olfactory, visual-periphery, spinal-cord, and whole-brain bands.

## Slice C: Provenance and Confidence

Goal: distinguish strong anatomical evidence from weak text mentions.

- Add evidence tiers to the regional map:
  `verified_label`, `structured_metadata`, `file_level_location`,
  `title_description_candidate`, `task_inferred`, and `manual_reviewed`.
- Store counts by tier per region.
- Gate search boosting on verified or structured evidence first.
- Let candidate-only evidence power review queues and UI warnings, not hard
  ranking constraints.

## Slice D: Complete Map View

Goal: power a complete regional map view for exploration and demos.

- Emit a compact frontend-friendly artifact from `regional_map.json`:
  region id, display label, system, verified count, candidate count, species,
  modalities, sources, and example records.
- Add per-system totals and coverage percentages.
- Add query examples for each populated region:
  species + modality + region + analysis affordance.
- Surface uncovered atlas regions as acquisition targets rather than failures.

## Slice E: Coverage Gates

Goal: prevent regressions while corpus ingestion scales.

- Add a lightweight CI test that regional coverage does not drop unexpectedly
  on the checked-in normalized corpus.
- Add a stricter local gate for release builds:
  minimum verified region coverage, minimum atlas target coverage, and maximum
  candidate-only backlog.
- Save historical regional map JSON snapshots to compare extraction phases.

## Near-Term Commands

```bash
make regional-map-build
make regional-signals-build
python -m pytest tests/test_regional_map.py tests/test_coverage_depth_pack.py tests/test_region_precision.py -q
```
