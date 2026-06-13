# Corpus Expansion Next Slices

Date: 2026-06-12

This plan follows the first coverage-depth implementation pass:

- `data/seed/coverage_depth_datasets.yaml`
- `data/corpus/normalized/coverage_depth/coverage_depth.records.jsonl`
- `data/corpus/normalized/coverage_depth/coverage_depth_report.md`
- `data/corpus/normalized/coverage_depth/coverage_depth.source_targets.json`

## Current Improvement

The reviewed coverage-depth pack adds precise labels for:

- macaque visual cortex V1/V2 imaging
- macaque dmFC, somatosensory area 2, M1, and PMd spiking
- rat mPFC/OFC LFP and spike/ephys records
- mouse dorsolateral/dorsal striatum photometry
- rat ventral striatum controls
- fMRI GLM-ready task-control metadata

It also records the largest remaining gap honestly: exact macaque area MT/MST single-unit visual-motion coverage is still missing.

## Next Source Targets

| Priority | Target | Why | Intake Path | Acceptance |
|---|---|---|---|---|
| critical | Macaque area MT/MST single-unit visual motion | Needed for the flagship failed query. | Search DANDI, CRCNS, NWB examples, linked-paper datasets. | At least 2 reviewed records with `macaque`, `area_mt` or `mst`, `extracellular_ephys`, `spikes`, visual-motion task evidence. |
| high | Rat PFC LFP theta | Prevents somatosensory LFP false positives. | DANDI rich metadata, Buzsaki lab datasets, existing DANDI 000065/000067 enrichment. | At least 5 records with `rat`, `lfp`, PFC child labels, and theta/oscillation evidence. |
| high | Mouse dorsal striatum spike trains | Fixes empty-modality top results. | DANDI ecephys search, GIN/CRCNS candidates, reviewed local DANDI striatum records. | At least 5 records with non-empty ephys/spike modalities and dorsal/ventral specificity. |
| high | BIDS fMRI GLM task datasets | Improves analysis-affordance query scores. | OpenNeuro metadata plus BIDS file inspection for `events.tsv` and task entities. | At least 10 records with `fmri_glm_analysis` medium/high, with rest-fMRI controls labeled low. |
| medium | Macaque visual cortex non-MT controls | Helps ranking distinguish exact MT from neighboring visual areas. | DANDI 000347, OpenNeuro macaque MRI, future visual cortex physiology records. | V1/V2/V4 records remain below exact MT for MT queries. |

## Implementation Slices

### Slice A - Coverage Pack Integration

1. Add coverage-depth records to any local evaluation corpus used for demo search.
2. Add a tiny benchmark file for the four known failure queries.
3. Run retrieval with and without the pack to verify that source coverage changes rank candidates.

Acceptance:

- `make coverage-depth-build` writes loadable normalized records and report.
- Query fixtures can load `coverage_depth.records.jsonl`.
- The MT query report explicitly says whether exact MT evidence exists.

### Slice B - Source Intake Search Lists

Add a machine-readable source target file that tracks candidate searches:

- query string
- archive
- expected species/modality/region
- review status
- accepted normalized record ID or rejection reason

Recommended path:

- `data/seed/coverage_depth_source_targets.yaml`
- `scripts/corpus/review_source_targets.py`

Acceptance:

- Every critical target has at least 5 candidate URLs or source IDs.
- Every accepted candidate has provenance notes.
- Rejected candidates explain the mismatch, such as wrong species, wrong region, summary-only data, or missing raw data.

### Slice C - File-Level Inspection

For BIDS and NWB records, enrich labels from file summaries instead of text alone.

NWB checks:

- `units` or sorted spike tables
- electrodes table brain-region labels
- processing modules for LFP
- trials table columns

BIDS checks:

- `dataset_description.json`
- modality folders
- `events.tsv`
- task entities
- derivatives or contrast/stat-map hints

Acceptance:

- Empty modality records are not promoted for ephys/spike queries.
- `fmri_glm_analysis` high support is based on task/event evidence, not just the word "GLM".

### Slice D - Ranking Validation

Build a small deterministic benchmark around hard negatives:

- macaque MT query vs human Neuropixels
- rat PFC LFP query vs rat somatosensory LFP
- dorsal striatum spike query vs ventral striatum and empty-modality records
- BIDS task GLM query vs rest fMRI and non-BIDS fMRI

Acceptance:

- Species mismatch rate is zero in top 3 for strict macaque queries.
- Strict subregion mismatch is penalized in top 5.
- Empty modality records are below explicit ephys records.
- Rest fMRI is below task fMRI for GLM queries.

## Near-Term Commands

```bash
make coverage-depth-build
python -m pytest tests/test_coverage_depth_pack.py tests/test_region_precision.py -q
```

After benchmark fixtures are added:

```bash
make eval-coverage-depth
```

## Notes For Future Expansion

- Keep reviewed seed packs separate from broad harvested corpus files.
- Do not hide true gaps by widening aliases. Exact MT should beat visual cortex; if exact MT is missing, report it.
- Treat source-target candidates as an audit queue, not automatic truth.
- Prefer small reviewed packs that fix known failure modes over large noisy ingest runs.

