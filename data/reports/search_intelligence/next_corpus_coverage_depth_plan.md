# Next Corpus Coverage and Query Depth Plan

Date: 2026-06-12

## Goal

Improve Neural Search on deep neuroscience queries where excellence depends on specific species, brain regions, recording modality, and analysis affordance, not just broad semantic overlap.

Primary target failures:

- `macaque single-unit area MT visual motion` returns human Neuropixels because macaque visual-motion coverage is thin and MT is not represented as a precise region.
- `rat LFP prefrontal cortex theta` ranks rat LFP somatosensory above prefrontal because region precision is too coarse and weakly enforced.
- `ephys spike trains dorsal striatum mouse` surfaces records with empty modalities because corpus extraction misses usable spike-train/ephys labels.
- `BIDS fMRI GLM` queries score lower than expected because GLM/first-level/contrast-design affordances are not strong query or ranking signals.

Observed local corpus clues from `data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl`:

- About 14 macaque/primate text hits, mostly motor cortex, premotor cortex, macaque visual cortex V1/V2 imaging, and one macaque OpenNeuro MRI record.
- About 44 visual-motion/MT-style text hits, but these are not normalized into `area_mt`, `mst`, or visual-motion analysis concepts.
- About 54 prefrontal/PFC text hits, but `prefrontal_cortex`, `mPFC`, `OFC`, `ACC`, and dorsomedial frontal cortex are inconsistently represented.
- About 6 striatum text hits, with dorsal/ventral striatum often collapsed to `striatum`.
- About 319 fMRI/BIDS/GLM-style text hits, so fMRI is more a routing/affordance issue than a raw coverage issue.

## Success Criteria

This sprint is successful when these four demonstration queries have credible, explainable top results:

| Query | Expected behavior |
|---|---|
| macaque single-unit area MT visual motion | Top 5 includes macaque/primate visual-motion extracellular/single-unit records; human Neuropixels is demoted as species mismatch. |
| rat LFP prefrontal cortex theta | Top 3 includes rat LFP/ephys with mPFC/OFC/ACC/PFC evidence; somatosensory-only results are below prefrontal matches. |
| ephys spike trains dorsal striatum mouse | Top 5 results have non-empty ephys/spike modality labels and dorsal striatum or striatum-subregion evidence. |
| BIDS fMRI GLM first-level contrast | Top 5 prioritizes BIDS fMRI task datasets with events/condition labels or derivatives; generic rest fMRI and non-BIDS datasets are demoted. |

Quantitative gates:

- Region extraction recall on curated challenge strings: at least 90 percent.
- Region hard-negative violation rate for strict region queries: at most 10 percent in top 5.
- Species mismatch rate for macaque strict queries: 0 in top 3 unless no macaque candidates exist.
- Empty-modality rate for ephys queries: 0 in top 5.
- `BIDS fMRI GLM` benchmark score closes the current gap by at least 30 percent relative to the baseline score delta.

## Implementation Plan

### Phase 1 - Region and Species Precision

Add a first-class region lexicon and stop treating all region matches as flat Jaccard matches.

Implementation tasks:

1. Create `data/ontology/brain_regions.yaml` with canonical IDs, aliases, parents, and strictness tags.
2. Move `BRAIN_REGION_SYNONYMS` from `neural_search/extraction.py` into a loader-backed ontology while preserving the current dictionary as fallback.
3. Add canonical regions:
   - `area_mt`: MT, middle temporal, V5, middle temporal visual area.
   - `mst`: MST, medial superior temporal, dorsal MST.
   - `v1`, `v2`, `v4`, plus parent `visual_cortex`.
   - `prefrontal_cortex`, `mPFC`, `OFC`, `ACC`, `dlPFC`, `dmFC`, `PMd`.
   - `dorsal_striatum`, `dorsolateral_striatum`, `dorsomedial_striatum`, `ventral_striatum`, `nucleus_accumbens`, `caudate`, `putamen`.
   - `somatosensory_area_2` and parent `somatosensory_cortex`.
4. Add parent-child expansion rules:
   - Querying `prefrontal cortex` can match mPFC/OFC/ACC/dlPFC/dmFC with partial credit.
   - Querying `mPFC` requires exact or child match; generic PFC gets partial credit only.
   - Querying `area MT` should not be satisfied by generic `visual_cortex` unless no exact candidates exist, and then only with a warning.
   - Querying `dorsal striatum` should outrank generic `striatum` and demote `ventral_striatum`.
5. Add tests in `tests/test_extraction.py` and `tests/test_retrieval_query_parsing.py` for MT, MST, dmFC, PMd, dorsal striatum, ventral striatum, and somatosensory area 2.

Acceptance checks:

- `extract_dataset_labels("macaque area MT visual motion single-unit")` includes `macaque`, `area_mt`, and `extracellular_ephys`.
- `extract_dataset_labels("dorsomedial frontal cortex")` maps to `dmFC` and parent `prefrontal_cortex`.
- Query parsing identifies strict region constraints when the query contains a named subregion.

### Phase 2 - Coverage Pack for Macaque, Rat PFC, and Striatum

Add a small, high-quality coverage pack before scaling broad intake. The point is to fix the demo failures with reviewed records, not just increase corpus size.

Implementation tasks:

1. Add `data/seed/coverage_depth_datasets.yaml` with 20 to 30 reviewed records across:
   - Macaque visual motion single-unit or extracellular ephys, including area MT/MST/V1/V4 when available.
   - Macaque frontal/premotor/motor controls already visible in DANDI, such as current DANDI macaque motor records, normalized with better region labels.
   - Rat prefrontal LFP/ephys, including mPFC/OFC/ACC where metadata supports it.
   - Mouse dorsal striatum spike/ephys/photometry records with dorsal vs ventral evidence.
   - BIDS fMRI task datasets with events, contrasts, first-level GLM suitability, or derivatives.
2. Extend `scripts/corpus/enrich_dandi_metadata.py` or add `scripts/corpus/enrich_region_depth.py` to re-normalize only records whose title/description contains high-value region aliases.
3. Ensure each record has:
   - `species`
   - `modalities`
   - `brain_regions`
   - `data_standards`
   - `url`
   - provenance evidence text for region and modality
4. Add a report `data/reports/search_intelligence/coverage_depth_report.md` listing new coverage counts by species, modality, and brain-region subfamily.

Initial local records to revisit:

- `dataset:dandi:000347` - macaque visual cortex V1/V2 calcium imaging.
- `dataset:dandi:000130` - macaque dorsomedial frontal cortex spiking.
- `dataset:dandi:000127` - macaque somatosensory area 2 spiking.
- `dataset:dandi:000128`, `000138`, `000139`, `000140` - macaque M1/PMd spiking.
- `dataset:dandi:000067` - rat medial prefrontal cortex ephys/LFP.
- `dataset:dandi:000065` - polymer probe LFP in OFC, NAc, mPFC, hippocampus.
- `dataset:dandi:000559` - mouse dorsolateral striatum dopamine signals.
- `dataset:dandi:000476`, `000546` - rat ventral striatum controls.

Acceptance checks:

- At least 8 macaque records have non-empty modalities and regions.
- At least 4 macaque records have visual-region labels, with at least 1 exact MT/MST candidate or an explicit `coverage_gap: no_exact_mt_record` note.
- At least 5 rat PFC/LFP/ephys records or reviewed candidates exist.
- At least 5 striatum records distinguish dorsal, ventral, or generic striatum.
- All new records have source URLs and extractor/provenance notes.

### Phase 3 - Analysis Affordance Upgrade for BIDS fMRI GLM

Make GLM a first-class analysis affordance rather than an accidental text match.

Implementation tasks:

1. Add `fmri_glm_analysis` to `neural_search/analysis_affordances.py`.
2. Add `FMRI_GLM_ANALYSIS` to `neural_search/affordances/registry.py`.
3. Add query aliases to `data/config/intent_profiles.yaml` and `data/config/retrieval.yaml`:
   - GLM
   - general linear model
   - first-level model
   - design matrix
   - contrast
   - beta maps
   - BOLD events
   - task fMRI
4. Define support levels:
   - High: fMRI + BIDS + events.tsv/task events + subject/session count or derivatives.
   - Medium: fMRI + BIDS + task labels, but no explicit event/condition evidence.
   - Low: fMRI only, or rest fMRI without task events.
   - Unsupported: non-fMRI or no analysis-ready evidence.
5. Update `neural_search/affordances/validators/bids_validator.py` to expose feature checks for events files, task entities, contrast/derivatives hints, and BOLD modality.
6. Add tests in `tests/test_analysis_affordances.py`, `tests/test_affordance_registry.py`, and `tests/test_affordance_validators.py`.

Acceptance checks:

- A BIDS fMRI task dataset with `events.tsv` gets `fmri_glm_analysis: high`.
- Resting-state fMRI does not get high GLM support unless explicit task/contrast evidence exists.
- Query intent classification routes `BIDS fMRI GLM first-level contrast` to `analysis_affordance` or a new `fmri_glm` profile.

### Phase 4 - Constraint-Aware Ranking

Use strict species, modality, and region requirements when a query is specific.

Implementation tasks:

1. Extend the query parser to emit structured constraints:
   - `required_species`
   - `required_modalities`
   - `required_brain_regions`
   - `required_affordances`
   - `region_strictness`
2. Add a region score component that supports exact, child, parent, sibling, and mismatch scoring.
3. Add penalties:
   - strict species mismatch: strong demotion.
   - strict region mismatch: strong demotion.
   - empty modality on modality-specific query: strong demotion.
   - broad parent-only region match: mild demotion when an exact subregion was requested.
4. Update field weights:
   - Increase `brain_regions` field embedding weight from 0.08 to 0.14 for strict region queries.
   - Increase graph `brain_region_match` from 0.02 to 0.05 for strict region queries.
   - Increase affordance weight for `analysis_affordance` and `fmri_glm` queries.
5. Ensure explanation cards show:
   - matched region evidence
   - parent/child relationship if partial
   - missing exact-region warning
   - modality evidence source

Acceptance checks:

- Human Neuropixels cannot outrank macaque visual ephys on `macaque single-unit area MT visual motion` unless macaque candidates are missing required modality evidence.
- Somatosensory cortex cannot rank above PFC for `rat LFP prefrontal cortex theta` when a rat PFC LFP candidate exists.
- Records with empty modalities are below records with explicit ephys/spike evidence for spike-train queries.

### Phase 5 - Benchmark and Failure Harness

Turn the four observed gaps into repeatable regression tests.

Implementation tasks:

1. Add `data/eval/benchmark_queries_coverage_depth.yaml` with the four primary queries plus hard negatives:
   - macaque MT query with human/mouse visual cortex as hard negatives.
   - rat PFC LFP query with somatosensory cortex as hard negative.
   - mouse dorsal striatum spike query with empty-modality and ventral-striatum controls.
   - BIDS fMRI GLM query with resting-state fMRI and non-BIDS fMRI controls.
2. Add a tiny fixture corpus in `tests/fixtures/coverage_depth/records.jsonl` and `qrels.jsonl`.
3. Add tests:
   - `tests/test_coverage_depth_queries.py`
   - `tests/test_region_precision.py`
   - `tests/test_fmri_glm_affordance.py`
4. Extend `scripts/eval/analyze_failures.py` with optional concept-specific summaries:
   - species mismatch
   - region mismatch
   - empty modality
   - affordance miss
5. Add a make target:
   - `make eval-coverage-depth`

Acceptance checks:

- Fixture tests pass without network.
- Real-corpus eval writes `reports/eval/coverage_depth_report.md`.
- The failure report lists the four query families and top false-positive causes.

## Proposed Work Order

1. Implement region ontology and extraction tests.
2. Add the coverage-depth fixture corpus and benchmark tests.
3. Add the manually reviewed coverage seed records.
4. Add GLM affordance detection and BIDS validator features.
5. Tune strict ranking penalties using fixture tests first, real corpus second.
6. Rebuild corpus graph, embeddings, and reports.
7. Run the four demonstration queries and lock their results into a regression report.

## Files To Touch

Likely implementation files:

- `neural_search/extraction.py`
- `neural_search/ontology/loader.py`
- `neural_search/ontology/matcher.py`
- `neural_search/retrieval/constraint_parser.py`
- `neural_search/search/intent.py`
- `neural_search/search/weight_optimizer.py`
- `data/config/intent_profiles.yaml`
- `data/config/retrieval.yaml`
- `neural_search/analysis_affordances.py`
- `neural_search/affordances/registry.py`
- `neural_search/affordances/validators/bids_validator.py`
- `scripts/corpus/enrich_dandi_metadata.py`
- `scripts/eval/analyze_failures.py`

New files:

- `data/ontology/brain_regions.yaml`
- `data/seed/coverage_depth_datasets.yaml`
- `data/eval/benchmark_queries_coverage_depth.yaml`
- `tests/fixtures/coverage_depth/records.jsonl`
- `tests/fixtures/coverage_depth/qrels.jsonl`
- `tests/test_region_precision.py`
- `tests/test_coverage_depth_queries.py`
- `tests/test_fmri_glm_affordance.py`
- `data/reports/search_intelligence/coverage_depth_report.md`

## Risks and Guardrails

- Do not solve MT by making generic visual cortex overmatch. If exact MT coverage is absent, report the coverage gap honestly.
- Do not label all fMRI as GLM-ready. Rest fMRI and task fMRI need different affordance support.
- Do not inflate macaque coverage with human or mouse datasets. Species mismatch should be visible and penalized.
- Do not let broad parent labels erase useful subregions. Store both parent and child labels when evidence supports them.
- Keep all network-backed source expansion outside CI. Tests should use fixtures and reviewed seed records.

## Demo Script After Implementation

Run:

```bash
make eval-coverage-depth
python scripts/run_killer_demo.py --queries data/eval/benchmark_queries_coverage_depth.yaml
```

Expected talking points:

- The system recognizes `area MT` as a precise visual-motion region, not just "visual cortex".
- The system distinguishes rat PFC LFP from rat somatosensory LFP.
- The system penalizes empty modality fields for ephys queries.
- The system understands that BIDS fMRI GLM needs task/event/contrast affordance evidence.

