# Task 13: General Neuroscience Search Intelligence

## Status

In progress.

## Goal

Make Neural Search aware of the broad shapes of neuroscience data, not only the
demo corpus labels. Search should reason about data forms, measurement scales,
species, experimental structure, and analysis requirements across systems,
clinical, cognitive, computational, molecular, and multimodal neuroscience.

## Why This Matters

Task 12 focuses on learned semantic embeddings. Task 13 adds the deterministic
scientific awareness layer that embeddings should cooperate with:

- what kind of data the user is asking for,
- which data forms can support the requested analysis,
- which metadata is missing for responsible reuse,
- where a dataset is cross-modal or only superficially related.

## Workstream A: Data-Form Taxonomy

- [x] Add a canonical taxonomy for major neuroscience data forms.
- [x] Include aliases for ephys, optical imaging, EEG/MEG, MRI, behavior, clinical, molecular, connectomics, and modeling data.
- [x] Track scale, usual file standards, compatible analyses, and complementary data forms.
- [x] Add tests for alias detection across common query language.

## Workstream B: Query Awareness

- [x] Infer requested data forms from free-text queries.
- [x] Infer analysis families such as decoding, event-aligned analysis, connectivity, behavioral modeling, clinical prediction, and molecular profiling.
- [x] Preserve hard-negative language as a separate constraint layer.

## Workstream C: Dataset Awareness Scoring

- [x] Score dataset fit against query awareness without replacing the main retriever.
- [x] Explain matched data forms, matched analysis families, missing requirements, and cross-modal opportunities.
- [x] Support normalized records and legacy dict records.

## Workstream D: Corpus Awareness Reports

- [x] Add a report generator that summarizes data-form coverage, scale coverage, species coverage, and reusable gaps.
- [x] Add `make awareness-report` for demo + real normalized corpora.
- [x] Keep the report deterministic and local-only.

## Next Implementation Slices

- [ ] Wire awareness scores into `search_datasets` as an optional score head once concurrent Task 12 search-core changes settle.
- [ ] Add benchmark queries for underrepresented data forms such as fMRI, MEG, connectomics, single-cell transcriptomics, and closed-loop BCI.
- [ ] Use awareness reports to prioritize real-corpus expansion beyond the current fixture slice.
- [ ] Add ontology mappings from data forms to file-inspection claims.
- [ ] Add UI/API payload fields for `data_form_awareness` and `cross_modal_opportunities`.

## Quality Gates

```bash
pytest -q tests/test_neuroscience_awareness.py
ruff check neural_search/awareness tests/test_neuroscience_awareness.py
python -m neural_search.awareness.report --records data/corpus/normalized --out data/reports/awareness
```
