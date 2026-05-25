# Task 26: Source Intake Scaling

Status: planned

## Goal

Scale Neural Search beyond a narrow real-corpus fixture by adding source families that cover the major forms of neuroscience data while keeping local builds reproducible.

## Source Families

- DANDI: NWB animal physiology, imaging, behavior, and cellular recordings.
- OpenNeuro: BIDS fMRI, EEG, MEG, iEEG, clinical, and behavioral datasets.
- OpenAlex: paper context, dataset mentions, methods, and findings.
- ModelDB and Open Source Brain: computational and biophysical model records.
- cellxgene and Allen Brain Cell Atlas: single-cell molecular neuroscience.
- MICrONS and NeuroMorpho: connectomics, morphology, and circuit structure.
- Curated landmarks: manually reviewed records for benchmark-critical datasets.

## First Implementation Slices

- [ ] Add fixture-backed normalizers for ModelDB, cellxgene, MICrONS, and curated landmark records.
- [ ] Add source-specific manifest entries with explicit reason-for-inclusion notes.
- [ ] Add normalized examples for undercovered forms before tuning ranking weights.
- [ ] Add source-balance summaries to corpus and search-intelligence reports.
- [ ] Add benchmark seeds with expected IDs only after representative records exist.

## Acceptance Criteria

- Each new source family can build from local fixtures in CI.
- Network-backed expansion remains optional and never required for tests.
- Source counts and data-form counts improve together.
- Corpus expansion produces graph, embedding, benchmark, and review artifacts without changing public search APIs.
