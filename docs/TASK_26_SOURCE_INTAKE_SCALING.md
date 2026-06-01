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

- [x] Add fixture-backed source records for ModelDB, cellxgene, and MICrONS.
- [x] Add real_v07 benchmark queries for computational-model, molecular/single-cell, and connectomics lookup.
- [ ] Add fixture-backed curated landmark records for benchmark-critical gaps.
- [ ] Add source-specific manifest entries with explicit reason-for-inclusion notes.
- [ ] Add normalized examples for undercovered forms before tuning ranking weights.
- [ ] Add source-balance summaries to corpus and search-intelligence reports.
- [ ] Add benchmark seeds with expected IDs only after representative records exist.

## Current Fixture Expansion

- ModelDB fixture: `dataset:modeldb:modeldb_87284` for computational model and simulation search.
- cellxgene fixture: `dataset:cellxgene:allen_mouse_motor_cortex_cells` for molecular and single-cell transcriptomics search.
- MICrONS fixture: `dataset:microns:minnie65_public` for connectomics and morphology search.

The local `real_v07` corpus now contains 6 dataset records, 1 paper record, 44 graph nodes, and 100 graph edges after `make real-artifacts-build`.

## Acceptance Criteria

- Each new source family can build from local fixtures in CI.
- Network-backed expansion remains optional and never required for tests.
- Source counts and data-form counts improve together.
- Corpus expansion produces graph, embedding, benchmark, and review artifacts without changing public search APIs.
