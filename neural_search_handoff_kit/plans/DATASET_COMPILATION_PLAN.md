# Dataset Compilation Plan

## Objective

Build a credible first corpus of neural and behavioral datasets that demonstrates experiment-aware search across task, behavior, modality, brain region, species, data standard, source archive, provenance, and reuse readiness.

## Sources

### DANDI

Primary value: NWB datasets with rich neural data and metadata.

Target queries:

- `go no-go calcium imaging`
- `reversal learning electrophysiology`
- `reward omission calcium imaging`
- `Neuropixels visual decision making`
- `fiber photometry reward choice`
- `motor reaching electrophysiology`
- `two photon mPFC behavior`

Normalize:

- DANDI set ID/version
- title/description
- assets count
- NWB metadata if available
- species
- modalities
- brain regions
- tasks/behavior events
- license
- related publications
- raw response path

### OpenNeuro

Primary value: BIDS, human EEG/iEEG/ECoG/fMRI datasets.

Target queries:

- `iEEG motor BCI`
- `ECoG speech BCI`
- `EEG motor imagery BCI`
- `fMRI decision making reward`
- `human reinforcement learning EEG`
- `BIDS intracranial electrophysiology reaching`

Normalize:

- accession ID
- BIDS modality
- participants count if available
- species/human
- task labels
- data standard
- license
- paper links
- raw response path

### OpenAlex

Primary value: paper-level evidence and dataset-paper linking.

Target queries:

- `go no-go calcium imaging mPFC`
- `probabilistic reversal learning electrophysiology striatum`
- `delay discounting fiber photometry`
- `Neuropixels visual decision making mouse`
- `human ECoG brain computer interface reaching`

Normalize:

- title
- authors
- year
- DOI/OpenAlex ID
- abstract/inverted index if present
- concepts
- referenced datasets if detectable
- evidence snippets

## Corpus target for v0.2

Minimum viable credible corpus:

| Category | Target count |
| --- | ---: |
| DANDI datasets | 50 |
| OpenNeuro datasets | 30 |
| Linked papers | 100 |
| Task labels represented | 20 |
| Modalities represented | 8 |
| Brain regions represented | 12 |
| Dataset cards generated | 80 |
| Reviewed/trusted cards | 20 |

## Normalization workflow

1. Fetch raw payload.
2. Save raw JSON under `data/raw/<source>/<timestamp>-<query>.json`.
3. Normalize records into internal `Dataset`/`Paper` schema.
4. Extract ontology labels with confidence.
5. Compute readiness/missing metadata warnings.
6. Generate dataset card.
7. Add to benchmark candidates.
8. Review top examples manually.

## Manual QA protocol

For each source batch:

- Mark obviously irrelevant records as rejected.
- Mark high-confidence records as trusted.
- Add missing task/region/modality labels when extraction misses them.
- Record why the dataset is reusable or not.
- Add one benchmark query per five trusted datasets.

## Avoid these traps

- Do not treat paper abstracts as datasets.
- Do not over-infer brain region from vague paper terms.
- Do not mark a dataset analysis-ready without license, modality, task, and data standard evidence.
- Do not silently overwrite raw payloads.
- Do not tune retrieval only on easy positive examples.
