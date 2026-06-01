# Neural Search Corpus

This directory contains the dataset corpus for Neural Search.

## Files

- `demo_neural_datasets.yaml` - Primary demo corpus with 25+ datasets

## Dataset Types

| source_type | Description |
|-------------|-------------|
| `real_public` | Actual public datasets from DANDI, OpenNeuro, etc. |
| `curated_demo` | Scientifically plausible demos with realistic metadata |
| `synthetic_demo` | Synthetic examples for testing edge cases |

## Schema

Each dataset record contains:

### Required Fields
- `dataset_id` - Unique identifier
- `title` - Human-readable title
- `description` - Detailed description
- `source_type` - One of real_public, curated_demo, synthetic_demo
- `source_name` - demo, dandi, openneuro, etc.
- `species` - List of species
- `modalities` - List of recording modalities
- `tasks` - List of behavioral tasks
- `data_standards` - List of data formats (NWB, BIDS, etc.)

### Optional Fields
- `source_url` - URL to original dataset
- `brain_regions` - List of recorded brain regions
- `behavioral_events` - List of event types in data
- `analysis_goals` - Suggested analysis types
- `file_formats` - File types included
- `keywords` - Additional search terms
- `synonyms` - Alternative names
- `scientific_relevance` - Why this dataset is useful
- `expected_queries` - Queries that should find this dataset
- `negative_queries` - Queries that should NOT find this dataset
- `limitations` - Known issues or gaps
- `provenance` - Source and processing info

## Coverage

The demo corpus covers:

### Tasks (15+)
- Go/NoGo, response inhibition
- Reversal learning, probabilistic reversal
- Delay discounting, intertemporal choice
- Working memory, delayed match-to-sample
- Motor imagery, BCI control
- Reaching, grasping
- Spatial navigation
- Speech production, auditory processing
- Pavlovian/operant conditioning
- Foraging, explore-exploit

### Brain Regions (20+)
- Prefrontal: mPFC, OFC, ACC, DLPFC
- Motor: M1, premotor, SMA
- Visual: V1, higher visual areas
- Hippocampus: CA1, CA3, DG
- Striatum: dorsal, ventral, NAc
- Auditory: A1, auditory cortex
- Clinical: temporal lobe, amygdala

### Modalities (15+)
- Neuropixels, extracellular ephys
- Calcium imaging, two-photon, miniscope
- Fiber photometry
- ECoG, iEEG, sEEG
- EEG
- fMRI
- Behavior video, pose tracking (DLC, SLEAP)
- Pupil tracking, facemap

### Species
- Mouse
- Rat
- Human
- Non-human primate (macaque)

### Data Standards
- NWB (Neurodata Without Borders)
- BIDS (Brain Imaging Data Structure)
- DANDI archive
- OpenNeuro

## Adding New Datasets

1. Use the schema above
2. Include multiple synonyms for brain regions
3. List relevant analysis goals
4. Add expected queries that should match
5. Run benchmark to verify indexing

## Future: Real Public Datasets

The ingestion pipeline can populate this corpus from:
- DANDI Archive API
- OpenNeuro API
- OpenAlex paper metadata

Manual curation will be replaced with automated ingestion as APIs mature.
