# Neural Search MVP

Semantic discovery for neural and behavioral datasets.

## Product promise

Describe the experiment you want. Neural Search finds reusable datasets, papers, and starter analyses.

## MVP

This first version indexes:

- DANDI neurophysiology datasets
- OpenNeuro EEG/iEEG/ECoG/MEG datasets
- OpenAlex papers
- a behavioral task ontology
- generated dataset cards
- NWB starter notebooks

## Local setup

```bash
git clone <repo>
cd neural-search
cp .env.example .env
docker compose up --build
```

## First demo

```bash
make demo
```

## Example searches

```text
Find Go/NoGo datasets with neural recordings and lick events.
Find reversal learning datasets with reward omission.
Find ECoG datasets involving reaching.
Find visual decision-making datasets with Neuropixels recordings.
Find datasets where I can decode choice from neural activity.
```

## Architecture

```text
apps/api      FastAPI backend
apps/web      React frontend
packages      ingestion, ontology, extraction, indexing, cards, notebooks
data          ontology, seeds, benchmark queries
infra         Docker and deployment config
```

## Development priorities

1. Ontology and deterministic extraction
2. Dataset ingestion
3. Dataset-card generation
4. Hybrid search
5. NWB notebook generation
6. Evaluation benchmark
