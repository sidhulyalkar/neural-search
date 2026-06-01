# Claude Code Prompt: Build Neural Search MVP

You are building the first MVP of Neural Search.

## Product promise

Describe the experiment you want. Neural Search finds reusable datasets, papers, and starter analyses.

## Goal

Create a production-quality local MVP that can ingest DANDI, OpenNeuro, and OpenAlex records, normalize them into a shared schema, extract scientific labels using an ontology, generate dataset cards, search datasets by experimental meaning, and generate starter NWB notebooks.

## Important framing

This is not a generic RAG app. It is an experiment-aware neural data discovery system.

The MVP should focus on:
- metadata
- papers
- task labels
- behavioral labels
- modality labels
- dataset cards
- lightweight NWB/BIDS summaries
- notebook generation

Do not implement raw latent neural-state search yet.

## Build the monorepo

Create:

```text
neural-search/
  apps/api/
  apps/web/
  packages/ontology/
  packages/ingestion/
  packages/extraction/
  packages/indexing/
  packages/cards/
  packages/notebooks/
  packages/evaluation/
  data/ontology/
  data/seed/
  data/eval/
  docs/
  infra/
```

## Backend

Use:

- FastAPI
- SQLAlchemy
- Pydantic
- Alembic
- Postgres + pgvector
- Redis + RQ or Celery

Implement:

```text
GET  /healthz
POST /api/search
GET  /api/datasets
GET  /api/datasets/{dataset_id}
GET  /api/datasets/{dataset_id}/card
POST /api/datasets/{dataset_id}/notebook
GET  /api/ontology/tasks
POST /api/ingest/dandi
POST /api/ingest/openneuro
POST /api/ingest/openalex
POST /api/evaluation/run
```

## Frontend

Use:

- Vite
- React
- TypeScript
- Tailwind
- shadcn/ui if available

Pages:

1. Search home
2. Search results
3. Dataset card detail
4. Ontology browser
5. Evaluation/debug page

The UI should feel like a serious scientific tool: dark neural-data aesthetic, clear cards, no corny copy.

## Core features

### 1. Ontology

Load YAML from `data/ontology/behavioral_task_ontology.yaml`.

Implement:
- task lookup
- synonym expansion
- behavior label lookup
- category filtering
- suggested analysis lookup

### 2. Ingestion

Implement connectors:

#### DANDI connector

- Search or list Dandisets
- Fetch metadata
- List assets
- Identify NWB assets
- Save raw metadata snapshot
- Create normalized DatasetRecord

#### OpenNeuro connector

- Use GraphQL API
- Query dataset metadata
- Query snapshot/file-level metadata
- Identify BIDS modalities and task files
- Create normalized DatasetRecord

#### OpenAlex connector

- Search works by dataset title, DOI, task terms, and authors
- Store paper records
- Link candidate papers to datasets

### 3. Extraction

Implement deterministic extraction first.

Extract:
- task labels
- behavior labels
- modality labels
- brain region labels
- species
- data standards
- analysis affordances
- missing fields

Every extraction result should include:
- label
- confidence
- evidence
- source text span when possible

### 4. Dataset cards

Generate:
- structured JSON card
- Markdown card
- analysis readiness score
- strengths
- limitations
- missing metadata warnings
- suggested analyses

### 5. Search

Implement hybrid search:
- keyword search
- ontology matching
- vector search
- metadata filters
- analysis readiness weighting

Each search result must include:
- score
- why_matched
- warnings
- suggested_next_actions

### 6. NWB notebook generator

Use nbformat.

Generate notebook sections:
- install imports
- load NWB with PyNWB
- print session metadata
- inspect acquisition
- inspect processing modules
- inspect units if available
- inspect trials table
- summarize event columns
- plot simple trial/event summary
- TODO cells for event alignment and decoding

### 7. Evaluation

Add `data/eval/benchmark_queries.yaml`.

Implement:
- run benchmark queries
- store results
- precision@k skeleton
- manual relevance annotation file

## Testing

Add unit tests for:

- ontology synonym matching
- extraction from toy metadata
- readiness scoring
- card generation
- notebook generation validity
- search route returns explainable results

## Seed demo

Create a local demo script:

```bash
make demo
```

It should:

1. load ontology
2. ingest small sample/fixture records
3. generate cards
4. index embeddings
5. run example queries
6. generate one starter notebook

## Quality bar

- Typed code
- Clean modules
- Good README
- No hard-coded secrets
- Robust error handling
- Store raw source metadata for provenance
- Avoid hallucinated labels without confidence/evidence
- Build for local development first

## Output

After implementation, provide:

1. setup instructions
2. architecture summary
3. list of implemented files
4. demo commands
5. remaining TODOs
