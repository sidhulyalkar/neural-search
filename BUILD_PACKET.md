# Neural Search MVP Build Packet

## Project

**Neural Search** is a semantic discovery engine for neural and behavioral datasets.

## Product promise

> Describe the experiment you want. Neural Search finds reusable datasets, papers, and starter analyses.

## Core thesis

Neuroscience data discovery should not stop at filenames, paper keywords, or incomplete metadata. Neural Search should index:

- dataset metadata
- papers and abstracts
- protocols
- task structure
- behavioral labels
- neural modalities
- brain regions
- file standards such as NWB and BIDS
- analysis provenance
- dataset cards
- executable starter notebooks

The MVP is not the full latent-state search moonshot yet. The MVP is the first working substrate: experiment-aware search over open neural datasets and papers.

## MVP v0.1 scope

Seed with:

1. DANDI neurophysiology datasets
2. OpenNeuro EEG, iEEG, ECoG, MEG, and task-BIDS datasets
3. OpenAlex papers linked to dataset titles, DOIs, authors, archive IDs, and task phrases
4. A broad behavioral task ontology
5. Generated dataset cards
6. One NWB notebook generator

## Technical principles

1. Build modularly.
2. Preserve provenance.
3. Never hide uncertainty.
4. Keep raw data indexing out of v0.1.
5. Use metadata, papers, file-level summaries, and lightweight NWB/BIDS inspection first.
6. Every extracted scientific label should include evidence and confidence.
7. Search results should explain why they matched.
8. Dataset cards should summarize what is scientifically reusable.
9. Notebook generation should convert search into action.
10. Evaluation must be part of the MVP, not a future afterthought.

## Architecture

```text
neural-search/
  apps/
    api/
      app/
        main.py
        db/
        models/
        schemas/
        routes/
        services/
        workers/
      tests/
    web/
      src/
        pages/
        components/
        lib/
        types/
  packages/
    ontology/
    ingestion/
    extraction/
    indexing/
    cards/
    notebooks/
    evaluation/
  data/
    ontology/
    seed/
    cards/
    eval/
  infra/
    docker-compose.yml
    Dockerfile.api
    Dockerfile.web
  docs/
    architecture.md
    ontology.md
    data_sources.md
    evaluation.md
```

## Recommended stack

- Backend: FastAPI, SQLAlchemy, Pydantic, Alembic
- Frontend: Vite, React, TypeScript, Tailwind, shadcn/ui
- Database: Postgres + pgvector
- Queue: Redis + Celery or RQ
- Embeddings: sentence-transformers first, swappable embedding backend later
- Search: pgvector + text search first, OpenSearch later
- Data sources:
  - DANDI Python client / API
  - OpenNeuro GraphQL API
  - OpenAlex REST API
- NWB: PyNWB
- Notebook generation: nbformat + Jinja2
- Local orchestration: Docker Compose

## API endpoints

```text
POST /api/search
GET  /api/datasets
GET  /api/datasets/{dataset_id}
GET  /api/datasets/{dataset_id}/card
POST /api/datasets/{dataset_id}/notebook
GET  /api/ontology/tasks
GET  /api/ontology/terms
POST /api/ingest/dandi
POST /api/ingest/openneuro
POST /api/ingest/openalex
GET  /api/evaluation/queries
POST /api/evaluation/run
```

## Core database tables

### datasets

- id
- source
- source_id
- title
- description
- url
- license
- species
- modalities
- brain_regions
- tasks
- behaviors
- data_standards
- has_behavior
- has_trials
- has_raw_data
- has_processed_data
- metadata_json
- created_at
- updated_at

### dataset_assets

- id
- dataset_id
- path
- asset_type
- file_format
- size_bytes
- subject_id
- session_id
- modality
- metadata_json

### papers

- id
- openalex_id
- doi
- title
- abstract
- publication_year
- authors_json
- url
- concepts
- linked_dataset_ids
- metadata_json

### ontology_terms

- id
- term_id
- label
- category
- parent_id
- synonyms
- definition
- examples
- metadata_json

### dataset_cards

- id
- dataset_id
- summary
- why_relevant
- analysis_readiness_score
- strengths
- limitations
- missing_fields
- suggested_analyses
- provenance_json
- card_markdown

### embeddings

- id
- entity_type
- entity_id
- text_for_embedding
- embedding
- embedding_model
- metadata_json

### search_logs

- id
- query
- parsed_intent_json
- result_ids
- result_payload_json
- user_feedback_json
- created_at

## Retrieval flow

```text
natural language query
  ↓
query parser
  ↓
ontology expansion
  ↓
metadata filters
  ↓
keyword search
  ↓
vector search
  ↓
candidate merge
  ↓
reranking
  ↓
analysis-readiness weighting
  ↓
result explanations
```

## Ranking formula

```python
final_score = (
    0.30 * semantic_similarity
    + 0.25 * ontology_match_score
    + 0.20 * metadata_match_score
    + 0.15 * analysis_readiness_score
    + 0.10 * provenance_quality_score
)
```

## Result explanation format

Each result must include:

- matched task terms
- matched modalities
- matched behaviors
- matched brain regions
- source provenance
- linked paper evidence
- analysis readiness explanation
- missing metadata warnings

## Dataset card format

```yaml
dataset_id:
source:
source_url:
title:
one_sentence_summary:
scientific_match_summary:
species:
modalities:
brain_regions:
tasks:
behaviors:
data_standard:
file_types:
linked_papers:
analysis_readiness:
  score:
  strengths:
  limitations:
missing_metadata:
suggested_analyses:
starter_notebook_available:
provenance:
  metadata_sources:
  extraction_version:
  embedding_version:
```

## Analysis readiness scoring

```python
score = 0

if data_standard in ["NWB", "BIDS"]:
    score += 20
if has_behavior:
    score += 15
if has_trials:
    score += 15
if modalities:
    score += 10
if tasks:
    score += 10
if brain_regions:
    score += 10
if linked_papers:
    score += 10
if license:
    score += 5
if has_processed_data:
    score += 5

score = min(score, 100)
```

## Score bands

- 80-100: strong candidate for immediate reuse
- 60-79: promising but needs inspection
- 40-59: interesting but metadata incomplete
- below 40: low confidence or difficult to reuse

## MVP implementation order

### Phase 0: Scaffold

- Create monorepo
- Add backend, frontend, packages, infra, docs
- Add Docker Compose
- Add Postgres + pgvector
- Add Redis
- Add FastAPI healthcheck
- Add React search shell

### Phase 1: Ontology

- Add behavioral task ontology YAML
- Add behavior labels YAML
- Add modality labels YAML
- Add ontology loader
- Add synonym matcher
- Add unit tests

### Phase 2: Ingestion

- DANDI connector
- OpenNeuro connector
- OpenAlex connector
- Normalize source records into common DatasetRecord and PaperRecord models
- Save raw metadata snapshots
- Add ingestion CLI commands

### Phase 3: Extraction

- Deterministic term matching
- LLM-ready extractor interface
- Evidence/confidence output
- Missing-field detection
- Provenance capture

### Phase 4: Dataset cards

- Generate structured dataset cards
- Generate Markdown cards
- Add analysis readiness scoring
- Add suggested analyses from ontology
- Add card page in frontend

### Phase 5: Search

- Embed dataset cards, descriptions, paper abstracts, and ontology terms
- Implement hybrid search
- Implement result explanations
- Add search logs
- Build results UI

### Phase 6: NWB notebook generation

- Generate notebook with nbformat
- Include PyNWB load and inspection cells
- Inspect acquisition, processing modules, units, trials, intervals
- Add downloadable notebook endpoint

### Phase 7: Evaluation

- Add benchmark query file
- Add expected labels
- Add precision@k, recall@k, nDCG@k
- Add manual relevance annotation format
- Add evaluation report generator

## MVP success criteria

The MVP is successful when it can:

1. ingest at least 50 DANDI datasets
2. ingest at least 50 OpenNeuro datasets
3. ingest at least 200 linked or candidate OpenAlex papers
4. generate dataset cards for at least 25 useful datasets
5. answer 20 benchmark queries with sensible ranked results
6. show why each result matched
7. generate one valid NWB starter notebook
8. expose the whole system through a polished web UI
9. preserve provenance and missing-metadata warnings
10. provide enough evidence to support Stanford/faculty outreach or a technical demo

## Demo queries

```text
Find Go/NoGo datasets with neural recordings and lick events.
Find reversal learning datasets with reward omission and trial outcomes.
Find delay discounting datasets with neural activity and behavior.
Find ECoG or iEEG datasets involving reaching or motor control.
Find visual decision-making datasets with Neuropixels recordings.
Find datasets where I can decode choice from neural activity.
Find datasets with behavior video and electrophysiology.
Find datasets suitable for latent state modeling of reward-guided behavior.
Find datasets that include reaction times and visual stimuli.
Generate a starter notebook for this NWB dataset.
```

## Non-goals for v0.1

- Do not train a neural foundation model yet.
- Do not index all raw neural time series.
- Do not build cross-session latent neural similarity yet.
- Do not promise complete ontology coverage.
- Do not hide uncertainty behind polished language.
- Do not rely solely on LLM extraction.
