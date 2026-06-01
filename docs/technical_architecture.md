# Technical Architecture

Neural Search is organized as a demo-ready full stack:

- FastAPI backend in `apps/api`.
- React/Vite frontend in `apps/web`.
- Core Python package in `neural_search`.
- YAML-backed ontology, seed data, benchmark queries, notebook templates, and reports in `data`.

## Data Flow

1. **Ingestion**
   Demo records are loaded from curated fixtures and seed builders. Each record contains dataset metadata, assets, optional linked papers, and generated card data.

2. **Ontology Loading**
   The behavioral ontology defines tasks, synonyms, common events, modalities, brain regions, behavior labels, and suggested analyses.

3. **Extraction**
   Dataset titles, descriptions, assets, source metadata, and linked paper abstracts are used to extract scientific labels.

4. **Search**
   Query parsing combines free text with structured filters. Ranking uses ontology matches, metadata matches, text evidence, readiness signals, and provenance-oriented warnings.

5. **Dataset Cards**
   Cards summarize experimental structure, neural data, readiness, missing fields, reuse instructions, linked literature, provenance, QA status, and suggested analyses.

6. **Notebooks**
   Template selection generates starter notebooks for inspection and first analysis.

7. **Evaluation and Reports**
   Benchmark queries measure label recovery and precision. Compilation reports summarize corpus coverage and missing metadata.

## Backend Modules

| Module | Responsibility |
| --- | --- |
| `neural_search/ontology` | Load and match task ontology terms |
| `neural_search/search` | Query parsing and hybrid ranking |
| `neural_search/extraction.py` | Scientific label extraction |
| `neural_search/cards` | Dataset-card generation |
| `neural_search/notebooks` | Notebook template matching and generation |
| `neural_search/evaluation` | Benchmark execution and reporting |
| `neural_search/reports` | Corpus compilation report |
| `neural_search/qa` | Dataset-card QA state |
| `apps/api/main.py` | Demo API surface |

## Frontend Pages

| Page | Purpose |
| --- | --- |
| `/` | Search entry point and demo query launcher |
| `/search` | Results, structured query controls, match evidence, empty/error/loading states |
| `/datasets/:id` | Full dataset card, QA, notebook generation, literature, provenance |
| `/ontology` | Ontology browser |
| `/evaluation` | Benchmark report display |
| `/reports` | Corpus compilation report |

## Retrieval Model

The demo ranking is intentionally interpretable:

- Free-text query terms capture user wording.
- Ontology matching normalizes task, behavior, modality, and region synonyms.
- Structured filters constrain results by explicit experimental criteria.
- Metadata boosts prefer datasets with matching source, standard, species, modality, or brain region.
- Readiness and QA signals help identify reusable datasets.
- Provenance and missing metadata remain visible so ranking does not hide uncertainty.

This is a hybrid scientific retrieval surface, not a single-vector nearest-neighbor demo.

## Local Runtime

The local public-demo path uses in-memory seed data loaded at API startup. This keeps the demo reliable without requiring a live database or external archive credentials.

Optional database and Docker targets exist for development, but the recommended public demo is:

```bash
make demo
make api
make web
```

## Production Gaps

Production deployment would need durable indexing, scheduled ingestion, authenticated review workflows, persistent QA state, larger embedding infrastructure, observability, source-specific rate-limit handling, and explicit data governance around generated cards.
