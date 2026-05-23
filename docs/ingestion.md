# Ingestion

Neural Search ingests source records in two steps: fetch raw payloads for provenance, then normalize
records into the internal dataset or paper schema.

## CLI

Run dry normalization without writing files:

```bash
neural-search ingest dandi --query "go no-go calcium imaging" --limit 25
neural-search ingest openneuro --query "iEEG BCI motor" --limit 25
neural-search ingest openalex --query "reversal learning electrophysiology" --limit 25
```

Persist raw payloads under `data/raw/<source>/` and save normalized records to the local SQLite DB:

```bash
neural-search ingest dandi --query "go no-go calcium imaging" --limit 25 --save
```

The source-specific module CLIs still support `--save-raw` for raw payload capture:

```bash
python -m neural_search.ingestion.dandi --query "go no-go calcium imaging" --limit 5 --save-raw
```

## Service Layer

`neural_search.ingestion.services` provides deterministic functions for API and tests:

- `ingest_dandi`
- `ingest_openneuro`
- `ingest_openalex`
- `ingest_source`

Each returns `IngestionRunResult` with source, query, fetched/normalized/saved/skipped counts, raw
paths, warnings, dataset IDs, and paper IDs. Tests can inject frozen payloads or fetch functions so
default CI never depends on live archive availability.

## API

The FastAPI endpoints call the service layer:

```text
POST /api/ingest/dandi
POST /api/ingest/openneuro
POST /api/ingest/openalex
```

Request body:

```json
{
  "query": "go no-go calcium imaging",
  "limit": 10,
  "save": false,
  "force": false
}
```

`save: true` persists raw payloads and normalized records. `force: true` allows overwriting existing
records during save.
