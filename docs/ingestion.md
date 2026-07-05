# Ingestion

Neural Search ingests source records in two steps: fetch raw payloads for provenance, then normalize
records into the internal dataset or paper schema. The live corpus this produces has 7,171 normalized
records (625 unique datasets) across DANDI, OpenNeuro, NeuroVault, Zenodo, Figshare, NeuroMorpho,
Allen, GIN, and other sources.

Paper-dataset linking is a separate, multi-source pipeline (`neural_search/literature/`): OpenAlex,
DataCite, Crossref, PubMed/bioRxiv, and Semantic Scholar each run their own linker
(`link_corpus_to_*`), writing source-specific JSONL output that `neural_search/graph/paper_node_builder.py`
merges into real `paper` KG nodes. See [Technical Architecture](technical_architecture.md) for how this
feeds the knowledge graph, and [Known Limitations](known_limitations.md) for current per-source coverage.

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

The source-specific module CLIs use the same explicit `--save` behavior:

```bash
python -m neural_search.ingestion.dandi --query "go no-go calcium imaging" --limit 5 --save
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
