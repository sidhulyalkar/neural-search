# Codex Prompt - Backend, Retrieval, and Corpus Expansion

You are working in the `neural-search` repo as a backend/data systems engineer. Focus on ingestion, retrieval, evaluation, persistence, and tests. Do not spend time on visual polish except where API contracts require it.

## Goal

Turn Neural Search from a demo seed corpus into a real corpus compiler and measurable hybrid retrieval system for neuroscience datasets.

## Constraints

- Keep existing tests passing.
- Add tests for every new ingestion/retrieval behavior.
- Preserve SQLite/demo mode.
- No paid API requirements.
- Save raw source payloads for provenance/debugging.
- Normalize source records into stable internal schemas.
- Make everything runnable from CLI first, API second.

## Tasks

### 1. Fix packaging/CLI

`pyproject.toml` declares `neural-search = "neural_search.cli:main"`, but `neural_search/cli.py` is missing.

Implement a small Typer-free or argparse-based CLI with subcommands:

```bash
neural-search demo
neural-search search "go/nogo calcium imaging"
neural-search benchmark
neural-search report
neural-search ingest dandi --query "go no-go calcium imaging" --limit 25 --save
neural-search ingest openneuro --query "iEEG BCI motor" --limit 25 --save
neural-search ingest openalex --query "reversal learning electrophysiology" --limit 25 --save
```

Keep dependencies minimal. Use existing modules.

### 2. Build ingestion service layer

Add:

```text
neural_search/ingestion/services.py
```

Create typed dataclasses or Pydantic models:

```python
class IngestionRunResult:
    source: str
    query: str
    fetched: int
    normalized: int
    saved: int
    skipped: int
    raw_response_paths: list[str]
    warnings: list[str]
    dataset_ids: list[str]
    paper_ids: list[str]
```

Wrap existing connector modules. Make functions deterministic and testable by allowing injected payloads/client functions.

### 3. Wire API ingestion endpoints

Replace stub responses in `apps/api/main.py` with calls to the ingestion service. API response should include source, counts, warnings, and raw paths. Validate query/limit. Do not make source-specific code messy inside endpoint functions.

### 4. Improve retrieval benchmark harness

Enhance benchmark query schema with:

- `expected_dataset_ids`
- `expected_task_labels`
- `expected_modalities`
- `expected_regions`
- `hard_negative_dataset_ids`
- `analysis_intent`
- `minimum_precision_at_5`
- `minimum_label_recall_at_10`

Reports should show:

- precision@1/3/5/10
- recall@k where expected IDs exist
- label recall
- MRR
- NDCG if practical
- top false positives
- missed expected datasets
- per-query `why_failed`

### 5. Add hybrid embedding interface

Do not require sentence-transformers in base tests. Add a provider interface:

```python
class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
```

Implement:

- `HashingEmbeddingProvider` for deterministic offline tests.
- Optional `SentenceTransformerProvider` behind optional dependency.

Use embeddings as one component in the score, not as the only score.

### 6. Add hard-negative tests

Add tests like:

- Query for Go/NoGo should not rank pure visual decision datasets above actual inhibition datasets.
- Query for human ECoG BCI should not return rodent calcium imaging as top result.
- Query for reversal learning with reward omission should prioritize PRL/reversal datasets and penalize unrelated reward tasks.

## Required commands before completion

```bash
pytest -q
ruff check neural_search apps/api scripts tests
python -m neural_search.evaluation.run_benchmark
python -m neural_search.reports
```

## Deliverables

- Code changes.
- Tests.
- Updated docs for ingestion and retrieval.
- Summary of benchmark deltas.
- Any known limitations or remaining TODOs.
