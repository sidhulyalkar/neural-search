# Codex Prompt: Implement Neural Search Backend Core

Implement the backend/package core for Neural Search.

Do not focus on frontend in this pass.

## Tasks

### 1. Database models

Create SQLAlchemy models:

- Dataset
- DatasetAsset
- Paper
- OntologyTerm
- DatasetCard
- Embedding
- SearchLog

Use UUID primary keys.
Use JSONB for flexible metadata.
Use ARRAY fields where appropriate or JSONB if portability is easier.
Prepare pgvector integration but allow a fallback if pgvector is unavailable.

### 2. Pydantic schemas

Create schemas for:

- DatasetCreate
- DatasetRead
- DatasetAssetRead
- PaperRead
- SearchRequest
- SearchResponse
- SearchResult
- DatasetCardRead
- NotebookGenerationResponse
- ExtractionResult
- OntologyTermRead

### 3. Ontology loader

Implement:

- load_ontology(path)
- get_all_tasks()
- get_task_by_id(task_id)
- expand_query_terms(query)
- match_tasks(text)
- match_behavior_labels(text)

Return evidence and confidence.

### 4. Extraction

Implement deterministic extraction from text and metadata.

Input:
- title
- description
- file paths
- source metadata
- linked paper abstracts

Output:
```json
{
  "tasks": [{"id": "...", "label": "...", "confidence": 0.0, "evidence": "..."}],
  "behaviors": [],
  "modalities": [],
  "brain_regions": [],
  "species": [],
  "data_standards": [],
  "missing_fields": []
}
```

### 5. Readiness scoring

Implement `compute_analysis_readiness(dataset, extraction, linked_papers)`.

Score:
- NWB/BIDS standard: +20
- behavior found: +15
- trial/event structure found: +15
- modality found: +10
- task found: +10
- brain region found: +10
- linked paper found: +10
- license found: +5
- processed data found: +5

Return:
- score
- strengths
- limitations

### 6. Dataset cards

Implement:

- generate_dataset_card_json()
- generate_dataset_card_markdown()

Include:
- summary
- why matched
- scientific labels
- analysis readiness
- suggested analyses
- missing metadata
- provenance

### 7. Notebook generation

Implement:

- generate_nwb_starter_notebook(dataset, asset, output_path)

Use nbformat.
Make sure output is a valid `.ipynb`.

Notebook sections:
- title/metadata
- imports
- load NWB with PyNWB
- session metadata
- acquisition objects
- processing modules
- units table if available
- trials table if available
- event column summary
- simple placeholder plots

### 8. Search skeleton

Implement:

- parse_query(query)
- score_dataset_against_query(dataset, card, parsed_query)
- search_datasets(query, filters)

Return explainable result objects with:
- score
- why_matched
- warnings
- dataset_card_preview

### 9. CLI

Add CLI commands:

```bash
python -m neural_search.ontology validate data/ontology/behavioral_task_ontology.yaml
python -m neural_search.ingestion.demo_seed
python -m neural_search.cards.generate_all
python -m neural_search.search.run "Find reversal learning datasets with reward omission"
python -m neural_search.notebooks.generate --dataset-id DEMO --asset-id DEMO
```

### 10. Tests

Add tests for:

- ontology loading
- synonym matching
- behavior label matching
- readiness scoring
- card generation
- notebook generation
- search scoring

## Constraints

- Do not require live API calls in tests.
- Use fixtures.
- Keep modules clean.
- Use type hints.
- Do not invent claims in generated cards without provenance.
- Store confidence/evidence with every extracted label.
