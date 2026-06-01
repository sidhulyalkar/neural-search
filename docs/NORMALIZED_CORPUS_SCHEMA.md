# Normalized Corpus Schema

Neural Search v0.3 uses normalized corpus records as the boundary between real source ingestion and retrieval. DANDI, OpenNeuro, OpenAlex, manually curated records, and future NWB/BIDS inspection outputs should all write into this layer before records are searched, scored, reported, or reviewed.

The goal is not just to store scientific labels. The goal is to preserve why each label exists.

## Why Normalized Records Exist

Source APIs expose different shapes:

- DANDI records emphasize dandiset metadata, assets, and NWB provenance.
- OpenNeuro records emphasize BIDS dataset metadata and snapshot summaries.
- OpenAlex records emphasize paper metadata, abstracts, concepts, and citations.
- Future NWB/BIDS inspectors will expose file-level structure such as trials, units, events, electrodes, and time series.

Normalized records provide one inspectable contract for these sources:

- Stable source-aware IDs.
- Original source IDs and URLs.
- Raw payload references.
- Evidence-backed labels.
- Usability flags for scientific reuse.
- Missing metadata lists instead of silent assumptions.

## Core Models

The Pydantic models live in `neural_search/schemas.py`. Serialization and stable ID helpers live in `neural_search/normalized.py`.

### EvidenceLabel

`EvidenceLabel` is the provenance unit for extracted scientific labels.

```json
{
  "id": "label:modality:neuropixels",
  "label": "Neuropixels",
  "label_type": "modality",
  "confidence": 0.92,
  "evidence_text": "Neuropixels",
  "source_field": "metadata.measurementTechnique",
  "source_value": "Neuropixels extracellular electrophysiology",
  "extractor_name": "neural_search.rule_extractor",
  "extractor_version": "v0.3.0"
}
```

Validation rules:

- `confidence` must be between `0.0` and `1.0`.
- `id`, `label`, `label_type`, `extractor_name`, and `extractor_version` cannot be empty.
- `label_type` is normalized to lowercase underscore form, for example `Data Standard` becomes `data_standard`.

### Confidence Scale

Use confidence conservatively:

| Range | Meaning |
|-------|---------|
| `0.90-1.00` | Structured metadata or controlled-vocabulary match. |
| `0.75-0.90` | Strong curated synonym or common abbreviation. |
| `0.60-0.80` | Free-text phrase match in title, description, abstract, or README. |
| `0.35-0.60` | Weak inference that should be visibly uncertain. |
| `0.00-0.30` | Unsupported candidate; normally do not include as a positive label. |

## Dataset Example

```json
{
  "dataset_id": "dataset:dandi:000026",
  "source": "dandi",
  "source_id": "000026",
  "title": "Mouse Neuropixels decision task",
  "description": null,
  "url": "https://dandiarchive.org/dandiset/000026",
  "raw_payload_path": "data/raw/dandi/000026.json",
  "species": [],
  "modalities": [
    {
      "id": "label:modality:neuropixels",
      "label": "Neuropixels",
      "label_type": "modality",
      "confidence": 0.92,
      "evidence_text": "Neuropixels",
      "source_field": "metadata.measurementTechnique",
      "source_value": "Neuropixels extracellular electrophysiology",
      "extractor_name": "fixture",
      "extractor_version": "v0.3.0"
    }
  ],
  "brain_regions": [],
  "tasks": [],
  "behavioral_events": [],
  "analysis_goals": [],
  "data_standards": [],
  "file_formats": [],
  "linked_papers": [],
  "usability_flags": {
    "has_trials": null,
    "has_behavior": null,
    "has_neural_data": true,
    "has_continuous_behavior": null,
    "has_event_timestamps": null,
    "has_raw_data": true,
    "has_processed_data": null,
    "has_standard_format": true
  },
  "missing_fields": ["description"],
  "created_at": "2026-05-23T00:00:00+00:00",
  "extractor_version": "v0.3.0"
}
```

## Paper Example

```json
{
  "paper_id": "paper:openalex:W123456789",
  "source": "openalex",
  "source_id": "W123456789",
  "title": "Neuropixels recordings during decision making",
  "abstract": null,
  "doi": "https://doi.org/10.0000/example",
  "url": "https://openalex.org/W123456789",
  "year": 2024,
  "authors": ["Demo Author"],
  "linked_datasets": ["dataset:dandi:000026"],
  "extracted_labels": [
    {
      "id": "label:modality:neuropixels",
      "label": "Neuropixels",
      "label_type": "modality",
      "confidence": 0.82,
      "evidence_text": "Neuropixels recordings",
      "source_field": "title",
      "source_value": "Neuropixels recordings during decision making",
      "extractor_name": "fixture",
      "extractor_version": "v0.3.0"
    }
  ],
  "raw_payload_path": "data/raw/openalex/W123456789.json",
  "created_at": "2026-05-23T00:00:00+00:00",
  "extractor_version": "v0.3.0"
}
```

## Stable IDs

Use helpers from `neural_search.normalized`:

```python
make_dataset_id("dandi", "000026")
make_dataset_id("openneuro", "ds004148")
make_paper_id("openalex", "W123456789")
make_evidence_label_id("modality", "Neuropixels")
```

Examples:

- `dataset:dandi:000026`
- `dataset:openneuro:ds004148`
- `paper:openalex:W123456789`
- `label:modality:neuropixels`

ID components are deterministic and sanitized so slashes, spaces, and punctuation inside source IDs become safe tokens.

## Serialization

Use these utilities from `neural_search.normalized`:

- `record_to_dict(record)`
- `record_from_dict(payload)`
- `write_json(record, path)`
- `read_json(path)`
- `write_jsonl(records, path)`
- `read_jsonl(path)`

JSON is useful for individual fixture records. JSONL is useful for normalized corpus batches under `data/corpus/normalized`.

## Ingestion Usage

Ingestion modules should:

1. Fetch or receive a raw payload.
2. Save the raw payload when requested.
3. Generate a stable normalized ID from `source` and `source_id`.
4. Extract labels only when evidence exists.
5. Store each label as an `EvidenceLabel`.
6. Fill usability flags conservatively.
7. Record missing fields explicitly.
8. Write `NormalizedDatasetRecord` or `NormalizedPaperRecord` to JSON/JSONL.

Existing database-oriented ingestion can continue to write legacy `Dataset` and `Paper` rows. The normalized schema is the contract for v0.3 corpus artifacts and future migration.

## Future Consumers

Retrieval, scoring, and reporting will consume normalized records to:

- Explain matched labels with evidence snippets.
- Weight high-confidence structured metadata above weak text matches.
- Penalize or disclose missing metadata.
- Build corpus coverage and label confidence reports.
- Link datasets and papers with evidence.
- Feed future NWB/BIDS inspection outputs into the same search path.
