# Sprint 1 — Literature at Scale

**Goal:** Index 4.36M neuroscience papers from OpenAlex, link them to existing
datasets, extract structured findings via LLM, and extend search + KG to span
literature alongside datasets.

**Why this first:** 99% of neuroscience knowledge lives in papers, not datasets.
The current 7,171-dataset corpus covers experimental artifacts but almost none of
the *findings* those experiments produced. This sprint adds the knowledge layer.

## Scale (as of 2026-06-17)
- Tier 1 (≥100 citations): 255,940 papers — ingest first, ~hours
- Tier 2 (OA + has abstract): 1,389,240 papers — ingest over days
- Tier 3 (all neuroscience): 4,360,916 papers — eventual full corpus

## Architecture

```
OpenAlex API (cursor pagination)
         ↓
openalex_bulk.py  (checkpoint/resume, rate-limited)
         ↓
data/corpus/normalized/openalex_neuro/   (JSONL shards, one per 10K papers)
         ↓
link_papers_to_datasets.py               (DOI bridge to existing corpus)
         ↓
extract_findings.py  (Claude Haiku batch, structured finding extraction)
         ↓
artifacts/literature/findings_v1.jsonl
         ↓
rebuild_full_corpus_graph.py             (adds paper/finding/venue KG nodes)
         ↓
search/core.py + API                     (literature results in search)
```

## Tasks

| Task | File | Description |
|------|------|-------------|
| 01 | openalex_bulk.py | Cursor paginator + schema extension |
| 02 | bulk_ingest_openalex.py | CLI script + tier config |
| 03 | test_openalex_bulk.py | Tests |
| 04 | link_papers_to_datasets.py | DOI bridge corpus↔literature |
| 05 | finding_extractor.py | LLM finding extraction module |
| 06 | extract_findings.py | CLI + batch pipeline |
| 07 | literature_kg.py | KG integration: paper/finding nodes + edges |
| 08 | search integration | Literature results in search API |

## Key Design Decisions

1. **Shard output by tier + batch**: `openalex_neuro/tier1_batch_000.jsonl`, etc.
   Same directory-of-shards pattern as `combined_corpus.jsonl`.

2. **Schema extension**: Add `citation_count`, `venue`, `concept_ids`,
   `open_access_url`, `topics` to `NormalizedPaperRecord`. Add
   `paper_type: Literal["literature", "dataset"]` discriminator to `NormalizedRecord`.

3. **Checkpoint file**: `data/corpus/normalized/openalex_neuro/.checkpoint.json`
   stores last cursor + count. Script resumes from there if interrupted.

4. **Finding schema**: `FindingRecord(paper_id, region, species, modality, task,
   finding_text, result_direction, confidence, extraction_model)`. Stored as JSONL.

5. **KG node types added**: `paper`, `finding`, `venue`, `author_group`
   **KG edge types added**: `paper_reports_finding`, `finding_involves_region`,
   `finding_involves_task`, `finding_involves_modality`, `dataset_linked_to_paper`,
   `paper_cites_paper`

6. **Search result union**: API returns `SearchResult` which is either a
   `DatasetResult` or `PaperResult`. Query param `?result_types=datasets,papers`
   controls what's returned.
