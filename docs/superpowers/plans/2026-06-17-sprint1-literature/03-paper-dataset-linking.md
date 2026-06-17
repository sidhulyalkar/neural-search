# Task 03 — Paper-to-Dataset DOI Bridge

**File to create:** `scripts/ingestion/link_papers_to_datasets.py`
**File to create:** `neural_search/literature/linking.py`

---

## Goal

For every dataset in `combined_corpus.jsonl` that has a DOI, find the
corresponding OpenAlex paper record. For datasets without DOIs, try
title + year fuzzy matching. Write `dataset_linked_to_paper` edges to
`artifacts/literature/dataset_paper_links.jsonl`.

## linking.py Spec

```python
@dataclass
class DatasetPaperLink:
    dataset_record_id: str   # e.g. "dandi:000004"
    paper_openalex_id: str   # e.g. "W2741809807"
    paper_doi: str | None
    match_method: str        # "doi_exact" | "title_fuzzy" | "author_title"
    confidence: float        # 0.0–1.0

def lookup_by_doi(doi: str) -> dict | None:
    """Query OpenAlex /works/{doi}. Returns normalized work or None."""
    ...

def lookup_by_title(title: str, year: int | None) -> dict | None:
    """Search OpenAlex for title match. Returns top result if score >= 0.85."""
    ...

def link_corpus_to_literature(
    corpus_path: Path,
    out_path: Path,
    *,
    max_workers: int = 4,
) -> list[DatasetPaperLink]:
    """Process all corpus records. Uses ThreadPoolExecutor for parallelism."""
    ...
```

## CLI Spec

```
python scripts/ingestion/link_papers_to_datasets.py \
    --corpus data/corpus/normalized/combined_corpus.jsonl \
    --out artifacts/literature/dataset_paper_links.jsonl \
    [--max-workers 4]
```

## Output format (JSONL)

```json
{"dataset_record_id": "dandi:000004", "paper_openalex_id": "W2741809807",
 "paper_doi": "10.1038/s41593-020-0636-4", "match_method": "doi_exact",
 "confidence": 1.0}
```

## Tests (tests/test_paper_dataset_linking.py)

```python
def test_lookup_by_doi_found(monkeypatch)
def test_lookup_by_doi_not_found(monkeypatch)
def test_lookup_by_title_high_confidence(monkeypatch)
def test_lookup_by_title_low_score_returns_none(monkeypatch)
def test_link_corpus_to_literature_smoke(tmp_path, monkeypatch)
def test_doi_exact_confidence_is_one()
def test_fuzzy_confidence_range()
```
