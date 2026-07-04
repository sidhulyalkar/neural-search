# Task 06 — Search Integration

**Files to create:** `neural_search/literature/search.py`
**Files to modify:** `neural_search/search/core.py`, `neural_search/api/` (routes)

---

## Goal

Extend search to return literature results (papers + findings) alongside
dataset results. Results are ranked by a unified relevance score.

## New Types

```python
@dataclass
class PaperResult:
    result_type: Literal["paper"] = "paper"
    paper_id: str
    title: str
    abstract_snippet: str | None   # first 300 chars
    year: int | None
    citation_count: int
    venue: str | None
    doi: str | None
    url: str | None
    relevance_score: float
    why_matched: list[str]
    linked_datasets: list[str]     # dataset_record_ids linked to this paper
    top_findings: list[str]        # finding_text from top 2 findings

@dataclass
class FindingResult:
    result_type: Literal["finding"] = "finding"
    finding_id: str
    finding_text: str
    result_direction: str
    regions: list[str]
    species: list[str]
    modalities: list[str]
    paper_id: str
    paper_title: str
    paper_year: int | None
    relevance_score: float
    why_matched: list[str]
```

## literature/search.py Spec

```python
def search_papers(
    query: str,
    *,
    limit: int = 10,
    shard_dir: Path,
    filters: dict | None = None,
) -> list[PaperResult]:
    """BM25-style search over paper titles + abstracts in JSONL shards.
    
    Loads shards lazily. Returns top-k by TF-IDF-style relevance.
    On first call, builds an in-memory inverted index (cached).
    """
    ...

def search_findings(
    query: str,
    *,
    limit: int = 10,
    findings_path: Path,
    filters: dict | None = None,
) -> list[FindingResult]:
    """BM25 over finding_text. Also filters by region/species/modality."""
    ...
```

## API Change

Add `result_types` query param to `/api/search`:

```
GET /api/search?q=hippocampus+memory&result_types=datasets,papers,findings&limit=10
```

Response:
```json
{
  "results": [
    {"result_type": "dataset", ...},
    {"result_type": "paper", ...},
    {"result_type": "finding", ...}
  ],
  "total_datasets": 3,
  "total_papers": 4,
  "total_findings": 3,
  "query": "hippocampus memory"
}
```

Default: `result_types=datasets` (backward compatible).

## Tests (tests/test_literature_search.py)

```python
def test_search_papers_returns_results(tmp_path)
def test_search_papers_empty_shards_returns_empty(tmp_path)
def test_search_findings_filters_by_region(tmp_path)
def test_unified_search_backward_compat(tmp_path)
def test_paper_result_has_linked_datasets(tmp_path)
```
