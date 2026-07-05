# Task 01 — OpenAlex Bulk Ingest Module

**File to create:** `neural_search/ingestion/openalex_bulk.py`
**File to modify:** `neural_search/schemas.py` (extend NormalizedPaperRecord)

---

## Schema Extension

Add to `NormalizedPaperRecord` in `neural_search/schemas.py`:

```python
citation_count: int = 0
venue: str | None = None
concept_ids: list[str] = Field(default_factory=list)
open_access_url: str | None = None
topics: list[str] = Field(default_factory=list)
```

## openalex_bulk.py Spec

### Constants
```python
OPENALEX_BASE = "https://api.openalex.org"
POLITE_EMAIL = "neuralsearch@example.com"  # moves to polite pool (10 req/s)
NEURO_CONCEPT_ID = "C169760540"            # OpenAlex neuroscience concept
RATE_LIMIT_DELAY = 0.12                    # seconds between requests (~8/s)
PAGE_SIZE = 200                            # max per cursor page
SELECT_FIELDS = "id,doi,title,abstract_inverted_index,publication_year,concepts,cited_by_count,authorships,primary_location,open_access,topics"
```

### Tier filters
```python
TIER_FILTERS = {
    "tier1": "concepts.id:C169760540,type:article,cited_by_count:>99",
    "tier2": "concepts.id:C169760540,type:article,has_abstract:true,open_access.is_oa:true",
    "tier3": "concepts.id:C169760540,type:article",
}
```

### Functions

```python
def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Reconstruct abstract text from OpenAlex inverted index."""
    ...

def normalize_bulk_work(work: dict) -> dict:
    """Convert raw OpenAlex work to flat dict matching NormalizedPaperRecord fields.
    
    Returns dict with all NormalizedPaperRecord fields including the new ones.
    Does NOT call extract_dataset_labels (too slow for bulk — done separately).
    """
    ...

class BulkIngester:
    """Cursor-based bulk downloader with checkpoint/resume and rate limiting."""
    
    def __init__(self, out_dir: Path, tier: str = "tier1", shard_size: int = 10_000):
        self.out_dir = out_dir
        self.tier = tier
        self.shard_size = shard_size
        self._checkpoint_path = out_dir / ".checkpoint.json"
    
    def load_checkpoint(self) -> tuple[str, int]:
        """Return (cursor, records_fetched). cursor='*' if no checkpoint."""
        ...
    
    def save_checkpoint(self, cursor: str, count: int) -> None:
        ...
    
    def _fetch_page(self, cursor: str) -> tuple[list[dict], str | None]:
        """Fetch one page. Returns (works, next_cursor). next_cursor=None at end."""
        ...
    
    def run(self, max_records: int | None = None) -> int:
        """Download until exhausted or max_records reached. Returns total saved."""
        ...
    
    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write a batch of normalized records to a JSONL shard file."""
        ...
```

### Checkpoint format
```json
{
  "cursor": "IlsxMDAuMCwgOT...",
  "records_fetched": 12400,
  "tier": "tier1",
  "last_updated": "2026-06-17T10:30:00Z"
}
```

### Output file naming
```
data/corpus/normalized/openalex_neuro/
  tier1_batch_0000.jsonl    (records 0–9999)
  tier1_batch_0001.jsonl    (records 10000–19999)
  ...
  .checkpoint.json
```

## Tests (tests/test_openalex_bulk.py)

```python
class TestReconstructAbstract:
    def test_empty_returns_none()
    def test_single_word()
    def test_multi_word_correct_order()  # positions matter

class TestNormalizeBulkWork:
    def test_full_record()
    def test_no_abstract()
    def test_no_doi()
    def test_concept_ids_extracted()
    def test_citation_count()
    def test_venue_from_primary_location()
    def test_topics_extracted()

class TestBulkIngesterCheckpoint:
    def test_load_missing_checkpoint_returns_star_cursor()
    def test_save_and_reload_checkpoint(tmp_path)
    def test_checkpoint_survives_restart(tmp_path)

class TestBulkIngesterShard:
    def test_writes_valid_jsonl(tmp_path)
    def test_shard_naming(tmp_path)
    def test_max_records_respected(tmp_path, monkeypatch)  # mock _fetch_page
```

## Implementation Notes

- `reconstruct_abstract`: sort (position, word) pairs, join with space
- `normalize_bulk_work`: extracts `concept_ids` as bare IDs like `C169760540`
  (strip `https://openalex.org/` prefix)
- `_fetch_page`: uses `httpx.Client` (sync), adds `time.sleep(RATE_LIMIT_DELAY)`
  after each call, raises on 4xx/5xx, returns `([], None)` on 200 with empty results
- `run()`: accumulates into a buffer, flushes when buffer hits `shard_size`,
  saves checkpoint after each flush
- On `KeyboardInterrupt`, saves checkpoint before exit so progress is preserved
