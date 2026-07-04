# Task 02 — Bulk Ingest CLI + Tier Config

**File to create:** `scripts/ingestion/bulk_ingest_openalex.py`
**File to create:** `configs/ingestion/openalex_tiers.yaml`

---

## CLI Spec

```
python scripts/ingestion/bulk_ingest_openalex.py \
    --tier tier1 \
    --out data/corpus/normalized/openalex_neuro \
    --max-records 50000 \
    [--resume]  \
    [--shard-size 10000]
```

### Arguments
- `--tier`: `tier1` | `tier2` | `tier3` (default: `tier1`)
- `--out`: output directory (created if absent)
- `--max-records`: stop after N records (default: unlimited)
- `--resume`: load checkpoint and continue (default: start fresh for new tier)
- `--shard-size`: records per JSONL file (default: 10000)
- `--dry-run`: fetch one page, print stats, exit

### Behaviour
- Prints progress every 1000 records: `[12400/255940] tier1 | shard=1 | rate=7.8/s`
- On completion prints summary:
  ```json
  {"tier": "tier1", "total_records": 255940, "shards_written": 26, "elapsed_s": 3200}
  ```
- Saves checkpoint on Ctrl+C

## configs/ingestion/openalex_tiers.yaml

```yaml
tiers:
  tier1:
    description: "High-impact papers (>=100 citations)"
    filter: "concepts.id:C169760540,type:article,cited_by_count:>99"
    estimated_count: 255940
    priority: 1

  tier2:
    description: "Open-access papers with abstracts"
    filter: "concepts.id:C169760540,type:article,has_abstract:true,open_access.is_oa:true"
    estimated_count: 1389240
    priority: 2
    depends_on: tier1

  tier3:
    description: "All neuroscience articles"
    filter: "concepts.id:C169760540,type:article"
    estimated_count: 4360916
    priority: 3
    depends_on: tier2

settings:
  page_size: 200
  rate_limit_delay: 0.12
  shard_size: 10000
  polite_email: "neuralsearch@example.com"
  select_fields: "id,doi,title,abstract_inverted_index,publication_year,concepts,cited_by_count,authorships,primary_location,open_access,topics"
```
