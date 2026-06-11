# Known-Item Lookup Evaluation

Generated: 2026-06-11T19:42:29.441415+00:00
Queries: 6

---

## Summary Metrics

| System | R@1 | R@3 | R@10 | MRR | Found/N |
|--------|-----|-----|------|-----|---------|
| BM25 raw | 0.8333 | 0.8333 | 0.8333 | 0.8345 | 5/6 |
| BM25 + source-dedup | 0.8333 | 0.8333 | 1.0000 | 0.8611 | 6/6 |
| Hybrid RRF + source-dedup | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 5/6 |
| Hybrid RRF + dedup + alias-boost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 6/6 |

---

## Per-Query Results

### ki_0001: Steinmetz 2019 Neuropixels visual coding mouse

Expected: `dandi:000040` — Neuropixels recordings in mouse visual system
> Note: dandi:000026 was originally listed but is incorrect (it is a human brain cell census). Correct ID is dandi:000040.

| System | Rank | Top-3 results |
|--------|------|---------------|
| BM25 raw | 148 | [1] ecephys_847657808 (Allen Visual Coding Neuropixel) | [2] ecephys_840012044 (Allen Visual Coding Neuropixel) | [3] ecephys_839557629 (Allen Visual Coding Neuropixel) |
| BM25 + source-dedup | 6 ✓ | [1] ecephys_847657808 (Allen Visual Coding Neuropixel) | [2] 000022 (Allen Institute - Visual Codin) | [3] 000021 (Allen Institute - Visual Codin) |
| Hybrid RRF + source-dedup | not found | [1] ecephys_847657808 (Allen Visual Coding Neuropixel) |
| Hybrid RRF + dedup + alias-boost | 1 ✓ | [1] 000040 (Neuropixels recordings in mous) | [2] 000169 (Milti-probe Neuropixels record) | [3] ecephys_847657808 (Allen Visual Coding Neuropixel) |

### ki_0002: Allen Institute Visual Coding Neuropixels Brain Observatory stimulus set

Expected: `dandi:000021` — Allen Institute - Visual Coding - Neuropixels (Brain Observatory 1.1 Stimulus Set)

| System | Rank | Top-3 results |
|--------|------|---------------|
| BM25 raw | 1 ✓ | [1] 000021 (Allen Institute - Visual Codin) | [2] 000022 (Allen Institute - Visual Codin) | [3] ecephys_799864342 (Allen Visual Coding Neuropixel) |
| BM25 + source-dedup | 1 ✓ | [1] 000021 (Allen Institute - Visual Codin) | [2] 000022 (Allen Institute - Visual Codin) | [3] ecephys_799864342 (Allen Visual Coding Neuropixel) |
| Hybrid RRF + source-dedup | 1 ✓ | [1] 000021 (Allen Institute - Visual Codin) | [2] 000022 (Allen Institute - Visual Codin) | [3] ecephys_799864342 (Allen Visual Coding Neuropixel) |
| Hybrid RRF + dedup + alias-boost | 1 ✓ | [1] 000021 (Allen Institute - Visual Codin) | [2] 000022 (Allen Institute - Visual Codin) | [3] ecephys_799864342 (Allen Visual Coding Neuropixel) |

### ki_0003: Allen Institute Visual Coding Neuropixels functional connectivity stimulus

Expected: `dandi:000022` — Allen Institute - Visual Coding - Neuropixels (Functional Connectivity Stimulus Set)

| System | Rank | Top-3 results |
|--------|------|---------------|
| BM25 raw | 1 ✓ | [1] 000022 (Allen Institute - Visual Codin) | [2] ecephys_847657808 (Allen Visual Coding Neuropixel) | [3] ecephys_840012044 (Allen Visual Coding Neuropixel) |
| BM25 + source-dedup | 1 ✓ | [1] 000022 (Allen Institute - Visual Codin) | [2] ecephys_847657808 (Allen Visual Coding Neuropixel) | [3] 000021 (Allen Institute - Visual Codin) |
| Hybrid RRF + source-dedup | 1 ✓ | [1] 000022 (Allen Institute - Visual Codin) | [2] ecephys_847657808 (Allen Visual Coding Neuropixel) | [3] 000021 (Allen Institute - Visual Codin) |
| Hybrid RRF + dedup + alias-boost | 1 ✓ | [1] 000022 (Allen Institute - Visual Codin) | [2] ecephys_847657808 (Allen Visual Coding Neuropixel) | [3] 000021 (Allen Institute - Visual Codin) |

### ki_0004: Allen Institute calcium imaging contrast tuning mouse visual cortex

Expected: `dandi:000039` — Allen Institute – Contrast tuning in mouse visual cortex with calcium imaging

| System | Rank | Top-3 results |
|--------|------|---------------|
| BM25 raw | 1 ✓ | [1] 000039 (Allen Institute – Contrast tun) | [2] 000049 (Allen Institute – TF x SF tuni) | [3] 000050 (Allen Institute - Run Tuning i) |
| BM25 + source-dedup | 1 ✓ | [1] 000039 (Allen Institute – Contrast tun) | [2] 000049 (Allen Institute – TF x SF tuni) | [3] 000050 (Allen Institute - Run Tuning i) |
| Hybrid RRF + source-dedup | 1 ✓ | [1] 000039 (Allen Institute – Contrast tun) | [2] 000049 (Allen Institute – TF x SF tuni) | [3] 000050 (Allen Institute - Run Tuning i) |
| Hybrid RRF + dedup + alias-boost | 1 ✓ | [1] 000039 (Allen Institute – Contrast tun) | [2] 000049 (Allen Institute – TF x SF tuni) | [3] 000050 (Allen Institute - Run Tuning i) |

### ki_0005: Allen Institute calcium imaging temporal frequency spatial frequency tuning visual cortex mouse

Expected: `dandi:000049` — Allen Institute – TF x SF tuning in mouse visual cortex with calcium imaging

| System | Rank | Top-3 results |
|--------|------|---------------|
| BM25 raw | 1 ✓ | [1] 000039 (Allen Institute – Contrast tun) | [2] 000049 (Allen Institute – TF x SF tuni) | [3] 000050 (Allen Institute - Run Tuning i) |
| BM25 + source-dedup | 1 ✓ | [1] 000039 (Allen Institute – Contrast tun) | [2] 000049 (Allen Institute – TF x SF tuni) | [3] 000050 (Allen Institute - Run Tuning i) |
| Hybrid RRF + source-dedup | 1 ✓ | [1] 000039 (Allen Institute – Contrast tun) | [2] 000049 (Allen Institute – TF x SF tuni) | [3] 000050 (Allen Institute - Run Tuning i) |
| Hybrid RRF + dedup + alias-boost | 1 ✓ | [1] 000039 (Allen Institute – Contrast tun) | [2] 000049 (Allen Institute – TF x SF tuni) | [3] 000050 (Allen Institute - Run Tuning i) |

### ki_0006: Steinmetz multi-probe Neuropixels mouse visual system additional recordings

Expected: `dandi:000169` — Milti-probe Neuropixels recordings in mouse visual system (additional data)
> Note: Note typo 'Milti' in corpus title — this is the actual DANDI title.

| System | Rank | Top-3 results |
|--------|------|---------------|
| BM25 raw | 1 ✓ | [1] 000169 (Milti-probe Neuropixels record) | [2] 7739750 (Eight-probe Neuropixels record) | [3] ibl_np1_np2_kim (IBL NP1 vs NP2 Neuropixels Com) |
| BM25 + source-dedup | 1 ✓ | [1] 000169 (Milti-probe Neuropixels record) | [2] 7739750 (Eight-probe Neuropixels record) | [3] ibl_np1_np2_kim (IBL NP1 vs NP2 Neuropixels Com) |
| Hybrid RRF + source-dedup | 1 ✓ | [1] 000169 (Milti-probe Neuropixels record) | [2] 7739750 (Eight-probe Neuropixels record) | [3] ibl_np1_np2_kim (IBL NP1 vs NP2 Neuropixels Com) |
| Hybrid RRF + dedup + alias-boost | 1 ✓ | [1] 000169 (Milti-probe Neuropixels record) | [2] 7739750 (Eight-probe Neuropixels record) | [3] ibl_np1_np2_kim (IBL NP1 vs NP2 Neuropixels Com) |

---

## Retrieval Fix: Source-Deduplication

**Problem:** The corpus contains thousands of individual Allen and IBL session records that share similar titles (e.g., `Allen Visual Coding Neuropixels: session_XXXXXXX`). These flood BM25 rankings for any Neuropixels query, burying the parent DANDI datasets.

**Fix implemented:** `source_dedup_rerank()` in `evaluate_known_item_lookup.py` — collapse repeated session-level records from `allen` and `ibl` sources to at most 1 per title-family prefix. This promotes DANDI parent datasets to the top of the ranking.

**How to integrate into main search:** Apply `source_dedup_rerank()` as a post-processing step when the query intent is `EXACT_LOOKUP` or `REPLICATION` (detected from query patterns like author + year + method). A shallow heuristic: if the query contains a year (\d{4}) and an author-like capitalized word, apply session deduplication.

```bash
# Re-run this evaluation
python scripts/eval/evaluate_known_item_lookup.py
```