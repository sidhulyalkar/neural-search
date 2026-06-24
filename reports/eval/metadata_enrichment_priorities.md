# Metadata Enrichment Priorities

Priorities are derived from the current qrels failure-analysis false positives and false negatives.

| Rank | Field | Failure count | Recommended action | Related mismatch modes |
|---:|---|---:|---|---|
| 1 | `task` | 14904 | Extract task and behavioral paradigm labels from titles, descriptions, and protocol files. | `task_mismatch` |
| 2 | `affordance` | 14341 | Audit and normalize `affordance` evidence. | none |
| 3 | `brain_region` | 9848 | Extract anatomical regions from repository metadata, paper links, and file annotations. | `brain_region_mismatch` |
| 4 | `species` | 7034 | Normalize species aliases and source-specific organism fields. | `species_mismatch` |
| 5 | `modality` | 5979 | Add or normalize modality labels from source metadata and file manifests. | `modality_mismatch` |
| 6 | `raw_data` | 5571 | Verify raw-data availability and expose raw/processed evidence separately. | `raw_data_missing` |
| 7 | `other` | 1446 | Audit and normalize `other` evidence. | none |
| 8 | `behavioral_event` | 407 | Audit and normalize `behavioral_event` evidence. | `behavioral_event_mismatch` |
| 9 | `data_standard` | 70 | Normalize NWB, BIDS, and source-specific file-standard metadata. | `raw_data_missing` |

## Top False-Positive Sources

| Source | Count |
|---|---:|
| `dandi` | 2599 |
| `zenodo` | 1769 |
| `neurovault` | 1347 |
| `harvard_dataverse` | 1071 |
| `gin` | 699 |
| `crcns` | 612 |
| `osf` | 400 |
| `openneuro` | 186 |
| `allen` | 172 |
| `figshare` | 149 |
