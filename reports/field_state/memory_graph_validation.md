# Memory Graph Validation Report

**Total nodes:** 2200
**Total edges:** 3788
**Total datasets:** 629
**Orphan nodes:** 692

## Completeness Issues
- Missing modality: 127
- Missing species: 331
- Missing region: 383
- Missing task: 489
- Missing description: 395
- Missing raw/processed evidence: 0

## Node Counts by Type
- `neuro_judge_evidence_packet`: 675
- `dataset`: 629
- `raw_data_signal`: 618
- `brain_region`: 59
- `task`: 56
- `modality`: 36
- `behavioral_event`: 29
- `paper`: 22
- `analysis_affordance`: 18
- `feedback_signal`: 17
- `processed_data_signal`: 11
- `file_format`: 9
- `data_standard`: 8
- `species`: 7
- `source_archive`: 6

## Edge Counts by Type
- `dataset_has_modality`: 742
- `dataset_from_source`: 629
- `dataset_has_raw_signal`: 618
- `dataset_records_region`: 604
- `dataset_has_species`: 384
- `dataset_uses_standard`: 323
- `dataset_has_task`: 195
- `dataset_has_behavioral_event`: 147
- `dataset_lacks_required_evidence`: 65
- `dataset_supports_analysis`: 36
- `dataset_linked_to_paper`: 23
- `dataset_has_processed_signal`: 11
- `dataset_has_file_format`: 11

## Average Graph Degree by Source
- modeldb: 30.0
- microns: 30.0
- cellxgene: 26.0
- spark: 7.4
- dandi: 6.4
- openneuro: 5.4

## Top Disconnected Datasets (Curation Targets)
- `dataset:dandi:000012` (missing 5/6 label categories)
- `dataset:dandi:000029` (missing 5/6 label categories)
- `dataset:dandi:000031` (missing 5/6 label categories)
- `dataset:dandi:000032` (missing 5/6 label categories)
- `dataset:dandi:000033` (missing 5/6 label categories)
- `dataset:dandi:000038` (missing 5/6 label categories)
- `dataset:dandi:000042` (missing 5/6 label categories)
- `dataset:dandi:000046` (missing 5/6 label categories)
- `dataset:dandi:000047` (missing 5/6 label categories)
- `dataset:dandi:000051` (missing 5/6 label categories)

## Guardrail Violations
None — all guardrails passed.
