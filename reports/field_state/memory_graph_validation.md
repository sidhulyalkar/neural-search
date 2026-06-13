# Memory Graph Validation Report

**Total nodes:** 2088  
**Total edges:** 2303  
**Total datasets:** 622  
**Orphan nodes:** 686  

## Completeness Issues
- Missing modality: 504
- Missing species: 485
- Missing region: 542
- Missing task: 526
- Missing description: 596
- Missing raw/processed evidence: 0

## Node Counts by Type
- `neuro_judge_evidence_packet`: 675
- `dataset`: 622
- `raw_data_signal`: 611
- `task`: 49
- `behavioral_event`: 28
- `modality`: 20
- `analysis_affordance`: 18
- `brain_region`: 14
- `processed_data_signal`: 11
- `feedback_signal`: 11
- `file_format`: 9
- `data_standard`: 8
- `source_archive`: 6
- `species`: 5
- `paper`: 1

## Edge Counts by Type
- `dataset_from_source`: 622
- `dataset_has_raw_signal`: 611
- `dataset_uses_standard`: 324
- `dataset_has_species`: 146
- `dataset_has_modality`: 136
- `dataset_has_behavioral_event`: 129
- `dataset_has_task`: 119
- `dataset_records_region`: 92
- `dataset_lacks_required_evidence`: 65
- `dataset_supports_analysis`: 36
- `dataset_has_processed_signal`: 11
- `dataset_has_file_format`: 11
- `dataset_linked_to_paper`: 1

## Average Graph Degree by Source
- modeldb: 30.0
- microns: 30.0
- cellxgene: 26.0
- spark: 7.3
- openneuro: 3.7
- dandi: 3.2

## Top Disconnected Datasets (Curation Targets)
- `dataset:dandi:000785` (missing 5/6 label categories)
- `dataset:dandi:000045` (missing 5/6 label categories)
- `dataset:dandi:001550` (missing 5/6 label categories)
- `dataset:dandi:000007` (missing 5/6 label categories)
- `dataset:dandi:000009` (missing 5/6 label categories)
- `dataset:dandi:000011` (missing 5/6 label categories)
- `dataset:dandi:000012` (missing 5/6 label categories)
- `dataset:dandi:000027` (missing 5/6 label categories)
- `dataset:dandi:000029` (missing 5/6 label categories)
- `dataset:dandi:000031` (missing 5/6 label categories)

## Guardrail Violations
None — all guardrails passed.