# Memory Graph Validation Report

**Total nodes:** 860  
**Total edges:** 370  
**Total datasets:** 35  
**Orphan nodes:** 680  

## Completeness Issues
- Missing modality: 7
- Missing species: 6
- Missing region: 17
- Missing task: 15
- Missing description: 9
- Missing raw/processed evidence: 0

## Node Counts by Type
- `neuro_judge_evidence_packet`: 675
- `dataset`: 35
- `raw_data_signal`: 24
- `task`: 23
- `analysis_affordance`: 18
- `behavioral_event`: 17
- `modality`: 15
- `processed_data_signal`: 11
- `brain_region`: 10
- `file_format`: 9
- `data_standard`: 8
- `source_archive`: 6
- `feedback_signal`: 5
- `species`: 3
- `paper`: 1

## Edge Counts by Type
- `dataset_lacks_required_evidence`: 65
- `dataset_has_modality`: 39
- `dataset_supports_analysis`: 36
- `dataset_from_source`: 35
- `dataset_has_task`: 34
- `dataset_has_species`: 34
- `dataset_has_behavioral_event`: 32
- `dataset_records_region`: 26
- `dataset_has_raw_signal`: 24
- `dataset_uses_standard`: 22
- `dataset_has_processed_signal`: 11
- `dataset_has_file_format`: 11
- `dataset_linked_to_paper`: 1

## Average Graph Degree by Source
- openneuro: 30.0
- modeldb: 30.0
- microns: 30.0
- cellxgene: 26.0
- dandi: 9.8
- spark: 7.3

## Top Disconnected Datasets (Curation Targets)
- `dataset:dandi:000785` (missing 5/6 label categories)
- `dataset:dandi:000045` (missing 5/6 label categories)
- `dataset:dandi:001550` (missing 5/6 label categories)
- `dataset:dandi:001079` (missing 4/6 label categories)
- `dataset:dandi:001371` (missing 4/6 label categories)
- `dataset:dandi:000640` (missing 3/6 label categories)
- `dataset:dandi:000678` (missing 3/6 label categories)
- `dataset:spark:SPARK-GENO-001` (missing 3/6 label categories)
- `dataset:spark:SPARK-GENO-002` (missing 3/6 label categories)

## Guardrail Violations
None — all guardrails passed.