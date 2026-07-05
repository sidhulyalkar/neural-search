# Relationship Edge Quality

- Top-K analyzed: 10
- Promotions into top-K: 650
- Judged promotions into top-K: 590

| Edge type | Promotions | Judged | Helpful | Harmful | Helpful rate | Mean relevance |
|---|---:|---:|---:|---:|---:|---:|
| `dataset_reprocessing_candidate` | 449 | 406 | 202 | 204 | 0.498 | 1.320 |
| `same_region_same_task` | 449 | 406 | 202 | 204 | 0.498 | 1.320 |
| `dataset_reanalysis_bridge_dataset` | 393 | 366 | 191 | 175 | 0.522 | 1.377 |
| `same_region_cross_modality` | 393 | 366 | 191 | 175 | 0.522 | 1.377 |
| `dataset_reinterpretation_candidate` | 309 | 275 | 130 | 145 | 0.473 | 1.309 |
| `same_task_cross_species` | 309 | 275 | 130 | 145 | 0.473 | 1.309 |
| `no_relationship_edge` | 43 | 39 | 13 | 26 | 0.333 | 1.128 |

## Helpful Promotion Examples

| Query | Record | Relevance | Base rank | Graph rank | Edge types |
|---|---|---:|---:|---:|---|
| `can_0002` | `neurovault:3396` | 2 | 6 | 5 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0003` | `neurovault:2581` | 2 | 10 | 9 | `dataset_reinterpretation_candidate`, `same_task_cross_species` |
| `can_0003` | `neurovault:5617` | 3 | 13 | 10 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0004` | `dandi:001209` | 2 | 8 | 6 | `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_same_task`, `same_task_cross_species` |
| `can_0006` | `dandi:001641` | 2 | 5 | 4 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0007` | `gin:2697` | 2 | 6 | 5 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0007` | `dandi:000488` | 2 | 12 | 8 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0008` | `openneuro:ds002338` | 2 | 2 | 1 | `dataset_reanalysis_bridge_dataset`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task` |
| `can_0008` | `openneuro:ds002336` | 2 | 3 | 2 | `dataset_reanalysis_bridge_dataset`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task` |
| `can_0009` | `dandi:001044` | 2 | 4 | 3 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0011` | `crcns:ofc-3` | 2 | 2 | 1 | `dataset_reanalysis_bridge_dataset`, `same_region_cross_modality` |
| `can_0012` | `crcns:fcx-2` | 2 | 12 | 10 | `dataset_reanalysis_bridge_dataset`, `same_region_cross_modality` |
| `can_0013` | `dandi:001692` | 2 | 8 | 5 | `dataset_reanalysis_bridge_dataset`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task` |
| `can_0013` | `dandi:000971` | 2 | 10 | 9 | `dataset_reanalysis_bridge_dataset`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task` |
| `can_0014` | `dandi:001695` | 2 | 3 | 2 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0014` | `dandi:001754` | 2 | 5 | 4 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0014` | `dandi:001361` | 2 | 12 | 10 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0015` | `dandi:000688` | 2 | 6 | 5 | `dataset_reanalysis_bridge_dataset`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task` |
| `can_0015` | `crcns:pmd-1` | 2 | 9 | 7 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0016` | `dandi:001692` | 2 | 2 | 1 | `dataset_reanalysis_bridge_dataset`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task` |
| `can_0017` | `dandi:001532` | 2 | 11 | 10 | `dataset_reprocessing_candidate`, `same_region_same_task` |
| `can_0018` | `spark:SPARK-FMRI-002` | 2 | 2 | 1 | `dataset_reprocessing_candidate`, `same_region_same_task` |
| `can_0018` | `neurovault:826` | 2 | 13 | 8 | `dataset_reanalysis_bridge_dataset`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task` |
| `can_0019` | `zenodo:7037327` | 2 | 7 | 6 | `dataset_reinterpretation_candidate`, `same_task_cross_species` |
| `can_0020` | `zenodo:4307883` | 2 | 2 | 1 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |

## Harmful Promotion Examples

| Query | Record | Relevance | Base rank | Graph rank | Edge types |
|---|---|---:|---:|---:|---|
| `can_0001` | `dandi:001056` | 0 | 4 | 3 | `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_same_task`, `same_task_cross_species` |
| `can_0001` | `zenodo:4307883` | 1 | 5 | 4 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0002` | `dandi:000462` | 1 | 9 | 6 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `same_region_cross_modality`, `same_task_cross_species` |
| `can_0004` | `dandi:000125` | 1 | 6 | 5 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0004` | `dandi:000941` | 0 | 10 | 7 | `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_same_task`, `same_task_cross_species` |
| `can_0004` | `zenodo:1321265` | 0 | 9 | 8 | `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_same_task`, `same_task_cross_species` |
| `can_0004` | `zenodo:3854034` | 1 | 11 | 10 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0005` | `dandi:001079` | 1 | 7 | 6 | `dataset_reanalysis_bridge_dataset`, `same_region_cross_modality` |
| `can_0005` | `figshare:29052116` | 1 | 11 | 10 | `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_same_task`, `same_task_cross_species` |
| `can_0007` | `gin:2361` | 1 | 5 | 4 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0010` | `openneuro:ds001740` | 0 | 5 | 4 | `dataset_reanalysis_bridge_dataset`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task` |
| `can_0011` | `neurovault:599` | 1 | 10 | 9 | `dataset_reinterpretation_candidate`, `same_task_cross_species` |
| `can_0012` | `dandi:000574` | 1 | 7 | 5 | `dataset_reanalysis_bridge_dataset`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task` |
| `can_0014` | `dandi:000053` | 0 | 9 | 7 | `dataset_reprocessing_candidate`, `same_region_same_task` |
| `can_0014` | `dandi:000405` | 1 | 11 | 9 | `dataset_reprocessing_candidate`, `same_region_same_task` |
| `can_0016` | `zenodo:17457318` | 1 | 9 | 8 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0018` | `neurovault:2138` | 1 | 10 | 6 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `same_region_cross_modality`, `same_task_cross_species` |
| `can_0019` | `dandi:000951` | 1 | 6 | 5 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `same_region_cross_modality`, `same_task_cross_species` |
| `can_0022` | `dandi:001631` | 1 | 7 | 5 | `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_same_task`, `same_task_cross_species` |
| `can_0024` | `zenodo:12818267` | 1 | 12 | 10 | `dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task`, `same_task_cross_species` |
| `can_0025` | `zenodo:2598755` | 1 | 3 | 1 | `dataset_reanalysis_bridge_dataset`, `same_region_cross_modality` |
| `can_0026` | `dandi:000231` | 1 | 10 | 6 | `dataset_reinterpretation_candidate`, `dataset_reprocessing_candidate`, `same_region_same_task`, `same_task_cross_species` |
| `can_0026` | `dandi:000016` | 1 | 11 | 9 | `dataset_reanalysis_bridge_dataset`, `same_region_cross_modality` |
| `can_0028` | `zenodo:10697024` | 1 | 8 | 7 | `dataset_reanalysis_bridge_dataset`, `same_region_cross_modality` |
| `can_0028` | `dandi:000574` | 1 | 10 | 9 | `dataset_reanalysis_bridge_dataset`, `dataset_reprocessing_candidate`, `same_region_cross_modality`, `same_region_same_task` |
