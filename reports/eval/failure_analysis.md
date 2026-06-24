# Failure Analysis Report

Queries evaluated: 317  
Variants: bm25, bm25_structured, dense_bge, full, hybrid_graph, hybrid_rrf

---

## Variant: `bm25`

| Metric | Count |
|--------|-------|
| False positives (top-K, relevance ≤ 1) | 1739 |
| False negatives (relevant, not in top-K) | 3308 |
| Hard-negative violations | 71 |


### Top False Positives (rank ≤ 10)

| Rank | Record | Relevance | HN | Query | Rationale |
|------|--------|-----------|----|----|-----------|
| 1 | `neurovault:3078` | 0 | no | can_0001 | The dataset uses fMRI in humans, which does not match the query's requirement fo |
| 1 | `zenodo:5889167` | 1 | no | can_0004 | The dataset has a matching brain region but lacks explicit evidence for required |
| 1 | `neurovault:835` | 0 | no | can_0005 | The dataset is not relevant because it does not match the required modality of n |
| 1 | `osf:ag5ym` | 0 | no | can_0006 | The dataset lacks core requirements such as the specific analysis affordances ne |
| 1 | `neurovault:1467` | 0 | no | can_0007 | The dataset lacks pupil or running arousal measurements, which are critical for  |
| 1 | `harvard_dataverse:10.7910_DVN_J0PQIW` | 1 | no | can_0008 | The dataset has correct EEG modality, but it focuses on olfactory tasks rather t |
| 1 | `neurovault:4487` | 1 | no | can_0011 | The dataset matches species and task but has a mismatch in brain region, making  |
| 1 | `osf:8j7g2` | 1 | no | can_0016 | The dataset is weakly related due to matching modality but lacks explicit eviden |
| 1 | `osf:vn4yq` | 0 | no | can_0021 | The dataset is not relevant as it does not contain any of the required neuroimag |
| 1 | `harvard_dataverse:10.7910_DVN_ZQQABJ` | 1 | no | can_0023 | The dataset matches on brain regions but lacks explicit evidence of the required |
| 1 | `crcns:ofc-3` | 0 | no | can_0026 | The dataset is not relevant because it involves human subjects and uses a differ |
| 1 | `zenodo:13134148` | 1 | no | can_0028 | The dataset is weakly related as it matches the language comprehension task, but |
| 1 | `dandi:000020` | 1 | no | can_0030 | The dataset matches the species and brain region but lacks explicit evidence for |
| 1 | `dandi:000020` | 0 | no | can_0033 | The dataset does not match any hard-negative patterns but fails to meet the requ |
| 1 | `dandi:000070` | 0 | **YES** | can_0034 | The dataset uses Utah Arrays, which is a hard-negative pattern for this query. |
| 1 | `neurovault:3264` | 0 | **YES** | can_0035 | The dataset is labeled as fMRI, which is a hard-negative for this query requirin |
| 1 | `osf:8j7g2` | 0 | **YES** | can_0036 | Dataset contains modalities that are explicitly excluded in the query constraint |
| 1 | `dandi:000468` | 1 | no | can_0037 | The dataset matches key dimensions like brain region and task but uses virtual r |
| 1 | `dandi:001641` | 1 | no | can_0038 | The dataset matches species, modality, and brain regions but lacks explicit evid |
| 1 | `harvard_dataverse:10.7910_DVN_FB0MZT` | 1 | no | can_0040 | The dataset matches species and brain regions but lacks explicit support for the |

### False Negatives (relevant, not in top-K)

| Record | Relevance | Best Rank Outside K | Query |
|--------|-----------|--------------------|----|
| `ibl:session_7f56a60c-92c9-42` | 3 | 12 | can_0230 |
| `dandi:000941` | 3 | 13 | can_0136 |
| `dandi:001209` | 3 | 14 | can_0136 |
| `zenodo:15446526` | 3 | 21 | can_0075 |
| `neurovault:190` | 3 | 26 | can_0107 |
| `crcns:hc-8` | 3 | 33 | can_0185 |
| `crcns:pvc-12` | 3 | not retrieved | can_0073 |
| `figshare:7666892` | 3 | not retrieved | can_0086 |
| `zenodo:15098469` | 3 | not retrieved | can_0086 |
| `openneuro:ds002422` | 3 | not retrieved | can_0279 |
| `gin:13314` | 2 | 11 | can_0001 |
| `crcns:pfc-4` | 2 | 11 | can_0012 |
| `dandi:001371` | 2 | 11 | can_0014 |
| `dandi:000128` | 2 | 11 | can_0015 |
| `zenodo:16914371` | 2 | 11 | can_0016 |
| `allen:ophys_539540432` | 2 | 11 | can_0017 |
| `harvard_dataverse:10.7910_DVN_YF9MGO` | 2 | 11 | can_0019 |
| `zenodo:1321257` | 2 | 11 | can_0027 |
| `dandi:000139` | 2 | 11 | can_0034 |
| `dandi:000139` | 2 | 11 | can_0039 |

### Source Breakdown

| Source | FP Count | FN Count |
|--------|----------|----------|
| allen | 34 | 350 |
| bluebrain | 26 | 9 |
| brain_image_library | 8 | 0 |
| buzsaki | 14 | 113 |
| crcns | 130 | 141 |
| dandi | 466 | 981 |
| figshare | 19 | 54 |
| gin | 124 | 135 |
| harvard_dataverse | 229 | 134 |
| ibl | 2 | 347 |
| nemo | 11 | 7 |
| neuromorpho | 9 | 14 |
| neurovault | 227 | 401 |
| openneuro | 35 | 65 |
| osf | 108 | 83 |
| spark | 22 | 8 |
| zenodo | 275 | 466 |

### By Query Intent

| Intent | FP Count | FN Count |
|--------|----------|----------|
| CROSS_DATASET_COMPARISON | 2 | 20 |
| EXPLORATION | 1588 | 2983 |
| PIPELINE_REUSE | 5 | 58 |
| REANALYSIS_FEASIBILITY | 131 | 121 |
| REPLICATION | 4 | 10 |
| STRICT_LOOKUP | 9 | 116 |

### False-Positive Mismatch Breakdown

| Mismatch bucket | Count |
|-----------------|-------|
| task_mismatch | 1127 |
| brain_region_mismatch | 893 |
| modality_mismatch | 707 |
| species_mismatch | 699 |
| raw_data_missing | 174 |
| behavioral_event_mismatch | 18 |

### Metadata Missingness Breakdown

| Missing dimension | FP Count | FN Count |
|-------------------|----------|----------|
| affordance | 963 | 1541 |
| behavioral_event | 19 | 46 |
| brain_region | 894 | 810 |
| data_standard | 7 | 4 |
| modality | 836 | 361 |
| other | 83 | 145 |
| raw_data | 88 | 810 |
| species | 692 | 602 |
| task | 1121 | 1388 |

### Top False-Positive Failure Modes

| Failure mode | Count |
|--------------|-------|
| wrong_modality | 338 |
| wrong_task | 173 |
| wrong_species | 169 |
| wrong_brain_region | 161 |
| no_raw_data | 111 |
| no_task_match | 61 |
| hard_negative_detected | 38 |
| no_species | 23 |
| no_brain_region | 19 |
| no_species_match | 17 |
| no_modality_match | 16 |
| no_affordance_evidence | 16 |
| no_required_modalities | 14 |
| no_task_evidence | 14 |
| no_modality_evidence | 12 |
| no_modality | 12 |
| no_tasks | 11 |
| wrong_tasks | 10 |
| no_brain_region_match | 10 |
| task_mismatch | 8 |

### Hard-Negative Failure Modes

| Hard-negative failure mode | Count |
|----------------------------|-------|
| hard_negative_detected | 71 |

## Variant: `bm25_structured`

| Metric | Count |
|--------|-------|
| False positives (top-K, relevance ≤ 1) | 1686 |
| False negatives (relevant, not in top-K) | 3372 |
| Hard-negative violations | 125 |


### Top False Positives (rank ≤ 10)

| Rank | Record | Relevance | HN | Query | Rationale |
|------|--------|-----------|----|----|-----------|
| 1 | `neurovault:3078` | 0 | no | can_0001 | The dataset uses fMRI in humans, which does not match the query's requirement fo |
| 1 | `zenodo:8207948` | 0 | no | can_0004 | The dataset does not meet the query's requirements for ECoG or iEEG modalities a |
| 1 | `zenodo:4837010` | 0 | no | can_0005 | The dataset lacks Neuropixels recordings and the tasks are unrelated to visual d |
| 1 | `osf:ag5ym` | 0 | no | can_0006 | The dataset lacks core requirements such as the specific analysis affordances ne |
| 1 | `neurovault:1467` | 0 | no | can_0007 | The dataset lacks pupil or running arousal measurements, which are critical for  |
| 1 | `harvard_dataverse:10.7910_DVN_J0PQIW` | 1 | no | can_0008 | The dataset has correct EEG modality, but it focuses on olfactory tasks rather t |
| 1 | `dandi:001467` | 1 | no | can_0009 | The dataset matches the query in terms of modality and task but lacks required i |
| 1 | `ibl:ibl_brain_wide_map_2022` | 1 | no | can_0011 | The dataset matches on modality and a related task but does not explicitly menti |
| 1 | `dandi:001181` | 1 | no | can_0016 | The dataset matches the modality requirement but lacks explicit evidence of rewa |
| 1 | `dandi:000167` | 1 | no | can_0017 | The dataset matches species and modality requirements but has a mismatch in brai |
| 1 | `neurovault:3732` | 1 | no | can_0018 | The dataset matches species and modality but lacks explicit evidence for relevan |
| 1 | `osf:vn4yq` | 0 | no | can_0021 | The dataset is not relevant as it does not contain any of the required neuroimag |
| 1 | `harvard_dataverse:10.7910_DVN_ZQQABJ` | 1 | no | can_0023 | The dataset matches on brain regions but lacks explicit evidence of the required |
| 1 | `osf:e23v9` | 0 | no | can_0027 | The dataset is scored as 0 because it involves human subjects, which does not me |
| 1 | `neurovault:4560` | 0 | **YES** | can_0029 | The dataset uses fMRI which is explicitly excluded by the query constraints. |
| 1 | `gin:11399` | 0 | **YES** | can_0030 | Dataset uses EEG, which is a hard-negative pattern for this query. |
| 1 | `neurovault:4560` | 0 | **YES** | can_0031 | The dataset uses fMRI, which is a hard-negative pattern for this query that excl |
| 1 | `harvard_dataverse:10.7910_DVN_ZAELMD` | 0 | **YES** | can_0033 | The dataset is not relevant due to mismatch in species (human vs mouse) and moda |
| 1 | `neuromorpho:Medalla` | 1 | no | can_0034 | The dataset matches the species and brain region but lacks explicit evidence of  |
| 1 | `neurovault:3264` | 0 | **YES** | can_0035 | The dataset is labeled as fMRI, which is a hard-negative for this query requirin |

### False Negatives (relevant, not in top-K)

| Record | Relevance | Best Rank Outside K | Query |
|--------|-----------|--------------------|----|
| `ibl:session_7f56a60c-92c9-42` | 3 | 12 | can_0230 |
| `dandi:000941` | 3 | 13 | can_0136 |
| `dandi:001209` | 3 | 14 | can_0136 |
| `zenodo:15098469` | 3 | 16 | can_0086 |
| `zenodo:15446526` | 3 | 17 | can_0075 |
| `crcns:pvc-12` | 3 | not retrieved | can_0073 |
| `figshare:7666892` | 3 | not retrieved | can_0086 |
| `crcns:ieeg-1` | 3 | not retrieved | can_0151 |
| `openneuro:ds002422` | 3 | not retrieved | can_0279 |
| `gin:13314` | 2 | 11 | can_0001 |
| `ibl:session_1b966923-de4a-4a` | 2 | 11 | can_0005 |
| `zenodo:7140061` | 2 | 11 | can_0009 |
| `crcns:pfc-4` | 2 | 11 | can_0012 |
| `dandi:000128` | 2 | 11 | can_0015 |
| `dandi:001434` | 2 | 11 | can_0016 |
| `allen:ophys_529179127` | 2 | 11 | can_0017 |
| `harvard_dataverse:10.7910_DVN_YF9MGO` | 2 | 11 | can_0019 |
| `zenodo:1301046` | 2 | 11 | can_0020 |
| `ibl:session_28b45b7a-8100-41` | 2 | 11 | can_0026 |
| `crcns:pvc-6` | 2 | 11 | can_0033 |

### Source Breakdown

| Source | FP Count | FN Count |
|--------|----------|----------|
| allen | 30 | 398 |
| bluebrain | 21 | 8 |
| brain_image_library | 4 | 3 |
| buzsaki | 14 | 118 |
| crcns | 130 | 135 |
| dandi | 387 | 1036 |
| figshare | 11 | 59 |
| gin | 194 | 130 |
| harvard_dataverse | 229 | 148 |
| ibl | 25 | 258 |
| nemo | 6 | 5 |
| neuromorpho | 18 | 13 |
| neurovault | 186 | 413 |
| openneuro | 24 | 90 |
| osf | 85 | 78 |
| spark | 13 | 13 |
| zenodo | 309 | 467 |

### By Query Intent

| Intent | FP Count | FN Count |
|--------|----------|----------|
| CROSS_DATASET_COMPARISON | 1 | 19 |
| EXPLORATION | 1523 | 3017 |
| PIPELINE_REUSE | 6 | 59 |
| REANALYSIS_FEASIBILITY | 145 | 151 |
| REPLICATION | 4 | 10 |
| STRICT_LOOKUP | 7 | 116 |

### False-Positive Mismatch Breakdown

| Mismatch bucket | Count |
|-----------------|-------|
| task_mismatch | 1214 |
| brain_region_mismatch | 919 |
| species_mismatch | 608 |
| modality_mismatch | 543 |
| raw_data_missing | 163 |
| behavioral_event_mismatch | 27 |

### Metadata Missingness Breakdown

| Missing dimension | FP Count | FN Count |
|-------------------|----------|----------|
| affordance | 901 | 1570 |
| behavioral_event | 31 | 41 |
| brain_region | 924 | 832 |
| data_standard | 11 | 5 |
| modality | 638 | 389 |
| other | 106 | 146 |
| raw_data | 94 | 834 |
| species | 598 | 637 |
| task | 1206 | 1410 |

### Top False-Positive Failure Modes

| Failure mode | Count |
|--------------|-------|
| wrong_modality | 220 |
| wrong_task | 181 |
| wrong_brain_region | 170 |
| wrong_species | 107 |
| no_raw_data | 91 |
| hard_negative_detected | 70 |
| no_task_match | 62 |
| no_brain_region | 18 |
| no_affordance_evidence | 17 |
| no_task_evidence | 12 |
| no_species_match | 11 |
| no_species | 11 |
| no_modality_match | 10 |
| no_task | 9 |
| wrong_tasks | 8 |
| no_tasks_match | 8 |
| no_brain_region_match | 7 |
| no_causal_manipulation | 7 |
| no_task_specified | 6 |
| no_tasks | 6 |

### Hard-Negative Failure Modes

| Hard-negative failure mode | Count |
|----------------------------|-------|
| hard_negative_detected | 125 |

## Variant: `dense_bge`

| Metric | Count |
|--------|-------|
| False positives (top-K, relevance ≤ 1) | 1532 |
| False negatives (relevant, not in top-K) | 3655 |
| Hard-negative violations | 43 |


### Top False Positives (rank ≤ 10)

| Rank | Record | Relevance | HN | Query | Rationale |
|------|--------|-----------|----|----|-----------|
| 1 | `zenodo:17425744` | 0 | no | can_0001 | The dataset explicitly states 'Modalities: none', which means it lacks the requi |
| 1 | `harvard_dataverse:10.3886_E112812V1` | 0 | no | can_0003 | The dataset lacks explicit mention of delay discounting as a task, and criticall |
| 1 | `crcns:dream` | 1 | no | can_0004 | The dataset matches the task requirement but lacks the required modalities, maki |
| 1 | `dandi:001282` | 1 | no | can_0005 | The dataset matches in modality and species but lacks explicit evidence for requ |
| 1 | `dandi:001631` | 1 | no | can_0006 | The dataset is weakly related due to matching the flexible species requirement b |
| 1 | `crcns:eye-1` | 1 | no | can_0007 | The dataset is weakly related as it matches the required species and modality bu |
| 1 | `gin:811` | 1 | no | can_0008 | The dataset is weakly related as it only confirms the EEG modality but lacks spe |
| 1 | `harvard_dataverse:10.34894_9BAJTD` | 0 | no | can_0016 | The dataset is not relevant because it uses fMRI instead of fiber photometry as  |
| 1 | `bluebrain:6dbfb587e17cb7a1` | 0 | no | can_0020 | The dataset does not contain the required electrophysiology modalities and is th |
| 1 | `osf:vn4yq` | 0 | no | can_0021 | The dataset is not relevant as it does not contain any of the required neuroimag |
| 1 | `dandi:001631` | 1 | no | can_0022 | The dataset has some broad scientific concept matches but does not support the i |
| 1 | `zenodo:17425744` | 1 | no | can_0023 | The dataset has some broad scientific concept matches, particularly in mentionin |
| 1 | `dandi:001086` | 0 | no | can_0025 | The dataset does not provide any explicit information about the necessary dimens |
| 1 | `neurovault:671` | 0 | **YES** | can_0029 | The dataset uses fMRI, which is explicitly excluded by the query constraints. |
| 1 | `neurovault:671` | 0 | **YES** | can_0031 | The dataset uses fMRI modality which is explicitly excluded by the query constra |
| 1 | `zenodo:17457318` | 1 | no | can_0032 | The dataset includes relevant brain regions but lacks explicit evidence for dopa |
| 1 | `zenodo:11550255` | 0 | **YES** | can_0034 | The dataset matches several query requirements but fails due to the use of Utah  |
| 1 | `osf:k4jp8` | 0 | **YES** | can_0035 | The dataset is not relevant because it includes fMRI, which is a hard-negative f |
| 1 | `crcns:hc-12` | 0 | no | can_0037 | The dataset matches several required dimensions but fails due to involvement in  |
| 1 | `dandi:001631` | 0 | no | can_0038 | The dataset is not relevant as it does not match the required tasks or analysis  |

### False Negatives (relevant, not in top-K)

| Record | Relevance | Best Rank Outside K | Query |
|--------|-----------|--------------------|----|
| `dandi:000952` | 3 | 20 | can_0003 |
| `crcns:pvc-12` | 3 | 23 | can_0073 |
| `harvard_dataverse:10.34894_WPNGHT` | 3 | 25 | can_0159 |
| `figshare:7666892` | 3 | 29 | can_0086 |
| `crcns:hc-8` | 3 | 31 | can_0185 |
| `dandi:001177` | 3 | not retrieved | can_0003 |
| `osf:sb82w` | 3 | not retrieved | can_0003 |
| `neurovault:5617` | 3 | not retrieved | can_0003 |
| `neurovault:2860` | 3 | not retrieved | can_0003 |
| `zenodo:19729161` | 3 | not retrieved | can_0023 |
| `zenodo:15098469` | 3 | not retrieved | can_0086 |
| `neurovault:190` | 3 | not retrieved | can_0107 |
| `dandi:000941` | 3 | not retrieved | can_0136 |
| `dandi:001209` | 3 | not retrieved | can_0136 |
| `harvard_dataverse:10.7910_DVN_BQNOMZ` | 3 | not retrieved | can_0165 |
| `ibl:session_7f56a60c-92c9-42` | 3 | not retrieved | can_0230 |
| `zenodo:4307883` | 3 | not retrieved | can_0231 |
| `openneuro:ds002422` | 3 | not retrieved | can_0279 |
| `osf:2ypk4` | 2 | 11 | can_0003 |
| `dandi:000039` | 2 | 11 | can_0017 |

### Source Breakdown

| Source | FP Count | FN Count |
|--------|----------|----------|
| allen | 25 | 405 |
| bluebrain | 29 | 16 |
| brain_image_library | 2 | 3 |
| buzsaki | 42 | 70 |
| crcns | 65 | 176 |
| dandi | 385 | 1075 |
| figshare | 50 | 48 |
| gin | 112 | 159 |
| harvard_dataverse | 152 | 180 |
| ibl | 2 | 401 |
| nemo | 3 | 7 |
| neuromorpho | 0 | 18 |
| neurovault | 278 | 424 |
| openneuro | 37 | 70 |
| osf | 59 | 114 |
| spark | 2 | 15 |
| zenodo | 289 | 474 |

### By Query Intent

| Intent | FP Count | FN Count |
|--------|----------|----------|
| CROSS_DATASET_COMPARISON | 2 | 20 |
| EXPLORATION | 1407 | 3311 |
| PIPELINE_REUSE | 3 | 59 |
| REANALYSIS_FEASIBILITY | 111 | 132 |
| REPLICATION | 6 | 12 |
| STRICT_LOOKUP | 3 | 121 |

### False-Positive Mismatch Breakdown

| Mismatch bucket | Count |
|-----------------|-------|
| task_mismatch | 1183 |
| brain_region_mismatch | 773 |
| modality_mismatch | 496 |
| species_mismatch | 486 |
| raw_data_missing | 128 |
| behavioral_event_mismatch | 22 |

### Metadata Missingness Breakdown

| Missing dimension | FP Count | FN Count |
|-------------------|----------|----------|
| affordance | 846 | 1666 |
| behavioral_event | 23 | 45 |
| brain_region | 772 | 928 |
| data_standard | 2 | 7 |
| modality | 588 | 403 |
| other | 84 | 170 |
| raw_data | 79 | 913 |
| species | 475 | 741 |
| task | 1176 | 1425 |

### Top False-Positive Failure Modes

| Failure mode | Count |
|--------------|-------|
| wrong_modality | 195 |
| wrong_task | 97 |
| wrong_species | 79 |
| wrong_brain_region | 74 |
| no_raw_data | 70 |
| no_task_match | 52 |
| hard_negative_detected | 25 |
| no_affordance_evidence | 20 |
| no_modality_evidence | 19 |
| no_modality_match | 18 |
| no_species | 14 |
| wrong_tasks | 10 |
| no_task_evidence | 10 |
| wrong_modalities | 9 |
| no_task | 8 |
| no_modality | 8 |
| no_required_modalities | 8 |
| no_species_match | 8 |
| no_required_tasks | 6 |
| no_tasks | 6 |

### Hard-Negative Failure Modes

| Hard-negative failure mode | Count |
|----------------------------|-------|
| hard_negative_detected | 43 |

## Variant: `full`

| Metric | Count |
|--------|-------|
| False positives (top-K, relevance ≤ 1) | 1417 |
| False negatives (relevant, not in top-K) | 3328 |
| Hard-negative violations | 67 |


### Top False Positives (rank ≤ 10)

| Rank | Record | Relevance | HN | Query | Rationale |
|------|--------|-----------|----|----|-----------|
| 1 | `dandi:001056` | 0 | no | can_0001 | The dataset's task ('center_out_reaching') does not match the query's required ' |
| 1 | `dandi:000070` | 1 | no | can_0004 | The dataset involves reaching tasks, which matches the query's task requirement. |
| 1 | `dandi:000575` | 1 | no | can_0012 | The dataset is weakly related due to species match but lacks explicit evidence f |
| 1 | `zenodo:4307883` | 0 | no | can_0021 | The dataset does not meet the specific requirement for BIDS-formatted neuroimagi |
| 1 | `dandi:001631` | 1 | no | can_0022 | The dataset has some broad scientific concept matches but does not support the i |
| 1 | `zenodo:2598755` | 1 | no | can_0025 | The dataset matches on species and modality but lacks explicit evidence for the  |
| 1 | `dandi:000574` | 1 | no | can_0028 | The dataset matches species and modality but fails to meet the required task cri |
| 1 | `dandi:000987` | 1 | no | can_0030 | The dataset matches the required species and modalities but fails to meet the br |
| 1 | `neurovault:4487` | 1 | no | can_0032 | The dataset matches the brain region and modality but lacks the specific task re |
| 1 | `dandi:000987` | 1 | no | can_0033 | The dataset matches the species and modality but has a different brain region th |
| 1 | `zenodo:11550255` | 0 | **YES** | can_0034 | The dataset matches several query requirements but fails due to the use of Utah  |
| 1 | `harvard_dataverse:10.11588_DATA_V57GNG` | 0 | **YES** | can_0035 | The dataset does not meet the modality requirement for iEEG/ECOG and lacks expli |
| 1 | `crcns:hc-12` | 0 | no | can_0037 | The dataset matches several required dimensions but fails due to involvement in  |
| 1 | `gin:2479` | 1 | no | can_0041 | The dataset matches calcium imaging and hippocampal CA1 region but lacks necessa |
| 1 | `zenodo:10277145` | 1 | no | can_0043 | The dataset is weakly related due to mismatched species and brain region, with a |
| 1 | `dandi:001253` | 1 | no | can_0045 | The dataset matches species, modality, and brain region but not the task or affo |
| 1 | `dandi:001631` | 0 | no | can_0046 | The dataset does not contain data from any of the required brain regions (striat |
| 1 | `dandi:001057` | 1 | no | can_0047 | The dataset matches on species and brain regions, but the task does not align we |
| 1 | `dandi:000231` | 1 | no | can_0050 | The dataset matches the species and includes relevant modalities but does not ma |
| 1 | `dandi:000053` | 0 | no | can_0051 | The dataset does not contain any relevant data for the specified brain regions,  |

### False Negatives (relevant, not in top-K)

| Record | Relevance | Best Rank Outside K | Query |
|--------|-----------|--------------------|----|
| `neurovault:190` | 3 | 11 | can_0107 |
| `zenodo:15446526` | 3 | 15 | can_0075 |
| `dandi:000941` | 3 | 22 | can_0136 |
| `dandi:001209` | 3 | 23 | can_0136 |
| `zenodo:19729161` | 3 | 25 | can_0023 |
| `harvard_dataverse:10.7910_DVN_BQNOMZ` | 3 | 30 | can_0165 |
| `crcns:hc-8` | 3 | 33 | can_0185 |
| `figshare:7666892` | 3 | 34 | can_0086 |
| `openneuro:ds002422` | 3 | 35 | can_0279 |
| `osf:sb82w` | 3 | 44 | can_0003 |
| `zenodo:15098469` | 3 | 45 | can_0086 |
| `crcns:pvc-12` | 3 | not retrieved | can_0073 |
| `ibl:session_7f56a60c-92c9-42` | 3 | not retrieved | can_0230 |
| `neurovault:4560` | 2 | 11 | can_0002 |
| `ibl:session_034e726f-b35f-41` | 2 | 11 | can_0005 |
| `spark:SPARK-MULTI-001` | 2 | 11 | can_0010 |
| `dandi:001340` | 2 | 11 | can_0013 |
| `dandi:000140` | 2 | 11 | can_0015 |
| `dandi:000351` | 2 | 11 | can_0016 |
| `allen:ophys_529693740` | 2 | 11 | can_0017 |

### Source Breakdown

| Source | FP Count | FN Count |
|--------|----------|----------|
| allen | 26 | 369 |
| bluebrain | 4 | 15 |
| brain_image_library | 3 | 0 |
| buzsaki | 10 | 105 |
| crcns | 92 | 147 |
| dandi | 457 | 908 |
| figshare | 22 | 53 |
| gin | 74 | 164 |
| harvard_dataverse | 149 | 147 |
| ibl | 3 | 366 |
| nemo | 5 | 7 |
| neuromorpho | 0 | 18 |
| neurovault | 203 | 395 |
| openneuro | 25 | 66 |
| osf | 46 | 106 |
| spark | 10 | 7 |
| zenodo | 288 | 455 |

### By Query Intent

| Intent | FP Count | FN Count |
|--------|----------|----------|
| CROSS_DATASET_COMPARISON | 0 | 19 |
| EXPLORATION | 1312 | 3007 |
| PIPELINE_REUSE | 1 | 60 |
| REANALYSIS_FEASIBILITY | 96 | 116 |
| REPLICATION | 3 | 10 |
| STRICT_LOOKUP | 5 | 116 |

### False-Positive Mismatch Breakdown

| Mismatch bucket | Count |
|-----------------|-------|
| task_mismatch | 986 |
| brain_region_mismatch | 669 |
| modality_mismatch | 443 |
| species_mismatch | 417 |
| raw_data_missing | 105 |
| behavioral_event_mismatch | 21 |

### Metadata Missingness Breakdown

| Missing dimension | FP Count | FN Count |
|-------------------|----------|----------|
| affordance | 713 | 1523 |
| behavioral_event | 16 | 51 |
| brain_region | 670 | 862 |
| data_standard | 7 | 4 |
| modality | 510 | 374 |
| other | 83 | 146 |
| raw_data | 68 | 847 |
| species | 408 | 646 |
| task | 978 | 1380 |

### Top False-Positive Failure Modes

| Failure mode | Count |
|--------------|-------|
| wrong_modality | 197 |
| wrong_task | 171 |
| wrong_brain_region | 121 |
| wrong_species | 106 |
| no_raw_data | 60 |
| hard_negative_detected | 32 |
| no_task_match | 31 |
| no_affordance_evidence | 17 |
| no_species | 16 |
| wrong_tasks | 11 |
| no_modality | 9 |
| wrong_modalities | 8 |
| no_modality_match | 7 |
| no_task_evidence | 7 |
| no_modality_evidence | 6 |
| no_brain_region | 6 |
| no_task | 6 |
| no_required_modalities | 6 |
| incorrect_modality | 5 |
| no_tasks | 5 |

### Hard-Negative Failure Modes

| Hard-negative failure mode | Count |
|----------------------------|-------|
| hard_negative_detected | 67 |

## Variant: `hybrid_graph`

| Metric | Count |
|--------|-------|
| False positives (top-K, relevance ≤ 1) | 1417 |
| False negatives (relevant, not in top-K) | 3328 |
| Hard-negative violations | 67 |


### Top False Positives (rank ≤ 10)

| Rank | Record | Relevance | HN | Query | Rationale |
|------|--------|-----------|----|----|-----------|
| 1 | `dandi:001056` | 0 | no | can_0001 | The dataset's task ('center_out_reaching') does not match the query's required ' |
| 1 | `dandi:000070` | 1 | no | can_0004 | The dataset involves reaching tasks, which matches the query's task requirement. |
| 1 | `dandi:000575` | 1 | no | can_0012 | The dataset is weakly related due to species match but lacks explicit evidence f |
| 1 | `zenodo:4307883` | 0 | no | can_0021 | The dataset does not meet the specific requirement for BIDS-formatted neuroimagi |
| 1 | `dandi:001631` | 1 | no | can_0022 | The dataset has some broad scientific concept matches but does not support the i |
| 1 | `zenodo:2598755` | 1 | no | can_0025 | The dataset matches on species and modality but lacks explicit evidence for the  |
| 1 | `dandi:000574` | 1 | no | can_0028 | The dataset matches species and modality but fails to meet the required task cri |
| 1 | `dandi:000987` | 1 | no | can_0030 | The dataset matches the required species and modalities but fails to meet the br |
| 1 | `neurovault:4487` | 1 | no | can_0032 | The dataset matches the brain region and modality but lacks the specific task re |
| 1 | `dandi:000987` | 1 | no | can_0033 | The dataset matches the species and modality but has a different brain region th |
| 1 | `zenodo:11550255` | 0 | **YES** | can_0034 | The dataset matches several query requirements but fails due to the use of Utah  |
| 1 | `harvard_dataverse:10.11588_DATA_V57GNG` | 0 | **YES** | can_0035 | The dataset does not meet the modality requirement for iEEG/ECOG and lacks expli |
| 1 | `crcns:hc-12` | 0 | no | can_0037 | The dataset matches several required dimensions but fails due to involvement in  |
| 1 | `gin:2479` | 1 | no | can_0041 | The dataset matches calcium imaging and hippocampal CA1 region but lacks necessa |
| 1 | `zenodo:10277145` | 1 | no | can_0043 | The dataset is weakly related due to mismatched species and brain region, with a |
| 1 | `dandi:001253` | 1 | no | can_0045 | The dataset matches species, modality, and brain region but not the task or affo |
| 1 | `dandi:001631` | 0 | no | can_0046 | The dataset does not contain data from any of the required brain regions (striat |
| 1 | `dandi:001057` | 1 | no | can_0047 | The dataset matches on species and brain regions, but the task does not align we |
| 1 | `dandi:000231` | 1 | no | can_0050 | The dataset matches the species and includes relevant modalities but does not ma |
| 1 | `dandi:000053` | 0 | no | can_0051 | The dataset does not contain any relevant data for the specified brain regions,  |

### False Negatives (relevant, not in top-K)

| Record | Relevance | Best Rank Outside K | Query |
|--------|-----------|--------------------|----|
| `neurovault:190` | 3 | 11 | can_0107 |
| `zenodo:15446526` | 3 | 15 | can_0075 |
| `dandi:000941` | 3 | 22 | can_0136 |
| `dandi:001209` | 3 | 23 | can_0136 |
| `zenodo:19729161` | 3 | 25 | can_0023 |
| `harvard_dataverse:10.7910_DVN_BQNOMZ` | 3 | 30 | can_0165 |
| `crcns:hc-8` | 3 | 33 | can_0185 |
| `openneuro:ds002422` | 3 | 35 | can_0279 |
| `osf:sb82w` | 3 | 44 | can_0003 |
| `ibl:session_7f56a60c-92c9-42` | 3 | 48 | can_0230 |
| `crcns:pvc-12` | 3 | not retrieved | can_0073 |
| `figshare:7666892` | 3 | not retrieved | can_0086 |
| `zenodo:15098469` | 3 | not retrieved | can_0086 |
| `neurovault:4560` | 2 | 11 | can_0002 |
| `ibl:session_034e726f-b35f-41` | 2 | 11 | can_0005 |
| `spark:SPARK-MULTI-001` | 2 | 11 | can_0010 |
| `dandi:001340` | 2 | 11 | can_0013 |
| `dandi:000140` | 2 | 11 | can_0015 |
| `dandi:000351` | 2 | 11 | can_0016 |
| `allen:ophys_529693740` | 2 | 11 | can_0017 |

### Source Breakdown

| Source | FP Count | FN Count |
|--------|----------|----------|
| allen | 26 | 369 |
| bluebrain | 4 | 15 |
| brain_image_library | 3 | 0 |
| buzsaki | 10 | 105 |
| crcns | 92 | 147 |
| dandi | 457 | 908 |
| figshare | 22 | 53 |
| gin | 74 | 164 |
| harvard_dataverse | 149 | 147 |
| ibl | 3 | 366 |
| nemo | 5 | 7 |
| neuromorpho | 0 | 18 |
| neurovault | 203 | 395 |
| openneuro | 25 | 66 |
| osf | 46 | 106 |
| spark | 10 | 7 |
| zenodo | 288 | 455 |

### By Query Intent

| Intent | FP Count | FN Count |
|--------|----------|----------|
| CROSS_DATASET_COMPARISON | 0 | 19 |
| EXPLORATION | 1312 | 3007 |
| PIPELINE_REUSE | 1 | 60 |
| REANALYSIS_FEASIBILITY | 96 | 116 |
| REPLICATION | 3 | 10 |
| STRICT_LOOKUP | 5 | 116 |

### False-Positive Mismatch Breakdown

| Mismatch bucket | Count |
|-----------------|-------|
| task_mismatch | 986 |
| brain_region_mismatch | 669 |
| modality_mismatch | 443 |
| species_mismatch | 417 |
| raw_data_missing | 105 |
| behavioral_event_mismatch | 21 |

### Metadata Missingness Breakdown

| Missing dimension | FP Count | FN Count |
|-------------------|----------|----------|
| affordance | 713 | 1523 |
| behavioral_event | 16 | 51 |
| brain_region | 670 | 862 |
| data_standard | 7 | 4 |
| modality | 510 | 374 |
| other | 83 | 146 |
| raw_data | 68 | 847 |
| species | 408 | 646 |
| task | 978 | 1380 |

### Top False-Positive Failure Modes

| Failure mode | Count |
|--------------|-------|
| wrong_modality | 197 |
| wrong_task | 171 |
| wrong_brain_region | 121 |
| wrong_species | 106 |
| no_raw_data | 60 |
| hard_negative_detected | 32 |
| no_task_match | 31 |
| no_affordance_evidence | 17 |
| no_species | 16 |
| wrong_tasks | 11 |
| no_modality | 9 |
| wrong_modalities | 8 |
| no_modality_match | 7 |
| no_task_evidence | 7 |
| no_modality_evidence | 6 |
| no_brain_region | 6 |
| no_task | 6 |
| no_required_modalities | 6 |
| incorrect_modality | 5 |
| no_tasks | 5 |

### Hard-Negative Failure Modes

| Hard-negative failure mode | Count |
|----------------------------|-------|
| hard_negative_detected | 67 |

## Variant: `hybrid_rrf`

| Metric | Count |
|--------|-------|
| False positives (top-K, relevance ≤ 1) | 1617 |
| False negatives (relevant, not in top-K) | 3345 |
| Hard-negative violations | 78 |


### Top False Positives (rank ≤ 10)

| Rank | Record | Relevance | HN | Query | Rationale |
|------|--------|-----------|----|----|-----------|
| 1 | `dandi:001633` | 0 | no | can_0001 | The dataset lacks explicit evidence for the required Go/NoGo task and lick event |
| 1 | `zenodo:8207948` | 1 | no | can_0008 | The dataset has matching modality (EEG), but the task mismatch and lack of expli |
| 1 | `osf:8j7g2` | 1 | no | can_0016 | The dataset is weakly related due to matching modality but lacks explicit eviden |
| 1 | `neurovault:3732` | 1 | no | can_0018 | The dataset matches species and modality but lacks explicit evidence for relevan |
| 1 | `dandi:000027` | 1 | no | can_0020 | The dataset lacks explicit evidence for required dimensions, though it has raw d |
| 1 | `osf:vn4yq` | 0 | no | can_0021 | The dataset is not relevant as it does not contain any of the required neuroimag |
| 1 | `zenodo:17425744` | 1 | no | can_0023 | The dataset has some broad scientific concept matches, particularly in mentionin |
| 1 | `dandi:001086` | 0 | no | can_0025 | The dataset does not provide any explicit information about the necessary dimens |
| 1 | `gin:1275` | 1 | no | can_0028 | The dataset matches human species and iEEG modality but lacks explicit mention o |
| 1 | `zenodo:11550255` | 0 | **YES** | can_0034 | The dataset matches several query requirements but fails due to the use of Utah  |
| 1 | `zenodo:10697024` | 1 | no | can_0035 | The dataset matches species but lacks explicit evidence for speech tasks and has |
| 1 | `crcns:hc-12` | 0 | no | can_0037 | The dataset matches several required dimensions but fails due to involvement in  |
| 1 | `neurovault:1323` | 1 | no | can_0038 | The dataset has explicit evidence for species and modality but lacks required ta |
| 1 | `crcns:mesmerize` | 1 | no | can_0041 | The dataset supports calcium imaging but does not provide sufficient information |
| 1 | `zenodo:20402132` | 1 | no | can_0042 | The dataset matches the EEG modality but fails to meet the task requirements spe |
| 1 | `zenodo:10277145` | 1 | no | can_0043 | The dataset is weakly related due to mismatched species and brain region, with a |
| 1 | `neurovault:1323` | 1 | no | can_0045 | The dataset matches species and modality but lacks explicit evidence for tasks a |
| 1 | `dandi:001631` | 0 | no | can_0046 | The dataset does not contain data from any of the required brain regions (striat |
| 1 | `dandi:000072` | 1 | no | can_0047 | The dataset lacks explicit information about required dimensions but has inferre |
| 1 | `dandi:000231` | 1 | no | can_0050 | The dataset matches the species and includes relevant modalities but does not ma |

### False Negatives (relevant, not in top-K)

| Record | Relevance | Best Rank Outside K | Query |
|--------|-----------|--------------------|----|
| `neurovault:5617` | 3 | 13 | can_0003 |
| `zenodo:19729161` | 3 | 13 | can_0023 |
| `dandi:001177` | 3 | 14 | can_0003 |
| `neurovault:2860` | 3 | 16 | can_0003 |
| `neurovault:190` | 3 | 16 | can_0107 |
| `crcns:hc-8` | 3 | 17 | can_0185 |
| `osf:sb82w` | 3 | 20 | can_0003 |
| `harvard_dataverse:10.7910_DVN_BQNOMZ` | 3 | 22 | can_0165 |
| `zenodo:4307883` | 3 | 22 | can_0231 |
| `openneuro:ds002422` | 3 | 26 | can_0279 |
| `dandi:000941` | 3 | 39 | can_0136 |
| `dandi:001209` | 3 | 40 | can_0136 |
| `crcns:pvc-12` | 3 | 47 | can_0073 |
| `figshare:7666892` | 3 | 47 | can_0086 |
| `zenodo:15098469` | 3 | not retrieved | can_0086 |
| `ibl:session_7f56a60c-92c9-42` | 3 | not retrieved | can_0230 |
| `neurovault:1408` | 2 | 11 | can_0013 |
| `dandi:000128` | 2 | 11 | can_0015 |
| `zenodo:16912311` | 2 | 11 | can_0016 |
| `dandi:001532` | 2 | 11 | can_0017 |

### Source Breakdown

| Source | FP Count | FN Count |
|--------|----------|----------|
| allen | 31 | 364 |
| bluebrain | 11 | 11 |
| brain_image_library | 4 | 0 |
| buzsaki | 31 | 81 |
| crcns | 103 | 142 |
| dandi | 447 | 969 |
| figshare | 25 | 54 |
| gin | 121 | 155 |
| harvard_dataverse | 163 | 160 |
| ibl | 1 | 373 |
| nemo | 5 | 7 |
| neuromorpho | 0 | 18 |
| neurovault | 250 | 385 |
| openneuro | 40 | 62 |
| osf | 56 | 101 |
| spark | 9 | 13 |
| zenodo | 320 | 450 |

### By Query Intent

| Intent | FP Count | FN Count |
|--------|----------|----------|
| CROSS_DATASET_COMPARISON | 2 | 19 |
| EXPLORATION | 1485 | 3017 |
| PIPELINE_REUSE | 2 | 58 |
| REANALYSIS_FEASIBILITY | 119 | 124 |
| REPLICATION | 5 | 11 |
| STRICT_LOOKUP | 4 | 116 |

### False-Positive Mismatch Breakdown

| Mismatch bucket | Count |
|-----------------|-------|
| task_mismatch | 1139 |
| brain_region_mismatch | 775 |
| species_mismatch | 541 |
| modality_mismatch | 529 |
| raw_data_missing | 143 |
| behavioral_event_mismatch | 20 |

### Metadata Missingness Breakdown

| Missing dimension | FP Count | FN Count |
|-------------------|----------|----------|
| affordance | 834 | 1548 |
| behavioral_event | 18 | 50 |
| brain_region | 773 | 851 |
| data_standard | 8 | 4 |
| modality | 616 | 380 |
| other | 98 | 156 |
| raw_data | 89 | 834 |
| species | 528 | 653 |
| task | 1135 | 1327 |

### Top False-Positive Failure Modes

| Failure mode | Count |
|--------------|-------|
| wrong_modality | 207 |
| wrong_brain_region | 122 |
| wrong_task | 121 |
| wrong_species | 117 |
| no_raw_data | 83 |
| no_task_match | 55 |
| hard_negative_detected | 42 |
| no_species | 24 |
| no_task_evidence | 21 |
| no_affordance_evidence | 17 |
| no_modality_evidence | 16 |
| no_modality_match | 13 |
| no_modality | 11 |
| no_brain_region | 11 |
| no_tasks | 9 |
| no_task | 9 |
| wrong_modalities | 9 |
| no_species_evidence | 8 |
| no_required_modalities | 7 |
| no_species_match | 7 |

### Hard-Negative Failure Modes

| Hard-negative failure mode | Count |
|----------------------------|-------|
| hard_negative_detected | 77 |
| hard_negative_modality | 1 |
