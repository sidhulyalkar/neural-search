# Neuro-Judge Mock Collapse Diagnostics

> DIAGNOSTIC ONLY - neuro-judge/mock labels are not human gold and must not be reported as scientific validation.

## Collapse Assessment

- Severe label collapse: `False`
- Label-2 fraction: `0.3659`
- Likely cause: `no_severe_label_collapse_detected`

## Label Distribution

| Label | Count |
|---|---:|
| 0 | 117 |
| 1 | 175 |
| 2 | 247 |
| 3 | 136 |

## Confidence Distribution

| Confidence | Count |
|---|---:|
| <0.5 | 0 |
| 0.5-0.7 | 401 |
| 0.7-0.9 | 138 |
| >=0.9 | 136 |

## Evidence Match Rates

| Dimension | Required | Matched | Missing | Match Rate |
|---|---:|---:|---:|---:|
| species | 435 | 255 | 180 | 0.5862 |
| modality | 407 | 292 | 115 | 0.7174 |
| brain_region | 313 | 92 | 221 | 0.2939 |
| task | 485 | 186 | 299 | 0.3835 |
| affordance | 99 | 17 | 82 | 0.1717 |
| raw_data | 56 | 0 | 56 | 0.0 |

## Raw And Processed Evidence

- Raw data present: `0`
- Raw data absent/uncertain: `675`
- Processed-only evidence count: `0`
- Hard-negative warning count: `2`

## Missing Expected Dimensions

| Dimension | Count |
|---|---:|
| task | 299 |
| brain_region | 221 |
| species | 180 |
| modality | 115 |
| affordance | 82 |
| raw_data | 56 |

## Commonly Absent Evidence Fields

| Field | Count |
|---|---:|
| file_format_evidence | 331 |
| dataset_brain_regions | 321 |
| dataset_tasks | 210 |
| dataset_species | 185 |
| dataset_modalities | 130 |
| description | 109 |

## Label-2 Rule Reasons

| Reason | Count |
|---|---:|
| mock_rule:task_mismatch | 160 |
| mock_rule:missing_raw_data | 54 |
| mock_rule:missing_required_evidence | 23 |
| mock_rule:missing_affordance | 10 |

## Label-2 Examples By Reason

### mock_rule:task_mismatch
- `q_0001` / `neurovault:4778` confidence=0.66 missing=['task'] rationale=mock: task evidence does not match the query
- `q_0001` / `neurovault:1323` confidence=0.66 missing=['task'] rationale=mock: task evidence does not match the query
- `q_0001` / `openneuro:ds004323` confidence=0.66 missing=['task'] rationale=mock: task evidence does not match the query

### mock_rule:missing_raw_data
- `q_0003` / `ibl:ibl_np1_np2_kim` confidence=0.62 missing=['raw_data'] rationale=mock: correct core match but explicit raw data evidence is missing
- `q_0003` / `ibl:ibl_repeated_site` confidence=0.62 missing=['raw_data'] rationale=mock: correct core match but explicit raw data evidence is missing
- `q_0003` / `ibl:ibl_brain_wide_map_2023` confidence=0.62 missing=['raw_data'] rationale=mock: correct core match but explicit raw data evidence is missing

### mock_rule:missing_affordance
- `q_0005` / `zenodo:5592702` confidence=0.64 missing=['affordance'] rationale=mock: required analysis affordance is not evidenced
- `q_0005` / `zenodo:16997337` confidence=0.64 missing=['affordance'] rationale=mock: required analysis affordance is not evidenced
- `q_0005` / `osf:7k5w4` confidence=0.64 missing=['affordance'] rationale=mock: required analysis affordance is not evidenced

### mock_rule:missing_required_evidence
- `q_0008` / `zenodo:19115395` confidence=0.68 missing=['brain_region'] rationale=mock: partial match with missing required evidence
- `q_0009` / `neurovault:1883` confidence=0.7 missing=['brain_region'] rationale=mock: partial match with missing required evidence
- `q_0009` / `neurovault:109` confidence=0.7 missing=['brain_region'] rationale=mock: partial match with missing required evidence
