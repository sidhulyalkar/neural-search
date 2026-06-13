# Neuro-Judge Audit Report

> PRELIMINARY NEURO-JUDGE EVALUATION — RAG-GROUNDED LLM LABELS, NOT PURE HUMAN GOLD
These labels are produced by an LLM judge with retrieval-augmented evidence.
They have NOT been reviewed by domain experts and must not be reported as final results.

**Judgment file**: `artifacts/field_state/neuro_qrels_consensus_mock.jsonl`  
**Conflicts file**: `artifacts/field_state/neuro_qrels_conflicts_mock.jsonl`  

## Summary

| Metric | Value |
|--------|-------|
| Total pairs | 675 |
| Conflicts | 0 |
| Hard-negative detected | 2 |
| Abstain recommended | 247 |

## Label Distribution

| Label | Count | % |
|-------|-------|---|
| 0 | 117 | 17% |
| 1 | 175 | 25% |
| 2 | 247 | 36% |
| 3 | 136 | 20% |

## Confidence Distribution

| Range | Count | % |
|-------|-------|---|
| <0.5 | 0 | 0% |
| 0.5–0.7 | 401 | 59% |
| 0.7–0.9 | 138 | 20% |
| >=0.9 | 136 | 20% |

## Evidence Completeness

| Range | Count | % |
|-------|-------|---|
| 0 | 144 | 21% |
| (0,0.5) | 125 | 18% |
| [0.5,0.8) | 270 | 40% |
| [0.8,1.0) | 0 | 0% |
| 1.0 | 136 | 20% |

## Missing Required Dimensions

| Mode | Count |
|------|-------|
| task | 299 |
| brain_region | 221 |
| species | 180 |
| modality | 115 |
| affordance | 82 |
| raw_data | 56 |

## Top Failure Modes

| Mode | Count |
|------|-------|
| needs_human_review | 247 |
| mock_rule:task_mismatch | 161 |
| mock_rule:direct_match | 136 |
| mock_rule:missing_species_evidence | 115 |
| mock_rule:wrong_modality | 115 |
| mock_rule:species_mismatch | 105 |
| mock_rule:wrong_region_for_replication | 69 |
| mock_rule:missing_raw_data | 54 |
| mock_rule:missing_modality_evidence | 33 |
| mock_rule:missing_required_evidence | 23 |

### Label 0 Examples

**q_0001 / zenodo:16914534**
- Label: 0 | Confidence: 0.6
- Rationale: mock: wrong modality for the requested analysis
- Missing: missing_required_dimension:species, missing_required_dimension:modality, missing_required_dimension:task

**q_0001 / openneuro:ds004515**
- Label: 0 | Confidence: 0.6
- Rationale: mock: wrong modality for the requested analysis
- Missing: missing_required_dimension:species, missing_required_dimension:modality, missing_required_dimension:task

**q_0001 / zenodo:16914371**
- Label: 0 | Confidence: 0.82
- Rationale: mock: wrong modality for the requested analysis
- Missing: missing_required_dimension:species, missing_required_dimension:modality, missing_required_dimension:task

### Label 1 Examples

**q_0001 / openneuro:ds005230**
- Label: 1 | Confidence: 0.6
- Rationale: mock: species mismatch prevents direct relevance
- Missing: missing_required_dimension:species

**q_0001 / zenodo:16965124**
- Label: 1 | Confidence: 0.6
- Rationale: mock: species mismatch prevents direct relevance
- Missing: missing_required_dimension:species, missing_required_dimension:task

**q_0001 / openneuro:ds005598**
- Label: 1 | Confidence: 0.6
- Rationale: mock: species mismatch prevents direct relevance
- Missing: missing_required_dimension:species

### Label 2 Examples

**q_0001 / neurovault:4778**
- Label: 2 | Confidence: 0.66
- Rationale: mock: task evidence does not match the query
- Missing: missing_required_dimension:task

**q_0001 / neurovault:1323**
- Label: 2 | Confidence: 0.66
- Rationale: mock: task evidence does not match the query
- Missing: missing_required_dimension:task

**q_0001 / openneuro:ds004323**
- Label: 2 | Confidence: 0.66
- Rationale: mock: task evidence does not match the query
- Missing: missing_required_dimension:task

### Label 3 Examples

**q_0001 / openneuro:ds007486**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0001 / openneuro:ds007436**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0001 / openneuro:ds004920**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

### High-Confidence Examples (≥0.9)

**q_0001 / openneuro:ds007486**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0001 / openneuro:ds007436**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0001 / openneuro:ds004920**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0001 / openneuro:ds006576**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0002 / zenodo:17752967**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0004 / crcns:pmd-1**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0004 / dandi:000129**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0004 / dandi:000128**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0004 / dandi:000140**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0004 / dandi:000139**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0004 / dandi:000138**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0004 / gin:11134**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0004 / gin:11184**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0004 / zenodo:11550255**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0004 / gin:11115**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0004 / zenodo:15487063**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0004 / zenodo:18785985**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0006 / ibl:ibl_brain_wide_map_2024**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0006 / crcns:ssc-8**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0006 / zenodo:17990626**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0006 / zenodo:17900730**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0006 / crcns:ssc-2**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0006 / zenodo:18672975**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0006 / zenodo:18601917**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0006 / zenodo:17761469**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0006 / zenodo:17704682**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0006 / zenodo:17360954**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0008 / osf:psbcw**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0008 / zenodo:19118616**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0008 / zenodo:18734619**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0008 / zenodo:11235921**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0008 / crcns:hc-26**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / dandi:001210**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / crcns:pvc-10**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / zenodo:17360954**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / crcns:ssc-8**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_675478137**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_636930038**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_613599793**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / dandi:000167**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_717214654**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_714778358**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_711590640**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_707917316**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_652737678**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_704826374**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_700659215**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_682732631**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_663478400**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_662960692**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_662348706**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / dandi:000579**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / crcns:cai-3**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_685494041**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_650389887**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / dandi:001605**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_663868345**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_657649672**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_660510593**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_654920038**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_681673022**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_674678616**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / dandi:001778**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / dandi:001532**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_667364442**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / zenodo:18613596**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_715923832**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_713568018**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0010 / allen:ophys_710504563**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0012 / openneuro:ds000105**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0012 / zenodo:16697913**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / osf:etuwq**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / zenodo:15222132**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / openneuro:ds005121**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / crcns:ieeg-1**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / openneuro:ds005530**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / openneuro:ds005178**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / openneuro:ds003555**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / openneuro:ds004348**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / openneuro:ds004148**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / zenodo:16850208**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / openneuro:ds006576**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / zenodo:10200482**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / openneuro:ds003768**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / zenodo:7779375**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / openneuro:ds006801**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / zenodo:7899655**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / openneuro:ds003574**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / openneuro:ds006695**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / openneuro:ds005207**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / openneuro:ds006366**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / zenodo:17138539**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0013 / openneuro:ds005555**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / osf:bf7yz**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds007369**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / neurovault:3857**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds005267**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds003342**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / neurovault:594**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / osf:zpnmt**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds006805**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds004103**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds004562**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds005559**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds001419**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds003965**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds005295**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds004909**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds004496**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds004829**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds007354**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds004693**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds001246**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds002685**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds000232**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds005934**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds003452**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds004489**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds007046**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds005226**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds003812**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds006642**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / zenodo:6614683**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds007378**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / neurovault:833**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / neurovault:4743**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / figshare:20328111**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / figshare:20328108**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / figshare:20328102**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / figshare:20328099**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / figshare:20328096**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / figshare:19189889**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds007384**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds007329**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds007275**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

**q_0014 / openneuro:ds006826**
- Label: 3 | Confidence: 0.9
- Rationale: mock: direct match on required dimensions

### Low-Confidence Examples (<0.5)

None.

### High Label but Severe Missing Information

**q_0002 / zenodo:18613596**
- Label: 2 | Confidence: 0.66
- Rationale: mock: task evidence does not match the query
- Missing: missing_required_dimension:brain_region, missing_required_dimension:task

**q_0002 / dandi:000579**
- Label: 2 | Confidence: 0.66
- Rationale: mock: task evidence does not match the query
- Missing: missing_required_dimension:brain_region, missing_required_dimension:task

**q_0002 / gin:13458**
- Label: 2 | Confidence: 0.66
- Rationale: mock: task evidence does not match the query
- Missing: missing_required_dimension:brain_region, missing_required_dimension:task

**q_0002 / gin:12110**
- Label: 2 | Confidence: 0.66
- Rationale: mock: task evidence does not match the query
- Missing: missing_required_dimension:brain_region, missing_required_dimension:task

**q_0003 / crcns:pfc-4**
- Label: 2 | Confidence: 0.62
- Rationale: mock: correct core match but explicit raw data evidence is missing
- Missing: missing_required_dimension:affordance, missing_required_dimension:raw_data

## Human Calibration

No human labels provided. Run with `--human <path>` to enable calibration.
