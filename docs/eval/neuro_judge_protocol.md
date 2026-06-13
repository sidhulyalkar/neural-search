# Neuro-Judge Protocol

> PRELIMINARY NEURO-JUDGE EVALUATION — RAG-GROUNDED LLM LABELS, NOT PURE HUMAN GOLD

## Overview

The neuro_judge is a RAG-grounded LLM adjudicator for query-dataset relevance scoring. It assigns 0–3 relevance labels to pooled qrels candidates using explicit evidence assembled from dataset metadata, concept memory, linked papers, affordance probes, and neuroscience-specific rules.

Neuro-judge labels are **not** human gold. They are preliminary silver labels with richer evidence grounding than the basic silver pipeline. Use them to prioritize annotation and validation, not to replace human review on high-impact pairs. Do not scale judge labels across the benchmark until collapse diagnostics and human calibration exist.

## Label Provenance

| Provenance | Meaning |
|---|---|
| `neuro_judge` | Single judge, no RAG context |
| `neuro_judge_rag` | Single judge, concept-memory evidence injected |
| `neuro_judge_consensus` | Two or more judges agree (per consensus rules) |
| `expert_audited_consensus` | Consensus reviewed and accepted by a domain expert |
| `human_gold` | Reserved — never emitted by the judge pipeline |

## Scoring Rubric

| Score | Meaning |
|---|---|
| 0 | Not relevant. Fails on species, modality, or task. May share only surface keywords. |
| 1 | Weakly related. Broad concept match but significant gaps. Unlikely to support the intended analysis. |
| 2 | Useful with caveats. Likely supports part of the goal; missing or imperfect evidence. |
| 3 | Highly relevant. Directly supports the query. Correct species, modality, region, task, and affordance. |

## Neuroscience-Specific Rules

These rules override general scoring heuristics:

- fMRI-only datasets are **0** for spike/LFP/theta/place-cell queries.
- Correct modality + wrong brain region → **1** for replication queries.
- Correct species + modality, uncertain raw data → **2**.
- Human imaging and rodent electrophysiology are **not interchangeable**.
- MT/V5 ≠ LIP; PFC ≠ LIP; entorhinal cortex ≠ hippocampus (unless query explicitly allows).
- Calcium-imaging events ≠ extracellular spikes.
- ALF/processed spike tables ≠ raw AP-band data.
- NWB/BIDS/ALF format = evidence of accessibility, not suitability.
- Do **not** infer raw data availability without explicit evidence.

## Pipeline

```
artifacts/benchmark_queries.jsonl
  + artifacts/field_state/qrels_candidates_pooled.jsonl
  + (optional) concept rerank results
        ↓
scripts/eval/build_neuro_judge_evidence.py
        ↓
artifacts/field_state/neuro_judge_evidence_packets.jsonl
        ↓
scripts/eval/run_neuro_qrels_judge.py   (--backend anthropic|openai|local_hf|mock)
        ↓
artifacts/field_state/neuro_qrels_judgments.jsonl
        ↓
scripts/eval/consensus_neuro_qrels.py
        ↓
artifacts/field_state/neuro_qrels_consensus.jsonl
artifacts/field_state/neuro_qrels_conflicts.jsonl
        ↓
scripts/eval/audit_neuro_qrels.py  (requires human labels)
        ↓
reports/eval/neuro_judge_audit.md
```

## Consensus Rules

| Condition | Outcome |
|---|---|
| Exact agreement + mean_confidence ≥ 0.75 | `neuro_judge_consensus` |
| Label diff = 1 + mean_confidence ≥ 0.80 | Consensus with `minor_disagreement=True` |
| Label diff ≥ 2 | Conflict → `neuro_qrels_conflicts.jsonl` |
| HN detection differs between judges | Conflict |
| Label ≥ 2 but missing required dimension | Flagged for human audit |
| High NDCG@10 impact | Flagged for human audit |

## Backends

| Backend | Requirements |
|---|---|
| `mock` | None — deterministic plumbing, diagnostics, and regression tests only; not scientific relevance labels |
| `anthropic` | `ANTHROPIC_API_KEY` + `anthropic` package |
| `openai` | `OPENAI_API_KEY` + `openai` package |
| `local_hf` | `transformers` + `torch`; optional `bitsandbytes` for quantization |
| `braingpt` | `transformers` + BrainGPT model weights (skips gracefully if absent) |

## Evidence Packet Fields

Each packet contains:

- **Query side**: id, text, intent, hard negatives, expected species/modalities/regions/tasks/affordances
- **Dataset side**: id, title, source archive/URL, description, modalities, species, brain regions, tasks, data standards, license
- **Derived evidence**: linked papers (title + abstract), affordance matches, concept explanation summary, matched concept names, missing evidence from concept memory, hard-negative conflicts
- **Raw data signals**: has_raw_data (inferred), has_processed_data (inferred), file format evidence
- **Warnings**: pre-screened hard-negative warnings

## Output Fields per Judgment

| Field | Type | Description |
|---|---|---|
| `label` | int 0–3 | Relevance score |
| `confidence` | float 0–1 | Judge confidence |
| `rationale_short` | str | 1–2 sentence explanation |
| `evidence_for` | list[str] | Supporting signals |
| `evidence_against` | list[str] | Counter signals |
| `missing_information` | list[str] | Evidence gaps |
| `matched_dimensions` | list[str] | Which dimensions matched |
| `failure_modes` | list[str] | Detected failure modes |
| `hard_negative_detected` | bool | Hard-negative violation flag |
| `judge_model` | str | Model used |
| `prompt_version` | str | Prompt version |
| `evidence_packet_hash` | str | SHA-256 prefix of evidence packet |
| `label_provenance` | str | One of the provenance types above |
| `evidence_completeness` | float 0–1 | Fraction of required dimensions with explicit supporting evidence |
| `required_dimensions_present` | list[str] | Required dimensions supported by evidence |
| `required_dimensions_missing` | list[str] | Required dimensions absent or uncertain |
| `abstain_recommended` | bool | True when the judge should not be treated as a usable label without human review |
| `abstain_reason` | str/null | Short explanation for abstention |

## Evidence Completeness And Abstention

Evidence completeness is computed over required dimensions such as species, modality, brain region, task, affordance, and raw data. Format signals such as NWB, BIDS, and ALF count as accessibility evidence, not suitability evidence. Raw-data availability must be explicit; processed spike tables or ALF files do not prove raw AP-band access.

If a record receives label ≥2 while required dimensions are missing or uncertain, the judge must set `abstain_recommended=true` or an equivalent human-audit failure mode. Abstention is not a negative label; it is a routing signal saying the pair requires expert review before it can support metrics or claims.

## Collapse Diagnostics

Before scaling any neuro-judge labels, run:

```bash
python scripts/eval/diagnose_neuro_judge_collapse.py \
  --evidence artifacts/field_state/neuro_judge_evidence_packets.jsonl \
  --judgments artifacts/field_state/neuro_qrels_judgments.jsonl \
  --consensus artifacts/field_state/neuro_qrels_consensus.jsonl \
  --out reports/eval/neuro_judge_mock_collapse_diagnostics.md \
  --json-out reports/eval/neuro_judge_mock_collapse_diagnostics.json
```

The report checks label distribution, confidence distribution, evidence completeness, missing dimensions, raw-data evidence, hard-negative warnings, and label-2 rule reasons. If one label dominates, treat the backend as uncalibrated.

## Validation Sample Protocol

Select a stratified sample before running a real backend:

```bash
python scripts/eval/select_neuro_judge_validation_set.py \
  --evidence artifacts/field_state/neuro_judge_evidence_packets.jsonl \
  --judgments artifacts/field_state/neuro_qrels_judgments.jsonl \
  --candidates artifacts/field_state/qrels_candidates_pooled.jsonl \
  --n 150 --seed 42 --require-diversity --include-missing-evidence \
  --out artifacts/field_state/neuro_judge_validation_sample.jsonl \
  --summary reports/eval/neuro_judge_validation_sample_summary.md
```

Run Anthropic/OpenAI only on this validation sample first. Human reviewers should audit the sample, calibrate judge agreement, inspect false-high/false-low cases, and only then decide whether broader silver-labeling is warranted.

## Calibration Metrics

The audit script reports:

- **Exact agreement**: fraction of pairs where judge label = human label
- **Agreement within 1**: fraction where |judge − human| ≤ 1
- **Quadratic weighted kappa (QWK)**: standard ordinal agreement statistic
- **Confusion matrix**: 4×4 (rows = human, columns = judge)
- **False-high / false-low examples**: for qualitative error analysis
- **Breakdowns**: by query intent, modality, failure mode

## Safeguards

- `human_gold` provenance **cannot** be emitted by the judge pipeline (validated at model level).
- All report headers include the watermark: `PRELIMINARY NEURO-JUDGE EVALUATION — RAG-GROUNDED LLM LABELS, NOT PURE HUMAN GOLD`
- Gold qrels at `artifacts/qrels.jsonl` are never written by the silver/judge pipeline.
- Neuro-judge qrels used in metric reports require `--allow-silver` and receive a `neuro_judge_warning` field in the JSON report.
- Silver/judge labels used in metric reports must display a warning watermark and must be reported separately from expert-validated results.
- Mock-backend labels are diagnostic artifacts only. They must not be reported as scientific relevance labels.

## Scientific Basis

The rubric and neuroscience-specific rules are informed by empirical findings on how researchers evaluate dataset reuse potential, including modality specificity, species generalizability, and data accessibility constraints. See also: Saxe et al. (2024), *Nature Human Behaviour* — empirical benchmarking of neuroscience dataset retrieval systems.
