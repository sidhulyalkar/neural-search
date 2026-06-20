# Weak Supervision Labeling Pipeline

## Overview

Instead of requiring a human to label every query-dataset pair, the weak
supervision pipeline applies 13 deterministic labeling functions (LFs) and
an optional LLM rubric judge, then aggregates their votes into tiered qrels.

## Pipeline Steps

```
corpus + queries
       ↓
build_evidence.py          → artifacts/eval/pair_evidence.jsonl
       ↓
run_labeling_functions.py  → artifacts/eval/label_function_votes.jsonl
       ↓
judge_candidates.py        → artifacts/eval/llm_judgments.jsonl  (optional)
       ↓
build_qrels_from_votes.py  → qrels_gold / qrels_silver / qrels_bronze
       ↓
export_audit_queue.py      → obsidian_vault/05_Annotations/Human Audits/
       ↓
[human audits in Obsidian]
       ↓
import_audits.py           → qrels_gold.jsonl (gold labels)
```

## Labeling Functions

| LF | What it checks |
|---|---|
| `lf_hard_negative` | Candidate matches a known failure mode → label 0 |
| `lf_required_modality` | Required modalities present? |
| `lf_partial_modality` | Preferred modalities present? |
| `lf_species_constraint` | Required species present? |
| `lf_task_constraint` | Task constraints met? |
| `lf_region_constraint` | Brain region constraints met? |
| `lf_data_level_required` | Raw/processed data requirements met? |
| `lf_raw_data_available` | Raw data available? |
| `lf_license_reusable` | License is open? |
| `lf_metadata_completeness` | Metadata completeness score |
| `lf_analysis_affordance` | Analysis affordance signals present? |
| `lf_pipeline_reuse` | Standardised format (NWB/BIDS) for pipeline-reuse queries? |
| `lf_meta_analysis_depth` | Behavioral metadata depth for meta-analysis queries? |

Source: `neural_search/eval/labeling_functions.py`

## Vote Aggregation

Votes are aggregated in `neural_search/eval/label_ensemble.py`:

1. **Hard-negative override**: if `lf_hard_negative` fires, the pair is
   immediately labeled 0 at confidence 0.95 — no other votes matter.
2. **Confidence-weighted average**: remaining active votes are weighted by
   their per-LF confidence and averaged to a label in [0, 3].
3. **Tier assignment**: a pair reaches **silver** only when ≥3 LFs agree,
   mean confidence ≥ 0.75, and variance < 0.5. Everything else is bronze.

## Qrel Tiers

| Tier | Source | Use for |
|---|---|---|
| **Gold** | Human-audited labels | Scientific claims, whitepaper metrics |
| **Silver** | ≥3 LFs agree, conf ≥ 0.75, low variance | Development, ablation studies |
| **Bronze** | All remaining weak labels | Exploratory debugging only |

**Never cite silver or bronze metrics as final scientific validation.**
The `--qrels-tier` flag in `compute_ir_metrics.py` emits a stderr warning
when the tier is not gold.

## Running the Full Pipeline

```bash
# 1. Build evidence pairs from the candidate pool
python scripts/eval/build_evidence.py \
    --pool reports/eval/benchmark_pool.jsonl \
    --queries artifacts/benchmark_queries.jsonl \
    --corpus data/corpus/normalized/combined_corpus.jsonl \
    --out artifacts/eval/pair_evidence.jsonl

# 2. Run labeling functions
python scripts/eval/run_labeling_functions.py \
    --evidence artifacts/eval/pair_evidence.jsonl \
    --out artifacts/eval/label_function_votes.jsonl

# 3. (Optional) LLM rubric judge — requires ANTHROPIC_API_KEY
python scripts/eval/judge_candidates.py \
    --evidence artifacts/eval/pair_evidence.jsonl \
    --config configs/judges/rubric_judge_v1.yaml \
    --out artifacts/eval/llm_judgments.jsonl

# 4. Build qrel tiers
python scripts/eval/build_qrels_from_votes.py \
    --evidence artifacts/eval/pair_evidence.jsonl \
    --votes artifacts/eval/label_function_votes.jsonl \
    --llm artifacts/eval/llm_judgments.jsonl \
    --out-gold artifacts/qrels_gold.jsonl \
    --out-silver artifacts/qrels_silver.jsonl \
    --out-bronze artifacts/qrels_bronze.jsonl \
    --audit-queue artifacts/eval/audit_queue.jsonl
```

## Hard-Negative Analysis

After running retrieval, check whether hard negatives appear in top-k:

```bash
python scripts/eval/hard_negative_analysis.py \
    --qrels-silver artifacts/qrels_silver.jsonl \
    --qrels-bronze artifacts/qrels_bronze.jsonl \
    --run reports/eval/runs/usefulness.jsonl \
    --out reports/eval/hard_negative_report.json
```

## Current Corpus Stats (as of 2026-06-16)

- Pair evidence: 494 pairs (15 queries × corpus)
- Silver qrels: 175 pairs
- Bronze qrels: 319 pairs
- Audit queue: 242 pairs (high-priority for human review)
