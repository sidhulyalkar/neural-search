# Silver Qrels Protocol

> **SILVER LABEL DIAGNOSTIC — NOT EXPERT VALIDATION**
> Silver qrels are machine-generated and have not been reviewed by human annotators.
> Do not report silver metrics as final scientific results or include them in the whitepaper.

## What are silver qrels?

Silver qrels are machine-generated relevance judgements produced by combining
rule-based labeling functions, affordance probes, concept-memory signals, and
(optionally) an LLM judge.  They are a development tool to:

- Bootstrap a large coverage of (query, dataset) pairs before human annotation.
- Route the highest-value examples to human reviewers.
- Identify hard-negative violations and missing metadata early.
- Enable rapid iteration on retrieval variants during development.

## Why they are not gold qrels

Gold qrels require human review.  The minimum credible benchmark targets are:

- 100 queries
- 1,500 judged pairs
- 2 annotators on ≥ 30% of pairs
- Adjudication for disagreements

Silver labels cannot substitute for this because:

- Labeling functions use keyword matching, not semantic understanding.
- Concept Memory itself is under evaluation — using it to label its own inputs is circular.
- Affordance probes check structural signals, not whether the science works.
- LLM judges can hallucinate; they see only metadata, not the actual data.

## How labelers vote

Each (query, dataset) pair is evaluated by multiple labelers in parallel:

| Source | Type | Vote range |
|--------|------|------------|
| `rules` | Deterministic rule LFs | 0–3 or abstain |
| `affordance_probe` | Structural affordance check | 0–2 or abstain |
| `concept_memory` | Concept-graph weak signal | 0–2 or abstain |
| `llm_judge` | Optional LLM assessor | 0–3 or abstain |

### Hard-negative override

If **any** labeler votes 0 and marks the vote as a hard-negative violation, that
overrides all other votes.  The final label is 0 regardless of other labeler scores.

### Vote aggregation

1. Check for hard-negative override.
2. Remove abstentions.
3. Compute confidence-weighted mean of remaining votes.
4. Round to nearest integer 0–3.
5. Confidence = 0.5 × agreement_score + 0.5 × mean_labeler_confidence.

## How confidence is computed

Confidence is a [0, 1] float:

- High confidence (≥ 0.70): labelers agree and have high individual confidence.
- Medium confidence (0.45–0.69): some labeler disagreement or missing metadata.
- Low confidence (< 0.45): strong disagreement or mostly abstentions.

**Confidence does NOT equal accuracy.**  It is the system's estimate of how
sure it is; calibration against gold labels (once available) will reveal the
true accuracy curve.

## How review queue selection works

The review queue selector scores every silver entry by:

1. Low confidence → higher priority.
2. Strong labeler disagreement → higher priority.
3. Hard-negative conflict → higher priority.
4. Top-ranked results in any retrieval run → higher priority.
5. Large ranking gap between baseline and concept-full → higher priority.
6. Candidates likely to affect NDCG@10 (relevance 1–2, uncertain).
7. Sparse metadata but non-zero relevance signal.
8. Per-intent undercoverage.
9. Per-source undercoverage.

The `--limit` flag controls how many examples are selected.  The default is 300.

## How to use silver labels for development

### Exploring retrieval behaviour

```bash
python scripts/eval/build_silver_qrels.py \
    --queries artifacts/benchmark_queries.jsonl \
    --pool reports/eval/benchmark_pool.jsonl \
    --corpus data/corpus/normalized/combined_corpus.jsonl \
    --out artifacts/qrels_silver.jsonl \
    --seed 13
```

### Running metrics (development only)

```bash
python scripts/eval/compute_ir_metrics.py \
    --qrels artifacts/qrels_silver.jsonl \
    --run reports/eval/runs/bm25.jsonl \
    --out reports/eval/eval_report_silver.json \
    --allow-silver
```

The output will include `"silver_label_warning"` at the top.  Do not cite
these numbers in the whitepaper.

### Selecting examples for human review

```bash
python scripts/eval/select_human_review_set.py \
    --silver artifacts/qrels_silver.jsonl \
    --queries artifacts/benchmark_queries.jsonl \
    --pool reports/eval/benchmark_pool.jsonl \
    --out artifacts/qrels_review_queue.jsonl \
    --limit 300
```

## How NOT to use silver labels in the whitepaper

- Do not report NDCG, MRR, or any retrieval metric computed from silver labels.
- Do not claim that silver labels demonstrate improvement over a baseline.
- Do not mix silver and gold labels in a single metric computation.
- If you mention the silver qrels system, describe it as a development tool
  that reduces annotation burden, not as a substitute for human labels.

**Acceptable whitepaper wording:**
> To reduce annotation burden, we implemented an automatic silver labeling layer
> that uses rule-based labeling functions, affordance probes, and concept-memory
> signals to pre-screen candidates and prioritise the review queue.  Final
> evaluation metrics are computed exclusively from human-adjudicated gold labels.

## How to calibrate silver against gold labels

Once gold labels exist at `artifacts/qrels.jsonl`:

```bash
python scripts/eval/calibrate_silver_qrels.py \
    --silver artifacts/qrels_silver.jsonl \
    --gold artifacts/qrels.jsonl \
    --out reports/eval/silver_gold_calibration.md
```

This computes:

- Agreement accuracy
- Weighted kappa
- Confusion matrix
- Per-intent and per-source agreement
- High-confidence silver accuracy
- Hard-negative violation precision
- Calibration by confidence bin

A high kappa (≥ 0.60) would justify using silver labels to fill coverage gaps
in future iterations—always with explicit disclosure.

## Evaluation modes

| Mode | Qrels source | Paper credible? | Command flag |
|------|-------------|-----------------|--------------|
| Gold only | `artifacts/qrels.jsonl` | Yes | (default) |
| Silver only | `artifacts/qrels_silver.jsonl` | **No** | `--allow-silver` |
| Gold+silver diagnostic | Gold where available | Not for paper | `--allow-silver` + manual split |

Silver mode reports always include the watermark:

> SILVER LABEL DIAGNOSTIC — NOT EXPERT VALIDATION
