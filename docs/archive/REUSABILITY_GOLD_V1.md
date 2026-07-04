# Reusability Gold v1 Benchmark

**Version:** 1.0
**Created:** 2026-05-27
**Queries:** 30

## Purpose

This benchmark evaluates whether Neural Search can identify datasets that are **experimentally reusable** for a target scientific analysis, not merely datasets that are **topically similar**.

The key distinction:

| System Type | Answer to "delay discounting datasets" |
|-------------|----------------------------------------|
| **Weak** | Here are datasets that mention "delay" |
| **Strong** | Here are datasets with trial-level choices, reward magnitudes, delay durations, and outcomes - suitable for fitting delay discounting models |

## Benchmark Design

### Query Categories

1. **Ambiguity (5 queries)**: Overloaded scientific terms requiring sense disambiguation
   - delay (discounting vs motor vs signal vs working memory)
   - reward (value vs delivery)

2. **Affordance (7 queries)**: Queries targeting specific analysis capabilities
   - choice decoding
   - Q-learning model fitting
   - motor decoding
   - trial-aligned neural analysis
   - cross-session generalization

3. **Cross-modal/Cross-species (5 queries)**: Queries spanning modalities or species
   - mouse hippocampus spatial navigation
   - primate Neuropixels decision making
   - human fMRI working memory

4. **Natural Language (5 queries)**: Realistic messy user queries
   - "what datasets can i use to study impulsivity"
   - "ephys data with choices and rewards"
   - "something with running speed"

5. **Exact Lookup (3 queries)**: Precise dataset or paper lookups
   - DANDI 000026
   - Steinmetz et al 2019 datasets
   - OpenNeuro ds000117

6. **Complex (5 queries)**: Multi-faceted analysis requirements
   - drift-diffusion modeling
   - multimodal recordings
   - reversal learning

### Graded Relevance

| Score | Meaning |
|-------|---------|
| **3** | Perfect match - dataset fully supports the analysis |
| **2** | Good match - dataset mostly supports, minor gaps |
| **1** | Partial match - some support, missing key elements |
| **0** | Not relevant - wrong sense, missing critical requirements |

### Query Specification Fields

Each query includes:

```yaml
id: rg_v1_001
query: "Find datasets suitable for fitting delay discounting models"
category: ambiguity
intent: analysis_reusability
constructs:
  - delay_discounting
must_have:
  - trial_id
  - choice
  - reward_magnitude
  - delay_duration
should_have:
  - reaction_time
hard_negative_senses:
  - motor_delay
  - signal_delay
affordance_required: delay_discounting_modeling  # optional
notes: "..."
```

## Metrics

### Primary Metrics

1. **Precision@5**: Fraction of top-5 results that are relevant (score ≥ 2)
2. **NDCG@10**: Normalized discounted cumulative gain at 10
3. **MRR**: Mean reciprocal rank of first highly relevant (score = 3) result

### Secondary Metrics

1. **Hard-negative violation rate**: Fraction of results matching wrong senses
2. **Claim support rate**: Fraction of results backed by provenance claims
3. **Missing requirement rate**: How often must-have variables are absent
4. **Affordance accuracy**: Correctness of affordance support predictions

## Delay Discounting: The Key Demo Case

The EBRAINS GUI search for "delay discounting" returns lexical matches including:
- delayed reach-to-grasp (motor delay)
- reward/motivation mentions (could be any reward task)
- signal propagation delay (latency analysis)

Neural Search should distinguish true delay discounting datasets that have:
- ✅ Trial-level choices between immediate and delayed rewards
- ✅ Reward magnitude variables
- ✅ Delay duration variables
- ✅ Outcome labels

From false positives that have:
- ❌ Motor delay periods (reach-to-grasp, instructed delay)
- ❌ Signal propagation delays (conduction latency)
- ❌ Working memory delay periods (maintenance)

## Running the Benchmark

```bash
# Load and validate benchmark
python -m neural_search.evaluation.run_benchmark \
  --suite reusability_gold_v1

# Run with specific metrics
python -m neural_search.evaluation.run_benchmark \
  --suite reusability_gold_v1 \
  --metrics precision_at_5,ndcg_at_10,mrr
```

## Expected Results

A well-functioning system should achieve:

| Metric | Target | Notes |
|--------|--------|-------|
| Precision@5 | ≥ 70% | Top-5 results are usable |
| NDCG@10 | ≥ 0.75 | Good ranking quality |
| MRR | ≥ 0.8 | First good result in top 2 |
| Hard-negative rate | ≤ 5% | Rarely returns wrong senses |

## Benchmark Evolution

This is version 1 with 30 queries. Future versions will:
- Expand to 100+ queries
- Add multi-annotator labels
- Include actual corpus labels (not just specs)
- Add statistical significance testing

## File Locations

- **Benchmark spec**: `data/eval/reusability_gold_v1.yaml`
- **Labels (future)**: `data/eval/reusability_gold_v1_labels.jsonl`
- **Documentation**: `docs/REUSABILITY_GOLD_V1.md`
