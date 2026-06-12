# Concept Memory Retrieval Integration

## Overview

Graph-Indexed Concept Memory (v0.5) connects the concept graph to the main retrieval pipeline. This document describes what Concept Memory contributes to retrieval, how scoring is computed, how to interpret explanations, and how to run ablations.

---

## What Concept Memory Contributes to Retrieval

Before v0.5, retrieval was purely lexical (query tokens matched against dataset titles/descriptions). Concept Memory adds three additional signals:

| Signal | Description |
|--------|-------------|
| **Concept boost** | Datasets connected to query-matched concept nodes receive a bonus proportional to the concept match score and type weight |
| **Evidence boost** | Additional bonus for evidence links that are reviewed or carry strong provenance |
| **Hard-negative penalty** | Reduction for datasets linked to `failure_mode` concept nodes matching the query context |

All three signals are optional and composable. They are **disabled by default** in the base retrieval pipeline and only active when explicitly requested via CLI or config.

---

## How Scoring Is Computed

Every final score decomposes into four named components:

```
final_score = base_score + concept_boost + evidence_boost − hard_negative_penalty
```

### `base_score` (lexical)

Computed by `_lexical_score(query, dataset_concept)`:
- Query is tokenized (split on whitespace + punctuation, lowercased)
- Each token is matched against the dataset's `canonical_name` (weight 3.0), `aliases` (weight 2.0), `description` (weight 1.0), and `tags` (weight 1.5)
- Score = sum of matched weights / (3.0 × number of tokens), clipped to [0, 1]

### `concept_boost`

1. Query is run through `search_concepts()` to find the top 100 matching non-dataset concept nodes (tasks, modalities, species, brain regions, methods, etc.).
2. For each dataset, its graph adjacency is inspected for evidence links to those query-matched concepts.
3. Boost per link = `concept_match_score × type_weight × link.confidence`
4. Type weights (configured in `CONCEPT_TYPE_BOOST_WEIGHTS`):
   - `task`: 0.40
   - `modality`: 0.30
   - `species`, `brain_region`: 0.15 each
   - `method`, `analysis_affordance`: 0.10 each
   - Others: ≤ 0.08
5. Raw boost is capped at 1.0, then scaled by `concept_boost_scale` (default 0.30).

### `evidence_boost`

For each evidence link from the dataset to a query-matched concept:
- `strong` → +0.05
- `moderate` → +0.03
- `weak` → +0.01
- `none` → +0.0
- Any reviewed link → at least +0.01

Raw evidence boost is capped at 1.0, then scaled by `evidence_boost_scale` (default 0.10).

### `hard_negative_penalty`

For each evidence link connecting the dataset to a `failure_mode` concept node: −0.05, capped at −0.10 total.

---

## Running Reranking

### CLI

```bash
python -m neural_search.field_state.cli concept-rerank \
    --query "Neuropixels spike sorting in awake behaving mice" \
    --limit 10 \
    --field neuroscience_dataset_reuse
```

**Flags:**
- `--lexical-only`: disables all concept/evidence boosts, scores lexically only
- `--limit N`: number of results to return

**Output:** Ranked list with per-result score decomposition printed to stdout.

### Python API

```python
from neural_search.field_state.concept_memory.reranker import rerank_from_artifacts

results = rerank_from_artifacts(
    query="Neuropixels spike sorting in awake behaving mice",
    limit=10,
    enable_concept_boost=True,
    enable_evidence_boost=True,
    enable_hard_negative_penalty=True,
)
for r in results:
    print(r.dataset_title, r.final_score)
    print("  breakdown:", r.score_breakdown())
```

---

## Interpreting Explanations

`concept-explain` shows why a specific dataset scored as it did for a given query.

```bash
python -m neural_search.field_state.cli concept-explain \
    --query "Neuropixels spike sorting in awake behaving mice" \
    --dataset-id "dandiset_000001" \
    --field neuroscience_dataset_reuse
```

### Output structure

| Section | Description |
|---------|-------------|
| **Score Breakdown** | base, concept_boost, evidence_boost, penalty, and final scores |
| **Matched Concepts** | Concept nodes that were matched, with match score, concept type, and evidence texts |
| **Missing Evidence** | Expected concept types (task, modality, species, etc.) not found for this dataset |
| **Hard-Negative Conflicts** | Failure-mode concept nodes linked to this dataset |

### Python API

```python
from neural_search.field_state.concept_memory.explainer import explain_from_artifacts

explanation = explain_from_artifacts(
    query="spike sorting electrophysiology mouse",
    dataset_id="dandiset_000001",
)
print(explanation.explanation_markdown)
# Or serialize:
print(explanation.model_dump_json(indent=2))
```

---

## Running Ablation Evaluation

`concept-eval` compares four retrieval variants over benchmark queries and (if available) adjudicated qrels.

```bash
python -m neural_search.field_state.cli concept-eval \
    --qrels artifacts/field_state/adjudicated_qrels.jsonl \
    --queries artifacts/benchmark_queries.jsonl \
    --field neuroscience_dataset_reuse \
    --out reports/eval/concept_memory_eval.md
```

### Variants

| Variant | Description |
|---------|-------------|
| `lexical_only` | Base lexical score only |
| `concept_boost` | Lexical + concept graph boost |
| `concept_boost_ev` | Lexical + concept boost + evidence boost |
| `full` | All signals including hard-negative penalty |

### Metrics

| Metric | Description |
|--------|-------------|
| NDCG@10 | Normalized Discounted Cumulative Gain at rank 10 (graded relevance 0–3) |
| MRR | Mean Reciprocal Rank (binary relevance threshold ≥ 1) |
| Recall@10 | Fraction of relevant datasets retrieved in top 10 |
| Recall@50 | Fraction of relevant datasets retrieved in top 50 |
| HN Violation Rate | Fraction of top-10 results that are hard-negative violations |
| Source Skew | Concentration of top-10 results from a single data source |

### When qrels are absent

The command exits cleanly and writes a placeholder report with instructions for generating and labelling qrels. No metrics are fabricated.

---

## Concept Coverage Audit

`concept-coverage` shows how thoroughly the corpus has been annotated with concept links — helping identify enrichment targets.

```bash
python -m neural_search.field_state.cli concept-coverage \
    --field neuroscience_dataset_reuse \
    --out reports/field_state/concept_memory_coverage.md
```

### Report sections

- **Overall summary**: total datasets, well-covered vs. bare
- **Coverage by concept type**: count and % of datasets with each concept type linked
- **Coverage by source repository**: per-source breakdown
- **Underrepresented types**: types below 20% coverage threshold
- **Poor-coverage sources**: sources where <10% of datasets have any core concept link
- **Alias normalization examples**: sample of alias→canonical mappings applied at index time

---

## Limitations

The following limitations apply and should not be overstated:

1. **Concept matching is lexical, not semantic.** The query-concept search uses token overlap, not dense embeddings. Synonym misses are possible.

2. **Evidence links are sparsely reviewed.** Most evidence links have `review_status = "unreviewed"`. The evidence boost is small by design until more links are adjudicated.

3. **Qrels are incomplete.** As of v0.5, the adjudicated qrels file is empty. Ablation metrics cannot be computed until relevance judgements are provided.

4. **Graph boost has not been scientifically validated.** The concept boost increases ranking scores, but improvement over the lexical baseline has not been confirmed with qrels-backed evaluation. Do not cite these scores in publications until validated.

5. **Hard-negative detection is coarse.** The penalty is triggered only by explicit `failure_mode` concept links, which are rare in the current graph. This signal will strengthen as more failure-mode concepts are indexed.

---

## Validation Status

| Claim | Status |
|-------|--------|
| Concept boost improves NDCG@10 over lexical baseline | **Unvalidated** — qrels needed |
| Evidence boost improves ranking quality | **Unvalidated** — qrels needed |
| Hard-negative penalty reduces HN violation rate | **Unvalidated** — HN annotations incomplete |
| Score decomposition is correct (components sum to final) | **Validated** — unit tests pass |
| Graceful degradation when artifacts absent | **Validated** — integration tests pass |
