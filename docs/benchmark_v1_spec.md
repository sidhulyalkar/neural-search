# Neural Search Benchmark v1 Specification

**Status:** Frozen draft — pending first qrels batch  
**Version:** 1.0  
**Field:** neuroscience_dataset_reuse

---

## Purpose

This document defines the frozen benchmark specification for evaluating Neural Search retrieval quality under realistic neuroscience dataset reuse queries. It governs query construction, qrels annotation, metric computation, and what claims the benchmark can and cannot support.

This spec exists to prevent:
- Evaluation on trivially easy queries
- Post-hoc cherry-picking of favorable results
- Metric fabrication when qrels are incomplete
- Conflating structural correctness with scientific retrieval quality

---

## Query Intents

Seven intent types are defined. Every benchmark query must specify one.

| Intent | Description |
|--------|-------------|
| `EXACT_LOOKUP` | Retrieve a specific known dataset by title, ID, or precise description |
| `REPLICATION` | Find datasets suitable for replicating a published experiment |
| `PIPELINE_REUSE` | Find datasets compatible with an existing analysis pipeline |
| `CROSS_DATASET_COMPARISON` | Find datasets supporting cross-study or cross-modality comparisons |
| `META_ANALYSIS` | Find datasets for pooled statistical analysis across studies |
| `METHOD_TRANSFER` | Find datasets suitable for applying a method from a different domain |
| `EXPLORATION` | Open-ended discovery of datasets matching a broad scientific theme |

> **Note on existing queries:** The 15 queries in `artifacts/benchmark_queries.jsonl` use legacy intent labels (`MODEL_VALIDATION`, `REANALYSIS_FEASIBILITY`). These are acceptable aliases and are handled by the validator.

---

## Query Schema (v1)

Each benchmark query is a JSONL record with the following fields.

```json
{
  "query_id": "q_0001",
  "query_text": "human fMRI reward prediction error reinforcement learning task",
  "intent": "META_ANALYSIS",
  "scientific_goal": "Identify datasets suitable for cross-study reward-learning meta-analysis.",
  "must_have": ["species:human", "modality:fMRI", "task:reward_prediction_error"],
  "nice_to_have": ["doi", "preprocessing_pipeline", "sample_size", "events_file"],
  "hard_negatives": [
    "resting-state fMRI with reward words in description",
    "animal RL task when human-only requested"
  ],
  "expected_modalities": ["fMRI"],
  "expected_species": ["human"],
  "expected_tasks": ["reward_prediction_error", "reinforcement_learning"],
  "expected_brain_regions": [],
  "notes": ""
}
```

### Field definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query_id` | string | yes | Stable unique identifier: `q_NNNN` |
| `query_text` | string | yes | Natural language query (also accepted as `query` for backward compat) |
| `intent` | string | yes | One of the 7 intent types above |
| `scientific_goal` | string | yes | One sentence describing the scientific reuse goal |
| `must_have` | list[string] | yes | Constraints that must be met for relevance ≥ 2 |
| `nice_to_have` | list[string] | no | Fields that improve relevance but are not blocking |
| `hard_negatives` | list[string] | yes | Descriptions of datasets that look relevant but are not |
| `expected_modalities` | list[string] | no | Modalities expected in relevant results |
| `expected_species` | list[string] | no | Species expected in relevant results |
| `expected_tasks` | list[string] | no | Tasks expected in relevant results |
| `expected_brain_regions` | list[string] | no | Brain regions expected in relevant results |
| `notes` | string | no | Annotator notes, caveats, or ambiguity flags |

### Backward compatibility aliases

| Spec field | Alias accepted |
|------------|----------------|
| `query_text` | `query` |
| `must_have` | `required_evidence` |
| `hard_negatives` | `known_failure_modes` |

---

## Qrels Schema (v1)

Each relevance judgement is a JSONL record.

```json
{
  "query_id": "q_0001",
  "dataset_id": "neurovault:1323",
  "relevance": 2,
  "label": "partial_match",
  "rationale": "fMRI modality matches. Human subjects confirmed. Missing events file and task metadata.",
  "hard_negative_violation": false,
  "missing_metadata": ["events_file", "task_labels"],
  "annotator_id": "annotator_01",
  "timestamp": "2026-06-10T14:30:00+00:00",
  "adjudicated": false,
  "adjudication_notes": ""
}
```

### Field definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query_id` | string | yes | Must match a query in the benchmark query file |
| `dataset_id` | string | yes | Must match a dataset in the corpus (format: `source:id`) |
| `relevance` | int [0–3] | yes | Graded relevance score — see scale below |
| `label` | string | no | Human-readable label summarizing the judgement |
| `rationale` | string | yes for 0 and 3 | Explanation, required when relevance is 0 or 3 |
| `hard_negative_violation` | bool | yes | True if this result exemplifies a stated hard negative |
| `missing_metadata` | list[string] | no | Metadata fields absent from the dataset |
| `annotator_id` | string | yes | Identifier for the annotator |
| `timestamp` | string | yes | ISO 8601 timestamp |
| `adjudicated` | bool | yes | Whether disagreement has been adjudicated |
| `adjudication_notes` | string | no | Notes from adjudicator if applicable |

---

## Relevance Scale

| Score | Label | Definition |
|-------|-------|------------|
| `3` | **Highly relevant** | Strongly reusable for the stated scientific goal. Modality, species, task, brain region all match. Sufficient metadata. Clean provenance. |
| `2` | **Partially relevant** | Matches the primary scientific goal but is missing one important constraint (e.g., modality matches but no task metadata, or species matches but brain region does not). Plausibly reusable with caveats. |
| `1` | **Weakly relevant** | Superficially related but not directly reusable. Correct domain but wrong species, wrong modality, or critically missing metadata. |
| `0` | **Not relevant** | Off-topic, violates a hard negative, or is scientifically incompatible with the query goal. |

### Hard negative rule

A dataset with `relevance = 0` that also matches a stated `hard_negatives` description **must** have `hard_negative_violation = true`. Annotators should explicitly mark these — they drive the hard-negative violation rate metric.

---

## Judging Rules

### Task match

- Score ≥ 2 requires the dataset's task labels or task-related description to overlap with `expected_tasks`.
- Broad task categories (e.g., "reinforcement learning") match narrow subtypes (e.g., "reward prediction error") unless the query explicitly requires the narrow type.
- If `expected_tasks` is empty, task match is not required for ≥ 2.

### Modality match

- Score ≥ 2 requires modality to match `expected_modalities` where this field is non-empty.
- `"fMRI"` does not match `"EEG"`. `"two-photon imaging"` matches `"calcium imaging"`.
- If the dataset's modality is unlabeled but the description implies the correct modality, score 2 (not 3).

### Species match

- Score ≥ 2 requires species to match `expected_species` where this field is non-empty.
- `"mouse"` does not match `"rat"`. `"primate"` matches `"macaque"` and `"marmoset"`.
- If species is absent from metadata but title implies correct species, score 2 (not 3).

### Brain-region match

- Only required for score 3 when `expected_brain_regions` is non-empty.
- Broad region (e.g., "visual cortex") matches narrow region (e.g., "V1") unless query is specific.

### Analysis affordance

- Score 3 requires sufficient metadata for the stated reuse: trial structure for event-related analysis, raw traces for spike sorting, etc.
- Missing events files, raw data, or preprocessing details reduce score to 2.

### Missing metadata

- A dataset with unknown species, modality, or task cannot score ≥ 3.
- List missing fields in `missing_metadata` for every pair where this affects the score.

### Ambiguous datasets

- When a dataset could plausibly score 1 or 2, prefer 1 unless clear reusability is demonstrated.
- Flag ambiguity in `rationale`.

### Hard negatives

- When a dataset explicitly matches a `hard_negatives` description for the query, score 0 and set `hard_negative_violation = true`.
- Example: for a human fMRI query, a high-quality mouse ephys dataset scores 0 + hard_negative_violation = true (modality mismatch).

---

## Benchmark Splits

| Split | Purpose | Minimum size |
|-------|---------|--------------|
| `development` | Used during system development and iteration | 40 queries, 600 pairs |
| `held_out_test` | Frozen, evaluated only at submission/paper-writing time | 60 queries, 900 pairs |
| `smoke` | Tiny fixture for CI/automated tests | 5 queries, 30 pairs |

The development split is public to developers. The held-out test split is not used for any system tuning.

Each query file record should include a `split` field: `"development"`, `"held_out_test"`, or `"smoke"`.

---

## Minimum Credible Benchmark

For a benchmark report to support paper-ready claims:

| Requirement | Minimum |
|-------------|---------|
| Total queries | 100 |
| Judged query-dataset pairs | 1,500 |
| Dual-annotated pairs | ≥ 30% of all pairs |
| Intent types covered | All 7 intents |
| Sources covered | ≥ 5 distinct repositories |
| Adjudicated disagreements | 100% of disagreements adjudicated |
| Hard-negative pairs | ≥ 50 explicitly judged |

Current status: **0 adjudicated qrels. Benchmark is pending.**

---

## Metric Definitions

| Metric | Definition |
|--------|------------|
| NDCG@10 | Normalized Discounted Cumulative Gain at rank 10 (graded relevance 0–3) |
| NDCG@20 | NDCG at rank 20 |
| MRR | Mean Reciprocal Rank (binary threshold: relevance ≥ 2) |
| Precision@10 | Fraction of top-10 results with relevance ≥ 2 |
| Recall@10 | Fraction of all relevant pairs (≥ 2) retrieved in top 10 |
| Recall@50 | Same, top 50 |
| HN Violation Rate | Fraction of top-10 that are hard-negative violations |
| Source Skew | Entropy-normalized concentration of top-10 results by source |
| Bootstrap 95% CI | 1000-sample bootstrap CI for NDCG@10, MRR, Recall@50, HN rate |

---

## Variants for Ablation

| Variant | Description |
|---------|-------------|
| `bm25` | BM25 sparse retrieval only |
| `usefulness` | BM25 re-ranked by usefulness scorer |
| `concept_boost` | Usefulness + concept memory graph boost |
| `concept_boost_ev` | + evidence boost |
| `concept_full` | + hard-negative penalty |
| `concept_full_no_hn_penalty` | Without hard-negative penalty |
| `concept_full_no_missingness` | Without missingness penalty |
| `dense_bge` | BGE-large dense retrieval (if GPU available) |
| `dense_plus_concept` | Dense + concept memory (if GPU available) |

---

## What This Benchmark Can and Cannot Support

### Can support (when qrels are complete)

- "Variant X improves NDCG@10 over Y on judged queries."
- "Concept memory reduces hard-negative violations for pipeline reuse queries."
- "Performance degrades for queries with sparse metadata coverage."
- "Source skew is lower for concept-reranked results."

### Cannot support (ever)

- Claims about future reuse outcomes in the field
- Claims derived from qrels with fewer than 100 judged queries
- Claims based on non-adjudicated labels
- Claims that development-split results generalize to held-out test

---

## File Locations

| Artifact | Path |
|----------|------|
| Benchmark queries | `artifacts/benchmark_queries.jsonl` |
| Small fixture | `tests/fixtures/benchmark_queries_small.jsonl` |
| Candidate pool | `reports/eval/benchmark_pool.jsonl` |
| Qrels | `artifacts/qrels.jsonl` |
| Run files | `reports/eval/runs/*.jsonl` |
| Eval report | `reports/eval/benchmark_v1_results.md` |
| Eval JSON | `reports/eval/benchmark_v1_results.json` |
| By-intent CSV | `reports/eval/benchmark_v1_by_intent.csv` |
| By-source CSV | `reports/eval/benchmark_v1_by_source.csv` |
| Failures report | `reports/eval/benchmark_v1_failures.md` |
| Result card gallery | `reports/eval/result_card_gallery.md` |
