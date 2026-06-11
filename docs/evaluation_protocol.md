# Evaluation Protocol

**Field:** neuroscience_dataset_reuse  
**System:** Neural Search v0.6  
**Spec reference:** [docs/benchmark_v1_spec.md](benchmark_v1_spec.md)

---

## Overview

This document describes the end-to-end evaluation protocol for measuring retrieval quality in the Neural Search benchmark. It covers how runs are produced, how candidates are pooled for annotation, how metrics are computed, and what claims a completed evaluation can support.

---

## Retrieval Variants

Each evaluation compares variants of the retrieval pipeline:

| Variant | Script | Description |
|---------|--------|-------------|
| `bm25` | `run_retrieval_baselines.py` | BM25 sparse retrieval |
| `usefulness` | `run_retrieval_baselines.py` | BM25 + usefulness scorer reranking |
| `concept_boost` | `run_concept_ablation.py` | Usefulness + concept memory graph boost |
| `concept_boost_ev` | `run_concept_ablation.py` | + evidence boost |
| `concept_full` | `run_concept_ablation.py` | + hard-negative penalty |

Variants are run from the same query set and produce JSONL run files in `reports/eval/runs/`.

---

## Run File Format

Each run file contains one line per (query, result) pair:

```json
{
  "query_id": "q_0001",
  "record_id": "openneuro:ds000001",
  "rank": 1,
  "score": 0.923
}
```

`record_id` follows the format `{source}:{source_id}` (e.g., `dandi:000003`, `openneuro:ds000001`).

---

## Candidate Pooling

The annotation pool is built by merging the top-K results from all active variants. Candidates that appear in more variants are assigned higher priority for annotation.

```bash
python scripts/eval/sample_candidate_pool.py \
    --queries artifacts/benchmark_queries.jsonl \
    --runs-dir reports/eval/runs \
    --out reports/eval/benchmark_pool.jsonl \
    --depth 50
```

This adds the `concept_rerank` strategy when field-state artifacts are present.

Pool schema:

```json
{
  "query_id": "q_0001",
  "record_id": "openneuro:ds000001",
  "pooled_from": ["bm25", "concept_full", "usefulness"],
  "min_rank": 1,
  "priority": 3,
  "status": "needs_annotation"
}
```

Prioritize annotation by `priority` (descending) then `min_rank` (ascending).

---

## Annotation Workflow

Annotation is performed using the interactive CLI tool:

```bash
python scripts/eval/annotate_candidates.py \
    --pool reports/eval/benchmark_pool.jsonl \
    --corpus data/corpus/normalized/combined_corpus.jsonl \
    --queries artifacts/benchmark_queries.jsonl \
    --qrels artifacts/qrels.jsonl
```

For each candidate the annotator sees:
- The query and scientific goal
- `must_have` constraints
- `hard_negatives` descriptions
- Dataset title, description, and metadata

The annotator assigns a score from 0–3. See [qrels_annotation_guide.md](qrels_annotation_guide.md) for judging rules.

### Resuming

The annotation tool supports resume mode. Progress is persisted to `artifacts/qrels.jsonl` after each judgement.

### Dual annotation

At least 30% of pairs should be independently annotated by a second annotator. Disagreements (|score_A - score_B| ≥ 2) must be adjudicated.

---

## Qrels Validation

After annotating, validate the qrels file:

```bash
python scripts/eval/validate_qrels.py \
    artifacts/qrels.jsonl \
    --queries artifacts/benchmark_queries.jsonl
```

The validator checks:
- All records parse correctly
- No duplicate (query_id, dataset_id, annotator_id) triples
- All query_ids exist in the benchmark query file
- Rationale present for relevance 0 and 3
- `hard_negative_violation=True` only on relevance=0 entries

---

## Metric Computation

Once qrels are ready, compute IR metrics:

```bash
python scripts/eval/compute_ir_metrics.py \
    --qrels artifacts/qrels.jsonl \
    --run reports/eval/runs/bm25.jsonl \
    --run reports/eval/runs/usefulness.jsonl \
    --run reports/eval/runs/concept_full.jsonl \
    --out reports/eval/benchmark_v1_results.json
```

Metrics computed per variant:
- NDCG@10, NDCG@20
- MRR (relevance threshold ≥ 2)
- Precision@10, Recall@10, Recall@50
- Hard-negative violation rate
- Source skew (entropy-normalized)
- Bootstrap 95% CI for NDCG@10, MRR, Recall@50

---

## Failure Analysis

```bash
python scripts/eval/analyze_failures.py \
    --qrels artifacts/qrels.jsonl \
    --queries artifacts/benchmark_queries.jsonl \
    --runs-dir reports/eval/runs \
    --out reports/eval/benchmark_v1_failures.md
```

The failure report covers:
- False positives by rank, source, and intent
- False negatives (relevant, not retrieved in top-K)
- Hard-negative violation breakdown

---

## Result Card Gallery

```bash
python scripts/eval/build_result_card_gallery.py \
    --qrels artifacts/qrels.jsonl \
    --queries artifacts/benchmark_queries.jsonl \
    --corpus data/corpus/normalized/combined_corpus.jsonl \
    --runs-dir reports/eval/runs \
    --out reports/eval/result_card_gallery.md
```

Produces a Markdown file with success/failure/hard-negative examples.

---

## Coverage Audit

```bash
python -m neural_search.field_state.cli concept-coverage \
    --field neuroscience_dataset_reuse \
    --out reports/eval/concept_coverage.md
```

Shows which dataset types and concept types are covered by the concept memory graph.

---

## Minimum Credible Claims

Evaluation results support a paper-ready claim only when:

1. ≥ 100 queries with adjudicated qrels
2. ≥ 1,500 judged pairs
3. All 7 intent types represented
4. ≥ 5 distinct sources in the corpus
5. Bootstrap 95% CI does not include 0 improvement

If these conditions are not met, conclusions should be framed as preliminary results on a small development split.

---

## Output File Inventory

| File | Description |
|------|-------------|
| `reports/eval/benchmark_v1_results.json` | Full metric table across variants |
| `reports/eval/benchmark_v1_results.md` | Human-readable summary |
| `reports/eval/benchmark_v1_by_intent.csv` | Per-intent breakdown |
| `reports/eval/benchmark_v1_by_source.csv` | Per-source breakdown |
| `reports/eval/benchmark_v1_failures.md` | False positives and negatives |
| `reports/eval/result_card_gallery.md` | Example result cards |
| `reports/eval/concept_coverage.md` | Concept memory coverage audit |
| `artifacts/qrels_silver.jsonl` | Machine-generated silver qrels (development only) |
| `artifacts/qrels_review_queue.jsonl` | Examples selected for human review |
| `reports/eval/silver_qrels_summary.md` | Silver qrels coverage and confidence report |
| `reports/eval/silver_gold_calibration.md` | Calibration of silver vs gold (after annotation) |

---

## Reproducing Results

All evaluation scripts read from explicit file paths and produce deterministic output given the same inputs. To reproduce:

```bash
# 1. Run retrieval variants
python scripts/eval/run_retrieval_baselines.py \
    --queries artifacts/benchmark_queries.jsonl \
    --corpus data/corpus/normalized/combined_corpus.jsonl

# 2. Pool candidates
python scripts/eval/sample_candidate_pool.py

# 3. Annotate (interactive)
python scripts/eval/annotate_candidates.py

# 4. Validate qrels
python scripts/eval/validate_qrels.py artifacts/qrels.jsonl \
    --queries artifacts/benchmark_queries.jsonl

# 5. Compute metrics
python scripts/eval/compute_ir_metrics.py \
    --qrels artifacts/qrels.jsonl \
    --run reports/eval/runs/bm25.jsonl \
    --run reports/eval/runs/usefulness.jsonl

# 6. Analyze failures
python scripts/eval/analyze_failures.py

# 7. Build gallery
python scripts/eval/build_result_card_gallery.py

# --- Silver qrels workflow (development / annotation triage only) ---

# Build silver labels
python scripts/eval/build_silver_qrels.py \
    --queries artifacts/benchmark_queries.jsonl \
    --pool reports/eval/benchmark_pool.jsonl \
    --corpus data/corpus/normalized/combined_corpus.jsonl \
    --out artifacts/qrels_silver.jsonl \
    --seed 13

# Select review queue
python scripts/eval/select_human_review_set.py \
    --silver artifacts/qrels_silver.jsonl \
    --queries artifacts/benchmark_queries.jsonl \
    --pool reports/eval/benchmark_pool.jsonl \
    --out artifacts/qrels_review_queue.jsonl \
    --limit 300

# Calibrate silver vs gold (after annotation)
python scripts/eval/calibrate_silver_qrels.py \
    --silver artifacts/qrels_silver.jsonl \
    --gold artifacts/qrels.jsonl \
    --out reports/eval/silver_gold_calibration.md
```

> **Silver qrels are development diagnostics only.**
> See [docs/silver_qrels_protocol.md](silver_qrels_protocol.md) for full guidance on
> what silver labels are, how they are generated, and how NOT to use them in the whitepaper.
