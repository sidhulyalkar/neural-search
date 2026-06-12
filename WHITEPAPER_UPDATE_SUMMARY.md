# Whitepaper Update Summary

## Phase 2: Validation Gap Implementation (this session, continued)

### Infrastructure shipped

- **`scripts/eval/run_retrieval_baselines.py`** rewritten with real BM25 + usefulness-reranking retrieval. Handles stable `{source}:{source_id}` ID scheme. Supports `--variants bm25 usefulness` and `--top-k`.
- **`scripts/eval/compute_ir_metrics.py`** extended with: `precision_at_k`, `hard_negative_violation_rate`, `source_skew_at_k`, `bootstrap_ci` (with seed for reproducibility), `aggregate_metrics` (mean + CI95 per metric), multiple `--run` flag support.
- **`scripts/eval/build_benchmark_pool.py`** updated with priority ordering: consensus pairs (appearing in both BM25 and usefulness variants) sorted first, then by minimum rank.
- **`scripts/eval/annotate_candidates.py`** created: interactive annotation CLI with resume support, per-query limits, consensus-only filter, stats-only mode, validate mode.

### Real artifacts generated

- **`reports/eval/runs/bm25.jsonl`**: 15 queries × 100 results = 1,500 rows (field-weighted BM25 over 10K corpus).
- **`reports/eval/runs/usefulness.jsonl`**: 15 queries × 100 results = 1,500 rows (usefulness-reranked from BM25 top-200).
- **`reports/eval/benchmark_pool.jsonl`**: 914 total pairs; 586 consensus pairs (in both variants); priority-ordered, status=needs\_annotation.

### Test coverage

- `tests/test_eval_metrics.py` extended to 50 tests (was 34); covers `hard_negative_violation_rate`, `source_skew_at_k`, `bootstrap_ci`, `aggregate_metrics`, `precision_at_k`.
- Total: **135 tests passing**.

### Next required action (annotation)

The retrieval infrastructure is complete. The blocking gap is human-annotated qrels. Recommended session:

```bash
python scripts/eval/annotate_candidates.py \
    --pool reports/eval/benchmark_pool.jsonl \
    --queries artifacts/benchmark_queries.jsonl \
    --corpus data/corpus/normalized/combined_corpus.jsonl \
    --out artifacts/qrels.jsonl \
    --consensus-only \
    --limit-per-query 20 \
    --resume
```

`--consensus-only --limit-per-query 20` reduces the task to ~300 pairs (20 × 15 queries from the 586 consensus pool). Expected annotation time: 2–4 hours. Labels save after each pair; Ctrl-C is safe.

Once `artifacts/qrels.jsonl` exists:

```bash
# Compute NDCG, MRR, hard-neg violation rate, calibration
python scripts/eval/compute_ir_metrics.py \
    --qrels artifacts/qrels.jsonl \
    --run reports/eval/runs/bm25.jsonl \
    --run reports/eval/runs/usefulness.jsonl \
    --out reports/eval/eval_report.json

python scripts/eval/compute_calibration.py \
    --qrels artifacts/qrels.jsonl \
    --run reports/eval/runs/usefulness.jsonl \
    --out reports/eval/calibration_report.json
```

---

## Phase 1: Whitepaper and Test Infrastructure (this session, earlier)

## What changed (this session)

- Added `Section: From Engineering Validation to Retrieval Science Validation` to the whitepaper. This section defines the distinction between engineering metrics (corpus size, vector-index recall, latency, graph perturbation) and scientific retrieval metrics (NDCG, hard-negative violation rate, inter-annotator agreement, ECE, prospective case studies). Includes a reference table (Table 1).
- Added a `Scientific Benchmark Design` subsection with 9 intent classes and 4 hard-negative examples.
- Added a `Graded Relevance Labels` subsection with the 0–3 scale and annotation protocol requirements.
- Fixed two `theorem` environments to `definition`: BM25 Scoring Function and Compatibility Score are now `definition` environments. No true theorem/proof was provided for either; the `definition` label is scientifically accurate.
- Created `scripts/eval/compute_calibration.py`: computes Expected Calibration Error (ECE) per score bucket once adjudicated qrels are available. Outputs `reports/eval/calibration_report.json`.
- Created `tests/fixtures/mini_corpus/`: 10 deterministic records, 5 benchmark queries (5 intent classes), and 18 qrels. All query IDs and record IDs cross-validated. Used by smoke tests without touching the 10K corpus.
- Created `tests/test_eval_metrics.py`: 34 unit tests covering DCG, NDCG@K, MRR, Recall@K, mean, ECE, hard-negative violation rate logic, and mini-corpus schema validation.
- Created `tests/test_usefulness_scoring.py`: 29 unit tests covering inactive dimension renormalization, missing-metadata-not-rewarded, hard-negative scenarios, intent profile correctness, score bounds, and evidence/warning emission.
- Created `tests/test_corpus_artifacts.py`: 14 tests covering source distribution consistency, field completeness consistency, duplicate detection, license handling, malformed record detection, and embedding manifest checks.
- Created `tests/smoke/test_end_to_end_retrieval.py`: 22 smoke tests running mini corpus → usefulness scorer → IR metrics → calibration → script import verification. Deterministic, no GPU/network needed.
- Created `artifacts/benchmark_queries.jsonl`: 15 scientific benchmark queries with intents META_ANALYSIS, MODEL_VALIDATION, PIPELINE_REUSE, CROSS_DATASET_COMPARISON, REANALYSIS_FEASIBILITY, METHOD_TRANSFER, REPLICATION, EXPLORATION. Each includes required evidence fields and known failure modes.

## Previous session changes (Codex)

- Consolidated whitepaper around `docs/whitepaper/neural_search_whitepaper.tex`; removed duplicate ICLR file.
- Added validation protocol section, limitations section, and reproducibility artifacts section.
- Added artifact-generated tables (corpus/index state, source distribution, field completeness, claims vs evidence).
- Reframed `Recall@50 = 1.0` as vector-index validation, not scientific relevance.
- Updated usefulness scorer: inactive dimensions excluded, active weights renormalized, `s10` neural signature excluded.
- Fixed missing-metadata handling (empty Jaccard = 0.0, warnings emitted).
- Cleaned citations.
- Created all 7 `scripts/eval/` scripts (freeze, pool, baselines, metrics, ablation, audit, tables).

## Current test status

```
135 passed
```

All 135 tests pass across:
- `tests/test_usefulness_scorer.py` (17)
- `tests/test_eval_artifacts.py` (3)
- `tests/test_eval_metrics.py` (50)
- `tests/test_usefulness_scoring.py` (29)
- `tests/test_corpus_artifacts.py` (14)
- `tests/smoke/test_end_to_end_retrieval.py` (22)

## Preliminary results

- Corpus: 10,404 unique normalized records.
- Dense embedding/index: 60,175 BGE-large field embeddings, 10,404 indexed ids.
- Vector-index Recall@50 = 1.0 (engineering metric only; see whitepaper Section 2).
- Usefulness scorer signal: Spearman r ≈ 0.40 over 270 labeled pairs (preliminary; needs independent labels).
- Graph proximity changes 39% of ranked pairs (behavioral metric; relevance improvement pending benchmark).

These are not final end-to-end scientific retrieval claims.

## Required before publication claims

1. Freeze the 10K corpus snapshot: run `python scripts/eval/freeze_corpus_snapshot.py` and commit `corpus_manifest.json`, `embedding_manifest.json`, and `graph_manifest.json`.
2. Create adjudicated qrels: annotate 100–150 benchmark queries (start from `artifacts/benchmark_queries.jsonl`) with at least two annotators, compute inter-annotator agreement, and write `artifacts/qrels.jsonl`.
3. Run retrieval baselines: `python scripts/eval/run_retrieval_baselines.py` and populate run files with actual ranked results from BM25, dense BGE, RRF, hybrid, graph, and usefulness variants.
4. Compute IR metrics: `python scripts/eval/compute_ir_metrics.py` to get `eval_report.json` with NDCG@10, NDCG@20, MRR, Recall@50 (vs qrels), hard-negative violation rate.
5. Run ablation suite: `python scripts/eval/run_ablation_suite.py` to compare variants.
6. Compute calibration: `python scripts/eval/compute_calibration.py` for ECE once scored run files and qrels are available.
7. Manual extraction audits: per-source precision for modality, species, task, and affordance labels.
8. Prospective case studies: 3–5 studies where a researcher uses the system to find and evaluate real datasets.

## Commands

```bash
# Run all tests
python -m pytest tests/test_usefulness_scorer.py tests/test_eval_artifacts.py \
  tests/test_eval_metrics.py tests/test_usefulness_scoring.py \
  tests/test_corpus_artifacts.py tests/smoke/test_end_to_end_retrieval.py -q

# (Already done) Generate real run files
python scripts/eval/run_retrieval_baselines.py \
  --queries artifacts/benchmark_queries.jsonl \
  --corpus data/corpus/normalized/combined_corpus.jsonl \
  --out-dir reports/eval/runs/ \
  --top-k 100 --variants bm25 usefulness

# (Already done) Build annotation pool
python scripts/eval/build_benchmark_pool.py \
  --runs-dir reports/eval/runs --out reports/eval/benchmark_pool.jsonl --depth 100

# NEXT STEP: Annotate ~300 consensus pairs
python scripts/eval/annotate_candidates.py \
  --pool reports/eval/benchmark_pool.jsonl \
  --queries artifacts/benchmark_queries.jsonl \
  --corpus data/corpus/normalized/combined_corpus.jsonl \
  --out artifacts/qrels.jsonl \
  --consensus-only --limit-per-query 20 --resume

# After annotation: compute IR metrics
python scripts/eval/compute_ir_metrics.py \
  --qrels artifacts/qrels.jsonl \
  --run reports/eval/runs/bm25.jsonl \
  --run reports/eval/runs/usefulness.jsonl \
  --out reports/eval/eval_report.json

# After annotation: compute calibration
python scripts/eval/compute_calibration.py \
  --qrels artifacts/qrels.jsonl \
  --run reports/eval/runs/usefulness.jsonl \
  --out reports/eval/calibration_report.json

# Corpus artifact reports
python scripts/eval/freeze_corpus_snapshot.py
python scripts/eval/audit_extraction_quality.py
python scripts/eval/generate_paper_tables.py

# Compile whitepaper (requires TeX environment or Overleaf)
cd docs/whitepaper && pdflatex neural_search_whitepaper.tex && pdflatex neural_search_whitepaper.tex
```

Local note: `pdflatex` is not installed in this WSL environment. Compile in Overleaf or a TeX-enabled shell.

## Validation gaps remaining

| Gap | Status |
|-----|--------|
| Run files: BM25 + usefulness (15 queries × 100) | **Done** — `reports/eval/runs/` |
| Annotation pool built (914 pairs, 586 consensus) | **Done** — `reports/eval/benchmark_pool.jsonl` |
| Adjudicated qrels (≥150 pairs) | **Pending** — run annotate\_candidates.py |
| NDCG@10/MRR from qrels-based evaluation | Pending qrels |
| Hard-negative violation rate from qrels | Pending qrels |
| Source skew @ K | Pending qrels |
| ECE calibration report | Pending qrels |
| Inter-annotator agreement computed | Pending second annotator |
| Manual extraction quality audit | Pending per-source review |
| Prospective case studies (3–5) | Pending researcher evaluation |
| Content-level neural signature similarity | Not implemented (excluded from scorer) |
