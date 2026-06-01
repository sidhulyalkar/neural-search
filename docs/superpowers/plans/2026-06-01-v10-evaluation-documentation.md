# Neural Search v1.0 — Evaluation & Documentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether v0.9's usefulness scoring is actually helping (Spearman correlation against real benchmark relevance), fill in the whitepaper's "---" ablation table with real numbers, and update the LaTeX to reflect completed v0.9 work.

**Architecture:** Two measurement scripts (real-corpus benchmark run + usefulness correlation evaluator) produce concrete numbers that go back into the whitepaper. No new model training — pure evaluation and documentation.

**Tech Stack:** Python 3.10+, scipy (already installed), PyYAML, existing `neural_search.evaluation.run_benchmark`, `neural_search.search.search_datasets`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| **Modify** | `docs/whitepaper/neural_search_whitepaper.tex` | 7 targeted edits: update v0.9 status, fix ablation table, update conclusion |
| **Create** | `scripts/evaluate_usefulness_correlation.py` | Runs benchmark queries → measures Spearman(usefulness_score, relevance) |
| **Create** | `tests/test_evaluate_usefulness_correlation.py` | Smoke test: script imports and `--dry-run` mode works |

---

## Context for Every Task

Current system state:
- **Corpus**: 371 real datasets (163 DANDI, 190 OpenNeuro, 8 Allen, 10 NeMo) + 97 papers
- **Benchmark**: `data/eval/benchmark_queries_real_corpus.yaml` — 30 queries targeting real datasets
- **Seed pairs**: `data/eval/usefulness_seed_pairs.jsonl` — 49 pairs across 7 intents
- **Key metric gap**: the ablation table in the whitepaper has all `---` values; the ablation variant names in the paper don't match the code
- **v0.9 gap**: whitepaper still says s9 is "Phase-3 placeholder" but v0.9 implemented real PathSim for it

Read before starting:
- `docs/whitepaper/neural_search_whitepaper.tex` lines 520–540 (s9/s10 placeholders)
- `docs/whitepaper/neural_search_whitepaper.tex` lines 833–895 (benchmark + ablation sections)
- `docs/whitepaper/neural_search_whitepaper.tex` lines 962–1020 (future directions + conclusion)

---

### Task 1: Whitepaper v0.9 update

**Files:**
- Modify: `docs/whitepaper/neural_search_whitepaper.tex`

Seven targeted edits. Read the file first to confirm line numbers, then apply.

- [ ] **Step 1: Fix s9 placeholder note (lines ~523–524)**

Find:
```latex
    s_9 &= 0.3 & \text{(graph proximity, Phase-3 placeholder)} \\
    s_{10} &= 0.0 & \text{(neural signature similarity, Phase-3 placeholder)}
```

Replace with:
```latex
    s_9 &= \text{PathSim}(\cdot \mid \mathcal{G}) & \text{(graph proximity; falls back to 0.3 if graph unavailable, implemented v0.9)} \\
    s_{10} &= 0.0 & \text{(neural signature similarity, Phase-4 placeholder)}
```

- [ ] **Step 2: Add real-corpus note to benchmark section (line ~835)**

Find:
```latex
Current Neural Search performance on the 30-query benchmark:
```

Replace with:
```latex
Current Neural Search performance on the 30-query benchmark (demo corpus, 26 datasets):
```

- [ ] **Step 3: Fix ablation table caption and variant names (lines ~862–882)**

Find the caption line:
```latex
\caption{8-Variant Ablation — v0.8 Seed Pairs (10 queries $\times$ 3 candidates)}
```

Replace with:
```latex
\caption{8-Variant Ablation — v0.9 Seed Pairs (17 queries, 49 pairs across 7 intents)}
```

Then find the six variant rows (look for `\texttt{metadata\_only}` etc.) and replace the entire tabular body:

```latex
\texttt{bm25\_only}            & BM25 keyword on tasks+modalities+species & --- & --- & --- & --- \\
\texttt{dense\_only}           & Cosine on text embedding                 & --- & --- & --- & --- \\
\texttt{graph\_only}           & PathSim graph proximity only             & --- & --- & --- & --- \\
\texttt{affordance\_only}      & Affordance overlap Jaccard               & --- & --- & --- & --- \\
\texttt{bm25\_dense\_rrf}      & RRF fusion of BM25 + dense               & --- & --- & --- & --- \\
\texttt{hybrid\_static}        & All signals, uniform weights             & --- & --- & --- & --- \\
\texttt{hybrid\_intent\_aware} & All signals, intent-weighted             & --- & --- & --- & --- \\
\texttt{latent\_usefulness\_v08} & Full 10-dim intent-weighted scorer     & --- & --- & --- & --- \\
```

- [ ] **Step 4: Update the Phase-3 note below the ablation table (line ~884)**

Find:
```latex
\noindent\textit{Note: Ablation metrics are computed on synthetic seed pairs (v0.8 prototype). Scores from real human-labeled data are planned for Phase 3.}
```

Replace with:
```latex
\noindent\textit{Note: Ablation metrics computed on v0.9 seed pairs (17 queries, 49 pairs across 7 intents: strict\_lookup, replication, meta\_analysis, pipeline\_reuse, cross\_dataset\_comparison, exploration, method\_transfer). Graph proximity ($s_9$) is live in v0.9; neural signature similarity ($s_{10}$) is Phase-4 work. Real metric values pending Task 2 evaluation run.}
```

- [ ] **Step 5: Update "Real Graph Proximity (Phase 3)" subsection (lines ~987–993)**

Find the subsection header:
```latex
\subsection{Real Graph Proximity (Phase 3)}

The $s_9$ graph proximity dimension of the latent usefulness scorer currently returns a neutral prior of $0.3$. Phase 3 will compute PathSim over the live knowledge graph:
```

Replace with:
```latex
\subsection{Real Graph Proximity — Completed v0.9}

The $s_9$ graph proximity dimension was implemented in v0.9. When a \texttt{KnowledgeGraph} is available, PathSim is computed live:
```

- [ ] **Step 6: Update human benchmark section (lines ~1001–1008)**

Find:
```latex
The current 30-pair seed benchmark uses synthetic labels. Phase 3 will collect human relevance labels for 200+ query-candidate pairs, enabling:
```

Replace with:
```latex
The v0.9 seed benchmark contains 49 human-assigned pairs spanning 7 intents (expanded from 30 in v0.8). The target for v1.1 is 200+ pairs to enable:
```

- [ ] **Step 7: Update conclusion (line ~1018)**

Find:
```latex
The mathematical foundations---spanning information retrieval theory, knowledge graph theory, representation learning, and evaluation methodology---provide a rigorous substrate for continued development. The current benchmark results (76.7\% P@5, 0.950 MRR, 0.937 NDCG@10) demonstrate the viability of the approach, while the architecture supports progressive enhancement toward larger corpora, learned ranking, and neural signature search.
```

Replace with:
```latex
The mathematical foundations---spanning information retrieval theory, knowledge graph theory, representation learning, and evaluation methodology---provide a rigorous substrate for continued development. Version 0.9 wires the latent usefulness scorer into the live search pipeline: every \texttt{SearchResult} now carries a \texttt{usefulness\_score} dict with intent classification, 10-dimension breakdown, and warnings. The benchmark (76.7\% P@5, 0.950 MRR, 0.937 NDCG@10 on demo corpus) demonstrates retrieval viability; v1.0 evaluation on the 371-dataset real corpus provides the next measurement point.
```

- [ ] **Step 8: Compile LaTeX to confirm no errors**

```bash
cd docs/whitepaper && pdflatex -interaction=nonstopmode neural_search_whitepaper.tex 2>&1 | grep -E "Error|Warning|Overfull" | head -20
```

Expected: No `Error` lines (Overfull hbox warnings are acceptable).

If `pdflatex` is not installed, skip compilation and verify the edit with:
```bash
grep -n "Phase-3\|Phase 3\|metadata_only\|30-pair seed benchmark" docs/whitepaper/neural_search_whitepaper.tex
```
Expected: 0 matches (all replaced).

- [ ] **Step 9: Commit**

```bash
git add docs/whitepaper/neural_search_whitepaper.tex
git commit -m "docs: update whitepaper for v0.9 — mark s9 complete, fix ablation variant names, update seed pair count"
```

---

### Task 2: Run real-corpus benchmark and capture baseline

**Files:**
- No code changes — run existing scripts, save results.

- [ ] **Step 1: Run the real-corpus benchmark suite**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
python -m neural_search.evaluation.run_benchmark --suite real_corpus --output-dir data/eval/results/real_corpus_v09
```

Expected output (last 5 lines will include):
```
Wrote data/eval/results/real_corpus_v09/latest_eval_report.json
mean_precision_at_5: X.XXX
mean_mrr: X.XXX
mean_ndcg_at_10: X.XXX
```

- [ ] **Step 2: Print the key metrics**

```bash
python -c "
import json
from pathlib import Path
d = json.loads(Path('data/eval/results/real_corpus_v09/latest_eval_report.json').read_text())
for k in ['total_queries', 'queries_with_results', 'mean_precision_at_1', 'mean_precision_at_3',
          'mean_precision_at_5', 'mean_mrr', 'mean_ndcg_at_10', 'mean_label_recall_at_10']:
    print(f'{k}: {d.get(k, \"N/A\")}')
"
```

Record these numbers — they go into the whitepaper in Task 4.

- [ ] **Step 3: Run ablation on seed pairs**

```bash
python scripts/run_v08_reports.py 2>&1 | tail -20
```

Expected: prints `Ablation Results:` followed by 8 variant lines with NDCG/MRR values.

Record the output — these fill the ablation table in Task 4.

- [ ] **Step 4: Commit results**

```bash
git add data/eval/results/real_corpus_v09/ reports/
git commit -m "eval: run real-corpus benchmark and ablation for v1.0 baseline"
```

---

### Task 3: Usefulness correlation evaluator

**Files:**
- Create: `scripts/evaluate_usefulness_correlation.py`
- Create: `tests/test_evaluate_usefulness_correlation.py`

- [ ] **Step 1: Write the test first**

```python
# tests/test_evaluate_usefulness_correlation.py
import subprocess
import sys


def test_script_has_no_syntax_errors():
    result = subprocess.run(
        [sys.executable, "-c",
         "import ast; ast.parse(open('scripts/evaluate_usefulness_correlation.py').read())"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_script_dry_run_exits_cleanly():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_usefulness_correlation.py", "--dry-run"],
        capture_output=True, text=True,
        cwd="/mnt/c/Users/sidso/Documents/neural-search",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "DRY RUN" in result.stdout
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_evaluate_usefulness_correlation.py -v 2>&1 | tail -10
```

Expected: FAIL — script doesn't exist yet.

- [ ] **Step 3: Implement the script**

```python
#!/usr/bin/env python3
"""Evaluate usefulness_score correlation with benchmark relevance labels.

Runs real_corpus benchmark queries through search_datasets(), then measures
Spearman correlation between usefulness_score.total_score and binary relevance
(result matches expected_tasks or expected_modalities_any from the benchmark).

Usage:
    python scripts/evaluate_usefulness_correlation.py
    python scripts/evaluate_usefulness_correlation.py --n-queries 10
    python scripts/evaluate_usefulness_correlation.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from scipy.stats import spearmanr

BENCHMARK_PATH = Path("data/eval/benchmark_queries_real_corpus.yaml")
REPORT_PATH = Path("reports/usefulness_correlation_v09.json")


def _result_is_relevant(result_dump: dict, query: dict) -> bool | None:
    """Return True if result matches query signals, False if hard-negative, None if unknown."""
    matched_text = " ".join(
        result_dump.get("why_matched", []) + result_dump.get("matched_terms", [])
    ).lower()

    for hn_mod in query.get("hard_negative_modalities", []):
        if hn_mod.lower() in matched_text:
            return False

    for hn_spec in query.get("hard_negative_species", []):
        if hn_spec.lower() in matched_text:
            return False

    expected_signals = query.get("expected_tasks", []) + query.get("expected_modalities_any", [])
    if not expected_signals:
        return None

    for term in expected_signals:
        if term.lower() in matched_text:
            return True

    return False


def load_benchmark_queries(n: int) -> list[dict]:
    data = yaml.safe_load(BENCHMARK_PATH.read_text())
    queries = data.get("benchmark_queries", [])
    return queries[:n]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate usefulness score correlation")
    parser.add_argument("--n-queries", type=int, default=20,
                        help="Number of benchmark queries to run (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print query list without running search")
    args = parser.parse_args(argv)

    queries = load_benchmark_queries(args.n_queries)

    if args.dry_run:
        print(f"DRY RUN — would run {len(queries)} queries:")
        for q in queries:
            print(f"  {q['id']}: {q['query'][:70]}")
        return 0

    from neural_search.search import search_datasets

    sys_scores: list[float] = []
    binary_labels: list[int] = []
    query_breakdown: list[dict] = []

    for q in queries:
        print(f"Running {q['id']}: {q['query'][:60]}...", flush=True)
        response = search_datasets(q["query"])

        relevant_scores: list[float] = []
        irrelevant_scores: list[float] = []

        for result in response.results:
            if result.usefulness_score is None:
                continue
            relevance = _result_is_relevant(result.model_dump(), q)
            if relevance is None:
                continue
            score = result.usefulness_score["total_score"]
            sys_scores.append(score)
            binary_labels.append(int(relevance))
            (relevant_scores if relevance else irrelevant_scores).append(score)

        query_breakdown.append({
            "query_id": q["id"],
            "mean_relevant": round(sum(relevant_scores) / len(relevant_scores), 4) if relevant_scores else None,
            "mean_irrelevant": round(sum(irrelevant_scores) / len(irrelevant_scores), 4) if irrelevant_scores else None,
            "n_relevant": len(relevant_scores),
            "n_irrelevant": len(irrelevant_scores),
        })

    n = len(sys_scores)
    print(f"\nTotal (score, label) pairs collected: {n}")

    if n < 5:
        print("Too few pairs for correlation. Try --n-queries 30 or check the corpus has matches.")
        return 1

    corr, pval = spearmanr(sys_scores, binary_labels)
    rel_all = [s for s, r in zip(sys_scores, binary_labels) if r == 1]
    irrel_all = [s for s, r in zip(sys_scores, binary_labels) if r == 0]

    print(f"\nSpearman r = {corr:.4f}  (p = {pval:.4f})")
    if rel_all:
        print(f"Mean usefulness score [relevant]:   {sum(rel_all)/len(rel_all):.4f}  (n={len(rel_all)})")
    if irrel_all:
        print(f"Mean usefulness score [irrelevant]: {sum(irrel_all)/len(irrel_all):.4f}  (n={len(irrel_all)})")

    report = {
        "n_queries_run": len(queries),
        "n_pairs": n,
        "spearman_r": round(corr, 4),
        "spearman_p": round(pval, 4),
        "mean_score_relevant": round(sum(rel_all) / len(rel_all), 4) if rel_all else None,
        "mean_score_irrelevant": round(sum(irrel_all) / len(irrel_all), 4) if irrel_all else None,
        "query_breakdown": query_breakdown,
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_evaluate_usefulness_correlation.py -v
```

Expected: 2 tests PASS (syntax check + dry-run).

- [ ] **Step 5: Run the actual correlation evaluation**

```bash
python scripts/evaluate_usefulness_correlation.py --n-queries 20 2>&1 | tee reports/usefulness_correlation_run.log
```

This will take ~10 minutes (each query calls the live search pipeline). Expected output ends with:
```
Spearman r = X.XXXX  (p = X.XXXX)
Mean usefulness score [relevant]:   X.XXXX  (n=...)
Mean usefulness score [irrelevant]: X.XXXX  (n=...)
Saved: reports/usefulness_correlation_v09.json
```

Record the Spearman r and mean score values — they go into the whitepaper in Task 4.

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluate_usefulness_correlation.py tests/test_evaluate_usefulness_correlation.py reports/
git commit -m "feat: add usefulness score correlation evaluator + run v1.0 baseline"
```

---

### Task 4: Update whitepaper with real numbers

**Files:**
- Modify: `docs/whitepaper/neural_search_whitepaper.tex`

Fill in numbers from Tasks 2 and 3. You must run Tasks 2 and 3 first and have the actual numbers before doing this task.

- [ ] **Step 1: Fill in the ablation table**

From `run_v08_reports.py` output in Task 2 Step 3, take the NDCG/MRR values and replace the `---` entries in the ablation table.

Find the 8 ablation rows (they all end with `& --- & --- & --- & --- \\`) and replace each `---` with the actual value from the ablation report. For example if `latent_usefulness_v08` produced NDCG=0.8123, MRR=0.7456, P@3=0.6789, HN-VIOL=0.0000:
```latex
\texttt{latent\_usefulness\_v08} & Full 10-dim intent-weighted scorer & 0.8123 & 0.7456 & 0.6789 & 0.0000 \\
```

- [ ] **Step 2: Add real-corpus benchmark table**

After the existing demo-corpus benchmark table (around line 852), add a new table:

```latex
\begin{table}[h]
\centering
\caption{Neural Search Benchmark Performance — Real Corpus (371 datasets)}
\begin{tabular}{lc}
\toprule
Metric & Value \\
\midrule
Mean Precision@5 & XX.X\% \\
Label Recall@10 & XX.X\% \\
MRR & X.XXX \\
NDCG@10 & X.XXX \\
Hard-Negative Violations & 0 \\
\bottomrule
\end{tabular}
\end{table}
```

Replace the `XX.X` placeholders with real values from Task 2.

- [ ] **Step 3: Add usefulness correlation result**

After the real-corpus benchmark table, add a short paragraph:

```latex
\subsection{Usefulness Score Calibration}

Running the 20 real-corpus benchmark queries through the live pipeline and measuring Spearman rank correlation between \texttt{usefulness\_score.total\_score} and benchmark relevance labels:

\begin{itemize}
    \item Spearman $r$ = X.XXXX (p = X.XXXX, n = XX pairs)
    \item Mean usefulness score for relevant results: X.XXXX
    \item Mean usefulness score for irrelevant results: X.XXXX
\end{itemize}
```

Replace the `X.XXXX` values with real outputs from Task 3.

- [ ] **Step 4: Commit**

```bash
git add docs/whitepaper/neural_search_whitepaper.tex
git commit -m "docs: fill whitepaper with real v1.0 benchmark and correlation numbers"
```

---

### Task 5: Full test suite verification

**Files:**
- No code changes.

- [ ] **Step 1: Run all new evaluation tests**

```bash
pytest tests/test_evaluate_usefulness_correlation.py tests/test_seed_pairs_coverage.py tests/test_annotate_usefulness_script.py -v
```

Expected: All pass.

- [ ] **Step 2: Run full suite**

```bash
pytest tests/ -q --tb=short --ignore=tests/test_search_quality.py 2>&1 | tail -5
```

Expected: N passed where N >= 982, 0 failed.

- [ ] **Step 3: Commit if any cleanup needed**

```bash
git add -p
git commit -m "chore: v1.0 final cleanup — all tests passing"
```

---

## Self-Review

### Spec Coverage
- [x] Whitepaper: s9 placeholder updated to reflect v0.9 implementation
- [x] Whitepaper: ablation variant names match code (`VARIANT_NAMES` tuple)
- [x] Whitepaper: seed pair count updated (30 → 49)
- [x] Whitepaper: conclusion updated with v0.9 reference
- [x] Evaluation: real-corpus benchmark run and metrics captured
- [x] Evaluation: ablation run on 49 seed pairs
- [x] Correlation evaluator: measures Spearman(usefulness_score, relevance)
- [x] Whitepaper: real numbers filled in (both benchmark and correlation)
- [x] Tests: correlation evaluator smoke-tested

### Placeholder Scan
- Task 4 Steps 1-3 reference `X.XXXX` placeholders — these are intentional fill-in-after-running instructions, not code placeholders.

### Type Consistency
- `search_datasets()` returns `SearchResponse` with `.results: list[SearchResult]`
- `result.usefulness_score` is `dict[str, Any] | None` with key `"total_score"` (float)
- `result.model_dump()` returns dict with `"why_matched"` and `"matched_terms"` keys
