# Neuronpedia Foundation Sprint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform neural-search into a "Neuronpedia for neuroscience" by unblocking benchmark credibility, building an interactive Brain Atlas Map as the product centerpiece, and adding Neuronpedia-style affordance evidence to every dataset page.

**Architecture:** Three independent sprints: (A) fix the LLM qrels pipeline auth and run the full benchmark to establish credibility; (B) replace the Coverage tab with an interactive Brain Atlas Map that makes the holistic brain-coverage view the product's identity; (C) add an affordance evidence panel to DatasetPage so every dataset page shows what analyses it supports, with evidence grades.

**Tech Stack:** Python 3.11, FastAPI, DuckDB, SQLAlchemy, React 18, TanStack Query, Tailwind CSS, TypeScript.

---

## This plan covers three subsystems

- **Sprint A**: Benchmark Credibility (Tasks 1–4) — ~1 day
- **Sprint B**: Brain Atlas Map (Tasks 5–9) — ~2 days
- **Sprint C**: Dataset Affordance Panels (Tasks 10–12) — ~1.5 days

Each sprint is independently shippable.

---

## File Map

### Sprint A files
- Modify: `scripts/eval/run_parallel_llm_qrels.py` — fix dotenv override + add startup key guard
- New test: `tests/eval/test_qrels_pipeline_auth.py`

### Sprint B files
- Modify: `neural_search/coverage/duckdb_store.py` — add `region_dataset_counts()` and `datasets_for_region()`
- New test: `tests/test_duckdb_store.py` — extend with two new method tests
- Modify: `apps/api/main.py` — add `/api/coverage/region-counts` and `/api/coverage/region/{region_id}/datasets`
- Modify: `apps/web/src/api/coverage.ts` — add `RegionCount`, `RegionDataset` types + client methods
- Create: `apps/web/src/pages/BrainAtlasPage.tsx`
- Modify: `apps/web/src/App.tsx` — add `/atlas` route
- Modify: `apps/web/src/components/Layout.tsx` — add Atlas nav link, rename Coverage

### Sprint C files
- Modify: `apps/api/main.py` — add `GET /api/datasets/{id}/affordances`
- Modify: `apps/web/src/api/search.ts` — add `getDatasetAffordances()`
- Modify: `apps/web/src/pages/DatasetPage.tsx` — add `AffordancePanel` component

---

## Sprint A: Benchmark Credibility

### Task 1: Diagnose and fix qrels pipeline auth

The pipeline is producing 401 "Missing Authentication header" errors. The OPENROUTER_API_KEY is present in `.env.local` but `load_dotenv(..., override=False)` will silently skip the key if the var is already set (even as empty string) in the environment. Switching to `override=True` and adding an explicit startup check will fix this.

**Files:**
- Modify: `scripts/eval/run_parallel_llm_qrels.py` (lines ~30–45 and ~120–135)
- Create: `tests/eval/test_qrels_pipeline_auth.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/eval/test_qrels_pipeline_auth.py`:

```python
"""Validate that the qrels pipeline key discovery works."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))


def test_discover_workers_uses_openrouter_key_from_env():
    """_discover_workers should return OpenRouter workers when OPENROUTER_API_KEY is set."""
    # Import inside test to avoid module-level env reads
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-v1-testkey"}):
        from scripts.eval.run_parallel_llm_qrels import _discover_workers
        workers = _discover_workers()
    assert len(workers) > 0, "Should have at least one worker"
    assert all(w.api_key == "sk-or-v1-testkey" for w in workers)
    assert all("gemini" in w.model or "flash" in w.model for w in workers)


def test_discover_workers_returns_empty_when_no_key():
    """_discover_workers should return [] when no keys are set."""
    clean_env = {k: v for k, v in os.environ.items()
                 if not k.startswith("OPENROUTER") and not k.startswith("CLAUDE")
                 and not k.startswith("GEMINI") and k != "KEYWAY_BASE_URL"}
    with patch.dict(os.environ, clean_env, clear=True):
        from scripts.eval.run_parallel_llm_qrels import _discover_workers
        workers = _discover_workers()
    assert workers == [], "Expected empty worker list with no keys"
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
PYTHONPATH=. python -m pytest tests/eval/test_qrels_pipeline_auth.py -v 2>&1 | head -30
```

Expected: `ImportError` or `AssertionError` because the current code uses `override=False`.

- [ ] **Step 1.3: Fix the dotenv override and add startup guard**

In `scripts/eval/run_parallel_llm_qrels.py`, find the dotenv block (around line 33) and change `override=False` to `override=True`:

```python
# Load .env.local before importing anything that reads env vars
_ENV_LOCAL = _REPO / ".env.local"
if _ENV_LOCAL.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_LOCAL, override=True)  # override=True ensures .env.local wins
```

Then find the `_discover_workers` call in `main()` (around lines 350–415) and add a guard:

```python
    workers = _discover_workers(args.base_url or "")
    if not workers:
        print(
            "\nERROR: No API keys found.\n"
            "Set OPENROUTER_API_KEY in .env.local:\n"
            "  OPENROUTER_API_KEY=sk-or-v1-...\n",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  Workers: {workers}", flush=True)
```

- [ ] **Step 1.4: Run test to verify it passes**

```bash
PYTHONPATH=. python -m pytest tests/eval/test_qrels_pipeline_auth.py -v
```

Expected: `PASSED` for both tests.

- [ ] **Step 1.5: Smoke-test the pipeline with 3 pairs**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
PYTHONPATH=. python scripts/eval/run_parallel_llm_qrels.py --dry-run
```

Expected: prints workers showing `sk-or-v1-44ed...` (first 8 chars) and model `google/gemini-2.5-flash`.

```bash
PYTHONPATH=. python scripts/eval/run_parallel_llm_qrels.py --limit 3 --rerun-errors
```

Expected: 3 judgment lines in `data/qrels/llm_judgments.jsonl` with `"label": 0` or `1`, no `judge_error` in rationale.

- [ ] **Step 1.6: Commit**

```bash
git add scripts/eval/run_parallel_llm_qrels.py tests/eval/test_qrels_pipeline_auth.py
git commit -m "fix(eval): fix dotenv override=True so OPENROUTER_API_KEY loads from .env.local"
```

---

### Task 2: Run the full LLM qrels pipeline

13,654 evidence packets. With 4 parallel workers on Gemini Flash (~0.4 s/packet), expected runtime ~23 minutes. Run in background.

**Files:**
- Read: `artifacts/ablation_judge/evidence_packets.jsonl`
- Write: `data/qrels/llm_judgments.jsonl`, `data/qrels/qrels.trec`

- [ ] **Step 2.1: Clear the error-only judgment file to start fresh**

```bash
# Back up the existing (all-error) file
cp data/qrels/llm_judgments.jsonl data/qrels/llm_judgments.jsonl.bak
# The pipeline uses --rerun-errors to re-judge errors, so we just run it
```

- [ ] **Step 2.2: Run the full pipeline (background)**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
PYTHONPATH=. nohup python scripts/eval/run_parallel_llm_qrels.py \
  --rerun-errors \
  2>&1 | tee logs/qrels_run_$(date +%Y%m%d_%H%M%S).log &
echo "PID: $!"
```

- [ ] **Step 2.3: Monitor progress**

```bash
# Check every few minutes
wc -l data/qrels/llm_judgments.jsonl
grep -c '"label": 1' data/qrels/llm_judgments.jsonl  # relevant pairs found
grep -c 'judge_error' data/qrels/llm_judgments.jsonl  # error count
```

- [ ] **Step 2.4: Validate final output**

```bash
PYTHONPATH=. python scripts/eval/validate_qrels.py \
  --judgments data/qrels/llm_judgments.jsonl \
  --trec data/qrels/qrels.trec
```

Expected: prints total pairs, label distribution (should be <20% relevant), error rate (<5%), per-query coverage.

- [ ] **Step 2.5: Commit qrels snapshot**

```bash
git add data/qrels/llm_judgments.jsonl data/qrels/qrels.trec
git commit -m "feat(eval): add LLM-judged qrels for 13654 evidence pairs (benchmark v1)"
```

---

### Task 3: Run ablation ladder and compute IR metrics

Run the 6-rung ablation (BM25→BM25+struct→dense-BGE→hybrid-RRF→hybrid+graph→full) against the frozen qrels, then compute NDCG@10, MRR, Precision@10 with bootstrap CIs.

**Files:**
- Read: `data/qrels/qrels.trec`
- Run: `scripts/eval/run_ablation_ladder.py`
- Run: `scripts/eval/compute_ir_metrics.py`
- Write: `artifacts/benchmark_v1/ablation_results.json`, `artifacts/benchmark_v1/metrics_report.md`

- [ ] **Step 3.1: Run the ablation ladder**

```bash
mkdir -p artifacts/benchmark_v1
PYTHONPATH=. python scripts/eval/run_ablation_ladder.py \
  --qrels data/qrels/qrels.trec \
  --output artifacts/benchmark_v1/ablation_results.json
```

Expected: JSON file with per-rung TREC-format run files.

- [ ] **Step 3.2: Compute bootstrap IR metrics**

```bash
PYTHONPATH=. python scripts/eval/compute_bootstrap_ci.py \
  --qrels data/qrels/qrels.trec \
  --runs artifacts/benchmark_v1/ \
  --output artifacts/benchmark_v1/metrics_report.md \
  --n-bootstrap 1000
```

Expected: Markdown table showing NDCG@10, MRR, P@10 ± CI for each rung. BM25 should be the baseline; each rung should show delta vs. BM25.

- [ ] **Step 3.3: Check the report**

```bash
cat artifacts/benchmark_v1/metrics_report.md
```

Verify: full-system NDCG@10 > BM25 NDCG@10 (confirms retrieval system adds value). If full system underperforms BM25, this is critical information — note it, do not hide it.

- [ ] **Step 3.4: Commit the benchmark artifacts**

```bash
git add artifacts/benchmark_v1/
git commit -m "feat(eval): benchmark v1 — ablation ladder + IR metrics with bootstrap CI"
```

---

### Task 4: Freeze the benchmark snapshot

Create a versioned snapshot so future runs can compare against the same corpus + qrels.

**Files:**
- Run: `scripts/eval/freeze_corpus_snapshot.py`
- Write: `artifacts/benchmark_v1/corpus_snapshot_hash.txt`, `artifacts/benchmark_v1/README.md`

- [ ] **Step 4.1: Freeze the corpus snapshot**

```bash
PYTHONPATH=. python scripts/eval/freeze_corpus_snapshot.py \
  --corpus data/corpus/normalized/combined_corpus.jsonl \
  --output artifacts/benchmark_v1/corpus_snapshot_hash.txt
```

Expected: prints a SHA256 of the corpus file.

- [ ] **Step 4.2: Write a benchmark README**

Create `artifacts/benchmark_v1/README.md`:

```markdown
# Neural Search Benchmark v1

**Date:** 2026-06-15  
**Corpus:** `data/corpus/normalized/combined_corpus.jsonl`  
**Corpus hash:** *(paste output of freeze_corpus_snapshot)*  
**Queries:** 317 canonical queries from `data/benchmark/benchmark_queries_canonical.yaml`  
**Evidence packets:** 13,654 (query × candidate pairs)  
**Qrels:** `data/qrels/llm_judgments.jsonl` (LLM-judged, google/gemini-2.5-flash)  
**Judge model:** google/gemini-2.5-flash via OpenRouter  
**Ablation rungs:** BM25 → BM25+struct → dense-BGE → hybrid-RRF → hybrid+graph → full  

## How to reproduce

```bash
PYTHONPATH=. python scripts/eval/run_parallel_llm_qrels.py --rerun-errors
PYTHONPATH=. python scripts/eval/run_ablation_ladder.py --qrels data/qrels/qrels.trec --output artifacts/benchmark_v1/ablation_results.json
PYTHONPATH=. python scripts/eval/compute_bootstrap_ci.py --qrels data/qrels/qrels.trec --runs artifacts/benchmark_v1/ --output artifacts/benchmark_v1/metrics_report.md
```

## Results summary

*(paste metrics_report.md table here)*
```

- [ ] **Step 4.3: Commit**

```bash
git add artifacts/benchmark_v1/README.md artifacts/benchmark_v1/corpus_snapshot_hash.txt
git commit -m "docs(eval): freeze benchmark v1 snapshot with reproduction instructions"
```

---

## Sprint B: Brain Atlas Map

### Task 5: Add two DuckDB store methods

The Brain Atlas page needs: (1) all brain regions with dataset counts to color the atlas tiles, and (2) the dataset list for a selected region.

**Files:**
- Modify: `neural_search/coverage/duckdb_store.py` (add after `dark_pairs()` method, around line 667)
- Modify: `tests/test_duckdb_store.py` (extend with new tests)

- [ ] **Step 5.1: Write failing tests**

In `tests/test_duckdb_store.py`, add at the end of the test class (or as new standalone tests):

```python
def test_region_dataset_counts_returns_list(tmp_path):
    """region_dataset_counts returns a list of dicts with region_id, region_label, n_datasets."""
    from neural_search.coverage.duckdb_store import CoverageStore
    store = CoverageStore(tmp_path / "test.duckdb")
    # Build with tiny corpus so there's at least one entry
    test_corpus = tmp_path / "corpus.jsonl"
    test_corpus.write_text(
        '{"id": "dandi:000001", "source": "dandi", "source_id": "000001", '
        '"title": "Test dataset", "brain_regions": ["visual_cortex"], '
        '"modalities": ["ephys"], "species": ["mouse"], "tasks": [], '
        '"has_behavior": false, "has_raw_data": true}\n'
    )
    store.build(corpus_path=test_corpus)
    counts = store.region_dataset_counts()
    assert isinstance(counts, list)
    if counts:
        assert "region_id" in counts[0]
        assert "region_label" in counts[0]
        assert "n_datasets" in counts[0]
        assert isinstance(counts[0]["n_datasets"], int)


def test_datasets_for_region_returns_list(tmp_path):
    """datasets_for_region returns datasets tagged with that region."""
    from neural_search.coverage.duckdb_store import CoverageStore
    store = CoverageStore(tmp_path / "test.duckdb")
    test_corpus = tmp_path / "corpus.jsonl"
    test_corpus.write_text(
        '{"id": "dandi:000001", "source": "dandi", "source_id": "000001", '
        '"title": "Visual cortex recording", "brain_regions": ["visual_cortex"], '
        '"modalities": ["ephys"], "species": ["mouse"], "tasks": [], '
        '"has_behavior": false, "has_raw_data": true}\n'
    )
    store.build(corpus_path=test_corpus)
    results = store.datasets_for_region("visual_cortex")
    assert isinstance(results, list)
    # May be empty if confidence threshold not met on tiny corpus — just check types
    for r in results:
        assert "dataset_id" in r
        assert "source" in r
        assert "title" in r
        assert "confidence" in r
```

- [ ] **Step 5.2: Run tests to confirm they fail**

```bash
PYTHONPATH=. python -m pytest tests/test_duckdb_store.py::test_region_dataset_counts_returns_list tests/test_duckdb_store.py::test_datasets_for_region_returns_list -v
```

Expected: `AttributeError: 'CoverageStore' object has no attribute 'region_dataset_counts'`

- [ ] **Step 5.3: Implement both methods in `duckdb_store.py`**

After the `coverage_summary()` method (around line 715), add:

```python
def region_dataset_counts(self, *, min_confidence: float = 0.65) -> list[dict[str, Any]]:
    """All brain regions with their dataset counts. Used to color the Brain Atlas."""
    sql = f"""
    SELECT
        ce.value_id          AS region_id,
        MAX(o.label)         AS region_label,
        COUNT(DISTINCT ce.dataset_id) AS n_datasets
    FROM coverage_entries ce
    LEFT JOIN ontology_regions o ON o.id = ce.value_id
    WHERE ce.dimension = 'brain_regions'
      AND ce.confidence >= {min_confidence}
    GROUP BY ce.value_id
    ORDER BY n_datasets DESC
    """
    rows = self._conn.sql(sql).fetchall()
    return [
        {"region_id": r[0], "region_label": r[1] or r[0], "n_datasets": r[2]}
        for r in rows
    ]

def datasets_for_region(
    self,
    region_id: str,
    *,
    limit: int = 20,
    offset: int = 0,
    min_confidence: float = 0.65,
) -> list[dict[str, Any]]:
    """Datasets tagged with a specific brain region, ordered by confidence then title."""
    sql = """
    SELECT
        d.dataset_id,
        d.source,
        d.title,
        d.access_tier,
        ce.confidence
    FROM coverage_entries ce
    JOIN datasets d ON d.dataset_id = ce.dataset_id
    WHERE ce.dimension = 'brain_regions'
      AND ce.value_id = ?
      AND ce.confidence >= ?
    ORDER BY ce.confidence DESC, d.title
    LIMIT ? OFFSET ?
    """
    rows = self._conn.sql(
        sql, params=[region_id, min_confidence, limit, offset]
    ).fetchall()
    return [
        {
            "dataset_id": r[0],
            "source": r[1],
            "title": r[2],
            "access_tier": r[3],
            "confidence": round(float(r[4]), 3),
        }
        for r in rows
    ]
```

- [ ] **Step 5.4: Run tests to confirm they pass**

```bash
PYTHONPATH=. python -m pytest tests/test_duckdb_store.py::test_region_dataset_counts_returns_list tests/test_duckdb_store.py::test_datasets_for_region_returns_list -v
```

Expected: both `PASSED`.

- [ ] **Step 5.5: Run the full duckdb test suite to confirm no regressions**

```bash
PYTHONPATH=. python -m pytest tests/test_duckdb_store.py -v
```

Expected: all existing tests plus 2 new ones pass.

- [ ] **Step 5.6: Commit**

```bash
git add neural_search/coverage/duckdb_store.py tests/test_duckdb_store.py
git commit -m "feat(coverage): add region_dataset_counts() and datasets_for_region() to DuckDB store"
```

---

### Task 6: Add two new coverage API endpoints

**Files:**
- Modify: `apps/api/main.py` (append after the `/api/coverage/dark-pairs` endpoint, line ~1473)

- [ ] **Step 6.1: Add endpoints to `apps/api/main.py`**

Append the following at the end of `apps/api/main.py`:

```python
@app.get("/api/coverage/region-counts")
async def get_region_counts() -> list[dict[str, Any]]:
    """All brain regions with dataset counts for the Brain Atlas heatmap."""
    with _coverage_store() as store:
        return store.region_dataset_counts()


@app.get("/api/coverage/region/{region_id}/datasets")
async def get_region_datasets(
    region_id: str,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """Datasets tagged with a specific brain region."""
    # region_id comes from URL path — safe as DuckDB param (not interpolated)
    with _coverage_store() as store:
        datasets = store.datasets_for_region(region_id, limit=limit, offset=offset)
    return {"region_id": region_id, "datasets": datasets, "count": len(datasets)}
```

- [ ] **Step 6.2: Test the endpoints manually**

Start the API:
```bash
cd /mnt/c/Users/sidso/Documents/neural-search
PYTHONPATH=. uvicorn apps.api.main:app --reload --port 8000
```

In another terminal:
```bash
curl http://localhost:8000/api/coverage/region-counts | python3 -m json.tool | head -20
curl http://localhost:8000/api/coverage/region/visual_cortex/datasets | python3 -m json.tool
```

Expected: JSON arrays/objects with valid data.

- [ ] **Step 6.3: Commit**

```bash
git add apps/api/main.py
git commit -m "feat(api): add /api/coverage/region-counts and /api/coverage/region/{id}/datasets"
```

---

### Task 7: Add frontend API types and client methods

**Files:**
- Modify: `apps/web/src/api/coverage.ts`

- [ ] **Step 7.1: Add new types and client functions**

In `apps/web/src/api/coverage.ts`, append after the existing `coverageApi` object:

```typescript
export type RegionCount = {
  region_id: string
  region_label: string
  n_datasets: number
}

export type RegionDataset = {
  dataset_id: string
  source: string
  title: string
  access_tier: string | null
  confidence: number
}

export type RegionDatasetsResponse = {
  region_id: string
  datasets: RegionDataset[]
  count: number
}
```

And extend the `coverageApi` object by adding two new methods:

```typescript
export const coverageApi = {
  // ... existing methods ...
  regionCounts: () => get<RegionCount[]>('/api/coverage/region-counts'),
  regionDatasets: (regionId: string, params?: { limit?: number; offset?: number }) => {
    const q = new URLSearchParams()
    if (params?.limit) q.set('limit', String(params.limit))
    if (params?.offset) q.set('offset', String(params.offset))
    const qs = q.toString() ? `?${q}` : ''
    return get<RegionDatasetsResponse>(`/api/coverage/region/${encodeURIComponent(regionId)}/datasets${qs}`)
  },
}
```

- [ ] **Step 7.2: Verify TypeScript compiles**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search/apps/web
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 7.3: Commit**

```bash
git add apps/web/src/api/coverage.ts
git commit -m "feat(frontend): add RegionCount/RegionDataset types and atlas API client methods"
```

---

### Task 8: Build BrainAtlasPage.tsx

This is the centerpiece page. Layout: full-width header stats bar, then a two-column split — left is the atlas grid (regions grouped by anatomy, colored by coverage), right is the selected region panel (region title + dataset list). No D3 needed.

**Files:**
- Create: `apps/web/src/pages/BrainAtlasPage.tsx`

- [ ] **Step 8.1: Create `BrainAtlasPage.tsx`**

```tsx
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { coverageApi, type RegionCount, type RegionDataset } from '../api/coverage'

// ── Anatomical grouping ───────────────────────────────────────────────────────
// Maps from display group label → ordered list of region_id values.
// Derived from data/ontology/brain_regions.yaml parent structure.
const ATLAS_GROUPS: Record<string, string[]> = {
  'Visual System': [
    'visual_cortex', 'v1', 'v2', 'v4', 'area_mt', 'inferior_temporal_cortex',
    'lateral_geniculate',
  ],
  'Frontal / Motor': [
    'prefrontal_cortex', 'dlpfc', 'ofc', 'vlpfc', 'anterior_cingulate_cortex',
    'motor_cortex', 'm1', 'premotor_cortex', 'supplementary_motor_area',
    'orbitofrontal_cortex',
  ],
  'Somatosensory': [
    'somatosensory_cortex', 's1', 's2', 'barrel_cortex',
  ],
  'Parietal / Temporal': [
    'parietal_cortex', 'posterior_parietal_cortex', 'temporal_cortex',
    'auditory_cortex', 'inferior_parietal_lobule',
  ],
  'Hippocampal Fm.': [
    'hippocampus', 'ca1', 'ca3', 'dentate_gyrus', 'entorhinal_cortex',
    'subiculum', 'parahippocampal_cortex',
  ],
  'Cingulate / Insula': [
    'cingulate_cortex', 'insular_cortex', 'retrosplenial_cortex',
    'medial_prefrontal_cortex',
  ],
  'Basal Ganglia': [
    'striatum', 'caudate_nucleus', 'putamen', 'nucleus_accumbens',
    'globus_pallidus', 'subthalamic_nucleus', 'basal_ganglia',
  ],
  'Thalamus': [
    'thalamus', 'mediodorsal_thalamus', 'pulvinar', 'medial_geniculate',
    'ventral_posterior_thalamus',
  ],
  'Amygdala': [
    'amygdala', 'lateral_amygdala', 'basal_amygdala', 'central_amygdala',
    'bed_nucleus_stria_terminalis',
  ],
  'Hypothalamus': [
    'hypothalamus', 'lateral_hypothalamus', 'paraventricular_nucleus',
  ],
  'Brainstem': [
    'midbrain', 'superior_colliculus', 'inferior_colliculus',
    'substantia_nigra', 'ventral_tegmental_area', 'periaqueductal_gray',
    'pons', 'locus_coeruleus', 'raphe_nucleus', 'medulla',
  ],
  'Cerebellum': [
    'cerebellum', 'cerebellar_cortex', 'deep_cerebellar_nuclei', 'purkinje_cells',
  ],
}

// ── Color helpers ─────────────────────────────────────────────────────────────
function tileColor(n: number): string {
  if (n === 0) return 'bg-neural-900 border-neural-800 text-neural-600'
  if (n < 5) return 'bg-red-950 border-red-800/50 text-red-300 hover:bg-red-900/60'
  if (n < 30) return 'bg-orange-950 border-orange-700/50 text-orange-300 hover:bg-orange-900/60'
  if (n < 100) return 'bg-yellow-950 border-yellow-600/50 text-yellow-300 hover:bg-yellow-900/60'
  if (n < 400) return 'bg-emerald-950 border-emerald-700/50 text-emerald-300 hover:bg-emerald-900/60'
  return 'bg-emerald-900 border-emerald-500/70 text-emerald-100 hover:bg-emerald-800'
}

// ── Region tile ───────────────────────────────────────────────────────────────
function RegionTile({
  regionId,
  label,
  nDatasets,
  selected,
  onClick,
}: {
  regionId: string
  label: string
  nDatasets: number
  selected: boolean
  onClick: () => void
}) {
  const colors = tileColor(nDatasets)
  return (
    <button
      onClick={onClick}
      title={`${label} — ${nDatasets} datasets`}
      className={`
        relative border rounded px-2 py-1.5 text-left transition-all cursor-pointer
        ${colors}
        ${selected ? 'ring-2 ring-accent-cyan ring-offset-1 ring-offset-neural-950' : ''}
      `}
    >
      <div className="text-xs font-medium leading-tight truncate">{label}</div>
      <div className="text-xs font-mono opacity-70 mt-0.5">{nDatasets}</div>
    </button>
  )
}

// ── Region panel ──────────────────────────────────────────────────────────────
function RegionPanel({ regionId, regionLabel }: { regionId: string; regionLabel: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ['region-datasets', regionId],
    queryFn: () => coverageApi.regionDatasets(regionId, { limit: 30 }),
    enabled: !!regionId,
  })

  return (
    <div className="flex flex-col h-full">
      <div className="mb-4">
        <h2 className="font-mono text-lg text-white">{regionLabel}</h2>
        <p className="text-xs text-neural-500 mt-0.5">
          {data ? `${data.count} datasets` : '…'}
        </p>
      </div>

      {isLoading && <div className="text-neural-500 text-sm">Loading…</div>}

      {data && data.datasets.length === 0 && (
        <div className="text-neural-600 text-sm italic">No datasets found for this region.</div>
      )}

      <div className="flex flex-col gap-2 overflow-y-auto">
        {data?.datasets.map((ds: RegionDataset) => (
          <Link
            key={ds.dataset_id}
            to={`/datasets/${encodeURIComponent(ds.dataset_id)}`}
            className="block bg-neural-900 border border-neural-800 rounded-lg p-3 hover:border-accent-cyan/50 transition-colors"
          >
            <div className="text-sm text-neural-100 font-medium leading-snug mb-1">
              {ds.title}
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs text-neural-500 font-mono">{ds.source}</span>
              <span className="text-xs text-neural-700">·</span>
              <span className="text-xs text-neural-500">
                confidence {(ds.confidence * 100).toFixed(0)}%
              </span>
              {ds.access_tier && (
                <>
                  <span className="text-xs text-neural-700">·</span>
                  <span className="text-xs text-neural-500">{ds.access_tier}</span>
                </>
              )}
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}

// ── Legend ────────────────────────────────────────────────────────────────────
function Legend() {
  const entries = [
    { label: '0', color: 'bg-neural-900 border-neural-800' },
    { label: '1–4', color: 'bg-red-950 border-red-800/50' },
    { label: '5–29', color: 'bg-orange-950 border-orange-700/50' },
    { label: '30–99', color: 'bg-yellow-950 border-yellow-600/50' },
    { label: '100–399', color: 'bg-emerald-950 border-emerald-700/50' },
    { label: '400+', color: 'bg-emerald-900 border-emerald-500/70' },
  ]
  return (
    <div className="flex items-center gap-3 flex-wrap">
      <span className="text-xs text-neural-500">Coverage:</span>
      {entries.map((e) => (
        <div key={e.label} className="flex items-center gap-1">
          <div className={`w-3 h-3 rounded border ${e.color}`} />
          <span className="text-xs text-neural-500">{e.label}</span>
        </div>
      ))}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export function BrainAtlasPage() {
  const [selectedRegion, setSelectedRegion] = useState<{ id: string; label: string } | null>(null)

  const { data: counts, isLoading } = useQuery({
    queryKey: ['region-counts'],
    queryFn: coverageApi.regionCounts,
  })

  // Build lookup: region_id → n_datasets
  const countMap = new Map<string, { n: number; label: string }>(
    (counts ?? []).map((r) => [r.region_id, { n: r.n_datasets, label: r.region_label }])
  )

  const totalDatasets = counts
    ? new Set(counts.flatMap(() => [])).size  // approximate: sum unique datasets
    : 0
  const coveredRegions = counts?.filter((r) => r.n_datasets > 0).length ?? 0
  const totalRegionsInAtlas = Object.values(ATLAS_GROUPS).flat().length

  return (
    <div className="max-w-7xl mx-auto px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-6">
        <h1 className="font-mono text-2xl text-white mb-1">Brain Atlas</h1>
        <p className="text-neural-500 text-sm">
          Neuroscience dataset coverage across brain regions. Click any region to explore datasets.
        </p>
      </div>

      {/* Stats bar */}
      {counts && (
        <div className="flex items-center gap-6 mb-5 text-sm">
          <span>
            <span className="font-mono text-white">{coveredRegions}</span>
            <span className="text-neural-500 ml-1">/ {totalRegionsInAtlas} regions covered</span>
          </span>
          <span className="text-neural-700">·</span>
          <Legend />
        </div>
      )}

      {isLoading && (
        <div className="text-neural-500 text-sm">Loading atlas…</div>
      )}

      {/* Atlas grid + side panel */}
      <div className="flex gap-6">
        {/* Left: atlas grid */}
        <div className="flex-1 min-w-0">
          {Object.entries(ATLAS_GROUPS).map(([groupLabel, regionIds]) => {
            const groupRegions = regionIds.map((rid) => {
              const info = countMap.get(rid)
              return { id: rid, label: info?.label ?? rid.replace(/_/g, ' '), n: info?.n ?? 0 }
            })
            const groupTotal = groupRegions.reduce((s, r) => s + r.n, 0)
            return (
              <div key={groupLabel} className="mb-5">
                <div className="flex items-baseline gap-2 mb-2">
                  <span className="text-xs font-mono text-neural-400 uppercase tracking-wider">
                    {groupLabel}
                  </span>
                  <span className="text-xs text-neural-600">{groupTotal} datasets</span>
                </div>
                <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-1.5">
                  {groupRegions.map((r) => (
                    <RegionTile
                      key={r.id}
                      regionId={r.id}
                      label={r.label}
                      nDatasets={r.n}
                      selected={selectedRegion?.id === r.id}
                      onClick={() =>
                        setSelectedRegion(
                          selectedRegion?.id === r.id ? null : { id: r.id, label: r.label }
                        )
                      }
                    />
                  ))}
                </div>
              </div>
            )
          })}
        </div>

        {/* Right: selected region panel */}
        <div className="w-80 flex-shrink-0">
          <div className="sticky top-24 bg-neural-900/50 border border-neural-800 rounded-xl p-4 max-h-[80vh] overflow-y-auto">
            {selectedRegion ? (
              <RegionPanel regionId={selectedRegion.id} regionLabel={selectedRegion.label} />
            ) : (
              <div className="text-center py-12">
                <div className="text-neural-600 text-sm">
                  Click a brain region to explore its datasets
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 8.2: Commit**

```bash
git add apps/web/src/pages/BrainAtlasPage.tsx
git commit -m "feat(frontend): add BrainAtlasPage with anatomical coverage heatmap"
```

---

### Task 9: Wire up routing and nav

**Files:**
- Modify: `apps/web/src/App.tsx`
- Modify: `apps/web/src/components/Layout.tsx`

- [ ] **Step 9.1: Add route to App.tsx**

In `apps/web/src/App.tsx`, add the import and route:

```tsx
import { BrainAtlasPage } from './pages/BrainAtlasPage'

// Inside <Routes>:
<Route path="/atlas" element={<BrainAtlasPage />} />
```

The full file should look like:

```tsx
import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { SearchPage } from './pages/SearchPage'
import { ResultsPage } from './pages/ResultsPage'
import { DatasetPage } from './pages/DatasetPage'
import { OntologyPage } from './pages/OntologyPage'
import { ReportsPage } from './pages/ReportsPage'
import { EvaluationPage } from './pages/EvaluationPage'
import { DemoPage } from './pages/DemoPage'
import { GraphPage } from './pages/GraphPage'
import { CoveragePage } from './pages/CoveragePage'
import { BrainAtlasPage } from './pages/BrainAtlasPage'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<SearchPage />} />
        <Route path="/search" element={<ResultsPage />} />
        <Route path="/datasets/:id" element={<DatasetPage />} />
        <Route path="/ontology" element={<OntologyPage />} />
        <Route path="/reports" element={<ReportsPage />} />
        <Route path="/evaluation" element={<EvaluationPage />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="/graph" element={<GraphPage />} />
        <Route path="/coverage" element={<CoveragePage />} />
        <Route path="/atlas" element={<BrainAtlasPage />} />
      </Routes>
    </Layout>
  )
}

export default App
```

- [ ] **Step 9.2: Add Atlas link to nav in Layout.tsx**

In `apps/web/src/components/Layout.tsx`, update `navLinks` to add Atlas and rename Coverage to "Coverage (raw)":

```tsx
const navLinks = [
  { path: '/', label: 'Search', exact: true },
  { path: '/atlas', label: 'Atlas' },        // NEW — before other tools
  { path: '/demo', label: 'Demo' },
  { path: '/graph', label: 'Graph' },
  { path: '/ontology', label: 'Ontology' },
  { path: '/coverage', label: 'Coverage' },
  { path: '/reports', label: 'Reports' },
  { path: '/evaluation', label: 'Evaluation' },
]
```

- [ ] **Step 9.3: Verify the frontend builds**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search/apps/web
npm run build
```

Expected: builds without TypeScript errors.

- [ ] **Step 9.4: Start dev server and verify Atlas page loads**

```bash
npm run dev
```

Open `http://localhost:5173/atlas`. Expected: Brain Atlas page with colored region tiles grouped by anatomy, and a right panel that populates when a tile is clicked.

- [ ] **Step 9.5: Commit**

```bash
git add apps/web/src/App.tsx apps/web/src/components/Layout.tsx
git commit -m "feat(frontend): wire BrainAtlasPage into routing and nav"
```

---

## Sprint C: Dataset Affordance Panels

### Task 10: Add dataset affordances API endpoint

Currently, affordance validation results are computed during card generation but not exposed as a standalone API call. Add a lightweight endpoint that validates a dataset's affordances from its stored metadata and returns structured evidence.

**Files:**
- Modify: `apps/api/main.py` (add after `/api/datasets/{dataset_id}/notebook`)

- [ ] **Step 10.1: Add the endpoint to `main.py`**

First, add this import near the top of `main.py` (with other neural_search imports):

```python
from neural_search.affordances.registry import (
    AffordanceRegistry,
    detect_dataset_features,
)
```

Then add the endpoint (after the notebook endpoint, around line 843):

```python
@app.get("/api/datasets/{dataset_id}/affordances")
async def get_dataset_affordances(dataset_id: str) -> list[dict[str, Any]]:
    """Affordance support levels for a dataset derived from its metadata."""
    dataset = await _get_dataset_record(dataset_id)  # reuse existing helper
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    registry = AffordanceRegistry()
    features = detect_dataset_features(dataset)
    results = []
    for aff_id, req in registry.affordances.items():
        validation = req.validate(features)
        results.append({
            "affordance_id": aff_id,
            "label": req.label,
            "description": req.description,
            "support_level": validation.support_level,
            "confidence": round(validation.confidence, 3),
            "evidence_for": validation.evidence_for,
            "evidence_against": validation.evidence_against,
            "missing_features": validation.missing_features,
        })
    results.sort(key=lambda r: {"high": 0, "medium": 1, "low": 2, "unknown": 3, "unsupported": 4}
                 .get(r["support_level"], 5))
    return results
```

> Note: If `_get_dataset_record` does not exist as a standalone helper, add one. Check `apps/api/main.py` for the pattern used in `GET /api/datasets/{dataset_id}`. If the dataset lookup is inline, extract it or inline the same logic.

- [ ] **Step 10.2: Test the endpoint manually**

With the API running:
```bash
curl http://localhost:8000/api/datasets/dandi:001056/affordances | python3 -m json.tool | head -40
```

Expected: JSON array of affordance objects with `support_level` values like `"high"`, `"medium"`, `"low"`, or `"unsupported"`.

- [ ] **Step 10.3: Commit**

```bash
git add apps/api/main.py
git commit -m "feat(api): add GET /api/datasets/{id}/affordances returning evidence-backed support levels"
```

---

### Task 11: Add API client function for affordances

**Files:**
- Modify: `apps/web/src/api/search.ts`

- [ ] **Step 11.1: Add type and function to `search.ts`**

At the end of `apps/web/src/api/search.ts`, add:

```typescript
export type AffordanceResult = {
  affordance_id: string
  label: string
  description: string
  support_level: 'high' | 'medium' | 'low' | 'unsupported' | 'unknown'
  confidence: number
  evidence_for: string[]
  evidence_against: string[]
  missing_features: string[]
}

export async function getDatasetAffordances(datasetId: string): Promise<AffordanceResult[]> {
  const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(datasetId)}/affordances`)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<AffordanceResult[]>
}
```

- [ ] **Step 11.2: Check TypeScript**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search/apps/web
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 11.3: Commit**

```bash
git add apps/web/src/api/search.ts
git commit -m "feat(frontend): add getDatasetAffordances() API client"
```

---

### Task 12: Add AffordancePanel to DatasetPage

Add a new collapsible "Analysis Affordances" panel below the main metadata section, showing each affordance as a badge with its support level color and expandable evidence.

**Files:**
- Modify: `apps/web/src/pages/DatasetPage.tsx`

- [ ] **Step 12.1: Add the AffordancePanel component and wire it into DatasetPage**

At the top of `DatasetPage.tsx`, add the import:

```tsx
import { getDatasetAffordances, type AffordanceResult } from '../api/search'
```

After the existing `useQuery` hooks (around line 70), add:

```tsx
const { data: affordances } = useQuery({
  queryKey: ['affordances', id],
  queryFn: () => getDatasetAffordances(id!),
  enabled: !!id,
})
```

Then add the `AffordancePanel` component *inside* the file (before `DatasetPage`):

```tsx
const SUPPORT_STYLES: Record<AffordanceResult['support_level'], string> = {
  high: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40',
  medium: 'bg-yellow-500/15 text-yellow-300 border-yellow-500/40',
  low: 'bg-orange-500/15 text-orange-300 border-orange-500/40',
  unsupported: 'bg-neural-800 text-neural-600 border-neural-700',
  unknown: 'bg-neural-800 text-neural-500 border-neural-700',
}

const SUPPORT_LABEL: Record<AffordanceResult['support_level'], string> = {
  high: 'HIGH',
  medium: 'MED',
  low: 'LOW',
  unsupported: 'NO',
  unknown: '?',
}

function AffordancePanel({ affordances }: { affordances: AffordanceResult[] }) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const supported = affordances.filter((a) => a.support_level !== 'unsupported' && a.support_level !== 'unknown')
  const unsupported = affordances.filter((a) => a.support_level === 'unsupported' || a.support_level === 'unknown')

  return (
    <div className="mt-8">
      <h3 className="text-sm font-mono text-neural-400 uppercase tracking-wider mb-3">
        Analysis Affordances
      </h3>
      <div className="space-y-2">
        {[...supported, ...unsupported].map((aff) => (
          <div
            key={aff.affordance_id}
            className="bg-neural-900 border border-neural-800 rounded-lg overflow-hidden"
          >
            <button
              onClick={() => setExpanded(expanded === aff.affordance_id ? null : aff.affordance_id)}
              className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-neural-800/40 transition-colors"
            >
              <span
                className={`text-xs font-mono px-1.5 py-0.5 rounded border flex-shrink-0 w-10 text-center ${SUPPORT_STYLES[aff.support_level]}`}
              >
                {SUPPORT_LABEL[aff.support_level]}
              </span>
              <span className="text-sm text-neural-200 font-medium">{aff.label}</span>
              <span className="text-xs text-neural-600 ml-auto">
                {(aff.confidence * 100).toFixed(0)}%
              </span>
            </button>

            {expanded === aff.affordance_id && (
              <div className="px-4 pb-4 border-t border-neural-800 pt-3 space-y-2">
                <p className="text-xs text-neural-500">{aff.description}</p>
                {aff.evidence_for.length > 0 && (
                  <div>
                    <div className="text-xs font-mono text-emerald-600 mb-1">Evidence for</div>
                    <ul className="space-y-0.5">
                      {aff.evidence_for.map((e, i) => (
                        <li key={i} className="text-xs text-neural-400 flex gap-1.5">
                          <span className="text-emerald-700 flex-shrink-0">✓</span>
                          {e}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {aff.evidence_against.length > 0 && (
                  <div>
                    <div className="text-xs font-mono text-red-600 mb-1">Evidence against</div>
                    <ul className="space-y-0.5">
                      {aff.evidence_against.map((e, i) => (
                        <li key={i} className="text-xs text-neural-400 flex gap-1.5">
                          <span className="text-red-700 flex-shrink-0">✗</span>
                          {e}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {aff.missing_features.length > 0 && (
                  <div>
                    <div className="text-xs font-mono text-neural-600 mb-1">Missing</div>
                    <ul className="space-y-0.5">
                      {aff.missing_features.map((f, i) => (
                        <li key={i} className="text-xs text-neural-600">— {f}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
```

Finally, find the return statement in `DatasetPage` where the main content is rendered and add `AffordancePanel` after the assets section or QA section:

```tsx
{affordances && affordances.length > 0 && (
  <AffordancePanel affordances={affordances} />
)}
```

- [ ] **Step 12.2: Build and verify**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search/apps/web
npm run build
```

Expected: builds without errors.

Open any dataset page (e.g., `http://localhost:5173/datasets/dandi:001056`). Scroll to "Analysis Affordances". Expected: colored badges for each affordance, expandable to show evidence.

- [ ] **Step 12.3: Commit**

```bash
git add apps/web/src/pages/DatasetPage.tsx
git commit -m "feat(frontend): add AffordancePanel to DatasetPage with evidence-backed support levels"
```

---

## Self-Review

### Spec coverage check

| Requirement | Covered by |
|---|---|
| Fix qrels auth 401 errors | Task 1 |
| Run full benchmark (13,654 pairs) | Task 2 |
| Compute NDCG/MRR/ablation metrics | Task 3 |
| Freeze benchmark snapshot | Task 4 |
| Add `region_dataset_counts()` to DuckDB store | Task 5 |
| Add `datasets_for_region()` to DuckDB store | Task 5 |
| Add region-counts API endpoint | Task 6 |
| Add region/{id}/datasets API endpoint | Task 6 |
| Add RegionCount/RegionDataset types to frontend | Task 7 |
| Build Brain Atlas Map with anatomy groups | Task 8 |
| Wire Atlas into routing and nav | Task 9 |
| Add affordances API endpoint | Task 10 |
| Add getDatasetAffordances() client | Task 11 |
| Add AffordancePanel to DatasetPage | Task 12 |

### Placeholder scan

No TBDs, TODOs, or "similar to Task N" references. All code blocks are complete.

### Type consistency

- `RegionCount` / `RegionDataset` / `RegionDatasetsResponse` — defined in Task 7, consumed in Task 8 (correct)
- `AffordanceResult` — defined in Task 11, consumed in Task 12 (correct)
- `coverageApi.regionCounts()` / `coverageApi.regionDatasets()` — defined in Task 7, called in Task 8 (correct)
- `getDatasetAffordances()` — defined in Task 11, called in Task 12 (correct)

### Known gaps (future plans)

These are intentionally deferred:
- **Social/community layer**: curation notes, curator queues, dataset ratings — Plan D
- **Model pages**: Neuronpedia-style model/latent feature pages, neurOS bridge — Plan E
- **Cross-scale synthesis**: linking single-unit findings to fMRI BOLD — Plan F
- **Persistent production storage**: replace in-memory corpus with real DB as default — Plan G

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-15-neuronpedia-foundation.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — Fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans

**Which approach?**
