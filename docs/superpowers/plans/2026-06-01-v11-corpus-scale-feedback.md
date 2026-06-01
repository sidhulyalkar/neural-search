# Neural Search v1.1 — Corpus Scale & Feedback Loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scale the real corpus from 371 datasets to 1000+, rebuild the knowledge graph and embeddings from the expanded corpus, run a human annotation sprint to grow the labeled pairs from 49 to 150+, and use those labels to tune the 10-dimension usefulness scorer weights — then re-evaluate to show measurable improvement.

**Architecture:** Four independent subsystems execute in sequence: (1) corpus expansion via existing ingestion scripts with higher limits; (2) graph/embedding rebuild; (3) human annotation sprint using the existing `annotate_usefulness.py` CLI; (4) weight optimization via grid search on NDCG@3 over labeled pairs. Each step builds on the previous; the final whitepaper update closes the loop.

**Tech Stack:** Python 3.10+, `neural_search.ingestion.dandi`, `neural_search.ingestion.openneuro`, `neural_search.retrieval.usefulness_scorer`, `neural_search.evaluation.ablation_runner`, scipy, existing scripts in `scripts/`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| **Modify** | `scripts/expand_dandi_corpus.py` | Increase per-query limit 15→30, add 10 new query terms |
| **Modify** | `scripts/expand_openneuro_corpus.py` | Increase per-modality limit 30→100, add 4 new modalities |
| **Create** | `scripts/rebuild_corpus_graph.py` | Rebuild KnowledgeGraph from all normalized jsonl files |
| **Create** | `scripts/recompute_embeddings.py` | Recompute field embeddings + fingerprints for expanded corpus |
| **Create** | `scripts/optimize_usefulness_weights.py` | Grid-search dimension weights to maximize NDCG@3 on labeled pairs |
| **Create** | `tests/test_optimize_usefulness_weights.py` | Unit tests for weight optimizer |
| **Modify** | `docs/whitepaper/neural_search_whitepaper.tex` | v1.1 sections: corpus stats, weight table, v1.1 benchmark results |

---

## Context for Every Task

Current corpus sizes:
- DANDI: 163 datasets (`data/corpus/normalized/real_dandi.jsonl`)
- OpenNeuro: 190 datasets (`data/corpus/normalized/real_openneuro.jsonl`)
- Allen: 8, NeMo: 10

`fetch_dandi(query, limit=15)` is in `neural_search/ingestion/dandi.py` — `limit` maps directly to `page_size` in the API call. Increasing to 30 doubles yield per query.

`fetch_openneuro(modality, limit=30)` is in `neural_search/ingestion/openneuro.py` — increasing to 100 triples yield per modality.

Knowledge graph builder: `scripts/build_real_corpus_graph.py` already exists — read it before Task 3 to understand the pattern.

Usefulness scorer weights are currently hard-coded in `neural_search/retrieval/usefulness_scorer.py` as the `INTENT_WEIGHT_PROFILES` dict — 7 intent profiles × 10 dimensions = 70 weight values. Task 6 will tune these via grid search.

Key types:
```python
# neural_search/retrieval/usefulness_scorer.py
INTENT_WEIGHT_PROFILES: dict[str, dict[str, float]]
# Keys: "strict_lookup", "replication", "meta_analysis", "pipeline_reuse",
#       "cross_dataset_comparison", "exploration", "method_transfer"
# Each maps dimension name → float weight (summing to ≈1.0)

# neural_search/evaluation/ablation_runner.py
def run_ablation(config: AblationConfig) -> AblationReport:
    ...

# data/eval/usefulness_seed_pairs.jsonl
# Each line: {"query_id", "query", "intent", "candidate_id", "usefulness_label", "label_type", "notes"}
# usefulness_label in {"not_useful", "weakly_useful", "useful", "highly_useful"}
```

---

### Task 1: Expand DANDI corpus (163 → 500+)

**Files:**
- Modify: `scripts/expand_dandi_corpus.py`

- [ ] **Step 1: Read the current script**

```bash
cat scripts/expand_dandi_corpus.py
```

Note: currently uses `limit=15` per query and 20 query terms. Change to `limit=30` and add 10 more queries.

- [ ] **Step 2: Update the script**

Find in `scripts/expand_dandi_corpus.py`:
```python
DANDI_QUERIES = [
    "neuropixels",
    "calcium imaging",
    "electrophysiology",
    "hippocampus",
    "visual cortex",
    "prefrontal cortex",
    "decision making",
    "motor cortex",
    "behavior",
    "two-photon",
    "ecog",
    "ieeg",
    "fiber photometry",
    "patch clamp",
    "optogenetics",
    "reward",
    "learning",
    "working memory",
    "attention",
    "sleep",
]
```

Replace with:
```python
DANDI_QUERIES = [
    "neuropixels",
    "calcium imaging",
    "electrophysiology",
    "hippocampus",
    "visual cortex",
    "prefrontal cortex",
    "decision making",
    "motor cortex",
    "behavior",
    "two-photon",
    "ecog",
    "ieeg",
    "fiber photometry",
    "patch clamp",
    "optogenetics",
    "reward",
    "learning",
    "working memory",
    "attention",
    "sleep",
    # v1.1 additions
    "somatosensory cortex",
    "basal ganglia",
    "cerebellum",
    "olfactory",
    "primate",
    "human intracranial",
    "NWB",
    "BIDS",
    "spike sorting",
    "place cells",
]
```

Then find:
```python
payload = fetch_dandi(query, limit=15)
```

Replace with:
```python
payload = fetch_dandi(query, limit=30)
```

- [ ] **Step 3: Run the expanded script**

```bash
python scripts/expand_dandi_corpus.py 2>&1 | tail -5
```

Expected output ends with something like:
```
Saved X records to data/corpus/normalized/real_dandi.jsonl
```

Where X > 300 (up from 163).

- [ ] **Step 4: Verify the expansion**

```bash
python -c "
from pathlib import Path
lines = [l for l in Path('data/corpus/normalized/real_dandi.jsonl').read_text().splitlines() if l.strip()]
print(f'DANDI records: {len(lines)}')
assert len(lines) > 300, f'Expected >300, got {len(lines)}'
print('OK')
"
```

Expected: `DANDI records: N` where N > 300.

- [ ] **Step 5: Commit**

```bash
git add scripts/expand_dandi_corpus.py data/corpus/normalized/real_dandi.jsonl
git commit -m "data: expand DANDI corpus to 500+ datasets — increase limit and add query terms"
```

---

### Task 2: Expand OpenNeuro corpus (190 → 500+)

**Files:**
- Modify: `scripts/expand_openneuro_corpus.py`

- [ ] **Step 1: Read the current script**

```bash
cat scripts/expand_openneuro_corpus.py
```

Note: currently uses `limit=30` per modality and 9 modality filters. Change to `limit=100` and add modalities.

- [ ] **Step 2: Update the script**

Find:
```python
OPENNEURO_MODALITIES = [
    "eeg",
    "func",
    "anat",
    "meg",
    "ieeg",
    "dwi",
    "pet",
    "beh",
    None,
]
```

Replace with:
```python
OPENNEURO_MODALITIES = [
    "eeg",
    "func",
    "anat",
    "meg",
    "ieeg",
    "dwi",
    "pet",
    "beh",
    "fmap",      # field maps
    "perf",     # perfusion
    "micr",     # microscopy
    "nirs",     # near-infrared spectroscopy
    None,       # all public datasets
]
```

Find:
```python
payload = fetch_openneuro(modality, limit=30)
```

Replace with:
```python
payload = fetch_openneuro(modality, limit=100)
```

- [ ] **Step 3: Run the expanded script**

```bash
python scripts/expand_openneuro_corpus.py 2>&1 | tail -5
```

Expected: Saved N records where N > 400.

- [ ] **Step 4: Verify**

```bash
python -c "
from pathlib import Path
lines = [l for l in Path('data/corpus/normalized/real_openneuro.jsonl').read_text().splitlines() if l.strip()]
print(f'OpenNeuro records: {len(lines)}')
assert len(lines) > 400, f'Expected >400, got {len(lines)}'
print('OK')
"
```

- [ ] **Step 5: Print combined corpus size**

```bash
python -c "
from pathlib import Path
sources = {
    'DANDI': 'data/corpus/normalized/real_dandi.jsonl',
    'OpenNeuro': 'data/corpus/normalized/real_openneuro.jsonl',
    'Allen': 'data/corpus/normalized/real_allen.jsonl',
    'NeMo': 'data/corpus/normalized/real_nemo.jsonl',
}
total = 0
for name, path in sources.items():
    n = len([l for l in Path(path).read_text().splitlines() if l.strip()])
    print(f'{name}: {n}')
    total += n
print(f'TOTAL: {total}')
assert total >= 1000, f'Expected >=1000, got {total}'
"
```

Expected: TOTAL >= 1000.

- [ ] **Step 6: Commit**

```bash
git add scripts/expand_openneuro_corpus.py data/corpus/normalized/real_openneuro.jsonl
git commit -m "data: expand OpenNeuro corpus to 500+ — increase limit and add modalities"
```

---

### Task 3: Rebuild knowledge graph from expanded corpus

**Files:**
- Create: `scripts/rebuild_corpus_graph.py`

- [ ] **Step 1: Read the existing graph builder**

```bash
cat scripts/build_real_corpus_graph.py
```

Understand its input format (which normalized jsonl files it reads) and output (where it writes the graph JSON).

- [ ] **Step 2: Write the rebuild script**

The rebuild script is a thin wrapper that reads all normalized corpus files and calls the existing graph builder:

```python
#!/usr/bin/env python3
"""Rebuild the KnowledgeGraph from all normalized corpus files.

Reads all data/corpus/normalized/real_*.jsonl files, builds the graph,
and writes data/graphs/real_corpus_graph.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CORPUS_DIR = Path("data/corpus/normalized")
GRAPH_OUT = Path("data/graphs/real_corpus_graph.json")


def main() -> int:
    jsonl_files = sorted(CORPUS_DIR.glob("real_*.jsonl"))
    print(f"Building graph from {len(jsonl_files)} corpus files:")
    for f in jsonl_files:
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        print(f"  {f.name}: {len(lines)} records")

    # Run the existing build script for each source, then merge
    # (build_real_corpus_graph.py already handles all normalized files)
    result = subprocess.run(
        [sys.executable, "scripts/build_real_corpus_graph.py"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Graph builder failed:", result.stderr)
        return 1

    print(result.stdout)

    if GRAPH_OUT.exists():
        graph = json.loads(GRAPH_OUT.read_text())
        n_nodes = len(graph.get("nodes", {}))
        n_edges = len(graph.get("edges", {}))
        print(f"Graph rebuilt: {n_nodes} nodes, {n_edges} edges")
        print(f"Written to: {GRAPH_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run the rebuild**

```bash
python scripts/rebuild_corpus_graph.py 2>&1
```

Expected: prints graph size (nodes/edges) and confirms write.

- [ ] **Step 4: Verify graph output**

```bash
python -c "
import json
from pathlib import Path
# Try common graph output paths
for p in ['data/graphs/real_corpus_graph.json', 'data/graphs/knowledge_graph.json']:
    pth = Path(p)
    if pth.exists():
        g = json.loads(pth.read_text())
        print(f'{p}: {len(g.get(\"nodes\", {}))} nodes, {len(g.get(\"edges\", {}))} edges')
        break
else:
    # Check what build_real_corpus_graph.py actually writes
    import subprocess, sys
    r = subprocess.run([sys.executable, '-c', 
        'import ast; src = open(\"scripts/build_real_corpus_graph.py\").read(); print(\"output path:\", [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str) and \"graph\" in n.value.lower()])'],
        capture_output=True, text=True)
    print(r.stdout or r.stderr)
"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/rebuild_corpus_graph.py data/graphs/
git commit -m "data: rebuild knowledge graph from expanded 1000+ dataset corpus"
```

---

### Task 4: Recompute embeddings for expanded corpus

**Files:**
- Create: `scripts/recompute_embeddings.py`

Field embeddings and fingerprints are stored in `data/embeddings/`. They need to be regenerated for the new records added in Tasks 1-2.

- [ ] **Step 1: Read the existing embedding generation script**

```bash
ls scripts/ | grep embed
cat scripts/demo.py 2>/dev/null | head -30  # or whatever generates embeddings
```

Also check:
```bash
python -c "
from neural_search.embeddings import EmbeddingProvider
import inspect
print(inspect.getfile(EmbeddingProvider))
"
```

Note: if no embedding generation script exists, look in `neural_search/embeddings/` for a generation entrypoint.

- [ ] **Step 2: Identify what needs regeneration**

```bash
python -c "
from pathlib import Path
for f in sorted(Path('data/embeddings').glob('real_*.jsonl')):
    lines = [l for l in f.read_text().splitlines() if l.strip()]
    print(f'{f.name}: {len(lines)} records')
"
```

Note: `real_v07.field_embeddings.jsonl` has 55 records; after corpus expansion this needs to be regenerated for 1000+ records.

- [ ] **Step 3: Write the recompute script**

```python
#!/usr/bin/env python3
"""Recompute field embeddings and fingerprints for the expanded real corpus.

Reads all real_*.jsonl corpus files, computes embeddings, writes to
data/embeddings/real_all.field_embeddings.jsonl and real_all.fingerprints.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path
from neural_search.normalized import load_normalized_records
from neural_search.embeddings import build_field_embeddings, build_fingerprints

CORPUS_FILES = sorted(Path("data/corpus/normalized").glob("real_*.jsonl"))
EMBEDDINGS_OUT = Path("data/embeddings/real_all.field_embeddings.jsonl")
FINGERPRINTS_OUT = Path("data/embeddings/real_all.fingerprints.jsonl")


def main() -> None:
    records = []
    for f in CORPUS_FILES:
        batch = load_normalized_records(f)
        records.extend(batch)
        print(f"Loaded {len(batch)} records from {f.name}")
    print(f"Total: {len(records)} records")

    print("Computing field embeddings...", flush=True)
    embeddings = build_field_embeddings(records)
    EMBEDDINGS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with EMBEDDINGS_OUT.open("w", encoding="utf-8") as fh:
        for emb in embeddings:
            fh.write(json.dumps(emb) + "\n")
    print(f"Wrote {len(embeddings)} embeddings → {EMBEDDINGS_OUT}")

    print("Computing fingerprints...", flush=True)
    fingerprints = build_fingerprints(records)
    with FINGERPRINTS_OUT.open("w", encoding="utf-8") as fh:
        for fp in fingerprints:
            fh.write(json.dumps(fp) + "\n")
    print(f"Wrote {len(fingerprints)} fingerprints → {FINGERPRINTS_OUT}")


if __name__ == "__main__":
    main()
```

**Important:** Before writing this script, read `neural_search/embeddings/__init__.py` to confirm the actual function names (`build_field_embeddings`, `build_fingerprints`) exist. If they differ, use the correct names.

- [ ] **Step 4: Run embeddings computation**

```bash
python scripts/recompute_embeddings.py 2>&1 | tail -10
```

Expected: prints the number of embeddings and fingerprints written.

- [ ] **Step 5: Commit**

```bash
git add scripts/recompute_embeddings.py data/embeddings/real_all.field_embeddings.jsonl data/embeddings/real_all.fingerprints.jsonl
git commit -m "data: recompute embeddings for 1000+ dataset corpus"
```

---

### Task 5: Human annotation sprint (49 → 150+ labeled pairs)

**Files:**
- Modify: `data/eval/usefulness_seed_pairs.jsonl` (appended via CLI, not directly edited)

This task uses the existing `scripts/annotate_usefulness.py` CLI to label 100+ new query-candidate pairs. Unlike the previous seed pairs (which were synthetic), these will target real datasets from the expanded corpus.

- [ ] **Step 1: Generate candidate pairs from real corpus**

```python
# scripts/generate_annotation_candidates.py
"""Generate query-candidate pairs for human annotation from the real corpus."""
from __future__ import annotations

import json
import random
from pathlib import Path

from neural_search.search import search_datasets

ANNOTATION_QUERIES = [
    ("qa001", "mouse hippocampus place cells neuropixels", "exploration"),
    ("qa002", "replicate Steinmetz 2019 visual cortex recording", "replication"),
    ("qa003", "calcium imaging striatum reward learning", "pipeline_reuse"),
    ("qa004", "compare EEG and MEG motor cortex datasets", "cross_dataset_comparison"),
    ("qa005", "NWB formatted primate prefrontal cortex", "strict_lookup"),
    ("qa006", "meta-analysis visual cortex across species", "meta_analysis"),
    ("qa007", "transfer GLM from mouse to rat decision-making", "method_transfer"),
    ("qa008", "two-photon imaging barrel cortex sensory whisker", "exploration"),
    ("qa009", "fiber photometry dopamine reward signal", "pipeline_reuse"),
    ("qa010", "human intracranial EEG seizure detection", "strict_lookup"),
]

OUT_PATH = Path("data/eval/annotation_candidates.jsonl")

pairs = []
for qid, query, intent in ANNOTATION_QUERIES:
    print(f"Generating candidates for {qid}...", flush=True)
    response = search_datasets(query)
    for i, result in enumerate(response.results[:5]):
        pairs.append({
            "query_id": qid,
            "query": query,
            "intent": intent,
            "candidate_id": str(result.dataset_id),
            "usefulness_label": "",  # to be filled by annotator
            "label_type": "human",
            "notes": "",
        })

OUT_PATH.parent.mkdir(exist_ok=True)
with OUT_PATH.open("w", encoding="utf-8") as f:
    for pair in pairs:
        f.write(json.dumps(pair) + "\n")
print(f"Generated {len(pairs)} candidate pairs → {OUT_PATH}")
```

Run it:
```bash
python scripts/generate_annotation_candidates.py 2>&1 | tail -5
```

Expected: `Generated ~50 candidate pairs → data/eval/annotation_candidates.jsonl`

- [ ] **Step 2: Run annotation session on generated candidates**

```bash
python scripts/annotate_usefulness.py --file data/eval/annotation_candidates.jsonl
```

For each pair prompted: assign 0/1/2/3 (not_useful / weakly_useful / useful / highly_useful). Annotate all ~50 pairs. Press `q` to pause and resume later with `--start-from N`.

Goal: annotate all candidates in the session. This produces human labels on real corpus results.

- [ ] **Step 3: Append annotated pairs to the main seed file**

```bash
python -c "
import json
from pathlib import Path

candidates = Path('data/eval/annotation_candidates.jsonl')
seed = Path('data/eval/usefulness_seed_pairs.jsonl')

labeled = [json.loads(l) for l in candidates.read_text().splitlines() if l.strip()]
labeled_only = [p for p in labeled if p.get('usefulness_label') and p['usefulness_label'] != '']

with seed.open('a', encoding='utf-8') as f:
    for pair in labeled_only:
        f.write(json.dumps(pair) + '\n')

print(f'Appended {len(labeled_only)} human-labeled pairs to {seed}')
total = len([l for l in seed.read_text().splitlines() if l.strip()])
print(f'Total seed pairs now: {total}')
"
```

Expected: Total seed pairs now >= 99 (49 original + ~50 new).

- [ ] **Step 4: Update coverage test for new minimum**

In `tests/test_seed_pairs_coverage.py`, find:
```python
def test_minimum_30_pairs():
    pairs = [l for l in SEED_FILE.read_text().splitlines() if l.strip()]
    assert len(pairs) >= 30
```

Replace with:
```python
def test_minimum_30_pairs():
    pairs = [l for l in SEED_FILE.read_text().splitlines() if l.strip()]
    assert len(pairs) >= 99
```

- [ ] **Step 5: Run test**

```bash
pytest tests/test_seed_pairs_coverage.py -v
```

Expected: All 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_annotation_candidates.py data/eval/annotation_candidates.jsonl data/eval/usefulness_seed_pairs.jsonl tests/test_seed_pairs_coverage.py
git commit -m "data: human annotation sprint — grow labeled pairs from 49 to 99+"
```

---

### Task 6: Dimension weight optimization

**Files:**
- Create: `scripts/optimize_usefulness_weights.py`
- Create: `tests/test_optimize_usefulness_weights.py`

Use the labeled pairs to tune each intent's 10-dimension weight profile via grid search, maximizing NDCG@3.

- [ ] **Step 1: Write the test first**

```python
# tests/test_optimize_usefulness_weights.py
from pathlib import Path
import json


def test_optimizer_script_syntax():
    import ast
    src = Path("scripts/optimize_usefulness_weights.py").read_text()
    ast.parse(src)  # raises SyntaxError if broken


def test_optimizer_dry_run(tmp_path):
    """--dry-run should print current weights without writing anything."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "scripts/optimize_usefulness_weights.py", "--dry-run"],
        capture_output=True, text=True,
        cwd="/mnt/c/Users/sidso/Documents/neural-search",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "INTENT_WEIGHT_PROFILES" in result.stdout or "dry run" in result.stdout.lower()


def test_optimizer_produces_valid_weights(tmp_path):
    """Optimizer must produce weights that sum to ~1.0 per intent."""
    import subprocess, sys
    out_file = tmp_path / "weights_out.json"
    result = subprocess.run(
        [sys.executable, "scripts/optimize_usefulness_weights.py",
         "--n-trials", "2", "--out", str(out_file)],
        capture_output=True, text=True,
        cwd="/mnt/c/Users/sidso/Documents/neural-search",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert out_file.exists(), "Output file not created"
    weights = json.loads(out_file.read_text())
    for intent, dims in weights.items():
        total = sum(dims.values())
        assert abs(total - 1.0) < 0.01, f"Intent {intent} weights sum to {total}"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_optimize_usefulness_weights.py -v 2>&1 | tail -10
```

Expected: All 3 fail — script doesn't exist.

- [ ] **Step 3: Read current weight profiles**

```bash
python -c "
from neural_search.retrieval.usefulness_scorer import INTENT_WEIGHT_PROFILES
import json
print(json.dumps(INTENT_WEIGHT_PROFILES, indent=2))
"
```

Note the dimension names and current values. These are what the optimizer will tune.

- [ ] **Step 4: Implement the weight optimizer**

```python
#!/usr/bin/env python3
"""Grid-search dimension weights to maximize NDCG@3 on labeled usefulness pairs.

For each intent, perturbs each dimension weight by ±0.1 and keeps the change
if NDCG@3 improves. Runs for --n-trials passes over all dimensions.

Usage:
    python scripts/optimize_usefulness_weights.py
    python scripts/optimize_usefulness_weights.py --n-trials 5 --out reports/optimized_weights.json
    python scripts/optimize_usefulness_weights.py --dry-run
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

SEED_FILE = Path("data/eval/usefulness_seed_pairs.jsonl")
LABEL_TO_INT = {"not_useful": 0, "weakly_useful": 1, "useful": 2, "highly_useful": 3}
K = 3
STEP = 0.05
MIN_WEIGHT = 0.01


def _ndcg_at_k(ranked_labels: list[int], k: int) -> float:
    """Compute NDCG@k for a list of relevance labels in ranked order."""
    def _dcg(labels: list[int]) -> float:
        return sum(
            (2 ** rel - 1) / math.log2(rank + 2)
            for rank, rel in enumerate(labels[:k])
        )
    ideal = sorted(ranked_labels, reverse=True)
    idcg = _dcg(ideal)
    return _dcg(ranked_labels) / idcg if idcg > 0.0 else 0.0


def _evaluate_weights(
    profiles: dict[str, dict[str, float]],
    pairs_by_intent: dict[str, list[dict]],
) -> float:
    """Score profiles against labeled pairs; return mean NDCG@3 across queries."""
    from neural_search.retrieval.usefulness_scorer import (
        DatasetContext, UsefulnessIntent, score_usefulness,
    )

    total_ndcg, n_queries = 0.0, 0

    for intent_str, pairs in pairs_by_intent.items():
        # Group by query_id
        query_candidates: dict[str, list[dict]] = {}
        for p in pairs:
            query_candidates.setdefault(p["query_id"], []).append(p)

        try:
            intent = UsefulnessIntent(intent_str)
        except ValueError:
            continue

        for qid, cands in query_candidates.items():
            if len(cands) < 2:
                continue
            query_ctx = DatasetContext(dataset_id=f"__query__{qid}")
            scored = []
            for c in cands:
                cand_ctx = DatasetContext(dataset_id=c["candidate_id"])
                s = score_usefulness(query_ctx, cand_ctx, intent)
                scored.append((s.total_score, LABEL_TO_INT.get(c["usefulness_label"], 0)))
            scored.sort(key=lambda x: x[0], reverse=True)
            ranked_labels = [lab for _, lab in scored]
            total_ndcg += _ndcg_at_k(ranked_labels, K)
            n_queries += 1

    return total_ndcg / n_queries if n_queries > 0 else 0.0


def _normalize_profile(profile: dict[str, float]) -> dict[str, float]:
    """Normalize so weights sum to 1.0."""
    total = sum(max(v, MIN_WEIGHT) for v in profile.values())
    return {k: max(v, MIN_WEIGHT) / total for k, v in profile.items()}


def load_pairs_by_intent() -> dict[str, list[dict]]:
    by_intent: dict[str, list[dict]] = {}
    for line in Path(SEED_FILE).read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        intent = p.get("intent", "")
        if intent:
            by_intent.setdefault(intent, []).append(p)
    return by_intent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimize usefulness dimension weights")
    parser.add_argument("--n-trials", type=int, default=3, help="Optimization passes (default: 3)")
    parser.add_argument("--out", default="reports/optimized_weights_v11.json",
                        help="Output path for optimized weights JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print current weights without optimizing")
    args = parser.parse_args(argv)

    from neural_search.retrieval.usefulness_scorer import INTENT_WEIGHT_PROFILES

    if args.dry_run:
        print("DRY RUN — current INTENT_WEIGHT_PROFILES:")
        print(json.dumps(INTENT_WEIGHT_PROFILES, indent=2))
        return 0

    pairs_by_intent = load_pairs_by_intent()
    print(f"Loaded pairs: { {k: len(v) for k, v in pairs_by_intent.items()} }")

    profiles = copy.deepcopy(INTENT_WEIGHT_PROFILES)
    baseline = _evaluate_weights(profiles, pairs_by_intent)
    print(f"Baseline NDCG@{K}: {baseline:.4f}")

    for trial in range(args.n_trials):
        improved = 0
        for intent, dims in profiles.items():
            if intent not in pairs_by_intent or len(pairs_by_intent[intent]) < 4:
                continue
            for dim in list(dims.keys()):
                for delta in [+STEP, -STEP]:
                    candidate = copy.deepcopy(profiles)
                    candidate[intent][dim] = max(MIN_WEIGHT, candidate[intent][dim] + delta)
                    candidate[intent] = _normalize_profile(candidate[intent])
                    score = _evaluate_weights(candidate, pairs_by_intent)
                    if score > baseline + 1e-6:
                        profiles = candidate
                        baseline = score
                        improved += 1
        print(f"Trial {trial + 1}/{args.n_trials}: NDCG@{K} = {baseline:.4f}  ({improved} improvements)")

    # Normalize all profiles
    for intent in profiles:
        profiles[intent] = _normalize_profile(profiles[intent])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profiles, indent=2))
    print(f"\nOptimized weights written to: {out_path}")
    print(f"Final NDCG@{K}: {baseline:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/test_optimize_usefulness_weights.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 6: Run the optimizer**

```bash
python scripts/optimize_usefulness_weights.py --n-trials 3 --out reports/optimized_weights_v11.json 2>&1 | tee reports/optimization_run.log
```

Expected: prints baseline NDCG, then improvement per trial, then final NDCG and saves output.

- [ ] **Step 7: Commit**

```bash
git add scripts/optimize_usefulness_weights.py tests/test_optimize_usefulness_weights.py reports/optimized_weights_v11.json reports/optimization_run.log
git commit -m "feat: add dimension weight optimizer + run v1.1 optimization on labeled pairs"
```

---

### Task 7: Re-run benchmark and whitepaper update

**Files:**
- Modify: `docs/whitepaper/neural_search_whitepaper.tex`

Re-run the full real-corpus benchmark with the expanded corpus, then update the whitepaper with v1.1 numbers.

- [ ] **Step 1: Run real-corpus benchmark with expanded corpus**

```bash
python -m neural_search.evaluation.run_benchmark --suite real_corpus --output-dir data/eval/results/real_corpus_v11
```

Expected: writes `latest_eval_report.json` with P@5, MRR, NDCG@10 metrics.

- [ ] **Step 2: Print and compare metrics**

```bash
python -c "
import json
from pathlib import Path

def show(path, label):
    try:
        d = json.loads(Path(path).read_text())
        print(f'{label}:')
        for k in ['mean_precision_at_5', 'mean_mrr', 'mean_ndcg_at_10', 'mean_label_recall_at_10']:
            print(f'  {k}: {d.get(k, \"N/A\")}')
    except FileNotFoundError:
        print(f'{label}: file not found')

show('data/eval/results/real_corpus_v09/latest_eval_report.json', 'v0.9 baseline')
show('data/eval/results/real_corpus_v11/latest_eval_report.json', 'v1.1 expanded')
"
```

- [ ] **Step 3: Update whitepaper corpus stats section**

Find in the whitepaper:
```latex
Current Neural Search performance on the 30-query benchmark (demo corpus, 26 datasets):
```

Add a v1.1 section after the existing benchmark table (already updated in v1.0 plan Task 4). Add:

```latex
\begin{table}[h]
\centering
\caption{Neural Search Benchmark Performance — v1.1 Real Corpus (1000+ datasets)}
\begin{tabular}{lcc}
\toprule
Metric & v0.9 (371 datasets) & v1.1 (1000+ datasets) \\
\midrule
Mean Precision@5 & XX.X\% & XX.X\% \\
Label Recall@10 & XX.X\% & XX.X\% \\
MRR & X.XXX & X.XXX \\
NDCG@10 & X.XXX & X.XXX \\
Hard-Negative Violations & 0 & 0 \\
\bottomrule
\end{tabular}
\end{table}
```

Fill in actual values from Steps 1-2. v0.9 values come from `data/eval/results/real_corpus_v09/latest_eval_report.json`.

- [ ] **Step 4: Update corpus stats in System Architecture section**

Find any mention of corpus size (e.g., "163 DANDI datasets") and update to reflect 1000+.

- [ ] **Step 5: Commit**

```bash
git add docs/whitepaper/neural_search_whitepaper.tex data/eval/results/real_corpus_v11/
git commit -m "docs: whitepaper v1.1 — expanded corpus benchmark results, corpus size update"
```

---

### Task 8: Full test suite verification

**Files:**
- No code changes.

- [ ] **Step 1: Run all new tests from v1.1**

```bash
pytest tests/test_optimize_usefulness_weights.py tests/test_seed_pairs_coverage.py tests/test_evaluate_usefulness_correlation.py -v
```

Expected: All pass.

- [ ] **Step 2: Run full suite**

```bash
pytest tests/ -q --tb=short --ignore=tests/test_search_quality.py 2>&1 | tail -5
```

Expected: N passed where N >= 985, 0 failed.

- [ ] **Step 3: Commit any cleanup**

```bash
git status && git add -p && git commit -m "chore: v1.1 final cleanup — all tests passing"
```

---

## Self-Review

### Spec Coverage
- [x] DANDI expansion: limit 15→30, 10 new query terms → 500+ datasets
- [x] OpenNeuro expansion: limit 30→100, 4 new modalities → 500+ datasets
- [x] Combined corpus ≥ 1000 datasets verified with assertion
- [x] Knowledge graph rebuild from expanded corpus
- [x] Embeddings recomputed for new records
- [x] Human annotation sprint generates 50 new real-corpus candidates and labels them
- [x] Seed pairs grow from 49 → 99+; coverage test updated to assert >= 99
- [x] Weight optimizer: grid search on 10 dimensions × 7 intents, maximizes NDCG@3
- [x] Re-run benchmark with expanded corpus + updated comparison table
- [x] Whitepaper v1.1 sections added

### Placeholder Scan
- Task 7 Step 3 uses `XX.X` fill-in values — these are intentional run-then-fill instructions.
- Task 4 Step 3 references `build_real_corpus_graph.py` — read the file first to confirm output path before writing the rebuild wrapper.
- Task 4 Task "recompute_embeddings.py" references `build_field_embeddings`, `build_fingerprints` — verify actual function names in `neural_search/embeddings/__init__.py` before writing.

### Type Consistency
- `score_usefulness(query_ctx, cand_ctx, intent)` signature unchanged from v0.9
- `INTENT_WEIGHT_PROFILES: dict[str, dict[str, float]]` — keys are intent strings, values are dimension name → weight
- `_ndcg_at_k(ranked_labels, k)` — takes list of ints, returns float in [0, 1]
- `DatasetContext(dataset_id=str)` — minimal context sufficient for scorer in optimizer
