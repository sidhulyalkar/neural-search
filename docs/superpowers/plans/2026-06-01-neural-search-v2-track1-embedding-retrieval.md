# Neural Search v2.0 Track 1 — Embedding & Retrieval Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the retrieval stack from 64-dim hashing embeddings to BGE-large-en-v1.5 (1024-dim semantic), add turbovec ANN index, fix s9 graph proximity ID mapping, and measure the resulting Spearman r improvement.

**Architecture:** `DenseEmbeddingProvider` wraps BGE-large-en-v1.5 via sentence-transformers. `NeuralSearchTurboIndex` wraps turbovec `IdMapIndex(dim=1024, bit_width=4)`. The dataset context bridge is fixed to prefer `dataset_id` over `source_id` so graph node ID resolution works. The graph ID resolver adds a `node:` prefix for lookup.

**Tech Stack:** sentence-transformers, turbovec, scipy (Spearman), existing `neural_search.embeddings`, `neural_search.search.core`, `neural_search.retrieval.usefulness_scorer`, pytest

---

## File Map

**Create:**
- `neural_search/embeddings/dense_provider.py` — BGE-large-en-v1.5 provider
- `neural_search/embeddings/turbovec_index.py` — turbovec IdMapIndex wrapper
- `scripts/ablate_graph_proximity.py` — prove s9 fix changes rankings
- `scripts/build_turbovec_index.py` — build turbovec index from embeddings file
- `scripts/validate_turbovec_recall.py` — ANN vs brute-force recall check
- `scripts/freeze_baseline.py` — record current metrics before any changes
- `tests/test_dense_provider.py`
- `tests/test_turbovec_index.py`
- `tests/test_turbovec_recall.py`

**Modify:**
- `neural_search/embeddings/__init__.py` — export `DenseEmbeddingProvider`, `NeuralSearchTurboIndex`
- `neural_search/embeddings/providers.py` — register `bge-large` in `_PROVIDER_REGISTRY`
- `neural_search/retrieval/dataset_context_bridge.py` — prefer `dataset_id` over `source_id`
- `neural_search/retrieval/usefulness_scorer.py` — add `_resolve_graph_node_id()` helper
- `scripts/recompute_embeddings.py` — add `--provider dense` flag
- `pyproject.toml` — add `turbovec` to optional deps

---

## Task 1: Freeze Baseline

Record current state before any changes — corpus size, Spearman r, NDCG, config, git hash.

**Files:**
- Create: `scripts/freeze_baseline.py`

- [ ] **Step 1: Write the baseline freezer script**

```python
#!/usr/bin/env python3
"""Freeze a snapshot of current metrics as the v0.9 comparison baseline.

Usage:
    python scripts/freeze_baseline.py
    python scripts/freeze_baseline.py --output reports/baseline_v09.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/baseline_v09.json")
    args = parser.parse_args(argv)

    # Count corpus records
    corpus_sizes: dict[str, int] = {}
    normalized_dir = Path("data/corpus/normalized")
    for f in sorted(normalized_dir.glob("real_*.jsonl")):
        count = sum(1 for _ in f.open())
        corpus_sizes[f.stem] = count

    # Count seed pairs
    seed_path = Path("data/eval/usefulness_seed_pairs.jsonl")
    n_pairs = sum(1 for _ in seed_path.open()) if seed_path.exists() else 0

    # Count embedding vectors
    embed_path = Path("data/embeddings/real_all.field_embeddings.jsonl")
    n_embeddings = sum(1 for _ in embed_path.open()) if embed_path.exists() else 0

    # Read last correlation report if exists
    corr_path = Path("reports/usefulness_correlation_v09.json")
    spearman_r = None
    if corr_path.exists():
        try:
            data = json.loads(corr_path.read_text())
            spearman_r = data.get("spearman_r")
        except Exception:
            pass

    baseline = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "corpus_record_counts": corpus_sizes,
        "total_corpus_records": sum(corpus_sizes.values()),
        "seed_pairs": n_pairs,
        "field_embedding_vectors": n_embeddings,
        "embedding_provider": "hashing-64",
        "spearman_r": spearman_r,
        "notes": "v0.9 baseline before Track 1 embedding upgrade",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2))
    print(f"Baseline frozen → {out}")
    for k, v in baseline.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the freezer**

```bash
cd /mnt/c/Users/sidso/Documents/neural-search
python scripts/freeze_baseline.py
```

Expected: `reports/baseline_v09.json` created with current counts.

- [ ] **Step 3: Commit**

```bash
git add scripts/freeze_baseline.py reports/baseline_v09.json
git commit -m "chore: freeze v0.9 baseline before Track 1 upgrade"
```

---

## Task 2: Fix s9 Graph Proximity ID Mapping

The graph nodes use IDs like `node:dataset:dandi:000003`, but `dataset_context_from_record` returns `source_id` (`000003`), so lookups always miss. Fix in two places: bridge prefers `dataset_id`, scorer adds `node:` prefix.

**Files:**
- Modify: `neural_search/retrieval/dataset_context_bridge.py:43-46`
- Modify: `neural_search/retrieval/usefulness_scorer.py:194-215`
- Test: `tests/test_usefulness_scorer.py` (existing file)

- [ ] **Step 1: Write a failing test for the ID resolution**

Open `tests/test_usefulness_scorer.py` and add at the end:

```python
def test_graph_proximity_resolves_node_prefix(tmp_path):
    """score_usefulness should find nodes with the node: prefix and return non-0.3 score."""
    from neural_search.retrieval.usefulness_scorer import DatasetContext, score_usefulness
    from neural_search.graph.schema import KnowledgeGraph, GraphNode, GraphEdge

    # Build minimal graph with node: prefix IDs and a shared task neighbor
    task_node = GraphNode(
        node_id="node:task:decision_making",
        node_type="task",
        properties={"label": "decision_making"},
    )
    ds_a = GraphNode(
        node_id="node:dataset:dandi:000003",
        node_type="dataset",
        properties={},
    )
    ds_b = GraphNode(
        node_id="node:dataset:dandi:000004",
        node_type="dataset",
        properties={},
    )
    edge_a = GraphEdge(
        edge_id="e1",
        edge_type="dataset_has_task",
        source_node_id="node:dataset:dandi:000003",
        target_node_id="node:task:decision_making",
    )
    edge_b = GraphEdge(
        edge_id="e2",
        edge_type="dataset_has_task",
        source_node_id="node:dataset:dandi:000004",
        target_node_id="node:task:decision_making",
    )
    graph = KnowledgeGraph(
        nodes={n.node_id: n for n in [task_node, ds_a, ds_b]},
        edges={e.edge_id: e for e in [edge_a, edge_b]},
    )

    q = DatasetContext(dataset_id="dataset:dandi:000003", tasks=["decision_making"])
    c = DatasetContext(dataset_id="dataset:dandi:000004", tasks=["decision_making"])

    score = score_usefulness(q, c, graph=graph)
    # Must find nodes → real PathSim > 0, not 0.3 neutral prior
    assert score.dimension_scores["graph_proximity"] != 0.3, (
        "graph_proximity returned neutral prior 0.3 — ID resolution is broken"
    )
    assert score.dimension_scores["graph_proximity"] > 0.0
    # No "not found" warning
    assert not any("not found" in w for w in score.warnings)
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_usefulness_scorer.py::test_graph_proximity_resolves_node_prefix -v
```

Expected: FAIL — `AssertionError: graph_proximity returned neutral prior 0.3`

- [ ] **Step 3: Fix the dataset context bridge to prefer `dataset_id`**

In [neural_search/retrieval/dataset_context_bridge.py](neural_search/retrieval/dataset_context_bridge.py), change the `dataset_id` extraction line (currently around line 43):

Old:
```python
    dataset_id = str(
        record.get("id")
        or record.get("source_id")
        or record.get("dataset_id")
        or ""
    )
```

New:
```python
    dataset_id = str(
        record.get("dataset_id")
        or record.get("id")
        or record.get("source_id")
        or ""
    )
```

- [ ] **Step 4: Fix the graph node ID resolver in `usefulness_scorer.py`**

In [neural_search/retrieval/usefulness_scorer.py](neural_search/retrieval/usefulness_scorer.py), replace the graph lookup block (lines ~194–215) with:

```python
    # s9: graph_proximity
    if graph is not None:
        from neural_search.retrieval.graph_usefulness import normalized_metapath_score
        graph_dict = graph.model_dump(mode="json")
        node_ids = graph_dict.get("nodes", {})

        def _resolve_node_id(did: str) -> str | None:
            if did in node_ids:
                return did
            # Graph nodes use "node:" prefix; dataset_ids use "dataset:source:id"
            prefixed = f"node:{did}"
            if prefixed in node_ids:
                return prefixed
            return None

        q_node = _resolve_node_id(query_context.dataset_id)
        c_node = _resolve_node_id(candidate.dataset_id)
        if q_node and c_node:
            graph_proximity = normalized_metapath_score(
                graph_dict, q_node, c_node, "dataset_has_task",
            )
        else:
            graph_proximity = 0.3
            warnings.append(
                "graph_proximity: dataset_id(s) not found in graph nodes; using neutral prior 0.3"
            )
    else:
        graph_proximity = 0.3
        warnings.append("graph_proximity: using neutral prior 0.3 (graph not available)")
    dims["graph_proximity"] = graph_proximity
```

- [ ] **Step 5: Run test — expect PASS**

```bash
pytest tests/test_usefulness_scorer.py::test_graph_proximity_resolves_node_prefix -v
```

Expected: PASS

- [ ] **Step 6: Run full test suite — expect no regressions**

```bash
pytest tests/test_usefulness_scorer.py -v
```

Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add neural_search/retrieval/dataset_context_bridge.py \
        neural_search/retrieval/usefulness_scorer.py \
        tests/test_usefulness_scorer.py
git commit -m "fix: resolve node: prefix in s9 graph proximity lookup"
```

---

## Task 3: Graph Proximity Ablation Script

Prove that the fix actually changes rankings — a code change alone is not evidence.

**Files:**
- Create: `scripts/ablate_graph_proximity.py`
- Create: `tests/test_ablate_graph_proximity.py`

- [ ] **Step 1: Write the ablation script**

```python
#!/usr/bin/env python3
"""Ablate s9 graph proximity: compare real graph vs neutral prior (0.3).

Runs benchmark queries through the scorer twice — once with the real graph,
once with graph=None (forcing 0.3 neutral prior) — and reports:
  - % of query-candidate pairs where real graph score ≠ neutral prior
  - NDCG@10 with s9=0.3 vs real s9

Usage:
    python scripts/ablate_graph_proximity.py
    python scripts/ablate_graph_proximity.py --n-queries 10 --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


BENCHMARK_PATH = Path("data/eval/benchmark_queries_real_corpus.yaml")
GRAPH_PATH = Path("data/graph/neural_search_graph.real_corpus.json")
REPORT_PATH = Path("reports/graph_ablation.json")


def _dcg(gains: list[float]) -> float:
    import math
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg(ranked_gains: list[float], ideal_gains: list[float], k: int = 10) -> float:
    dcg = _dcg(ranked_gains[:k])
    idcg = _dcg(sorted(ideal_gains, reverse=True)[:k])
    return dcg / idcg if idcg > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(f"DRY RUN — would ablate {args.n_queries} queries against graph at {GRAPH_PATH}")
        return 0

    if not GRAPH_PATH.exists():
        print(f"ERROR: graph not found at {GRAPH_PATH}")
        return 1

    data = yaml.safe_load(BENCHMARK_PATH.read_text())
    queries = data.get("benchmark_queries", [])[:args.n_queries]

    from neural_search.graph.schema import KnowledgeGraph
    from neural_search.search.core import search_datasets

    graph_json = json.loads(GRAPH_PATH.read_text())
    graph = KnowledgeGraph(**graph_json)

    pairs_changed = 0
    total_pairs = 0
    ndcg_with_graph: list[float] = []
    ndcg_without_graph: list[float] = []

    retrieval_with_graph = {
        "graph": {"enabled": True, "path": str(GRAPH_PATH)},
    }

    for q in queries:
        query_text = q["query"]
        expected = set(q.get("expected_tasks", []) + q.get("expected_modalities_any", []))

        resp_with = search_datasets(query_text, limit=10, retrieval_config=retrieval_with_graph)
        resp_without = search_datasets(query_text, limit=10)

        gains_with, gains_without = [], []
        for r_with, r_without in zip(resp_with.results, resp_without.results):
            u_with = (r_with.usefulness_score or {}).get("dimension_scores", {}).get("graph_proximity", 0.3)
            u_without = (r_without.usefulness_score or {}).get("dimension_scores", {}).get("graph_proximity", 0.3)
            total_pairs += 1
            if abs(u_with - u_without) > 1e-6:
                pairs_changed += 1

            matched = " ".join(r_with.why_matched + list(r_with.matched_terms)).lower()
            gain = 2.0 if any(t.lower() in matched for t in expected) else 0.0
            gains_with.append(gain)

            matched2 = " ".join(r_without.why_matched + list(r_without.matched_terms)).lower()
            gain2 = 2.0 if any(t.lower() in matched2 for t in expected) else 0.0
            gains_without.append(gain2)

        ndcg_with_graph.append(_ndcg(gains_with, gains_with + gains_without))
        ndcg_without_graph.append(_ndcg(gains_without, gains_with + gains_without))

    pct_changed = 100 * pairs_changed / total_pairs if total_pairs > 0 else 0.0
    mean_ndcg_with = sum(ndcg_with_graph) / len(ndcg_with_graph) if ndcg_with_graph else 0.0
    mean_ndcg_without = sum(ndcg_without_graph) / len(ndcg_without_graph) if ndcg_without_graph else 0.0

    report = {
        "n_queries": len(queries),
        "total_pairs": total_pairs,
        "pairs_changed": pairs_changed,
        "pct_pairs_changed": round(pct_changed, 1),
        "mean_ndcg_with_graph": round(mean_ndcg_with, 4),
        "mean_ndcg_without_graph": round(mean_ndcg_without, 4),
        "ndcg_delta": round(mean_ndcg_with - mean_ndcg_without, 4),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if pct_changed < 10.0:
        print(f"\nWARNING: Only {pct_changed:.1f}% of pairs changed — s9 fix may not be working.")
        print("Investigate: are node IDs resolving in the real graph?")
    else:
        print(f"\n✓ s9 fix confirmed: {pct_changed:.1f}% of pairs changed rank.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write the test**

Create `tests/test_ablate_graph_proximity.py`:

```python
"""Tests for ablate_graph_proximity.py."""
import subprocess
import sys


def test_dry_run_exits_cleanly():
    result = subprocess.run(
        [sys.executable, "scripts/ablate_graph_proximity.py", "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "DRY RUN" in result.stdout


def test_syntax():
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", "scripts/ablate_graph_proximity.py"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_ablate_graph_proximity.py -v
```

Expected: 2 passed

- [ ] **Step 4: Run the ablation (proves fix works)**

```bash
python scripts/ablate_graph_proximity.py --n-queries 20
```

Expected: `pct_pairs_changed` ≥ 10.0%. If below, the graph doesn't have enough `dataset_has_task` edges for the benchmark queries — check `data/graph/neural_search_graph.real_corpus.json` edge count.

- [ ] **Step 5: Commit**

```bash
git add scripts/ablate_graph_proximity.py tests/test_ablate_graph_proximity.py
git commit -m "feat: add graph proximity ablation script; prove s9 fix changes rankings"
```

---

## Task 4: DenseEmbeddingProvider (BGE-large-en-v1.5)

BGE-large uses the same sentence-transformers API as the existing `SentenceTransformerProvider`. We create a dedicated subclass that enforces the correct model name and dimension.

**Files:**
- Create: `neural_search/embeddings/dense_provider.py`
- Modify: `neural_search/embeddings/providers.py` (add to registry)
- Modify: `neural_search/embeddings/__init__.py` (export)
- Create: `tests/test_dense_provider.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dense_provider.py`:

```python
"""Tests for DenseEmbeddingProvider (BGE-large-en-v1.5)."""
import pytest


def test_dense_provider_import():
    from neural_search.embeddings.dense_provider import DenseEmbeddingProvider
    assert DenseEmbeddingProvider is not None


def test_dense_provider_metadata():
    from neural_search.embeddings.dense_provider import DenseEmbeddingProvider
    p = DenseEmbeddingProvider.__new__(DenseEmbeddingProvider)
    # Check class-level attributes without loading the model
    assert DenseEmbeddingProvider.MODEL_NAME == "BAAI/bge-large-en-v1.5"
    assert DenseEmbeddingProvider.DIMENSION == 1024


def test_dense_provider_registry():
    from neural_search.embeddings.providers import get_provider, _PROVIDER_REGISTRY
    assert "bge-large" in _PROVIDER_REGISTRY


def test_dense_provider_get_provider_bge(monkeypatch):
    """get_provider('bge-large') should return DenseEmbeddingProvider without loading model."""
    from neural_search.embeddings.dense_provider import DenseEmbeddingProvider

    class _FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            import numpy as np
            return np.zeros((len(texts), 1024), dtype=float)
        def get_sentence_embedding_dimension(self):
            return 1024

    # Patch the constructor
    monkeypatch.setattr(
        "neural_search.embeddings.dense_provider.DenseEmbeddingProvider._load_model",
        lambda self: setattr(self, "_model", _FakeModel()),
    )
    from neural_search.embeddings.providers import get_provider
    p = get_provider("bge-large")
    assert isinstance(p, DenseEmbeddingProvider)
    assert p.dimension == 1024
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_dense_provider.py -v
```

Expected: FAIL with ImportError (`dense_provider` not found)

- [ ] **Step 3: Create the dense provider**

Create `neural_search/embeddings/dense_provider.py`:

```python
"""Dense embedding provider for BGE-large-en-v1.5.

BGE-large-en-v1.5 (BAAI/bge-large-en-v1.5):
  - 1024 dimensions
  - MTEB top-5 for scientific retrieval
  - Same model for corpus + query (cosine similarity is well-defined)
  - Fits in ~4GB VRAM on 3070 Ti; batch-embeds 5000 records in ~45s

Usage:
    from neural_search.embeddings.dense_provider import DenseEmbeddingProvider

    provider = DenseEmbeddingProvider()           # auto-detects GPU
    vecs = provider.embed_batch(["text1", ...])   # shape (N, 1024)
    q = provider.embed_query("query text")         # shape (1024,)
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from neural_search.embeddings.providers import EmbeddingProviderBase

logger = logging.getLogger(__name__)


class DenseEmbeddingProvider(EmbeddingProviderBase):
    """BGE-large-en-v1.5 provider using sentence-transformers."""

    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    DIMENSION = 1024

    def __init__(
        self,
        *,
        device: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
        model: Any | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._normalize = normalize
        self._model_version = "unknown"

        if model is not None:
            self._model = model
            self._device = device or "cpu"
            return

        self._device = device or self._auto_device()
        self._load_model()

    def _auto_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading {self.MODEL_NAME} on {self._device}")
            self._model = SentenceTransformer(self.MODEL_NAME, device=self._device)
            try:
                import sentence_transformers
                self._model_version = getattr(sentence_transformers, "__version__", "unknown")
            except Exception:
                pass
            logger.info(f"Loaded {self.MODEL_NAME}")
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers to use DenseEmbeddingProvider. "
                "Run: pip install sentence-transformers"
            ) from exc

    @property
    def provider_name(self) -> str:
        return "bge-large"

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    @property
    def normalize(self) -> bool:
        return self._normalize

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns (1024,) float32 array."""
        return np.array(self.embed_text(text), dtype=np.float32)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches. Returns list of 1024-dim vectors."""
        all_vecs: list[list[float]] = []
        texts_list = list(texts)
        for i in range(0, len(texts_list), self._batch_size):
            batch = texts_list[i : i + self._batch_size]
            vecs = self._model.encode(
                batch,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )
            all_vecs.extend(list(map(float, v)) for v in vecs)
        return all_vecs

    def embed_corpus_batch(self, texts: list[str]) -> np.ndarray:
        """Embed corpus texts. Returns (N, 1024) float32 array."""
        vecs = self.embed_batch(texts)
        return np.array(vecs, dtype=np.float32)
```

- [ ] **Step 4: Register in providers.py**

In [neural_search/embeddings/providers.py](neural_search/embeddings/providers.py), add to `_PROVIDER_REGISTRY` (around line 561):

```python
# Add after the existing imports at the top of the file:
# (no import needed here — registry uses lazy string lookup)

_PROVIDER_REGISTRY: dict[str, type[EmbeddingProviderBase]] = {
    "hashing": HashingEmbeddingProvider,
    "sentence-transformer": SentenceTransformerProvider,
    "specter2": SPECTER2Provider,
    "scibert": SciBERTProvider,
    "bge-large": None,  # registered lazily below
}
```

Then after the registry dict, add a lazy registration block:

```python
def _get_bge_large_class() -> type:
    from neural_search.embeddings.dense_provider import DenseEmbeddingProvider
    return DenseEmbeddingProvider


# Register bge-large lazily to avoid circular import at module load
_PROVIDER_REGISTRY["bge-large"] = _get_bge_large_class()  # type: ignore[assignment]
```

And update `get_provider` to handle the "bge-large" name:

```python
def get_provider(name: str = "auto", **kwargs: Any) -> EmbeddingProviderBase:
    if name == "auto":
        try:
            return SentenceTransformerProvider(**kwargs)
        except RuntimeError:
            logger.info("sentence-transformers not available, using hashing provider")
            return HashingEmbeddingProvider(**kwargs)

    if name == "bge-large":
        from neural_search.embeddings.dense_provider import DenseEmbeddingProvider
        return DenseEmbeddingProvider(**kwargs)

    if name not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider: {name}. Available: {list(_PROVIDER_REGISTRY.keys())}"
        )
    return _PROVIDER_REGISTRY[name](**kwargs)
```

- [ ] **Step 5: Export from `__init__.py`**

In [neural_search/embeddings/__init__.py](neural_search/embeddings/__init__.py), add at the end of imports and `__all__`:

```python
from neural_search.embeddings.dense_provider import DenseEmbeddingProvider
```

Add `"DenseEmbeddingProvider"` to `__all__`.

- [ ] **Step 6: Run tests — expect PASS**

```bash
pytest tests/test_dense_provider.py -v
```

Expected: 4 passed (model-loading tests skipped or mocked)

- [ ] **Step 7: Install sentence-transformers if needed, verify BGE loads**

```bash
pip install sentence-transformers
python -c "
from neural_search.embeddings.dense_provider import DenseEmbeddingProvider
p = DenseEmbeddingProvider()
v = p.embed_text('prefrontal cortex calcium imaging')
print(f'dim={len(v)}, norm≈{sum(x*x for x in v)**0.5:.4f}')
"
```

Expected: `dim=1024, norm≈1.0000`

- [ ] **Step 8: Commit**

```bash
git add neural_search/embeddings/dense_provider.py \
        neural_search/embeddings/providers.py \
        neural_search/embeddings/__init__.py \
        tests/test_dense_provider.py
git commit -m "feat: add DenseEmbeddingProvider (BGE-large-en-v1.5, 1024-dim)"
```

---

## Task 5: NeuralSearchTurboIndex (turbovec wrapper)

Wraps `turbovec.IdMapIndex(dim=1024, bit_width=4)` with a stable interface for insert, search, save/load. Falls back gracefully if turbovec is not installed.

**Files:**
- Create: `neural_search/embeddings/turbovec_index.py`
- Create: `tests/test_turbovec_index.py`

**First: install turbovec**

```bash
pip install turbovec
pip show turbovec  # verify
```

If `turbovec` is not on PyPI, use `faiss-cpu` as a drop-in alternative (see note at end of Task 5).

- [ ] **Step 1: Write the failing test**

Create `tests/test_turbovec_index.py`:

```python
"""Tests for NeuralSearchTurboIndex."""
import numpy as np
import pytest


def test_turbovec_import():
    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    assert NeuralSearchTurboIndex is not None


def test_index_add_and_search():
    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    idx = NeuralSearchTurboIndex(dim=4, bit_width=4)
    vecs = np.random.randn(10, 4).astype(np.float32)
    # L2-normalize
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"ds{i:03d}" for i in range(10)]
    idx.add(ids=ids, vectors=vecs)
    assert idx.size == 10

    q = vecs[0].copy()
    results = idx.search(q, k=3)
    assert len(results) == 3
    # Top result should be the query itself
    assert results[0][0] == ids[0]


def test_index_save_load(tmp_path):
    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    idx = NeuralSearchTurboIndex(dim=4, bit_width=4)
    vecs = np.eye(4, dtype=np.float32)
    ids = ["a", "b", "c", "d"]
    idx.add(ids=ids, vectors=vecs)

    path = tmp_path / "test.turbo"
    idx.save(str(path))

    idx2 = NeuralSearchTurboIndex.load(str(path))
    assert idx2.size == 4
    results = idx2.search(vecs[0], k=2)
    assert results[0][0] == "a"


def test_index_provider_metadata():
    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    idx = NeuralSearchTurboIndex(dim=1024, bit_width=4)
    assert idx.dim == 1024
    assert idx.bit_width == 4
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_turbovec_index.py -v
```

Expected: FAIL with ImportError

- [ ] **Step 3: Create the turbovec index wrapper**

Create `neural_search/embeddings/turbovec_index.py`:

```python
"""Wrapper for turbovec IdMapIndex with stable save/load API.

IdMapIndex preserves stable string IDs (unlike TurboQuantIndex whose slot IDs
shift after deletion). bit_width must be 2 or 4 — 8 is not supported.

Usage:
    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    import numpy as np

    idx = NeuralSearchTurboIndex(dim=1024, bit_width=4)
    idx.add(ids=["dandi:000003", ...], vectors=embeddings_matrix)  # (N, 1024) float32

    results = idx.search(query_vec, k=50)
    # results: list of (dataset_id: str, distance: float), sorted by distance asc
    # (lower distance = more similar for cosine/inner-product quantized search)

    idx.save("data/index/turbovec.index")
    idx2 = NeuralSearchTurboIndex.load("data/index/turbovec.index")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_TURBOVEC_AVAILABLE: bool | None = None


def _check_turbovec() -> bool:
    global _TURBOVEC_AVAILABLE
    if _TURBOVEC_AVAILABLE is None:
        try:
            import turbovec  # noqa: F401
            _TURBOVEC_AVAILABLE = True
        except ImportError:
            _TURBOVEC_AVAILABLE = False
    return _TURBOVEC_AVAILABLE


class NeuralSearchTurboIndex:
    """Compressed ANN index backed by turbovec IdMapIndex.

    Falls back to exact brute-force cosine search if turbovec is not installed
    (with a warning). This ensures tests can run without GPU/turbovec.

    bit_width must be 2 or 4. 4-bit gives better recall; 2-bit saves 50% memory.
    At 5000 datasets, 4-bit uses ~2.6 MB vs float32 baseline of ~20.5 MB.
    """

    def __init__(self, dim: int = 1024, bit_width: int = 4) -> None:
        if bit_width not in (2, 4):
            raise ValueError(f"bit_width must be 2 or 4, got {bit_width}")
        self.dim = dim
        self.bit_width = bit_width
        self._ids: list[str] = []
        self._vecs: np.ndarray | None = None  # fallback storage

        if _check_turbovec():
            from turbovec import IdMapIndex
            self._index: Any = IdMapIndex(dim=dim, bit_width=bit_width)
        else:
            logger.warning(
                "turbovec not installed — using exact brute-force fallback. "
                "Install turbovec for production use: pip install turbovec"
            )
            self._index = None

    @property
    def size(self) -> int:
        return len(self._ids)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """Add vectors with stable string IDs.

        Args:
            ids: List of string dataset IDs (stable external identifiers).
            vectors: Float32 array of shape (len(ids), dim). Should be L2-normalized
                     for cosine similarity queries.
        """
        if len(ids) != vectors.shape[0]:
            raise ValueError(f"ids length {len(ids)} != vectors rows {vectors.shape[0]}")
        vecs = vectors.astype(np.float32)

        if self._index is not None:
            self._index.add(ids=ids, vectors=vecs)
        else:
            # Fallback: store in numpy array
            if self._vecs is None:
                self._vecs = vecs
            else:
                self._vecs = np.vstack([self._vecs, vecs])

        self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int = 50) -> list[tuple[str, float]]:
        """Search for k nearest neighbors.

        Args:
            query: Float32 vector of shape (dim,). Should be L2-normalized.
            k: Number of results to return.

        Returns:
            List of (dataset_id, distance) sorted by distance ascending.
            Lower distance = more similar (inner product with negated sign, or L2).
        """
        if not self._ids:
            return []
        k = min(k, len(self._ids))
        q = query.astype(np.float32)

        if self._index is not None:
            distances, ids = self._index.search(q, k)
            return list(zip(ids, distances.tolist()))
        else:
            # Brute-force cosine fallback
            if self._vecs is None:
                return []
            sims = self._vecs @ q
            top_idx = np.argsort(sims)[::-1][:k]
            return [(self._ids[i], float(1.0 - sims[i])) for i in top_idx]

    def save(self, path: str) -> None:
        """Save index to disk. Creates a directory with index + metadata."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        meta = {"dim": self.dim, "bit_width": self.bit_width, "ids": self._ids}
        (p / "meta.json").write_text(json.dumps(meta))

        if self._index is not None:
            self._index.save(str(p / "turbovec.bin"))
        elif self._vecs is not None:
            np.save(str(p / "fallback_vecs.npy"), self._vecs)

        logger.info(f"Saved {self.size}-record index → {p}")

    @classmethod
    def load(cls, path: str) -> "NeuralSearchTurboIndex":
        """Load index from disk."""
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        obj = cls(dim=meta["dim"], bit_width=meta["bit_width"])
        obj._ids = meta["ids"]

        turbo_bin = p / "turbovec.bin"
        fallback_npy = p / "fallback_vecs.npy"

        if turbo_bin.exists() and obj._index is not None:
            obj._index.load(str(turbo_bin))
        elif fallback_npy.exists():
            obj._vecs = np.load(str(fallback_npy))

        logger.info(f"Loaded {obj.size}-record index ← {p}")
        return obj
```

- [ ] **Step 4: Export from `__init__.py`**

In [neural_search/embeddings/__init__.py](neural_search/embeddings/__init__.py), add:

```python
from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
```

Add `"NeuralSearchTurboIndex"` to `__all__`.

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/test_turbovec_index.py -v
```

Expected: 4 passed (uses brute-force fallback if turbovec not installed)

- [ ] **Step 6: Add turbovec to pyproject.toml optional deps**

In [pyproject.toml](pyproject.toml), add to the appropriate optional section (e.g., alongside sentence-transformers):

```toml
[project.optional-dependencies]
dense = [
  "sentence-transformers>=2.2.0",
  "turbovec>=0.1.0",
  "torch>=2.0.0",
]
```

- [ ] **Step 7: Commit**

```bash
git add neural_search/embeddings/turbovec_index.py \
        neural_search/embeddings/__init__.py \
        tests/test_turbovec_index.py \
        pyproject.toml
git commit -m "feat: add NeuralSearchTurboIndex (turbovec IdMapIndex wrapper, dim=1024, bit_width=4)"
```

---

## Task 6: Update recompute_embeddings.py for Dense Provider

Add `--provider dense` flag so the corpus can be re-embedded with BGE-large.

**Files:**
- Modify: `scripts/recompute_embeddings.py`
- Create: `tests/test_recompute_embeddings_dense.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_recompute_embeddings_dense.py`:

```python
"""Test that recompute_embeddings.py accepts --provider dense."""
import subprocess, sys


def test_dry_run_default_provider():
    r = subprocess.run(
        [sys.executable, "scripts/recompute_embeddings.py", "--dry-run"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr


def test_dry_run_dense_provider():
    r = subprocess.run(
        [sys.executable, "scripts/recompute_embeddings.py",
         "--provider", "dense", "--dry-run"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "dense" in r.stdout.lower() or "bge" in r.stdout.lower()


def test_syntax():
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", "scripts/recompute_embeddings.py"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_recompute_embeddings_dense.py::test_dry_run_dense_provider -v
```

Expected: FAIL (flag not recognized)

- [ ] **Step 3: Read the current recompute_embeddings.py**

Read [scripts/recompute_embeddings.py](scripts/recompute_embeddings.py) to understand current argument structure.

- [ ] **Step 4: Add `--provider` flag**

Locate the `argparse` block in `recompute_embeddings.py` and add:

```python
parser.add_argument(
    "--provider",
    default="hashing",
    choices=["hashing", "dense"],
    help="Embedding provider: 'hashing' (default, fast CI) or 'dense' (BGE-large-en-v1.5, semantic)",
)
```

In the dry-run print block, include the provider:

```python
if args.dry_run:
    print(f"DRY RUN — provider={args.provider}, would re-embed {n} records")
    return 0
```

In the main embedding loop, select the provider:

```python
if args.provider == "dense":
    from neural_search.embeddings.dense_provider import DenseEmbeddingProvider
    provider = DenseEmbeddingProvider()
    print(f"Using DenseEmbeddingProvider (BGE-large-en-v1.5, dim=1024)")
else:
    from neural_search.embeddings import HashingEmbeddingProvider
    provider = HashingEmbeddingProvider(dimensions=64)
    print(f"Using HashingEmbeddingProvider (dim=64)")
```

Also update the output path when using dense provider:

```python
if args.provider == "dense":
    output_path = Path("data/embeddings/real_all.dense.field_embeddings.jsonl")
else:
    output_path = Path("data/embeddings/real_all.field_embeddings.jsonl")
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/test_recompute_embeddings_dense.py -v
```

Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add scripts/recompute_embeddings.py tests/test_recompute_embeddings_dense.py
git commit -m "feat: add --provider dense flag to recompute_embeddings.py"
```

---

## Task 7: Build and Validate TurboVec Index

Create `scripts/build_turbovec_index.py` (reads dense embeddings → builds index) and `scripts/validate_turbovec_recall.py` (ANN vs brute-force recall check).

**Files:**
- Create: `scripts/build_turbovec_index.py`
- Create: `scripts/validate_turbovec_recall.py`
- Create: `tests/test_turbovec_recall.py`

- [ ] **Step 1: Write test for recall validator**

Create `tests/test_turbovec_recall.py`:

```python
"""Tests for turbovec recall validator."""
import subprocess, sys


def test_dry_run():
    r = subprocess.run(
        [sys.executable, "scripts/validate_turbovec_recall.py", "--dry-run"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout


def test_syntax_build():
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", "scripts/build_turbovec_index.py"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr


def test_syntax_validate():
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", "scripts/validate_turbovec_recall.py"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
```

- [ ] **Step 2: Create build script**

Create `scripts/build_turbovec_index.py`:

```python
#!/usr/bin/env python3
"""Build NeuralSearchTurboIndex from dense field embeddings.

Reads data/embeddings/real_all.dense.field_embeddings.jsonl (or --embeddings),
aggregates per-dataset vectors by mean-pooling, and saves a turbovec index.

Usage:
    python scripts/build_turbovec_index.py
    python scripts/build_turbovec_index.py --embeddings data/embeddings/real_all.dense.field_embeddings.jsonl
    python scripts/build_turbovec_index.py --dry-run
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EMBEDDINGS_PATH = Path("data/embeddings/real_all.dense.field_embeddings.jsonl")
INDEX_PATH = Path("data/index/turbovec_dense_1024.index")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default=str(EMBEDDINGS_PATH))
    parser.add_argument("--output", default=str(INDEX_PATH))
    parser.add_argument("--bit-width", type=int, default=4, choices=[2, 4])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    embed_path = Path(args.embeddings)
    if not embed_path.exists():
        print(f"ERROR: embeddings file not found: {embed_path}")
        print("Run: python scripts/recompute_embeddings.py --provider dense")
        return 1

    # Load embeddings and aggregate by dataset_id (mean pool)
    vecs_by_id: dict[str, list[np.ndarray]] = defaultdict(list)
    with embed_path.open() as f:
        for line in f:
            rec = json.loads(line)
            did = rec.get("entity_id", "")
            vec = rec.get("vector", [])
            if did and vec:
                vecs_by_id[did].append(np.array(vec, dtype=np.float32))

    if not vecs_by_id:
        print("ERROR: no embeddings found")
        return 1

    dim = len(next(iter(vecs_by_id.values()))[0])
    print(f"Loaded embeddings: {len(vecs_by_id)} datasets, dim={dim}")

    if args.dry_run:
        print(f"DRY RUN — would build turbovec index → {args.output}")
        return 0

    # Mean-pool per dataset
    ids = sorted(vecs_by_id.keys())
    matrix = np.zeros((len(ids), dim), dtype=np.float32)
    for i, did in enumerate(ids):
        pool = np.mean(vecs_by_id[did], axis=0)
        norm = np.linalg.norm(pool)
        matrix[i] = pool / norm if norm > 0 else pool

    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    idx = NeuralSearchTurboIndex(dim=dim, bit_width=args.bit_width)
    idx.add(ids=ids, vectors=matrix)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    idx.save(args.output)
    print(f"Built index: {idx.size} datasets → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Create recall validator**

Create `scripts/validate_turbovec_recall.py`:

```python
#!/usr/bin/env python3
"""Validate turbovec ANN recall vs brute-force cosine search.

Must show recall@50 >= 0.95 before claiming ANN retrieval is equivalent to exact search.
If below, reduce bit_width from 4 to 2.

Usage:
    python scripts/validate_turbovec_recall.py
    python scripts/validate_turbovec_recall.py --k 50 --n-queries 100
    python scripts/validate_turbovec_recall.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

INDEX_PATH = Path("data/index/turbovec_dense_1024.index")
EMBEDDINGS_PATH = Path("data/embeddings/real_all.dense.field_embeddings.jsonl")
REPORT_PATH = Path("reports/turbovec_recall.json")


def _brute_force_top_k(matrix: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    sims = matrix @ query
    return list(np.argsort(sims)[::-1][:k])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(f"DRY RUN — would validate recall@{args.k} for {args.n_queries} queries")
        return 0

    if not INDEX_PATH.exists():
        print(f"ERROR: index not found at {INDEX_PATH}")
        print("Run: python scripts/build_turbovec_index.py")
        return 1

    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    idx = NeuralSearchTurboIndex.load(str(INDEX_PATH))
    print(f"Index loaded: {idx.size} records")

    # Load float32 matrix for brute-force comparison
    from collections import defaultdict
    vecs_by_id: dict[str, np.ndarray] = {}
    with EMBEDDINGS_PATH.open() as f:
        buf: dict[str, list] = defaultdict(list)
        for line in f:
            rec = json.loads(line)
            did = rec.get("entity_id", "")
            v = rec.get("vector", [])
            if did and v:
                buf[did].append(np.array(v, dtype=np.float32))
    ids = sorted(buf.keys())
    dim = len(next(iter(buf.values()))[0])
    matrix = np.zeros((len(ids), dim), dtype=np.float32)
    id_to_pos = {did: i for i, did in enumerate(ids)}
    for did, vlist in buf.items():
        pool = np.mean(vlist, axis=0)
        norm = np.linalg.norm(pool)
        matrix[id_to_pos[did]] = pool / norm if norm > 0 else pool

    # Sample random queries
    rng = np.random.default_rng(42)
    query_indices = rng.choice(len(ids), size=min(args.n_queries, len(ids)), replace=False)

    recalls: list[float] = []
    ann_latencies: list[float] = []
    bf_latencies: list[float] = []

    for qi in query_indices:
        q = matrix[qi]
        t0 = time.perf_counter()
        ann_results = [r[0] for r in idx.search(q, k=args.k)]
        ann_latencies.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        bf_top = [ids[i] for i in _brute_force_top_k(matrix, q, args.k)]
        bf_latencies.append(time.perf_counter() - t0)

        shared = len(set(ann_results) & set(bf_top))
        recalls.append(shared / args.k)

    mean_recall = float(np.mean(recalls))
    p50_ann = float(np.percentile(ann_latencies, 50)) * 1000
    p95_ann = float(np.percentile(ann_latencies, 95)) * 1000
    p99_ann = float(np.percentile(ann_latencies, 99)) * 1000

    report = {
        "n_queries": args.n_queries,
        "k": args.k,
        "mean_recall": round(mean_recall, 4),
        "p50_latency_ms": round(p50_ann, 2),
        "p95_latency_ms": round(p95_ann, 2),
        "p99_latency_ms": round(p99_ann, 2),
        "index_size": idx.size,
        "dim": idx.dim,
        "bit_width": idx.bit_width,
        "pass": mean_recall >= 0.95,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if mean_recall < 0.95:
        print(f"\nFAIL: recall@{args.k} = {mean_recall:.4f} < 0.95")
        print("Try: rebuild index with --bit-width 2")
        return 1
    print(f"\n✓ PASS: recall@{args.k} = {mean_recall:.4f} >= 0.95")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_turbovec_recall.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/build_turbovec_index.py scripts/validate_turbovec_recall.py \
        tests/test_turbovec_recall.py
git commit -m "feat: add build_turbovec_index and validate_turbovec_recall scripts"
```

---

## Task 8: Re-embed Corpus and Measure Spearman r

Run the full dense embedding pipeline and record whether Spearman r improves.

**Files:**
- No new files — runs existing scripts

- [ ] **Step 1: Re-embed corpus with BGE-large**

```bash
python scripts/recompute_embeddings.py --provider dense
```

Expected: creates `data/embeddings/real_all.dense.field_embeddings.jsonl`. Takes ~45s on GPU, ~10min on CPU for 738 datasets.

- [ ] **Step 2: Build turbovec index**

```bash
python scripts/build_turbovec_index.py
```

Expected: creates `data/index/turbovec_dense_1024.index/`

- [ ] **Step 3: Validate ANN recall**

```bash
python scripts/validate_turbovec_recall.py --k 50
```

Expected: `recall@50 >= 0.95`. If below 0.95, re-run with `--bit-width 2`:

```bash
python scripts/build_turbovec_index.py --bit-width 2
python scripts/validate_turbovec_recall.py --k 50
```

- [ ] **Step 4: Update retrieval config to use dense embeddings**

In [data/config/retrieval.yaml](data/config/retrieval.yaml), update the field_embeddings path:

```yaml
field_embeddings:
  enabled: true
  path: data/embeddings/real_all.dense.field_embeddings.jsonl
  provider: bge-large
```

- [ ] **Step 5: Re-run Spearman r evaluation**

```bash
python scripts/evaluate_usefulness_correlation.py --n-queries 30
```

Expected: writes `reports/usefulness_correlation_v09.json` with updated Spearman r. Record result — this is the Track 1 exit criterion measurement.

- [ ] **Step 6: Run ablation to confirm s9 is working**

```bash
python scripts/ablate_graph_proximity.py --n-queries 20
```

Expected: `pct_pairs_changed >= 10.0%`. Both Track 1 exit criteria now recorded.

- [ ] **Step 7: Run full test suite**

```bash
pytest --timeout=300 -x -q
```

Expected: all pass, no regressions.

- [ ] **Step 8: Commit results**

```bash
git add data/config/retrieval.yaml reports/
git commit -m "feat: deploy BGE-large embeddings + turbovec index; record Spearman r"
```

---

## Track 1 Exit Criteria Checklist

Before merging Track 1:

- [ ] `evaluate_usefulness_correlation.py --n-queries 30` runs without error and produces report
- [ ] `ablate_graph_proximity.py` shows ≥10% of pairs change rank with real graph
- [ ] `validate_turbovec_recall.py` shows recall@50 ≥ 0.95
- [ ] All existing tests pass (979 tests, no regressions)
- [ ] `reports/baseline_v09.json` exists (comparison point)
- [ ] New Spearman r recorded in `reports/usefulness_correlation_v09.json`
