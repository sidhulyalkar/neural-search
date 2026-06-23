# IR-Evaluation Rigor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the completed LLM qrels (13,654 pairs, 317 queries) into a reproducible, CI-bounded evaluation of the retrieval ladder, deriving the eval corpus from the evidence packets so corpus and qrels are guaranteed aligned.

**Architecture:** Build the measurement apparatus around retrieval without touching retrieval algorithms. Derive a BM25/dense-ready corpus from `artifacts/ablation_judge/evidence_packets.jsonl` (the same source the qrels came from). Run the existing additive ablation ladder, compute metrics with bootstrap CIs, stratify by intent, validate the LLM judge against a second LLM, and gate it all behind `make eval`.

**Tech Stack:** Python 3.11 (conda env `neural-search`, interpreter `/opt/miniconda3/envs/neural-search/bin/python`), `sentence-transformers` + `torch` (BGE-large-en-v1.5), existing scripts under `scripts/eval/`, pytest.

**Reference spec:** `docs/superpowers/specs/2026-06-23-ir-evaluation-rigor-design.md`

---

## Environment note (applies to every command)

The eval scripts contain a stale hardcoded interpreter path (`/home/sid21/anaconda3`). Ignore it. All commands in this plan use:

```bash
export NSPY=/opt/miniconda3/envs/neural-search/bin/python
```

Run `export NSPY=/opt/miniconda3/envs/neural-search/bin/python` once per shell before the tasks below.

---

## Key facts the worker must know

- The qrels live at `data/qrels/qrels.trec` (TREC: `query_id 0 dataset_id grade`) and `data/qrels/llm_judgments.jsonl` (one NeuroJudgment JSON per line, has `query_id`, `dataset_id`, `label` 0–3, and `rationale_short`; error rows contain `"judge_error"` in `rationale_short`).
- `data/corpus/normalized/`, `data/embeddings/`, `data/graph/` are **gitignored and absent** on this machine. Do not assume any file under them exists.
- `artifacts/ablation_judge/evidence_packets.jsonl` (13,654 lines) IS present. Each line has keys: `dataset_id`, `title`, `description`, `dataset_modalities`, `dataset_species`, `dataset_tasks`, `dataset_brain_regions`, `data_standards`, `source_archive`, `query_id`, `query_intent`, `query_text`, plus others. Multiple lines share a `dataset_id` (one per query it was judged against).
- The BM25 `SparseIndex` (`neural_search/search/sparse.py`) indexes these record fields: `title`, `description`, `tasks`, `modalities`, `species`, `brain_regions`, `behaviors`, `analysis_goals`, `data_standards`, `source_id`. Document IDs come from `dataset_id`/`id`/`source_id`.
- The ablation ladder (`scripts/eval/run_ablation_ladder.py`) reads `--corpus` (JSONL of record dicts), `--embeddings` (dense field embeddings JSONL), `--graph` (JSON), `--queries` (YAML), writes one run file per rung to `reports/eval/runs/<rung>.jsonl` with fields `query_id, record_id, rank, score, rung`.
- Graph rungs (`hybrid_graph`, `full`) need `data/graph/...` which is absent. **This plan runs rungs 1–4** (`bm25`, `bm25_structured`, `dense_bge`, `hybrid_rrf`) — covering the structured/dense/hybrid claims — and defers graph rungs to a follow-up (graph rebuild is out of scope here; recorded as a caveat in Task 9).

---

## File structure (created/modified)

- Create: `scripts/eval/build_corpus_from_packets.py` — derive aligned eval corpus from evidence packets.
- Create: `scripts/eval/build_canonical_qrels.py` — one qrels source → `.trec` + JSONL, error rows dropped.
- Create: `scripts/eval/stratify_metrics_by_intent.py` — per-intent NDCG/MRR breakdown.
- Create: `scripts/eval/run_dual_judge_consensus.py` — re-judge a sample with a 2nd LLM, build judge-B JSONL for agreement.
- Create: `tests/eval/test_build_corpus_from_packets.py`, `tests/eval/test_build_canonical_qrels.py`, `tests/eval/test_stratify_metrics_by_intent.py`.
- Modify: `docs/CLAIM_LEDGER.md` — update core-retrieval rows with CI'd numbers.
- Modify: `Makefile` — add `eval` target + regression gate.

---

## Task 1: Derive the eval corpus from evidence packets

**Files:**
- Create: `scripts/eval/build_corpus_from_packets.py`
- Test: `tests/eval/test_build_corpus_from_packets.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_build_corpus_from_packets.py
import json
from pathlib import Path

from scripts.eval.build_corpus_from_packets import build_corpus_records


def test_dedup_and_field_mapping():
    packets = [
        {"dataset_id": "dandi:000003", "title": "Hippocampus ephys",
         "description": "theta", "dataset_modalities": ["extracellular_ephys"],
         "dataset_species": ["mouse"], "dataset_tasks": ["spatial_navigation"],
         "dataset_brain_regions": ["hippocampus"], "data_standards": ["nwb"],
         "source_archive": "dandi", "query_id": "q_1"},
        # same dataset judged against a different query -> must dedup to one record
        {"dataset_id": "dandi:000003", "title": "Hippocampus ephys",
         "description": "theta", "dataset_modalities": ["extracellular_ephys"],
         "dataset_species": ["mouse"], "dataset_tasks": ["spatial_navigation"],
         "dataset_brain_regions": ["hippocampus"], "data_standards": ["nwb"],
         "source_archive": "dandi", "query_id": "q_2"},
    ]
    records = build_corpus_records(packets)
    assert len(records) == 1
    r = records[0]
    assert r["dataset_id"] == "dandi:000003"
    assert r["source"] == "dandi"
    assert r["source_id"] == "000003"
    assert r["modalities"] == ["extracellular_ephys"]
    assert r["species"] == ["mouse"]
    assert r["tasks"] == ["spatial_navigation"]
    assert r["brain_regions"] == ["hippocampus"]
    assert r["data_standards"] == ["nwb"]
    assert r["title"] == "Hippocampus ephys"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export NSPY=/opt/miniconda3/envs/neural-search/bin/python; PYTHONPATH=. $NSPY -m pytest tests/eval/test_build_corpus_from_packets.py -v`
(If pytest is missing: `$NSPY -m pip install pytest` first.)
Expected: FAIL with `ModuleNotFoundError: scripts.eval.build_corpus_from_packets`.

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Derive a BM25/dense-ready eval corpus from evidence packets.

The evidence packets are the same source the qrels were built from, so a corpus
derived from them is guaranteed to cover exactly the judged datasets. One record
per unique dataset_id; fields renamed to what SparseIndex expects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_PACKETS = Path("artifacts/ablation_judge/evidence_packets.jsonl")
DEFAULT_OUT = Path("data/eval/ablation_corpus_from_packets.jsonl")


def build_corpus_records(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for p in packets:
        did = str(p.get("dataset_id", ""))
        if not did or did in by_id:
            continue
        source, _, source_id = did.partition(":")
        by_id[did] = {
            "dataset_id": did,
            "source": p.get("source_archive") or source,
            "source_id": source_id or did,
            "title": p.get("title", ""),
            "description": p.get("description", ""),
            "modalities": list(p.get("dataset_modalities", []) or []),
            "species": list(p.get("dataset_species", []) or []),
            "tasks": list(p.get("dataset_tasks", []) or []),
            "brain_regions": list(p.get("dataset_brain_regions", []) or []),
            "data_standards": list(p.get("data_standards", []) or []),
        }
    return list(by_id.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packets", type=Path, default=DEFAULT_PACKETS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    packets = [json.loads(l) for l in args.packets.read_text(encoding="utf-8").splitlines() if l.strip()]
    records = build_corpus_records(packets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} unique dataset records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Also create empty `tests/eval/__init__.py` if `tests/eval/` does not exist as a package (check with `ls tests/eval/`); if `tests/` uses bare test files without `__init__.py`, skip this.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. $NSPY -m pytest tests/eval/test_build_corpus_from_packets.py -v`
Expected: PASS.

- [ ] **Step 5: Generate the real corpus and sanity-check count**

Run:
```bash
PYTHONPATH=. $NSPY scripts/eval/build_corpus_from_packets.py
wc -l data/eval/ablation_corpus_from_packets.jsonl
```
Expected: a few thousand unique records (fewer than 13,654 packets, since datasets repeat across queries). Note the number.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval/build_corpus_from_packets.py tests/eval/test_build_corpus_from_packets.py
git commit -m "feat(eval): derive aligned ablation corpus from evidence packets"
```

---

## Task 2: Build the canonical qrels (one source → .trec + JSONL)

**Files:**
- Create: `scripts/eval/build_canonical_qrels.py`
- Test: `tests/eval/test_build_canonical_qrels.py`

The NDCG script reads `.trec`; the bootstrap-CI script reads JSONL with a `label`/`relevance` key. This task emits both from `llm_judgments.jsonl`, dropping `judge_error` rows and deduping.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_build_canonical_qrels.py
from scripts.eval.build_canonical_qrels import canonicalize


def test_drops_errors_dedups_and_emits_both_forms():
    judgments = [
        {"query_id": "q_1", "dataset_id": "dandi:1", "label": 3, "rationale_short": "great"},
        {"query_id": "q_1", "dataset_id": "dandi:2", "label": 0,
         "rationale_short": "judge_error: all_retries_failed"},   # dropped
        {"query_id": "q_1", "dataset_id": "dandi:1", "label": 2,  # dup -> first kept
         "rationale_short": "ok"},
    ]
    trec_lines, jsonl_rows = canonicalize(judgments)
    assert trec_lines == ["q_1 0 dandi:1 3"]
    assert jsonl_rows == [{"query_id": "q_1", "dataset_id": "dandi:1", "label": 3}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. $NSPY -m pytest tests/eval/test_build_canonical_qrels.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Emit canonical qrels in both TREC and JSONL form from llm_judgments.jsonl.

Drops judge_error rows and deduplicates (query_id, dataset_id) keeping first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_JUDGMENTS = Path("data/qrels/llm_judgments.jsonl")
DEFAULT_TREC = Path("data/qrels/qrels.canonical.trec")
DEFAULT_JSONL = Path("data/qrels/qrels.canonical.jsonl")


def canonicalize(judgments: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    seen: set[tuple[str, str]] = set()
    trec: list[str] = []
    rows: list[dict[str, Any]] = []
    for rec in judgments:
        if "judge_error" in str(rec.get("rationale_short", "")):
            continue
        qid, did = str(rec["query_id"]), str(rec["dataset_id"])
        if (qid, did) in seen:
            continue
        label = int(rec.get("label", -1))
        if label < 0:
            continue
        seen.add((qid, did))
        trec.append(f"{qid} 0 {did} {label}")
        rows.append({"query_id": qid, "dataset_id": did, "label": label})
    return trec, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    ap.add_argument("--trec", type=Path, default=DEFAULT_TREC)
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    args = ap.parse_args(argv)

    judgments = [json.loads(l) for l in args.judgments.read_text(encoding="utf-8").splitlines() if l.strip()]
    trec, rows = canonicalize(judgments)
    args.trec.parent.mkdir(parents=True, exist_ok=True)
    args.trec.write_text("\n".join(trec) + "\n", encoding="utf-8")
    with args.jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(trec)} qrels -> {args.trec} and {args.jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. $NSPY -m pytest tests/eval/test_build_canonical_qrels.py -v`
Expected: PASS.

- [ ] **Step 5: Generate real canonical qrels and verify counts match clean judgments**

Run:
```bash
PYTHONPATH=. $NSPY scripts/eval/build_canonical_qrels.py
wc -l data/qrels/qrels.canonical.trec data/qrels/qrels.canonical.jsonl
```
Expected: both files have the same line count (~13,640), a touch below the 13,654 raw qrels after dropping error rows.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval/build_canonical_qrels.py tests/eval/test_build_canonical_qrels.py
git commit -m "feat(eval): canonical qrels emitter (TREC + JSONL, errors dropped)"
```

---

## Task 3: Install dense deps and generate BGE field embeddings

**Files:** none created; produces `data/embeddings/real_all.dense.field_embeddings.jsonl`.

> NOTE: `recompute_embeddings.py` collects corpus from the gitignored `data/corpus/normalized/` dir, which is empty here. We instead point the embedding build at the packet-derived corpus from Task 1 by passing it through the same DenseEmbeddingProvider. Because `recompute_embeddings.py` has no `--corpus` flag, this task uses a thin inline driver.

- [ ] **Step 1: Install sentence-transformers + torch**

Run:
```bash
$NSPY -m pip install "sentence-transformers>=2.2" torch
$NSPY -c "import sentence_transformers, torch; print('OK', sentence_transformers.__version__, torch.__version__)"
```
Expected: `OK <ver> <ver>` with no ImportError.

- [ ] **Step 2: Write a one-shot embedding driver and run it**

Create `scripts/eval/embed_packet_corpus.py`:

```python
#!/usr/bin/env python3
"""Embed the packet-derived corpus with BGE-large into dense field embeddings.

Output format matches what run_ablation_ladder.load_field_embeddings expects:
one JSON object per line with keys: dataset_id, field_name, vector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_search.embeddings.dense_provider import DenseEmbeddingProvider

DEFAULT_CORPUS = Path("data/eval/ablation_corpus_from_packets.jsonl")
DEFAULT_OUT = Path("data/embeddings/real_all.dense.field_embeddings.jsonl")
FIELDS = ("title", "description")  # text fields worth dense-encoding


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    provider = DenseEmbeddingProvider()
    records = [json.loads(l) for l in args.corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for rec in records:
            did = rec["dataset_id"]
            for field in FIELDS:
                text = str(rec.get(field, "")).strip()
                if not text:
                    continue
                vec = provider.embed_text(text)
                vec = list(vec) if not isinstance(vec, list) else vec
                f.write(json.dumps({"dataset_id": did, "field_name": field, "vector": vec}) + "\n")
                n += 1
    print(f"Wrote {n} field-embedding rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> VERIFY BEFORE RUNNING: confirm the DenseEmbeddingProvider method name. Run
> `$NSPY -c "from neural_search.embeddings.dense_provider import DenseEmbeddingProvider as D; print([m for m in dir(D) if 'embed' in m or 'encode' in m])"`
> and if the method is not `embed_text`, replace `provider.embed_text(text)` with the
> correct single-string method (e.g. `provider.embed([text])[0]`). Also cross-check the
> field-embedding row schema against `load_field_embeddings()` in
> `scripts/eval/run_ablation_ladder.py:207` (it reads `field_name` and a vector key —
> match the exact vector key name it expects, `vector` vs `embedding`).

Run:
```bash
PYTHONPATH=. $NSPY scripts/eval/embed_packet_corpus.py
wc -l data/embeddings/real_all.dense.field_embeddings.jsonl
```
Expected: ~2× the corpus record count (title + description per dataset). This step is the slow one (BGE-large over a few thousand texts on CPU/MPS — minutes, not seconds).

- [ ] **Step 3: Commit the driver (not the gitignored embeddings)**

```bash
git add scripts/eval/embed_packet_corpus.py
git commit -m "feat(eval): BGE dense field-embedding driver for packet corpus"
```

---

## Task 4: Validate qrels and run the ablation ladder (rungs 1–4)

**Files:** none created; produces `reports/eval/runs/{bm25,bm25_structured,dense_bge,hybrid_rrf}.jsonl`.

- [ ] **Step 1: Validate the canonical qrels**

Run:
```bash
PYTHONPATH=. $NSPY scripts/eval/validate_qrels.py data/qrels/qrels.canonical.jsonl
```
Expected: validation passes (or reports only known, acceptable warnings). If it hard-fails on schema (it expects `QrelsEntryV1`), note the mismatch and proceed — the `.trec`/JSONL forms are already verified in Task 2; record any validator gap in the Task 9 ledger notes rather than blocking.

- [ ] **Step 2: Run the BM25 rungs first (fast, no embeddings) to de-risk corpus wiring**

Run:
```bash
rm -f reports/eval/runs/*.jsonl
PYTHONPATH=. $NSPY scripts/eval/run_ablation_ladder.py \
  --corpus data/eval/ablation_corpus_from_packets.jsonl \
  --skip-rungs dense_bge hybrid_rrf hybrid_graph full
wc -l reports/eval/runs/bm25.jsonl reports/eval/runs/bm25_structured.jsonl
```
Expected: non-empty run files (≥ 1 line per query that retrieved anything). If `bm25.jsonl` is empty, the corpus field mapping is wrong — re-check Task 1 field names against `SparseIndexConfig.field_weights`.

- [ ] **Step 3: Run the dense + hybrid rungs**

Run:
```bash
PYTHONPATH=. $NSPY scripts/eval/run_ablation_ladder.py \
  --corpus data/eval/ablation_corpus_from_packets.jsonl \
  --embeddings data/embeddings/real_all.dense.field_embeddings.jsonl \
  --skip-rungs hybrid_graph full
ls -la reports/eval/runs/
```
Expected: four run files present (`bm25`, `bm25_structured`, `dense_bge`, `hybrid_rrf`), all non-empty.

- [ ] **Step 4: Commit the run files**

```bash
git add -f reports/eval/runs/bm25.jsonl reports/eval/runs/bm25_structured.jsonl \
  reports/eval/runs/dense_bge.jsonl reports/eval/runs/hybrid_rrf.jsonl
git commit -m "eval: ablation ladder run files (rungs 1-4) over packet corpus"
```
(`-f` because `reports/` may be partially gitignored — verify with `git status` first; if reports are tracked normally, drop `-f`.)

---

## Task 5: Compute headline metrics (NDCG / MRR / Recall)

**Files:** produces `reports/eval/ndcg_report.{json,md}`.

- [ ] **Step 1: Run the metrics script against canonical TREC qrels**

Run:
```bash
PYTHONPATH=. $NSPY scripts/eval/compute_ndcg_from_qrels.py \
  --qrels data/qrels/qrels.canonical.trec \
  --runs-dir reports/eval/runs
cat reports/eval/ndcg_report.md
```
Expected: a table with one row per rung, columns NDCG@10 / MRR / Recall@50, monotonic-ish improvement across rungs (not guaranteed — record actuals).

- [ ] **Step 2: Commit the report**

```bash
git add reports/eval/ndcg_report.json reports/eval/ndcg_report.md
git commit -m "eval: NDCG/MRR/Recall report for rungs 1-4"
```

---

## Task 6: Bootstrap confidence intervals + significance

**Files:** produces `reports/eval/bootstrap_ci_report.json`.

- [ ] **Step 1: Run bootstrap CI over the JSONL qrels and run files**

Run:
```bash
PYTHONPATH=. $NSPY scripts/eval/compute_bootstrap_ci.py \
  --qrels data/qrels/qrels.canonical.jsonl \
  --runs reports/eval/runs/bm25.jsonl reports/eval/runs/bm25_structured.jsonl \
         reports/eval/runs/dense_bge.jsonl reports/eval/runs/hybrid_rrf.jsonl \
  --out reports/eval/bootstrap_ci_report.json \
  --n-bootstrap 2000
$NSPY -m json.tool reports/eval/bootstrap_ci_report.json | head -60
```
Expected: per-system NDCG@10 / MRR / P@5 with 95% CIs and pairwise significance. If it complains about silver labels, add `--allow-silver`.

- [ ] **Step 2: Commit**

```bash
git add reports/eval/bootstrap_ci_report.json
git commit -m "eval: bootstrap 95% CIs + pairwise significance for rungs 1-4"
```

---

## Task 7: Per-intent stratification

**Files:**
- Create: `scripts/eval/stratify_metrics_by_intent.py`
- Test: `tests/eval/test_stratify_metrics_by_intent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_stratify_metrics_by_intent.py
from scripts.eval.stratify_metrics_by_intent import group_queries_by_intent


def test_group_queries_by_intent():
    queries = [
        {"query_id": "q_1", "intent": "REPLICATION"},
        {"query_id": "q_2", "intent": "META_ANALYSIS"},
        {"query_id": "q_3", "intent": "REPLICATION"},
    ]
    groups = group_queries_by_intent(queries)
    assert groups["REPLICATION"] == {"q_1", "q_3"}
    assert groups["META_ANALYSIS"] == {"q_2"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. $NSPY -m pytest tests/eval/test_stratify_metrics_by_intent.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Stratify NDCG@10 / MRR by query intent for each rung.

Reuses the metric functions from compute_bootstrap_ci so numbers match.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from scripts.eval.compute_bootstrap_ci import (
    load_qrels, ndcg_at_k, mrr,
)


def group_queries_by_intent(queries: list[dict[str, Any]]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for q in queries:
        intent = str(q.get("intent", "UNKNOWN")).upper()
        qid = str(q.get("query_id") or q.get("id"))
        groups[intent].add(qid)
    return dict(groups)


def _load_run(path: Path) -> dict[str, list[str]]:
    ranked: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ranked[str(r["query_id"])].append((int(r["rank"]), str(r["record_id"])))
    return {qid: [rid for _, rid in sorted(items)] for qid, items in ranked.items()}


def _load_queries(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "queries" in raw:
        return raw["queries"]
    return raw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qrels", type=Path, default=Path("data/qrels/qrels.canonical.jsonl"))
    ap.add_argument("--queries", type=Path, default=Path("data/eval/benchmark_queries_canonical.yaml"))
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/eval/per_intent_report.md"))
    args = ap.parse_args(argv)

    qrels = load_qrels(args.qrels)
    groups = group_queries_by_intent(_load_queries(args.queries))

    lines = ["# Per-Intent Metrics (NDCG@10 / MRR)", ""]
    for run_path in args.runs:
        run = _load_run(run_path)
        lines.append(f"## {run_path.stem}")
        lines.append("| intent | n | NDCG@10 | MRR |")
        lines.append("|--------|---|---------|-----|")
        for intent, qids in sorted(groups.items()):
            present = [q for q in qids if q in run and q in qrels]
            if not present:
                continue
            ndcgs = [ndcg_at_k(run[q], qrels[q], 10) for q in present]
            mrrs = [mrr(run[q], qrels[q]) for q in present]
            lines.append(f"| {intent} | {len(present)} | "
                         f"{sum(ndcgs)/len(ndcgs):.3f} | {sum(mrrs)/len(mrrs):.3f} |")
        lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote per-intent report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> VERIFY: confirm `load_qrels`, `ndcg_at_k`, `mrr` are importable from
> `scripts.eval.compute_bootstrap_ci` (they are defined there at lines ~37–60 and ~151).
> If `compute_bootstrap_ci` isn't import-safe (runs code at import), wrap its CLI in
> `if __name__ == "__main__":` first as a tiny separate commit.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. $NSPY -m pytest tests/eval/test_stratify_metrics_by_intent.py -v`
Expected: PASS.

- [ ] **Step 5: Generate the real per-intent report**

Run:
```bash
PYTHONPATH=. $NSPY scripts/eval/stratify_metrics_by_intent.py \
  --runs reports/eval/runs/bm25.jsonl reports/eval/runs/hybrid_rrf.jsonl
cat reports/eval/per_intent_report.md
```
Expected: per-intent NDCG/MRR for the weakest and strongest rung. Intents with n<30 are under-powered — that caveat goes in Task 9.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval/stratify_metrics_by_intent.py tests/eval/test_stratify_metrics_by_intent.py reports/eval/per_intent_report.md
git commit -m "feat(eval): per-intent NDCG/MRR stratification"
```

---

## Task 8: Dual-judge reliability

**Files:**
- Create: `scripts/eval/run_dual_judge_consensus.py`
- Produces: a judge-B JSONL and `reports/eval/dual_judge_agreement.md`.

The reliability check reuses `audit_neuro_qrels.py`, which computes QWK / within-1 / confusion between `--judgments` (judge A = primary) and `--human` (here: judge B = second LLM). Judge B plays the "human" role.

- [ ] **Step 1: Select a stratified validation sample**

Run:
```bash
PYTHONPATH=. $NSPY scripts/eval/select_neuro_judge_validation_set.py \
  --evidence artifacts/ablation_judge/evidence_packets.jsonl \
  --judgments data/qrels/llm_judgments.jsonl \
  --candidates artifacts/field_state/qrels_candidates_pooled.jsonl \
  --n 250 --seed 42 --require-diversity \
  --out artifacts/eval/judge_validation_sample.jsonl \
  --summary reports/eval/judge_validation_sample_summary.md
wc -l artifacts/eval/judge_validation_sample.jsonl
```
Expected: ~250 evidence-packet lines selected for re-judging.

- [ ] **Step 2: Re-judge the sample with a DIFFERENT model**

`run_dual_judge_consensus.py` is a thin wrapper over the existing parallel judge that (a) points at the sample packets, (b) forces a different model, (c) writes to a separate output. Create it:

```python
#!/usr/bin/env python3
"""Re-judge a validation sample with a second (different) LLM for reliability.

Wraps run_parallel_llm_qrels by overriding the model + output path so the second
judge's labels land in a separate file for judge-vs-judge agreement.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packets", type=Path, default=Path("artifacts/eval/judge_validation_sample.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/qrels/llm_judgments_judgeB.jsonl"))
    ap.add_argument("--model", required=True,
                    help="Second judge model id, DIFFERENT from the primary run "
                         "(e.g. a Groq model distinct from the one used originally)")
    args = ap.parse_args(argv)

    # Force the parallel runner's model + output via env/args it already understands.
    os.environ["GROQ_MODEL"] = args.model
    from scripts.eval.run_parallel_llm_qrels import main as run_main
    return run_main([
        "--packets", str(args.packets),
        "--out", str(args.out),
        "--trec", str(args.out.with_suffix(".trec")),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
```

> VERIFY: confirm which backend env var the original run used (Groq, per the priority
> order in `run_parallel_llm_qrels.py`). If the primary judge was Groq `llama-3.3-70b-versatile`,
> pick a genuinely different model (e.g. another Groq model) so the agreement isn't
> self-agreement. Set the appropriate `*_API_KEY` in `.env.local` if needed.

Run:
```bash
PYTHONPATH=. $NSPY scripts/eval/run_dual_judge_consensus.py --model <different-model-id>
wc -l data/qrels/llm_judgments_judgeB.jsonl
```
Expected: ~250 judge-B judgments.

- [ ] **Step 3: Compute judge-vs-judge agreement (QWK, within-1, confusion)**

Run:
```bash
PYTHONPATH=. $NSPY scripts/eval/audit_neuro_qrels.py \
  --judgments data/qrels/llm_judgments.jsonl \
  --human data/qrels/llm_judgments_judgeB.jsonl \
  --out reports/eval/dual_judge_agreement.md
cat reports/eval/dual_judge_agreement.md
```
Expected: a QWK value, within-1 agreement %, and a confusion matrix over the ~250 overlapping pairs. Record the QWK — it's the reliability bound for the claims.

- [ ] **Step 4: Commit**

```bash
git add scripts/eval/run_dual_judge_consensus.py reports/eval/dual_judge_agreement.md \
  reports/eval/judge_validation_sample_summary.md
git commit -m "feat(eval): dual-judge reliability (QWK over 250-pair sample)"
```

---

## Task 9: Claim ledger update + `make eval` + regression gate

**Files:**
- Modify: `docs/CLAIM_LEDGER.md`
- Modify: `Makefile`

- [ ] **Step 1: Update the CLAIM_LEDGER core-retrieval rows**

In `docs/CLAIM_LEDGER.md`, for each core-retrieval claim now backed by a CI'd number, update the Status and Evidence columns. Use the actual numbers from `reports/eval/ndcg_report.md` and `reports/eval/bootstrap_ci_report.json`. Example edit for the structured-metadata row (replace bracketed values with real outputs):

```markdown
| Structured metadata improves retrieval over keyword search | statistically_validated | 317-query benchmark: bm25_structured vs bm25 ΔNDCG@10 = +[X.X] (95% CI [a, b]), reports/eval/bootstrap_ci_report.json | Single LLM judge (QWK=[κ] vs 2nd judge); neuroscience-only corpus | Add human gold labels; cross-domain corpus |
```

Add a new status definition row near the top:

```markdown
| **statistically_validated** | Measured on the 317-query benchmark with a 95% bootstrap CI that excludes zero |
```

Add a **Reliability & Limitations** note: qrels are LLM-generated; inter-judge QWK = [κ] on a 250-pair stratified sample; graph rungs (hybrid_graph, full) deferred pending graph rebuild; per-intent figures with n<30 are directional only.

- [ ] **Step 2: Add the `make eval` target**

In `Makefile`, add (use TAB indentation, not spaces):

```makefile
NSPY ?= /opt/miniconda3/envs/neural-search/bin/python

.PHONY: eval eval-gate
eval:  ## Reproduce the IR evaluation (rungs 1-4) from cached corpus+embeddings
	PYTHONPATH=. $(NSPY) scripts/eval/build_corpus_from_packets.py
	PYTHONPATH=. $(NSPY) scripts/eval/build_canonical_qrels.py
	PYTHONPATH=. $(NSPY) scripts/eval/run_ablation_ladder.py \
	  --corpus data/eval/ablation_corpus_from_packets.jsonl \
	  --embeddings data/embeddings/real_all.dense.field_embeddings.jsonl \
	  --skip-rungs hybrid_graph full
	PYTHONPATH=. $(NSPY) scripts/eval/compute_ndcg_from_qrels.py \
	  --qrels data/qrels/qrels.canonical.trec --runs-dir reports/eval/runs
	PYTHONPATH=. $(NSPY) scripts/eval/compute_bootstrap_ci.py \
	  --qrels data/qrels/qrels.canonical.jsonl \
	  --runs reports/eval/runs/bm25.jsonl reports/eval/runs/bm25_structured.jsonl \
	         reports/eval/runs/dense_bge.jsonl reports/eval/runs/hybrid_rrf.jsonl \
	  --out reports/eval/bootstrap_ci_report.json --n-bootstrap 2000
```

- [ ] **Step 3: Add the regression gate**

Create `scripts/eval/check_regression.py`:

```python
#!/usr/bin/env python3
"""Fail (exit 1) if a frozen rung's NDCG@10 drops below a floor.

The floor is the lower bound of the rung's 95% bootstrap CI captured at baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", type=Path, default=Path("reports/eval/ndcg_report.json"))
    ap.add_argument("--rung", default="hybrid_rrf")
    ap.add_argument("--floor", type=float, required=True,
                    help="Minimum acceptable NDCG@10 (lower CI bound from baseline)")
    args = ap.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    # report shape: {"<rung>": {"ndcg@10": float, ...}, ...} OR a list of rows.
    rows = report.values() if isinstance(report, dict) else report
    ndcg = None
    for row in rows:
        if isinstance(row, dict) and row.get("rung", row.get("system")) == args.rung:
            ndcg = row.get("ndcg@10") or row.get("ndcg")
            break
        if isinstance(report, dict) and args.rung in report:
            ndcg = report[args.rung].get("ndcg@10")
            break
    if ndcg is None:
        sys.exit(f"[gate] rung {args.rung} not found in {args.report}")
    if ndcg < args.floor:
        sys.exit(f"[gate] FAIL: {args.rung} NDCG@10 {ndcg:.3f} < floor {args.floor:.3f}")
    print(f"[gate] OK: {args.rung} NDCG@10 {ndcg:.3f} >= floor {args.floor:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> VERIFY: open `reports/eval/ndcg_report.json` and confirm the actual key names
> (`ndcg@10` vs `ndcg_at_10`, `rung` vs `system`); adjust the lookup to match before
> committing. Set `--floor` to the `hybrid_rrf` lower CI bound from
> `reports/eval/bootstrap_ci_report.json`.

Add to `Makefile`:

```makefile
eval-gate:  ## Fail if hybrid_rrf NDCG@10 regresses below its baseline CI floor
	PYTHONPATH=. $(NSPY) scripts/eval/check_regression.py --rung hybrid_rrf --floor $(FLOOR)
```

- [ ] **Step 4: Run the gate once to confirm it passes at baseline**

Run:
```bash
PYTHONPATH=. $NSPY scripts/eval/check_regression.py --rung hybrid_rrf --floor 0.0
```
Expected: `[gate] OK ...`. (Floor 0.0 just smoke-tests the wiring; set the real floor in the Makefile.)

- [ ] **Step 5: Commit**

```bash
git add docs/CLAIM_LEDGER.md Makefile scripts/eval/check_regression.py
git commit -m "feat(eval): claim-ledger CI update, make eval target, regression gate"
```

---

## Final verification

- [ ] **Run the full reproducible pipeline from the make target**

Run: `make eval` (after `export FLOOR=<hybrid_rrf lower CI bound>`)
Expected: regenerates corpus → qrels → run files → NDCG report → bootstrap CI with no manual steps.

- [ ] **Confirm success criteria from the spec**

1. `make eval` reproduces all four rungs' metrics — ✅ when Final step passes.
2. Every headline number carries a 95% CI — ✅ `bootstrap_ci_report.json`.
3. Per-intent breakdown exists — ✅ `per_intent_report.md`.
4. Dual-judge QWK reported with sample size — ✅ `dual_judge_agreement.md`.
5. CLAIM_LEDGER core rows cite CI'd numbers — ✅ Task 9.
6. Regression gate wired — ✅ `eval-gate` + `check_regression.py`.

- [ ] **Run the full eval test suite**

Run: `PYTHONPATH=. $NSPY -m pytest tests/eval/ -v`
Expected: all new tests pass.
