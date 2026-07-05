#!/usr/bin/env python3
"""Validate turbovec ANN recall vs brute-force cosine search.

Recall@50 must be >= 0.95 before claiming ANN is equivalent to exact search.
If below, rebuild the index with --bit-width 2.

Usage:
    python scripts/validate_turbovec_recall.py
    python scripts/validate_turbovec_recall.py --k 50 --n-queries 100
    python scripts/validate_turbovec_recall.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
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
        print(f"DRY RUN -- would validate recall@{args.k} for {args.n_queries} queries")
        return 0

    if not INDEX_PATH.exists():
        print(f"ERROR: index not found at {INDEX_PATH}")
        print("Run: python scripts/build_turbovec_index.py")
        return 1

    if not EMBEDDINGS_PATH.exists():
        print(f"ERROR: embeddings not found at {EMBEDDINGS_PATH}")
        print("Run: python scripts/recompute_embeddings.py --provider dense")
        return 1

    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    idx = NeuralSearchTurboIndex.load(str(INDEX_PATH))
    print(f"Index loaded: {idx.size} records")

    # Build float32 matrix for brute-force comparison
    buf: dict[str, list] = defaultdict(list)
    with EMBEDDINGS_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            did = rec.get("record_id", "") or rec.get("entity_id", "")
            v = rec.get("embedding", []) or rec.get("vector", [])
            if did and v:
                buf[did].append(np.array(v, dtype=np.float32))

    if not buf:
        print("ERROR: no valid embeddings found in file")
        return 1

    ids = sorted(buf.keys())
    dim = len(next(iter(buf.values()))[0])
    matrix = np.zeros((len(ids), dim), dtype=np.float32)
    id_to_pos = {did: i for i, did in enumerate(ids)}
    for did, vlist in buf.items():
        pool = np.mean(vlist, axis=0)
        norm = np.linalg.norm(pool)
        matrix[id_to_pos[did]] = pool / norm if norm > 0 else pool

    rng = np.random.default_rng(42)
    query_indices = rng.choice(len(ids), size=min(args.n_queries, len(ids)), replace=False)

    recalls: list[float] = []
    ann_latencies: list[float] = []

    for qi in query_indices:
        q = matrix[qi]
        t0 = time.perf_counter()
        ann_results = [r[0] for r in idx.search(q, k=args.k)]
        ann_latencies.append(time.perf_counter() - t0)

        bf_top = [ids[i] for i in _brute_force_top_k(matrix, q, args.k)]
        shared = len(set(ann_results) & set(bf_top))
        recalls.append(shared / args.k)

    if not recalls:
        print("ERROR: no queries evaluated -- check index and embeddings file")
        return 1

    mean_recall = float(np.mean(recalls))
    p50_ms = float(np.percentile(ann_latencies, 50)) * 1000
    p95_ms = float(np.percentile(ann_latencies, 95)) * 1000

    report = {
        "n_queries": len(query_indices),
        "k": args.k,
        "mean_recall": round(mean_recall, 4),
        "p50_latency_ms": round(p50_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
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
        print("Try: python scripts/build_turbovec_index.py --bit-width 2")
        return 1
    print(f"\nPASS: recall@{args.k} = {mean_recall:.4f} >= 0.95")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
