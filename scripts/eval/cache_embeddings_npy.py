#!/usr/bin/env python3
"""Pre-compute aggregated per-record embedding matrix as .npy for fast loading.

Reads field_embeddings.jsonl (or FAISS sidecar), aggregates per record with
field weights, and saves two files:
  <stem>.agg.ids.json   – list[str] of record IDs (source:source_id format)
  <stem>.agg.mat.npy    – float32 matrix of shape (N, dim)

Loading with np.load(mmap_mode='r') takes ~10ms vs 20+ seconds for JSONL.

Usage::

    python scripts/eval/cache_embeddings_npy.py
    python scripts/eval/cache_embeddings_npy.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIELD_WEIGHTS = {
    "title": 3.0,
    "description": 2.0,
    "tasks": 1.5,
    "behavioral_events": 1.2,
    "modalities": 1.5,
    "brain_regions": 1.2,
    "analysis_goals": 1.0,
    "combined_scientific_summary": 2.0,
}
DEFAULT_JSONL = ROOT / "data" / "embeddings" / "real_all.dense.field_embeddings.jsonl"


def build_aggregated(jsonl_path: Path) -> tuple[list[str], np.ndarray]:
    """Read field embeddings and aggregate per record. Returns (ids, matrix)."""
    faiss_path = jsonl_path.with_suffix(".faiss")
    meta_path = jsonl_path.with_name(jsonl_path.stem + ".meta.jsonl")

    sums: dict[str, np.ndarray] = {}
    weights: dict[str, float] = {}

    if faiss_path.exists() and meta_path.exists():
        print(f"  Reading FAISS sidecar ({faiss_path.name} + {meta_path.name}) ...")
        try:
            from neural_search.embeddings.field_index import (
                read_field_embedding_cache_faiss,
            )
            records = read_field_embedding_cache_faiss(faiss_path, meta_path)
            for rec in records:
                rid = str(rec.record_id).removeprefix("dataset:")
                field = rec.field_name
                emb = np.array(rec.embedding, dtype=np.float32)
                w = FIELD_WEIGHTS.get(field, 0.5)
                if rid not in sums:
                    sums[rid] = np.zeros(len(emb), dtype=np.float32)
                    weights[rid] = 0.0
                sums[rid] += emb * w
                weights[rid] += w
            print(f"  Loaded {len(records)} field vectors via FAISS")
        except Exception as exc:
            print(f"  FAISS failed ({exc}), falling back to JSONL ...")
            sums.clear()
            weights.clear()

    if not sums:
        print(f"  Reading JSONL ({jsonl_path.name}) ...")
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = str(rec["record_id"]).removeprefix("dataset:")
            field = str(rec.get("field_name", ""))
            emb = np.array(rec["embedding"], dtype=np.float32)
            w = FIELD_WEIGHTS.get(field, 0.5)
            if rid not in sums:
                sums[rid] = np.zeros(len(emb), dtype=np.float32)
                weights[rid] = 0.0
            sums[rid] += emb * w
            weights[rid] += w
        print(f"  Loaded {len(sums)} records via JSONL")

    ids = list(sums.keys())
    dim = next(iter(sums.values())).shape[0]
    mat = np.zeros((len(ids), dim), dtype=np.float32)
    for i, rid in enumerate(ids):
        w = weights[rid]
        if w > 0:
            v = sums[rid] / w
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v = v / norm
            mat[i] = v

    return ids, mat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--force", action="store_true", help="Overwrite existing .npy cache")
    args = ap.parse_args()

    ids_path = args.embeddings.with_name(args.embeddings.stem + ".agg.ids.json")
    mat_path = args.embeddings.with_name(args.embeddings.stem + ".agg.mat.npy")

    if ids_path.exists() and mat_path.exists() and not args.force:
        existing_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        print(f"Cache already exists: {len(existing_ids)} records. Use --force to rebuild.")
        return

    print(f"Building aggregated embedding cache from {args.embeddings.name} ...")
    t0 = time.time()
    ids, mat = build_aggregated(args.embeddings)
    t_build = time.time() - t0
    print(f"  Aggregated {len(ids)} records ({mat.shape[1]}-dim) in {t_build:.1f}s")

    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    np.save(str(mat_path), mat)

    # Verify load time
    t1 = time.time()
    loaded_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    loaded_mat = np.load(str(mat_path), mmap_mode="r")
    t_load = time.time() - t1
    print(f"  Saved: {ids_path.name} ({len(loaded_ids)} IDs) + {mat_path.name} ({loaded_mat.shape})")
    print(f"  Cache load time: {t_load*1000:.1f}ms (vs {t_build:.1f}s to build)")
    print(f"  Speedup: {t_build/max(t_load, 0.001):.0f}x")


if __name__ == "__main__":
    main()
