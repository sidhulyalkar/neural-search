#!/usr/bin/env python3
"""Build NeuralSearchTurboIndex from dense field embeddings.

Reads data/embeddings/real_all.dense.field_embeddings.jsonl,
aggregates per-dataset vectors by mean-pooling, L2-normalizes,
and saves a turbovec index.

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
        print(f"DRY RUN -- would build turbovec index -> {args.output}" if args.dry_run
              else f"ERROR: embeddings file not found: {embed_path}")
        if not args.dry_run:
            print("Run: python scripts/recompute_embeddings.py --provider dense")
            return 1
        return 0

    # Load embeddings and aggregate by dataset_id (mean pool per dataset)
    vecs_by_id: dict[str, list[np.ndarray]] = defaultdict(list)
    with embed_path.open() as f:
        for line in f:
            rec = json.loads(line)
            did = rec.get("record_id", "") or rec.get("entity_id", "")
            vec = rec.get("embedding", []) or rec.get("vector", [])
            if did and vec:
                vecs_by_id[did].append(np.array(vec, dtype=np.float32))

    if not vecs_by_id:
        print("ERROR: no embeddings found in file")
        return 1

    dim = len(next(iter(vecs_by_id.values()))[0])
    print(f"Loaded embeddings: {len(vecs_by_id)} datasets, dim={dim}")

    if args.dry_run:
        print(f"DRY RUN -- would build turbovec index ({len(vecs_by_id)} datasets) -> {args.output}")
        return 0

    # Mean-pool then L2-normalize per dataset
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
    print(f"Built index: {idx.size} datasets -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
