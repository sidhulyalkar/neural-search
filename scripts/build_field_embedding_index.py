#!/usr/bin/env python3
"""Build a FAISS field-embedding index from the dense JSONL cache.

Reads ``data/embeddings/real_all.dense.field_embeddings.jsonl`` (or any
JSONL produced by ``write_field_embedding_cache``) and writes two sidecar
files next to the source:

    <stem>.faiss       — FAISS IndexFlatIP storing L2-normalized float32 vectors
    <stem>.meta.jsonl  — companion metadata, one JSON object per row (same order)

The companion metadata contains every FieldEmbeddingRecord field *except*
``embedding`` and ``dimension`` (those come from the FAISS index at load time).

Usage::

    python scripts/build_field_embedding_index.py
    python scripts/build_field_embedding_index.py \\
        --input  data/embeddings/real_all.dense.field_embeddings.jsonl \\
        --output data/embeddings/real_all.dense.field_embeddings.faiss
    python scripts/build_field_embedding_index.py --dry-run

The resulting ``.faiss`` / ``.meta.jsonl`` pair is consumed by
:func:`neural_search.embeddings.field_index.read_field_embedding_cache_faiss`,
which is called automatically by ``load_field_semantic_index`` when both
sidecar files are present alongside the JSONL path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EMBEDDINGS_PATH = Path("data/embeddings/real_all.dense.field_embeddings.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(EMBEDDINGS_PATH),
        help="Source JSONL field-embedding cache (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Destination .faiss path.  Defaults to <input-stem>.faiss "
            "in the same directory as --input."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without writing output files.",
    )
    args = parser.parse_args(argv)

    embed_path = Path(args.input)
    if not embed_path.exists():
        if args.dry_run:
            print(f"DRY RUN — source not found: {embed_path} (skipping)")
            return 0
        print(f"ERROR: embeddings file not found: {embed_path}")
        print("Run: python scripts/recompute_embeddings.py --provider dense")
        return 1

    faiss_path = Path(args.output) if args.output else embed_path.with_suffix(".faiss")
    meta_path = faiss_path.parent / (faiss_path.stem + ".meta.jsonl")

    try:
        import faiss  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        print("ERROR: faiss-cpu is not installed.  Run: pip install faiss-cpu")
        return 1

    # --- Pass 1: stream JSONL to collect vectors and metadata ---
    vectors: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []
    dim: int | None = None

    print(f"Reading embeddings from {embed_path} ...")
    with embed_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vec = rec.get("embedding") or rec.get("vector", [])
            if not vec:
                print(f"  WARNING: empty embedding at line {lineno}, skipping")
                continue

            arr = np.array(vec, dtype=np.float32)
            if dim is None:
                dim = len(arr)
            elif len(arr) != dim:
                print(
                    f"  WARNING: dimension mismatch at line {lineno} "
                    f"(expected {dim}, got {len(arr)}), skipping"
                )
                continue

            # L2-normalize so inner-product == cosine similarity
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm

            vectors.append(arr)
            meta_rows.append(
                {
                    "record_id": rec.get("record_id", ""),
                    "record_type": rec.get("record_type", "dataset"),
                    "field_name": rec.get("field_name", ""),
                    "text": rec.get("text", ""),
                    "provider_name": rec.get("provider_name", ""),
                    "model_name": rec.get("model_name", ""),
                    "normalize": rec.get("normalize", True),
                    "created_at": rec.get("created_at", ""),
                }
            )

            if lineno % 100_000 == 0:
                print(f"  ... processed {lineno:,} lines")

    if not vectors:
        print("ERROR: no valid embeddings found in file")
        return 1

    assert dim is not None
    n = len(vectors)
    print(f"Loaded {n:,} field embedding records (dim={dim})")

    if args.dry_run:
        print("DRY RUN — would write:")
        print(f"  {faiss_path}  ({n} vectors × {dim} floats)")
        print(f"  {meta_path}   ({n} metadata rows)")
        return 0

    # --- Build FAISS IndexFlatIP ---
    matrix = np.stack(vectors, axis=0)  # (n, dim) float32
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)  # type: ignore[arg-type]

    # --- Write output ---
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    print(f"Wrote FAISS index -> {faiss_path}  ({faiss_path.stat().st_size / 1e6:.1f} MB)")

    with meta_path.open("w", encoding="utf-8") as fh:
        for row in meta_rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    print(f"Wrote metadata   -> {meta_path}  ({meta_path.stat().st_size / 1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
