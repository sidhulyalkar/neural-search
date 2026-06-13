#!/usr/bin/env python3
"""Compute BGE-large dense embeddings for flat-format corpus records.

Reads combined_corpus.jsonl (flat legacy dict format from ingestion adapters),
generates per-field BGE-large embeddings, and rebuilds the turbovec index.

Usage:
    python scripts/embed_flat_corpus.py
    python scripts/embed_flat_corpus.py --corpus data/corpus/normalized/combined_corpus.jsonl
    python scripts/embed_flat_corpus.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl")
EMBEDDINGS_OUT = Path("data/embeddings/real_all.dense.field_embeddings.jsonl")
INDEX_PATH = Path("data/index/turbovec_dense_1024.index")

EMBED_FIELDS = [
    "title",
    "description",
    "modalities",
    "brain_regions",
    "tasks",
    "behaviors",
    "data_standards",
    "species",
]


def _field_texts(rec: dict) -> dict[str, str]:
    texts: dict[str, str] = {}
    for field in EMBED_FIELDS:
        val = rec.get(field)
        if isinstance(val, list):
            text = " ".join(str(v) for v in val if v)
        else:
            text = str(val or "")
        text = text.strip()
        if text:
            texts[field] = text

    combined = " ".join(texts.values())
    if combined.strip():
        texts["combined_scientific_summary"] = combined
    return texts


def _record_id(rec: dict) -> str:
    return f"dataset:{rec['source']}:{rec['source_id']}"


def embed_corpus(corpus_path: Path, out_path: Path, dry_run: bool = False) -> int:
    records = []
    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info("Loaded %d records from %s", len(records), corpus_path)

    from neural_search.embeddings.dense_provider import DenseEmbeddingProvider
    provider = DenseEmbeddingProvider()
    logger.info("Provider: %s dim=%d", provider.model_name, provider.dimension)

    # Build (record_id, field_name, text) triples
    triples: list[tuple[str, str, str]] = []
    for rec in records:
        rid = _record_id(rec)
        for field, text in _field_texts(rec).items():
            triples.append((rid, field, text))

    logger.info("Embedding %d field texts across %d records", len(triples), len(records))

    if dry_run:
        logger.info("[dry-run] skipping embed+write")
        return 0

    texts = [t for _, _, t in triples]
    vectors = provider.embed_batch(texts)
    timestamp = datetime.now(timezone.utc).isoformat()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for (rid, field, text), vec in zip(triples, vectors):
            f.write(json.dumps({
                "record_id": rid,
                "record_type": "dataset",
                "field_name": field,
                "text": text,
                "embedding": vec,
                "provider_name": provider.provider_name,
                "model_name": provider.model_name,
                "dimension": provider.dimension,
                "normalize": provider.normalize,
                "created_at": timestamp,
            }) + "\n")

    logger.info("Wrote %d field embeddings → %s", len(triples), out_path)
    return len(records)


def build_index(embed_path: Path, index_path: Path, bit_width: int = 4) -> int:
    from collections import defaultdict

    vecs_by_id: dict[str, list[np.ndarray]] = defaultdict(list)
    with embed_path.open() as f:
        for line in f:
            rec = json.loads(line)
            did = rec.get("record_id", "")
            vec = rec.get("embedding", [])
            if did and vec:
                vecs_by_id[did].append(np.array(vec, dtype=np.float32))

    if not vecs_by_id:
        logger.error("No embeddings found in %s", embed_path)
        return 0

    dim = len(next(iter(vecs_by_id.values()))[0])
    ids = sorted(vecs_by_id.keys())
    matrix = np.zeros((len(ids), dim), dtype=np.float32)
    for i, did in enumerate(ids):
        pool = np.mean(vecs_by_id[did], axis=0)
        norm = np.linalg.norm(pool)
        matrix[i] = pool / norm if norm > 0 else pool

    from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
    idx = NeuralSearchTurboIndex(dim=dim, bit_width=bit_width)
    idx.add(ids=ids, vectors=matrix)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    idx.save(str(index_path))
    logger.info("Built index: %d datasets → %s", idx.size, index_path)
    return idx.size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(CORPUS_PATH))
    parser.add_argument("--output", default=str(EMBEDDINGS_OUT))
    parser.add_argument("--index", default=str(INDEX_PATH))
    parser.add_argument("--bit-width", type=int, default=4, choices=[2, 4])
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip embedding; just rebuild index from existing embeddings file")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_embed:
        n = embed_corpus(Path(args.corpus), Path(args.output), dry_run=args.dry_run)
        if not args.dry_run and n == 0:
            return 1

    if not args.dry_run:
        n_idx = build_index(Path(args.output), Path(args.index), bit_width=args.bit_width)
        logger.info("Index rebuild complete: %d vectors", n_idx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
