#!/usr/bin/env python3
"""Embed the packet-derived corpus with BGE-large into dense field embeddings.

Output rows match what scripts/eval/run_ablation_ladder.py::load_field_embeddings
expects: one JSON object per line with keys record_id, field_name, embedding.

Usage:
    PYTHONPATH=. python scripts/eval/embed_packet_corpus.py   # run from repo root
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neural_search.embeddings.dense_provider import DenseEmbeddingProvider

DEFAULT_CORPUS = Path("data/eval/ablation_corpus_from_packets.jsonl")
DEFAULT_OUT = Path("data/embeddings/real_all.dense.field_embeddings.jsonl")
FIELDS = ("title", "description")  # text fields worth dense-encoding


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    if not args.corpus.exists():
        print(f"Corpus file not found: {args.corpus}", file=sys.stderr)
        return 1

    records = [
        json.loads(line)
        for line in args.corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    provider = DenseEmbeddingProvider()
    print(f"Provider {provider.model_name} dim={provider.dimension}; {len(records)} records", flush=True)

    # Build a flat list of (record_id, field_name, text) then batch-encode.
    items: list[tuple[str, str, str]] = []
    for rec in records:
        did = str(rec["dataset_id"])
        for field in FIELDS:
            text = str(rec.get(field, "")).strip()
            if text:
                items.append((did, field, text))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            vectors = provider.embed_batch([t for _, _, t in batch])
            for (did, field, _text), vec in zip(batch, vectors, strict=True):
                f.write(json.dumps({
                    "record_id": did,
                    "field_name": field,
                    "embedding": list(vec),
                }) + "\n")
                n += 1
            print(f"  embedded {min(start + args.batch_size, len(items))}/{len(items)}", flush=True)
    print(f"Wrote {n} field-embedding rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
