#!/usr/bin/env python3
"""Freeze corpus/index metadata into reproducibility artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CORPUS = Path("data/corpus/normalized/combined_corpus.jsonl")
DEFAULT_EMBEDDINGS = Path("data/embeddings/real_all.dense.field_embeddings.jsonl")
DEFAULT_INDEX_META = Path("data/index/turbovec_dense_1024.index/meta.json")
DEFAULT_REJECTIONS = Path("data/corpus/rejected/tier2_rejected.jsonl")
DEFAULT_OUT_DIR = Path("reports/eval")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def record_id(record: dict[str, Any]) -> str:
    source = record.get("source", "")
    source_id = record.get("source_id", "")
    if record.get("dataset_id"):
        return str(record["dataset_id"])
    return f"dataset:{source}:{source_id}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--index-meta", type=Path, default=DEFAULT_INDEX_META)
    parser.add_argument("--rejections", type=Path, default=DEFAULT_REJECTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        raise SystemExit(f"corpus not found: {args.corpus}")

    records = load_jsonl(args.corpus)
    ids = [record_id(record) for record in records]
    source_counts = Counter(str(record.get("source", "unknown")) for record in records)

    embedding_rows = 0
    embedding_model = None
    embedding_dimension = None
    if args.embeddings.exists():
        with args.embeddings.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                embedding_rows += 1
                if embedding_model is None:
                    row = json.loads(line)
                    embedding_model = row.get("model_name") or row.get("provider_name")
                    embedding_dimension = row.get("dimension")

    index_meta: dict[str, Any] = {}
    if args.index_meta.exists():
        index_meta = json.loads(args.index_meta.read_text(encoding="utf-8"))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(args.corpus),
        "corpus_sha256": sha256_file(args.corpus),
        "corpus_size": len(records),
        "unique_record_ids": len(set(ids)),
        "record_ids": ids,
        "source_distribution": dict(sorted(source_counts.items())),
        "ingestion_commit": git_commit(),
        "embedding_path": str(args.embeddings),
        "embedding_sha256": sha256_file(args.embeddings),
        "embedding_rows": embedding_rows,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "tokenizer_config": None,
        "index_meta_path": str(args.index_meta),
        "index_meta_sha256": sha256_file(args.index_meta),
        "index_size": len(index_meta.get("ids", [])) if index_meta else None,
        "index_dimension": index_meta.get("dim"),
        "index_bit_width": index_meta.get("bit_width"),
        "graph_build_hash": None,
        "rejection_log_path": str(args.rejections),
        "rejection_log_hash": sha256_file(args.rejections),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    with (args.out_dir / "source_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "records"])
        writer.writeheader()
        for source, count in sorted(source_counts.items()):
            writer.writerow({"source": source, "records": count})

    print(json.dumps({k: manifest[k] for k in ["corpus_size", "unique_record_ids", "index_size"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
