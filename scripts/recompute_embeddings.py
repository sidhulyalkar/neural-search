#!/usr/bin/env python3
"""Recompute field embeddings and fingerprints for the expanded real corpus.

Reads all real_*.jsonl corpus files (excluding backups), computes embeddings
using the hashing provider (64-d, matching the v07 convention), and writes:
  data/embeddings/real_all.field_embeddings.jsonl
  data/embeddings/real_all.fingerprints.jsonl

Usage:
    python scripts/recompute_embeddings.py [--dimensions N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_search.embeddings.field_index import (
    build_field_embedding_records,
    write_field_embedding_cache,
)
from neural_search.embeddings.fingerprint_builder import (
    DatasetFingerprintBuilder,
    build_fingerprints_from_corpus,
)
from neural_search.embeddings.hashing import HashingEmbeddingProvider
from neural_search.normalized import load_normalized_records
from neural_search.schemas import NormalizedDatasetRecord

CORPUS_DIR = Path("data/corpus/normalized")
EMBEDDINGS_DIR = Path("data/embeddings")
FIELD_EMBEDDINGS_OUT = EMBEDDINGS_DIR / "real_all.field_embeddings.jsonl"
FINGERPRINTS_OUT = EMBEDDINGS_DIR / "real_all.fingerprints.jsonl"

# Corpus files: all real_*.jsonl, excluding .backup.jsonl variants
# and the aggregated v07.*.jsonl files (which are derived, not source)
EXCLUDED_PATTERNS = {".backup.", "real_v07."}


def _is_source_corpus_file(path: Path) -> bool:
    return (
        path.name.startswith("real_")
        and path.suffix == ".jsonl"
        and not any(pat in path.name for pat in EXCLUDED_PATTERNS)
    )


def collect_corpus_files() -> list[Path]:
    return sorted(p for p in CORPUS_DIR.glob("real_*.jsonl") if _is_source_corpus_file(p))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dimensions",
        type=int,
        default=64,
        help="Hashing provider dimensions (default: 64, matching v07 convention)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and count records but do not write output files",
    )
    args = parser.parse_args(argv)

    corpus_files = collect_corpus_files()
    if not corpus_files:
        print(f"ERROR: no real_*.jsonl source files found under {CORPUS_DIR}", file=sys.stderr)
        return 1

    print(f"Corpus files ({len(corpus_files)}):")
    for f in corpus_files:
        print(f"  {f}")

    # Load all records from all source files, deduplicating by dataset_id
    all_records: list = []
    seen_ids: set[str] = set()
    for corpus_file in corpus_files:
        records = load_normalized_records(corpus_file)
        for record in records:
            rid = record.dataset_id if isinstance(record, NormalizedDatasetRecord) else record.paper_id
            if rid not in seen_ids:
                seen_ids.add(rid)
                all_records.append(record)

    dataset_records = [r for r in all_records if isinstance(r, NormalizedDatasetRecord)]
    paper_records = [r for r in all_records if not isinstance(r, NormalizedDatasetRecord)]

    print(f"\nLoaded {len(all_records)} unique records total:")
    print(f"  datasets : {len(dataset_records)}")
    print(f"  papers   : {len(paper_records)}")

    if args.dry_run:
        print("\n[dry-run] Skipping file writes.")
        return 0

    # --- Field embeddings (all record types) ---
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    provider = HashingEmbeddingProvider(dimensions=args.dimensions)
    print(
        f"\nBuilding field embeddings with provider={provider.provider_name}/"
        f"{provider.model_name} (dim={provider.dimension}) ..."
    )
    field_records = build_field_embedding_records(all_records, provider)
    write_field_embedding_cache(field_records, FIELD_EMBEDDINGS_OUT)
    print(f"Wrote {len(field_records)} field embedding records -> {FIELD_EMBEDDINGS_OUT}")

    # --- Fingerprints (dataset records only) ---
    print(f"\nBuilding fingerprints for {len(dataset_records)} datasets ...")
    builder = DatasetFingerprintBuilder(text_model="hashing", combined_dim=args.dimensions)
    fingerprints = builder.build_fingerprints(dataset_records)
    from neural_search.embeddings.fingerprint import write_fingerprints
    write_fingerprints(fingerprints, str(FINGERPRINTS_OUT))
    print(f"Wrote {len(fingerprints)} fingerprints -> {FINGERPRINTS_OUT}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
