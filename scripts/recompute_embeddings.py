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
)
from neural_search.embeddings.hashing import HashingEmbeddingProvider
from neural_search.normalized import load_normalized_records, stable_normalized_id
from neural_search.schemas import (
    EvidenceLabel,
    NormalizedDatasetRecord,
    NormalizedPaperRecord,
)

CORPUS_DIR = Path("data/corpus/normalized")
EMBEDDINGS_DIR = Path("data/embeddings")
FIELD_EMBEDDINGS_OUT = EMBEDDINGS_DIR / "real_all.field_embeddings.jsonl"
FIELD_EMBEDDINGS_DENSE_OUT = EMBEDDINGS_DIR / "real_all.dense.field_embeddings.jsonl"
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


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item.get("id") or item.get("label") or item) if isinstance(item, dict) else str(item)
                for item in value if item is not None]
    return [str(value)]


def _evidence_labels(values: object, label_type: str, source_field: str) -> list[EvidenceLabel]:
    labels: list[EvidenceLabel] = []
    for value in _as_list(values):
        if not value.strip():
            continue
        labels.append(
            EvidenceLabel(
                id=value.strip().replace(" ", "_").lower(),
                label=value.strip().replace("_", " ").title(),
                label_type=label_type,
                confidence=0.75,
                evidence_text=value,
                source_field=source_field,
                source_value=value,
                extractor_name="scripts.recompute_embeddings.flat_compat_loader",
            )
        )
    return labels


def _flat_record_id(raw: dict, *, prefix: str) -> str:
    source = str(raw.get("source") or "unknown")
    source_id = str(
        raw.get("source_id")
        or raw.get("dataset_id")
        or raw.get("paper_id")
        or raw.get("openalex_id")
        or raw.get("doi")
        or raw.get("identifier")
        or raw.get("title")
        or "unknown"
    ).strip() or "unknown"
    return stable_normalized_id(prefix, source, source_id)


def _flat_title(raw: dict, fallback: str) -> str:
    title = str(raw.get("title") or "").strip()
    return title or fallback.strip() or "Untitled record"


def _flat_to_normalized_record(raw: dict) -> NormalizedDatasetRecord | NormalizedPaperRecord:
    source = str(raw.get("source") or "unknown")
    source_id = str(
        raw.get("source_id")
        or raw.get("openalex_id")
        or raw.get("doi")
        or raw.get("identifier")
        or raw.get("title")
        or "unknown"
    ).strip() or "unknown"
    if source == "openalex" or raw.get("abstract") is not None or raw.get("paper_id"):
        return NormalizedPaperRecord(
            paper_id=_flat_record_id(raw, prefix="paper"),
            source=source,
            source_id=source_id,
            title=_flat_title(raw, source_id),
            abstract=raw.get("abstract"),
            doi=raw.get("doi"),
            url=raw.get("url"),
            year=raw.get("year") or raw.get("publication_year"),
            authors=_as_list(raw.get("authors") or raw.get("authors_json")),
            linked_datasets=_as_list(raw.get("linked_datasets") or raw.get("linked_dataset_ids")),
        )

    return NormalizedDatasetRecord(
        dataset_id=_flat_record_id(raw, prefix="dataset"),
        source=source,
        source_id=source_id,
        title=_flat_title(raw, source_id),
        description=raw.get("description"),
        url=raw.get("url"),
        species=_evidence_labels(raw.get("species"), "species", "flat.species"),
        modalities=_evidence_labels(raw.get("modalities"), "modality", "flat.modalities"),
        brain_regions=_evidence_labels(raw.get("brain_regions"), "brain_region", "flat.brain_regions"),
        tasks=_evidence_labels(raw.get("tasks"), "task", "flat.tasks"),
        behavioral_events=_evidence_labels(raw.get("behaviors"), "behavioral_event", "flat.behaviors"),
        data_standards=_evidence_labels(raw.get("data_standards"), "data_standard", "flat.data_standards"),
    )


def load_records_compatible(corpus_file: Path) -> list[NormalizedDatasetRecord | NormalizedPaperRecord]:
    """Load normalized records, falling back to flat adapter records."""
    try:
        return load_normalized_records(corpus_file)
    except ValueError:
        records: list[NormalizedDatasetRecord | NormalizedPaperRecord] = []
        with corpus_file.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(_flat_to_normalized_record(json.loads(line)))
        return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dimensions",
        type=int,
        default=64,
        help="Hashing provider dimensions (default: 64, matching v07 convention)",
    )
    parser.add_argument(
        "--provider",
        default="hashing",
        choices=["hashing", "dense"],
        help="Embedding provider: 'hashing' (default, fast CI) or 'dense' (BGE-large-en-v1.5, semantic)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and count records but do not write output files",
    )
    args = parser.parse_args(argv)

    corpus_files = collect_corpus_files()
    if not corpus_files:
        # Fallback: accept the full corpus file produced by build_full_corpus.py
        full_corpus = CORPUS_DIR / "combined_corpus.jsonl" / "full_corpus_v09.jsonl"
        if full_corpus.exists():
            corpus_files = [full_corpus]
        else:
            print(f"ERROR: no real_*.jsonl source files found under {CORPUS_DIR}", file=sys.stderr)
            return 1

    print(f"Corpus files ({len(corpus_files)}):")
    for f in corpus_files:
        print(f"  {f}")

    # Load all records from all source files, deduplicating by dataset_id
    all_records: list = []
    seen_ids: set[str] = set()
    for corpus_file in corpus_files:
        records = load_records_compatible(corpus_file)
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
        print(f"\n[dry-run] provider={args.provider}. Skipping file writes.")
        return 0

    # --- Field embeddings (all record types) ---
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if args.provider == "dense":
        from neural_search.embeddings.dense_provider import DenseEmbeddingProvider
        provider = DenseEmbeddingProvider()
        field_embeddings_out = FIELD_EMBEDDINGS_DENSE_OUT
        print("\nUsing DenseEmbeddingProvider (BGE-large-en-v1.5, dim=1024)")
    else:
        provider = HashingEmbeddingProvider(dimensions=args.dimensions)
        field_embeddings_out = FIELD_EMBEDDINGS_OUT
    print(
        f"\nBuilding field embeddings with provider={provider.provider_name}/"
        f"{provider.model_name} (dim={provider.dimension}) ..."
    )
    field_records = build_field_embedding_records(all_records, provider)
    write_field_embedding_cache(field_records, field_embeddings_out)
    print(f"Wrote {len(field_records)} field embedding records -> {field_embeddings_out}")

    # --- Fingerprints (dataset records only, hashing provider only) ---
    # Dense fingerprints would be dimensionally incompatible with existing hashing fingerprints.
    # The dense path only updates field embeddings; fingerprints are left unchanged.
    if args.provider == "hashing":
        print(f"\nBuilding fingerprints for {len(dataset_records)} datasets ...")
        builder = DatasetFingerprintBuilder(text_model="hashing", combined_dim=args.dimensions)
        fingerprints = builder.build_fingerprints(dataset_records)
        from neural_search.embeddings.fingerprint import write_fingerprints
        write_fingerprints(fingerprints, str(FINGERPRINTS_OUT))
        print(f"Wrote {len(fingerprints)} fingerprints -> {FINGERPRINTS_OUT}")
    else:
        print("\nSkipping fingerprints (dense provider — fingerprints use separate hashing pipeline)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
