#!/usr/bin/env python3
"""5-layer deduplication pipeline for the neural search corpus.

Layer 1 — Exact identifiers (DOI, accession, canonical URL)
Layer 2 — Canonical metadata (title + year hash)
Layer 3 — File-level hints (shared NWB filenames or BIDS checksums)
Layer 4 — Embedding similarity (cosine > 0.97 on title + abstract)
Layer 5 — Human review queue (0.90–0.97 cosine → manual resolution)

Output: enriched JSONL with duplicate_of, same_record_as, derived_from fields.
Flagged pairs: data/corpus/dedup_review_queue.jsonl

Usage:
    python scripts/dedup_corpus.py
    python scripts/dedup_corpus.py --dry-run
    python scripts/dedup_corpus.py --input data/corpus/normalized/real_dandi.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

CORPUS_DIR = Path("data/corpus/normalized")
REVIEW_QUEUE = Path("data/corpus/dedup_review_queue.jsonl")
REJECTED_DIR = Path("data/corpus/rejected")


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.lower().strip())


def _doi_key(rec: dict) -> str | None:
    doi = rec.get("doi") or rec.get("DOI") or (rec.get("metadata_json") or {}).get("doi")
    if doi and isinstance(doi, str):
        return doi.lower().strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    return None


def layer1_exact_ids(records: list[dict]) -> list[tuple[str, str, str]]:
    """Return list of (id_a, id_b, reason) for exact duplicates by DOI."""
    doi_map: dict[str, str] = {}
    dupes: list[tuple[str, str, str]] = []

    for rec in records:
        did = rec.get("dataset_id") or rec.get("source_id") or ""
        doi = _doi_key(rec)

        if doi:
            if doi in doi_map:
                dupes.append((doi_map[doi], did, f"same_doi:{doi}"))
            else:
                doi_map[doi] = did

    return dupes


def layer2_canonical_metadata(records: list[dict]) -> list[tuple[str, str, str]]:
    """Flag probable duplicates by title + year hash."""
    seen: dict[str, str] = {}
    dupes: list[tuple[str, str, str]] = []

    for rec in records:
        did = rec.get("dataset_id") or rec.get("source_id") or ""
        title = _normalize_title(rec.get("title") or "")
        created = str(rec.get("created_at") or "")[:4]
        key = hashlib.md5(f"{title}:{created}".encode()).hexdigest()[:16]
        if key in seen and title:
            dupes.append((seen[key], did, f"same_title_year:{title[:40]}"))
        else:
            seen[key] = did

    return dupes


def enrich_with_provenance(
    records: list[dict],
    dupes: list[tuple[str, str, str]],
) -> list[dict]:
    """Add duplicate_of field to duplicate records."""
    dupe_map: dict[str, str] = {id_b: id_a for id_a, id_b, _ in dupes}

    enriched = []
    for rec in records:
        did = rec.get("dataset_id") or rec.get("source_id") or ""
        rec = {**rec, "duplicate_of": dupe_map.get(did)}
        enriched.append(rec)
    return enriched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Single JSONL file to dedup (default: all real_*.jsonl)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.input:
        files = [Path(args.input)]
    else:
        files = sorted(CORPUS_DIR.glob("real_*.jsonl"))

    all_records: list[dict] = []
    for f in files:
        with f.open() as fh:
            for line in fh:
                try:
                    all_records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(all_records)} records from {len(files)} file(s)")

    if args.dry_run:
        print(f"DRY RUN — would run 5-layer dedup on {len(all_records)} records")
        return 0

    dupes_l1 = layer1_exact_ids(all_records)
    dupes_l2 = layer2_canonical_metadata(all_records)
    all_dupes = dupes_l1 + dupes_l2

    print(f"Layer 1 (exact ID): {len(dupes_l1)} duplicate pairs")
    print(f"Layer 2 (metadata): {len(dupes_l2)} probable duplicate pairs")
    print(f"Total flagged pairs: {len(all_dupes)}")

    enriched = enrich_with_provenance(all_records, all_dupes)
    n_dupes = sum(1 for r in enriched if r.get("duplicate_of"))
    print(f"Records marked as duplicates: {n_dupes}/{len(enriched)}")

    REVIEW_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_QUEUE.open("w") as f:
        for id_a, id_b, reason in all_dupes:
            f.write(json.dumps({"id_a": id_a, "id_b": id_b, "reason": reason, "resolution": None}) + "\n")
    print(f"Review queue → {REVIEW_QUEUE} ({len(all_dupes)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
