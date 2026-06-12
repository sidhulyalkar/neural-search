#!/usr/bin/env python3
"""Build the full normalized corpus from all available raw data sources.

Loads raw payloads from data/raw/, normalizes via source adapters,
deduplicates by dataset_id, and writes to data/corpus/normalized/.

Usage::

    python scripts/corpus/build_full_corpus.py
    python scripts/corpus/build_full_corpus.py --dry-run
    python scripts/corpus/build_full_corpus.py --max-per-source 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_full_corpus")

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/corpus/normalized/combined_corpus.jsonl")
OUT_FILENAME = "full_corpus_v09.jsonl"


def _load_raw_jsonl_dir(source_dir: Path) -> list[dict]:
    records: list[dict] = []
    if not source_dir.is_dir():
        return records
    for fpath in sorted(source_dir.glob("*.json")):
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            # Both list and paginated {results: [...]} formats
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict) and "results" in data:
                records.extend(data["results"])
            elif isinstance(data, dict):
                records.append(data)
        except Exception as exc:
            log.warning("Failed to load %s: %s", fpath, exc)
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-per-source", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_records: list[dict] = []
    counts: dict[str, int] = {}

    # --- DANDI ---
    try:
        from neural_search.ingestion.dandi import normalize_dandiset_record
        raw_dandi = _load_raw_jsonl_dir(RAW_DIR / "dandi")
        log.info("DANDI: %d raw records", len(raw_dandi))
        dandi_normalized = []
        for raw in raw_dandi[:args.max_per_source]:
            try:
                dandi_normalized.append(normalize_dandiset_record(raw).model_dump(mode="json", exclude_none=True))
            except Exception as exc:
                log.debug("DANDI normalize error: %s", exc)
        counts["dandi"] = len(dandi_normalized)
        all_records.extend(dandi_normalized)
        log.info("DANDI: normalized %d records", len(dandi_normalized))
    except Exception as exc:
        log.warning("DANDI ingestion failed: %s", exc)

    # --- OpenNeuro ---
    try:
        from neural_search.ingestion.openneuro import normalize_openneuro_record
        raw_openneuro = _load_raw_jsonl_dir(RAW_DIR / "openneuro")
        log.info("OpenNeuro: %d raw records", len(raw_openneuro))
        openneuro_normalized = []
        for raw in raw_openneuro[:args.max_per_source]:
            try:
                openneuro_normalized.append(normalize_openneuro_record(raw).model_dump(mode="json", exclude_none=True))
            except Exception as exc:
                log.debug("OpenNeuro normalize error: %s", exc)
        counts["openneuro"] = len(openneuro_normalized)
        all_records.extend(openneuro_normalized)
        log.info("OpenNeuro: normalized %d records", len(openneuro_normalized))
    except Exception as exc:
        log.warning("OpenNeuro ingestion failed: %s", exc)

    # --- SPARK stubs ---
    try:
        from neural_search.ingestion.spark import (
            fetch_spark_records,
            normalize_spark_record,
        )
        spark_raw = fetch_spark_records(limit=args.max_per_source or 20)
        spark_normalized = []
        for raw in spark_raw:
            try:
                spark_normalized.append(normalize_spark_record(raw).model_dump(mode="json", exclude_none=True))
            except Exception as exc:
                log.debug("SPARK normalize error: %s", exc)
        counts["spark"] = len(spark_normalized)
        all_records.extend(spark_normalized)
        log.info("SPARK: normalized %d records", len(spark_normalized))
    except Exception as exc:
        log.warning("SPARK ingestion failed: %s", exc)

    # --- Dedup by dataset_id ---
    seen: dict[str, dict] = {}
    dupes = 0
    for rec in all_records:
        did = rec.get("dataset_id", "")
        if did and did in seen:
            dupes += 1
        elif did:
            seen[did] = rec

    deduped = list(seen.values())
    log.info(
        "Total: %d records from %d sources, %d dupes removed → %d unique",
        len(all_records),
        len(counts),
        dupes,
        len(deduped),
    )
    for source, count in counts.items():
        log.info("  %s: %d", source, count)

    if args.dry_run:
        print(f"\nDry run — would write {len(deduped)} records to {args.out_dir / OUT_FILENAME}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / OUT_FILENAME
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in deduped:
            fh.write(json.dumps(rec, default=str) + "\n")
    log.info("Wrote %d records → %s", len(deduped), out_path)
    print(f"\nBuilt full corpus: {len(deduped)} records → {out_path}")
    for source, count in counts.items():
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
