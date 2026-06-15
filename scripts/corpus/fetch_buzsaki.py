"""Fetch Buzsaki Lab datasets and append to corpus.

Usage:
    python scripts/corpus/fetch_buzsaki.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("buzsaki_fetch")

CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")


def _has_region(record: dict) -> bool:
    br = record.get("brain_regions") or []
    return any((v.get("id") if isinstance(v, dict) else v) for v in br if v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from neural_search.ingestion.buzsaki import fetch_buzsaki_records

    corpus = [json.loads(l) for l in CORPUS_PATH.read_text().strip().splitlines() if l.strip()]
    existing_ids: set[str] = {r["source_id"] for r in corpus if r.get("source") == "buzsaki"}
    logger.info("Existing Buzsaki records in corpus: %d", len(existing_ids))

    records = fetch_buzsaki_records(limit=50)
    new_records = [r for r in records if r["source_id"] not in existing_ids]
    logger.info("New Buzsaki records: %d", len(new_records))

    if not new_records:
        logger.info("Nothing new to add")
        return 0

    with_region = sum(1 for r in new_records if _has_region(r))
    logger.info("New records with brain regions: %d/%d = %d%%",
                with_region, len(new_records),
                round(100 * with_region / len(new_records)) if new_records else 0)

    for r in new_records[:5]:
        logger.info("  %s | regions=%s | species=%s",
                    r["source_id"], r.get("brain_regions", [])[:3],
                    r.get("species", []))

    if args.dry_run:
        logger.info("[dry-run] would append %d records to %s", len(new_records), CORPUS_PATH)
        return 0

    all_records = corpus + new_records
    CORPUS_PATH.write_text("\n".join(json.dumps(r) for r in all_records) + "\n")

    total = len(all_records)
    total_with = sum(1 for r in all_records if _has_region(r))
    logger.info("Corpus: %d records (+%d Buzsaki)", total, len(new_records))
    logger.info("Brain region coverage: %d/%d = %d%%", total_with, total,
                round(100 * total_with / total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
