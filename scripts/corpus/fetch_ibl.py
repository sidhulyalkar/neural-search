"""Fetch IBL datasets and append them to the corpus.

The International Brain Laboratory (IBL) provides:
- Curated project-level records (Brain Wide Map releases, repeated-site)
- Session-level records from Neuropixels ephys across 12 collaborating labs

All records target the full mouse brain with regions: prefrontal_cortex,
parietal_cortex, hippocampus, thalamus, striatum, brainstem.

Usage:
    python scripts/corpus/fetch_ibl.py [--dry-run] [--projects-only]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ibl_fetch")

CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")


def _has_region(record: dict) -> bool:
    br = record.get("brain_regions") or []
    return any((v.get("id") if isinstance(v, dict) else v) for v in br if v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=200,
                        help="Max session records to fetch (default: 200)")
    parser.add_argument("--projects-only", action="store_true",
                        help="Only fetch curated project records, skip per-session S3 tables")
    args = parser.parse_args(argv)

    from neural_search.ingestion.ibl import fetch_ibl_records

    corpus = [json.loads(line) for line in CORPUS_PATH.read_text().strip().splitlines() if line.strip()]
    existing_ids: set[str] = {r["source_id"] for r in corpus if r.get("source") == "ibl"}
    logger.info("Existing IBL records in corpus: %d", len(existing_ids))

    limit = 10 if args.projects_only else args.limit
    logger.info("Fetching IBL records (limit=%d, projects_only=%s)…", limit, args.projects_only)
    ibl_records = fetch_ibl_records(limit=limit)
    logger.info("Fetched %d IBL records", len(ibl_records))

    new_records = [r for r in ibl_records if r["source_id"] not in existing_ids]
    logger.info("New (not yet in corpus): %d", len(new_records))

    if not new_records:
        logger.info("Nothing new to add")
        return 0

    with_region = sum(1 for r in new_records if _has_region(r))
    logger.info("New records with brain regions: %d/%d = %d%%",
                with_region, len(new_records),
                round(100 * with_region / len(new_records)) if new_records else 0)

    for r in new_records[:5]:
        logger.info("  %s | %s", r["source_id"], r["title"][:60])
        logger.info("    regions=%s mods=%s", r.get("brain_regions", [])[:4], r.get("modalities", [])[:3])

    if args.dry_run:
        logger.info("[dry-run] would append %d records to %s", len(new_records), CORPUS_PATH)
        return 0

    all_records = corpus + new_records
    CORPUS_PATH.write_text("\n".join(json.dumps(r) for r in all_records) + "\n")

    total = len(all_records)
    total_with_region = sum(1 for r in all_records if _has_region(r))
    logger.info("Corpus: %d records (+%d IBL)", total, len(new_records))
    logger.info("Brain region coverage: %d/%d = %d%%",
                total_with_region, total, round(100 * total_with_region / total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
