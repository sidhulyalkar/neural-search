"""Fetch NeuroVault collections and append them to the corpus.

NeuroVault hosts fMRI statistical maps with collection-level metadata. Each
collection normalizes to a corpus record via neural_search.ingestion.neurovault.

This script appends new NeuroVault records to the existing corpus without
requiring a full rebuild. Records are deduplicated by source_id.

Coverage note: Most NeuroVault collections are whole-brain fMRI contrasts,
so brain_region labels will be sparse (only when collection descriptions name
specific target regions). The value is volume + task/behavior richness.

Usage:
    python scripts/corpus/fetch_neurovault.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("neurovault_fetch")

CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")


def _has_region(record: dict) -> bool:
    br = record.get("brain_regions") or []
    return any((v.get("id") if isinstance(v, dict) else v) for v in br if v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=2000,
                        help="Max collections to fetch (default: 2000)")
    args = parser.parse_args(argv)

    from neural_search.ingestion.neurovault import fetch_neurovault

    corpus = [json.loads(line) for line in CORPUS_PATH.read_text().strip().splitlines() if line.strip()]
    existing_ids: set[str] = {r["source_id"] for r in corpus if r.get("source") == "neurovault"}
    logger.info("Existing NeuroVault records: %d", len(existing_ids))

    logger.info("Fetching up to %d NeuroVault collections…", args.limit)
    nv_records = fetch_neurovault(limit=args.limit)
    logger.info("Fetched %d NeuroVault collections", len(nv_records))

    new_records = [r for r in nv_records if r["source_id"] not in existing_ids]
    logger.info("New (not yet in corpus): %d", len(new_records))

    if not new_records:
        logger.info("Nothing new to add")
        return 0

    with_region = sum(1 for r in new_records if _has_region(r))
    logger.info("New records with brain regions: %d/%d = %d%%",
                with_region, len(new_records),
                round(100 * with_region / len(new_records)) if new_records else 0)

    # Show sample
    for r in new_records[:3]:
        logger.info("  %s: %s | modalities=%s regions=%s",
                    r["source_id"], r["title"][:60],
                    r.get("modalities", [])[:3],
                    r.get("brain_regions", [])[:3])

    if args.dry_run:
        logger.info("[dry-run] would append %d records to %s", len(new_records), CORPUS_PATH)
        return 0

    all_records = corpus + new_records
    CORPUS_PATH.write_text("\n".join(json.dumps(r) for r in all_records) + "\n")

    total = len(all_records)
    total_with_region = sum(1 for r in all_records if _has_region(r))
    logger.info("Corpus: %d records (+%d NeuroVault)", total, len(new_records))
    logger.info("Brain region coverage: %d/%d = %d%%",
                total_with_region, total, round(100 * total_with_region / total))

    by_src: dict[str, list[int]] = {}
    for r in all_records:
        s = r.get("source", "unknown")
        by_src.setdefault(s, [0, 0])
        by_src[s][1] += 1
        if _has_region(r):
            by_src[s][0] += 1
    for src, (w, t) in sorted(by_src.items()):
        logger.info("  %s: %d/%d = %d%%", src, w, t, round(100 * w / t))

    return 0


if __name__ == "__main__":
    sys.exit(main())
