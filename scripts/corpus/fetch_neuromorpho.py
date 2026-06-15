"""Fetch NeuroMorpho neuron morphology datasets and append to corpus.

NeuroMorpho.Org hosts ~200K digitally reconstructed neurons organized by
brain archive (lab/paper). Each archive gets one corpus record tagged with
the region(s) and species. Coverage: 78%+ region labels.

Usage:
    python scripts/corpus/fetch_neuromorpho.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("neuromorpho_fetch")

CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")


def _has_region(record: dict) -> bool:
    br = record.get("brain_regions") or []
    return any((v.get("id") if isinstance(v, dict) else v) for v in br if v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max archive records to fetch (default: 500)")
    args = parser.parse_args(argv)

    from neural_search.ingestion.neuromorpho import fetch_neuromorpho
    from neural_search.ontology.loader import get_brain_regions

    valid_ids = frozenset(r.id for r in get_brain_regions())

    corpus = [json.loads(l) for l in CORPUS_PATH.read_text().strip().splitlines() if l.strip()]
    existing_ids: set[str] = {r["source_id"] for r in corpus if r.get("source") == "neuromorpho"}
    logger.info("Existing NeuroMorpho records in corpus: %d", len(existing_ids))

    logger.info("Fetching NeuroMorpho archives (limit=%d)…", args.limit)
    records = fetch_neuromorpho(limit=args.limit)
    logger.info("Fetched %d NeuroMorpho records", len(records))

    # Filter to canonical region IDs only
    for rec in records:
        rec["brain_regions"] = [r for r in (rec.get("brain_regions") or []) if r in valid_ids]

    new_records = [r for r in records if r["source_id"] not in existing_ids]
    logger.info("New (not yet in corpus): %d", len(new_records))

    if not new_records:
        logger.info("Nothing new to add")
        return 0

    with_region = sum(1 for r in new_records if _has_region(r))
    logger.info("New records with brain regions: %d/%d = %d%%",
                with_region, len(new_records),
                round(100 * with_region / len(new_records)) if new_records else 0)

    for r in new_records[:5]:
        logger.info("  %s | regions=%s | mods=%s | species=%s",
                    r["source_id"][:40],
                    r.get("brain_regions", [])[:3],
                    r.get("modalities", [])[:2],
                    r.get("species", [])[:2])

    if args.dry_run:
        logger.info("[dry-run] would append %d records to %s", len(new_records), CORPUS_PATH)
        return 0

    all_records = corpus + new_records
    CORPUS_PATH.write_text("\n".join(json.dumps(r) for r in all_records) + "\n")

    total = len(all_records)
    total_with = sum(1 for r in all_records if _has_region(r))
    logger.info("Corpus: %d records (+%d NeuroMorpho)", total, len(new_records))
    logger.info("Brain region coverage: %d/%d = %d%%", total_with, total,
                round(100 * total_with / total))

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
