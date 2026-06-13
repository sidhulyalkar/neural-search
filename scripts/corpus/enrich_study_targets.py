"""Fetch studyTarget for DANDI records still missing brain regions.

The rich_metadata_cache was populated before studyTarget extraction was added.
This script re-fetches ONLY the ~413 DANDI records without brain_regions to get
their studyTarget text, then re-runs extraction using it as additional context.

Usage:
    python scripts/corpus/enrich_study_targets.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neural_search.ingestion.dandi import (
    fetch_dandiset_rich_metadata,
    normalize_dandiset,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("enrich_study_targets")

CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")
RICH_CACHE_PATH = Path("data/raw/dandi/rich_metadata_cache.jsonl")
RAW_BATCH_DIR = Path("data/raw/dandi")


def _load_raw_dandisets() -> dict[str, dict]:
    raw_map: dict[str, dict] = {}
    for jf in RAW_BATCH_DIR.glob("*.json"):
        if "rich_metadata" in jf.name or "nwb_electrode" in jf.name:
            continue
        try:
            data = json.loads(jf.read_text())
            batch = data if isinstance(data, list) else data.get("results", [])
            for r in batch:
                sid = str(r.get("identifier", ""))
                if sid:
                    raw_map[sid] = r
        except Exception:
            pass
    return raw_map


def _load_rich_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if RICH_CACHE_PATH.exists():
        for line in RICH_CACHE_PATH.read_text().splitlines():
            try:
                entry = json.loads(line)
                sid = entry.pop("_source_id", None)
                if sid:
                    cache[sid] = entry
            except Exception:
                pass
    return cache


def _save_to_rich_cache(source_id: str, data: dict) -> None:
    entry = {"_source_id": source_id, **data}
    with open(RICH_CACHE_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _flatten_ids(br: list) -> list[str]:
    return [(v.get("id") if isinstance(v, dict) else v) for v in (br or []) if v]


def _has_region(record: dict) -> bool:
    return any(_flatten_ids(record.get("brain_regions") or []))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    corpus = [
        json.loads(l)
        for l in CORPUS_PATH.read_text().strip().splitlines()
        if l.strip()
    ]
    raw_map = _load_raw_dandisets()
    rich_cache = _load_rich_cache()

    # Only re-fetch DANDI records still missing brain_regions
    targets = [
        r for r in corpus
        if r.get("source") == "dandi" and not _has_region(r)
    ]
    logger.info("DANDI records without brain regions: %d", len(targets))

    if args.limit:
        targets = targets[: args.limit]

    enriched_map: dict[str, dict] = {}
    refreshed = 0
    gained_region = 0

    for i, rec in enumerate(targets, 1):
        sid = rec["source_id"]

        # Force fresh fetch to pick up studyTarget
        cached = rich_cache.get(sid, {})
        has_study_target = bool(cached.get("studyTarget"))

        if has_study_target:
            rich = cached
        else:
            logger.info("[%d/%d] Re-fetching %s for studyTarget …", i, len(targets), sid)
            rich = fetch_dandiset_rich_metadata(sid)
            if not args.dry_run:
                _save_to_rich_cache(sid, rich)
            refreshed += 1
            time.sleep(0.15)

        if not rich.get("studyTarget"):
            continue

        raw = raw_map.get(sid, {"identifier": sid})
        new_norm = normalize_dandiset(raw, rich_metadata=rich)

        new_regions = _flatten_ids(new_norm.get("brain_regions") or [])
        if new_regions:
            logger.info("  %s: studyTarget gave regions %s", sid, new_regions)
            enriched_map[sid] = {**rec, "brain_regions": new_regions}
            gained_region += 1

    logger.info("Re-fetched: %d, Gained brain regions: %d", refreshed, gained_region)

    if args.dry_run or not enriched_map:
        if not enriched_map:
            logger.info("No new regions found via studyTarget; no changes written.")
        return 0

    output = []
    for r in corpus:
        sid = r.get("source_id", "")
        if sid in enriched_map:
            output.append(json.dumps(enriched_map[sid]))
        else:
            output.append(json.dumps(r))

    CORPUS_PATH.write_text("\n".join(output) + "\n")
    logger.info("Updated %d records → %s", len(enriched_map), CORPUS_PATH)

    refreshed_corpus = [
        json.loads(l)
        for l in CORPUS_PATH.read_text().strip().splitlines()
        if l.strip()
    ]
    total = len(refreshed_corpus)
    with_region = sum(1 for r in refreshed_corpus if _has_region(r))
    logger.info(
        "Brain region coverage: %d/%d = %d%%", with_region, total,
        round(100 * with_region / total)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
