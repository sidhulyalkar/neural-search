"""Enrich DANDI corpus records by fetching per-dandiset rich metadata.

For records missing modalities or species, fetches assetsSummary.variableMeasured,
assetsSummary.approach, and assetsSummary.species via the DANDI Python client,
then re-normalizes and writes an updated corpus file.

Usage:
    python scripts/corpus/enrich_dandi_metadata.py
    python scripts/corpus/enrich_dandi_metadata.py --dry-run --limit 10
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
logger = logging.getLogger("enrich_dandi")

CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")
RICH_CACHE_PATH = Path("data/raw/dandi/rich_metadata_cache.jsonl")
RAW_BATCH_DIR = Path("data/raw/dandi")


def _load_raw_dandisets() -> dict[str, dict]:
    """Load raw DANDI listing records keyed by identifier."""
    raw_map: dict[str, dict] = {}
    for jf in RAW_BATCH_DIR.glob("*.json"):
        if "rich_metadata" in jf.name:
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
    """Load previously-fetched rich metadata from cache file."""
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Fetch but do not write")
    parser.add_argument("--limit", type=int, default=None, help="Max records to enrich")
    parser.add_argument("--force-all", action="store_true", help="Re-enrich all DANDI records")
    args = parser.parse_args(argv)

    corpus = [json.loads(line) for line in CORPUS_PATH.read_text().strip().splitlines() if line.strip()]
    raw_map = _load_raw_dandisets()
    rich_cache = _load_rich_cache()

    logger.info("Corpus: %d records", len(corpus))
    logger.info("Raw DANDI listing records: %d", len(raw_map))
    logger.info("Rich metadata cache: %d entries", len(rich_cache))

    dandi_records = [r for r in corpus if r.get("source") == "dandi"]
    needs_enrichment = [
        r for r in dandi_records
        if args.force_all or not r.get("modalities") or not r.get("species") or not r.get("brain_regions")
    ]
    logger.info("DANDI records needing enrichment: %d", len(needs_enrichment))

    if args.limit:
        needs_enrichment = needs_enrichment[: args.limit]

    enriched_map: dict[str, dict] = {}  # source_id → enriched normalized record
    fetched = 0
    cache_hits = 0

    for i, rec in enumerate(needs_enrichment, 1):
        source_id = rec.get("source_id", "")
        if not source_id:
            continue

        if source_id in rich_cache:
            rich = rich_cache[source_id]
            cache_hits += 1
        else:
            logger.info("[%d/%d] Fetching %s …", i, len(needs_enrichment), source_id)
            rich = fetch_dandiset_rich_metadata(source_id)
            if not args.dry_run:
                _save_to_rich_cache(source_id, rich)
            fetched += 1
            time.sleep(0.15)  # gentle rate-limit

        raw = raw_map.get(source_id, {"identifier": source_id})
        new_norm = normalize_dandiset(raw, rich_metadata=rich)

        mods_before = rec.get("modalities") or []
        mods_after = new_norm.get("modalities") or []
        spc_before = rec.get("species") or []
        spc_after = new_norm.get("species") or []

        if mods_after != mods_before or spc_after != spc_before:
            logger.info(
                "  %s: modalities %s→%s  species %s→%s",
                source_id,
                [m.get("id", m) if isinstance(m, dict) else m for m in mods_before],
                mods_after,
                [s.get("id", s) if isinstance(s, dict) else s for s in spc_before],
                spc_after,
            )

        enriched_map[source_id] = {**rec, **{
            k: new_norm[k]
            for k in ("modalities", "species", "brain_regions", "tasks", "behaviors", "description")
            if k in new_norm
        }}

    logger.info("Fetched: %d  Cache hits: %d", fetched, cache_hits)

    if args.dry_run:
        logger.info("[dry-run] skipping corpus write")
        return 0

    # Write updated corpus
    updated = 0
    output: list[dict] = []
    for rec in corpus:
        sid = rec.get("source_id", "")
        if rec.get("source") == "dandi" and sid in enriched_map:
            output.append(enriched_map[sid])
            updated += 1
        else:
            output.append(rec)

    CORPUS_PATH.write_text("\n".join(json.dumps(r) for r in output) + "\n")
    logger.info("Updated %d records → %s", updated, CORPUS_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
