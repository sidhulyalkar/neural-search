"""Fetch Harvard Dataverse neuroscience datasets and append to corpus.

Usage:
    python scripts/corpus/fetch_harvard_dataverse.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hdv_fetch")
CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args(argv)
    from _fetch_common import append_to_corpus

    from neural_search.ingestion.harvard_dataverse import fetch_harvard_dataverse
    corpus = [json.loads(line) for line in CORPUS_PATH.read_text().strip().splitlines() if line.strip()]
    existing = {r["source_id"] for r in corpus if r.get("source") == "harvard_dataverse"}
    logger.info("Existing Harvard Dataverse records: %d", len(existing))
    records = fetch_harvard_dataverse(limit=args.limit)
    logger.info("Fetched %d Harvard Dataverse records", len(records))
    new_records = [r for r in records if r["source_id"] not in existing]
    logger.info("New: %d", len(new_records))
    if not new_records:
        logger.info("Nothing new")
        return 0
    append_to_corpus(CORPUS_PATH, corpus, new_records, "Harvard Dataverse", logger, args.dry_run)
    return 0

if __name__ == "__main__":
    sys.exit(main())
