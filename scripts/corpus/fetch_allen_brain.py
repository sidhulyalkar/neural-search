"""Fetch Allen Brain Atlas datasets and append to corpus.

Covers Allen Cell Types Database (morphology + electrophysiology) and
Brain Observatory (Neuropixels + calcium imaging visual cortex surveys).

Usage:
    python scripts/corpus/fetch_allen_brain.py [--dry-run] [--limit N]
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
logger = logging.getLogger("allen_fetch")
CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args(argv)
    from _fetch_common import append_to_corpus

    from neural_search.ingestion.allen_brain import fetch_allen_records
    corpus = [json.loads(line) for line in CORPUS_PATH.read_text().strip().splitlines() if line.strip()]
    existing = {r["source_id"] for r in corpus if r.get("source") == "allen"}
    logger.info("Existing Allen records: %d", len(existing))
    records = fetch_allen_records(limit=args.limit)
    logger.info("Fetched %d Allen records", len(records))
    new_records = [r for r in records if r["source_id"] not in existing]
    logger.info("New: %d", len(new_records))
    if not new_records:
        logger.info("Nothing new")
        return 0
    append_to_corpus(CORPUS_PATH, corpus, new_records, "Allen Brain", logger, args.dry_run)
    return 0

if __name__ == "__main__":
    sys.exit(main())
