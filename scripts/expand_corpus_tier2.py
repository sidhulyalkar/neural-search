#!/usr/bin/env python3
"""Expand corpus with all Tier 2 adapters (classifier-gated).

Tier 2: OSF, figshare, zenodo — all records pass DatasetInclusionClassifier.

Usage:
    python scripts/expand_corpus_tier2.py
    python scripts/expand_corpus_tier2.py --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUTPUT_DIR = Path("data/corpus/normalized")


def _save_records(records: list[dict], source: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  DRY RUN: would save {len(records)} {source} records")
        return
    out = OUTPUT_DIR / f"real_{source}.jsonl"
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Saved {len(records)} → {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args(argv)

    import neural_search.ingestion.figshare  # noqa: F401
    import neural_search.ingestion.osf  # noqa: F401
    import neural_search.ingestion.zenodo  # noqa: F401
    from neural_search.ingestion.registry import run_adapter

    for source in ["osf", "figshare", "zenodo"]:
        print(f"Fetching {source} (classifier-gated)...")
        try:
            records = run_adapter(source, limit=args.limit)
            _save_records(records, source, args.dry_run)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print("\nTier 2 expansion complete. Run scripts/dedup_corpus.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
