#!/usr/bin/env python3
"""Expand corpus with all Tier 1 adapters.

Tier 1: NeuroVault, GIN, EBRAINS, HCP + deeper DANDI and OpenNeuro.

Usage:
    python scripts/expand_corpus_tier1.py
    python scripts/expand_corpus_tier1.py --dry-run
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
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args(argv)

    import neural_search.ingestion.ebrains  # noqa: F401
    import neural_search.ingestion.gin  # noqa: F401
    import neural_search.ingestion.hcp  # noqa: F401
    import neural_search.ingestion.neurovault  # noqa: F401
    from neural_search.ingestion.registry import run_adapter

    for source in ["neurovault", "gin", "ebrains", "hcp"]:
        print(f"Fetching {source}...")
        try:
            records = run_adapter(source, limit=args.limit)
            _save_records(records, source, args.dry_run)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print("\nTier 1 expansion complete. Run scripts/dedup_corpus.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
