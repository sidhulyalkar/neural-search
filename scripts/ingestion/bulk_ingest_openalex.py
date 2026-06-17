#!/usr/bin/env python3
"""Bulk ingest neuroscience papers from OpenAlex using cursor pagination.

Usage (Tier 1 — high-cited papers, ~256K, ingest first):
    python scripts/ingestion/bulk_ingest_openalex.py \\
        --tier tier1 \\
        --out data/corpus/normalized/openalex_neuro

Resume after interruption:
    python scripts/ingestion/bulk_ingest_openalex.py \\
        --tier tier1 \\
        --out data/corpus/normalized/openalex_neuro \\
        --resume

Dry run (fetch one page, print stats, exit):
    python scripts/ingestion/bulk_ingest_openalex.py --tier tier1 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.ingestion.openalex_bulk import (
    TIER_FILTERS,
    BulkIngester,
    normalize_bulk_work,
    OPENALEX_BASE,
    POLITE_EMAIL,
    SELECT_FIELDS,
    PAGE_SIZE,
)

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False


def _dry_run(tier: str) -> None:
    if not _HAS_HTTPX:
        print("ERROR: httpx not installed", file=sys.stderr)
        sys.exit(1)
    import httpx as _httpx

    tier_filter = TIER_FILTERS[tier]
    params = {
        "filter": tier_filter,
        "per_page": 1,
        "select": "id",
        "cursor": "*",
        "mailto": POLITE_EMAIL,
    }
    with _httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{OPENALEX_BASE}/works", params=params)
        resp.raise_for_status()
        data = resp.json()
    count = data.get("meta", {}).get("count", "unknown")
    print(json.dumps({"tier": tier, "filter": tier_filter, "total_papers": count}, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bulk ingest neuroscience papers from OpenAlex.")
    parser.add_argument("--tier", choices=list(TIER_FILTERS), default="tier1")
    parser.add_argument("--out", type=Path, default=Path("data/corpus/normalized/openalex_neuro"))
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Fetch stats only, no ingest")
    args = parser.parse_args(argv)

    if args.dry_run:
        _dry_run(args.tier)
        return 0

    if not _HAS_HTTPX:
        print("ERROR: httpx not installed. Run: pip install httpx", file=sys.stderr)
        return 1

    ingester = BulkIngester(
        out_dir=args.out,
        tier=args.tier,
        shard_size=args.shard_size,
    )

    if not args.resume:
        checkpoint = ingester._checkpoint_path
        if checkpoint.exists():
            existing = json.loads(checkpoint.read_text())
            if existing.get("tier") == args.tier and existing.get("records_fetched", 0) > 0:
                print(
                    f"WARNING: checkpoint exists for {args.tier} with "
                    f"{existing['records_fetched']} records. Use --resume to continue, "
                    f"or delete {checkpoint} to start fresh.",
                    file=sys.stderr,
                )
                return 1

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Ingesting {args.tier} → {args.out}")
    print(f"Filter: {TIER_FILTERS[args.tier]}")
    if args.max_records:
        print(f"Max records: {args.max_records}")

    t0 = time.time()
    total = ingester.run(max_records=args.max_records)
    elapsed = time.time() - t0

    summary = {
        "tier": args.tier,
        "total_records": total,
        "elapsed_s": round(elapsed, 1),
        "rate_per_s": round(total / elapsed, 1) if elapsed > 0 else 0,
        "out_dir": str(args.out),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
