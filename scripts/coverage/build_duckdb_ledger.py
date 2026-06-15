"""Sprint 2 — Build the DuckDB coverage ledger from the full corpus.

Usage
-----
    # Full rebuild (replaces existing DB)
    python scripts/coverage/build_duckdb_ledger.py --replace

    # Incremental update (append new records only)
    python scripts/coverage/build_duckdb_ledger.py

    # Print coverage summary after build
    python scripts/coverage/build_duckdb_ledger.py --summary

    # Export to Parquet as well
    python scripts/coverage/build_duckdb_ledger.py --export-parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_duckdb_ledger")

CORPUS_PATH = ROOT / "data" / "corpus" / "normalized" / "combined_corpus.jsonl"
DB_PATH = ROOT / "data" / "coverage" / "ledger.duckdb"
REPORTS_DIR = ROOT / "data" / "reports" / "coverage"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH,
                        help="Path to corpus JSONL or directory")
    parser.add_argument("--db", type=Path, default=DB_PATH,
                        help="DuckDB output path")
    parser.add_argument("--snapshot-id", default="v09",
                        help="Snapshot label embedded in every entry")
    parser.add_argument("--replace", action="store_true",
                        help="Replace existing DB instead of incrementally updating")
    parser.add_argument("--summary", action="store_true",
                        help="Print coverage summary after build")
    parser.add_argument("--export-parquet", action="store_true",
                        help="Export tables to Parquet after build")
    parser.add_argument("--rebuild-report", action="store_true",
                        help="Regenerate coverage_gap_report.md from current corpus")
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        log.error("Corpus path not found: %s", args.corpus)
        return 1

    from neural_search.coverage.duckdb_store import CoverageStore
    from neural_search.coverage.ledger import CoverageLedger

    log.info("Building DuckDB ledger: %s", args.db)

    with CoverageStore(args.db) as store:
        stats = store.build(
            args.corpus,
            snapshot_id=args.snapshot_id,
            replace=args.replace,
        )
        log.info(
            "Done — new: %d datasets / %d entries | total: %d datasets / %d entries",
            stats["new_datasets"], stats["new_entries"],
            stats["total_datasets"], stats["total_entries"],
        )

        if args.summary:
            summary = store.coverage_summary()
            print("\n=== Coverage Summary ===")
            print(f"Datasets: {summary['total_datasets']:,}")
            print(f"Entries:  {summary['total_entries']:,}")
            print("\nDimension coverage (confidence ≥ 0.65):")
            for dim, data in summary.get("dimension_coverage", {}).items():
                print(f"  {dim:25s} {data['datasets']:5d} datasets  ({data['pct']:.1f}%)")

            print("\n=== Source Coverage Rates ===")
            store.source_coverage_rates().show(max_rows=20)

            print("\n=== Uncovered Regions ===")
            store.uncovered_regions().show(max_rows=30)

            print("\n=== Top Dark Region × Modality Gaps ===")
            store.dark_pairs("brain_regions", "modalities", top_n=15).show()

        if args.export_parquet:
            paths = store.export_parquet(args.db.parent)
            log.info("Exported Parquet:")
            for table, path in paths.items():
                log.info("  %s → %s", table, path)

    if args.rebuild_report:
        log.info("Rebuilding gap report from corpus…")
        ledger = CoverageLedger.from_path(args.corpus, snapshot_id=args.snapshot_id)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        paths = ledger.write(REPORTS_DIR)
        log.info("Gap report written:")
        for key, path in paths.items():
            log.info("  %s → %s", key, path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
