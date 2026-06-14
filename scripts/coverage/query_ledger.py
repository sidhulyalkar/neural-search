"""Query the DuckDB coverage ledger for gap analysis.

Preset queries
--------------
    gap-matrix          region × modality count table
    source-rates        per-source coverage percentages
    uncovered-regions   regions with zero datasets
    dark-pairs          highest-opportunity zero-coverage pairs
    atlas-coverage      Allen CCF / UBERON coverage per structure
    sql                 run a raw SQL query
    summary             print overall coverage stats

Usage
-----
    python scripts/coverage/query_ledger.py gap-matrix
    python scripts/coverage/query_ledger.py gap-matrix --species mus_musculus
    python scripts/coverage/query_ledger.py dark-pairs --dim-a brain_regions --dim-b modalities
    python scripts/coverage/query_ledger.py atlas-coverage --atlas uberon_id
    python scripts/coverage/query_ledger.py sql "SELECT * FROM ontology_regions LIMIT 5"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from neural_search.coverage.duckdb_store import CoverageStore

DB_PATH = ROOT / "data" / "coverage" / "ledger.duckdb"


def _require_db(db_path: Path) -> CoverageStore:
    if not db_path.exists():
        print(
            f"Ledger not found: {db_path}\n"
            "Run: python scripts/coverage/build_duckdb_ledger.py --replace",
            file=sys.stderr,
        )
        sys.exit(1)
    return CoverageStore(db_path)


def cmd_gap_matrix(args: argparse.Namespace) -> int:
    with _require_db(args.db) as store:
        result = store.gap_matrix(
            args.row_dim, args.col_dim,
            species_filter=args.species,
            min_confidence=args.min_confidence,
        )
        result.show(max_rows=args.limit)
    return 0


def cmd_source_rates(args: argparse.Namespace) -> int:
    with _require_db(args.db) as store:
        store.source_coverage_rates(min_confidence=args.min_confidence).show(max_rows=30)
    return 0


def cmd_uncovered(args: argparse.Namespace) -> int:
    with _require_db(args.db) as store:
        result = store.uncovered_regions(min_confidence=args.min_confidence)
        result.show(max_rows=args.limit)
    return 0


def cmd_dark_pairs(args: argparse.Namespace) -> int:
    with _require_db(args.db) as store:
        store.dark_pairs(
            args.dim_a, args.dim_b,
            top_n=args.limit,
            min_confidence=args.min_confidence,
        ).show(max_rows=args.limit)
    return 0


def cmd_atlas_coverage(args: argparse.Namespace) -> int:
    with _require_db(args.db) as store:
        store.atlas_coverage(atlas=args.atlas).show(max_rows=args.limit)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    with _require_db(args.db) as store:
        summary = store.coverage_summary()
        print(json.dumps(summary, indent=2))
    return 0


def cmd_sql(args: argparse.Namespace) -> int:
    with _require_db(args.db) as store:
        store.sql(args.query).show(max_rows=args.limit)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    sub = parser.add_subparsers(dest="cmd", required=True)

    gm = sub.add_parser("gap-matrix", help="region × modality dataset counts")
    gm.add_argument("--row-dim", default="brain_regions")
    gm.add_argument("--col-dim", default="modalities")
    gm.add_argument("--species", default=None, help="Filter by species value_id")
    gm.set_defaults(func=cmd_gap_matrix)

    sr = sub.add_parser("source-rates", help="Per-source coverage percentages")
    sr.set_defaults(func=cmd_source_rates)

    uc = sub.add_parser("uncovered-regions", help="Regions with zero datasets")
    uc.set_defaults(func=cmd_uncovered)

    dp = sub.add_parser("dark-pairs", help="Highest-opportunity zero-coverage pairs")
    dp.add_argument("--dim-a", default="brain_regions")
    dp.add_argument("--dim-b", default="modalities")
    dp.set_defaults(func=cmd_dark_pairs)

    ac = sub.add_parser("atlas-coverage", help="Coverage per atlas structure")
    ac.add_argument("--atlas", default="allen_ccf_mouse_id",
                    choices=["allen_ccf_mouse_id", "uberon_id", "waxholm_rat_id", "allen_human_id"])
    ac.set_defaults(func=cmd_atlas_coverage)

    sm = sub.add_parser("summary", help="Overall coverage stats (JSON)")
    sm.set_defaults(func=cmd_summary)

    sq = sub.add_parser("sql", help="Run raw SQL against the ledger")
    sq.add_argument("query")
    sq.set_defaults(func=cmd_sql)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
