"""Build (or rebuild) the KG DuckDB store from composed_kg.jsonl.

Usage
-----
    python scripts/kg/build_kg_duckdb.py [--db PATH] [--jsonl PATH] [--force]

Options
-------
    --db PATH      Path to the DuckDB file to write.
                   Default: data/kg/neural_search_kg.duckdb
    --jsonl PATH   Path to the source composed_kg.jsonl.
                   Default: artifacts/kg/composed_kg.jsonl
    --force        Drop and recreate tables before importing (full rebuild).

The script is idempotent by default: running it twice does not duplicate rows.
Use --force to do a clean rebuild from scratch.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import duckdb

from neural_search.graph.kg_store import (
    DEFAULT_DB_PATH,
    DEFAULT_JSONL_PATH,
    import_composed_kg,
    init_kg_tables,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the KG DuckDB store from composed_kg.jsonl"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to DuckDB file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL_PATH,
        help=f"Path to composed_kg.jsonl (default: {DEFAULT_JSONL_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate tables before importing (full rebuild)",
    )
    args = parser.parse_args(argv)

    db_path: Path = args.db
    jsonl_path: Path = args.jsonl

    if not jsonl_path.exists():
        log.error("JSONL not found: %s", jsonl_path)
        log.error("Run the KG composition pipeline first to produce composed_kg.jsonl")
        return 1

    db_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Opening DuckDB at %s", db_path)

    conn = duckdb.connect(str(db_path))

    if args.force:
        log.info("--force: dropping kg_nodes and kg_edges")
        conn.execute("DROP TABLE IF EXISTS kg_edges")
        conn.execute("DROP TABLE IF EXISTS kg_nodes")

    init_kg_tables(conn)

    t0 = time.perf_counter()
    stats = import_composed_kg(conn, jsonl_path)
    elapsed = time.perf_counter() - t0

    node_count: int = conn.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0]
    edge_count: int = conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]

    print(f"\nKG DuckDB store: {db_path}")
    print(f"  Nodes:           {node_count:,}")
    print(f"  Edges:           {edge_count:,}")
    print(f"  Inserted this run: {stats['nodes_inserted']:,} nodes, {stats['edges_inserted']:,} edges")
    print(f"  Import time:     {elapsed:.1f}s")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
