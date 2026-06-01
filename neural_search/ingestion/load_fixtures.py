"""CLI for loading deterministic dataset and paper fixtures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from neural_search.ingestion.demo_seed import DEFAULT_DATABASE_URL, seed_demo_database


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.ingestion.load_fixtures"
    )
    parser.add_argument("datasets_path", help="Path to demo_datasets.yaml")
    parser.add_argument(
        "--papers-path",
        default=None,
        help="Path to demo_papers.yaml. Defaults to demo_papers.yaml next to datasets.",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
        help="Database URL to populate. Defaults to local SQLite fixture DB.",
    )
    args = parser.parse_args(argv)

    papers_path = args.papers_path or str(Path(args.datasets_path).with_name("demo_papers.yaml"))
    summary = seed_demo_database(
        database_url=args.database_url,
        datasets_path=args.datasets_path,
        papers_path=papers_path,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

