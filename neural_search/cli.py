"""Top-level command-line interface for Neural Search."""

from __future__ import annotations

import argparse
import json
from typing import Any

from neural_search.evaluation.run_benchmark import main as benchmark_main
from neural_search.ingestion.demo_seed import DEFAULT_DATABASE_URL, seed_demo_database
from neural_search.ingestion.services import ingest_source
from neural_search.reports.dataset_compilation import main as report_main
from neural_search.search import search_datasets


def main(argv: list[str] | None = None) -> int:
    """Run the Neural Search CLI."""

    parser = argparse.ArgumentParser(prog="neural-search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Seed the local demo database.")
    demo_parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)

    search_parser = subparsers.add_parser("search", help="Search demo datasets.")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)

    subparsers.add_parser("benchmark", help="Run the retrieval benchmark.")
    subparsers.add_parser("report", help="Generate the dataset compilation report.")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest live source records.")
    ingest_subparsers = ingest_parser.add_subparsers(dest="source", required=True)
    for source in ("dandi", "openneuro", "openalex"):
        source_parser = ingest_subparsers.add_parser(source)
        source_parser.add_argument("--query", required=True)
        source_parser.add_argument("--limit", type=int, default=10)
        source_parser.add_argument("--save", action="store_true")
        source_parser.add_argument("--force", action="store_true")
        source_parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)

    args, remainder = parser.parse_known_args(argv)

    if args.command == "demo":
        _print_json(seed_demo_database(args.database_url))
        return 0
    if args.command == "search":
        response = search_datasets(args.query, {}, limit=args.limit)
        _print_json(response.model_dump(mode="json"))
        return 0
    if args.command == "benchmark":
        return benchmark_main(remainder)
    if args.command == "report":
        return report_main(remainder)
    if args.command == "ingest":
        result = ingest_source(
            args.source,
            args.query,
            args.limit,
            save=args.save,
            force=args.force,
            database_url=args.database_url,
        )
        _print_json(result.to_dict())
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
