"""Top-level command-line interface for Neural Search."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from neural_search.evaluation.run_benchmark import main as benchmark_main
from neural_search.ingestion.demo_seed import DEFAULT_DATABASE_URL, seed_demo_database
from neural_search.ingestion.services import ingest_source
from neural_search.reports.dataset_compilation import main as report_main
from neural_search.search import search_datasets


def _package_version() -> str:
    """Return the installed package version, with a source-checkout fallback."""

    try:
        return version("neural-search")
    except PackageNotFoundError:
        return "0.1.0"


def _doctor_payload() -> dict[str, Any]:
    """Collect lightweight diagnostics without requiring external services."""

    core_modules = ("fastapi", "pydantic", "yaml", "numpy", "pandas")
    optional_modules = (
        "sentence_transformers",
        "networkx",
        "redis",
        "psycopg",
        "specparam",
    )
    source_root = Path(__file__).resolve().parents[1]
    source_checkout = (source_root / "pyproject.toml").is_file()

    core = {name: importlib.util.find_spec(name) is not None for name in core_modules}
    optional = {
        name: importlib.util.find_spec(name) is not None for name in optional_modules
    }
    assets: dict[str, bool] = {}
    if source_checkout:
        for relative_path in (
            "behavioral_task_ontology.yaml",
            "apps/web/package.json",
            "data",
        ):
            assets[relative_path] = (source_root / relative_path).exists()

    python_supported = sys.version_info >= (3, 11)
    healthy = python_supported and all(core.values()) and all(assets.values())
    return {
        "healthy": healthy,
        "neural_search_version": _package_version(),
        "python": platform.python_version(),
        "python_supported": python_supported,
        "platform": platform.platform(),
        "source_checkout": source_checkout,
        "core_dependencies": core,
        "optional_dependencies": optional,
        "source_assets": assets,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the Neural Search CLI."""

    parser = argparse.ArgumentParser(prog="neural-search")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Seed the local demo database.")
    demo_parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)

    search_parser = subparsers.add_parser("search", help="Search demo datasets.")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)

    subparsers.add_parser(
        "doctor",
        help="Check the local Python environment and source-checkout assets.",
    )
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
    if args.command == "doctor":
        payload = _doctor_payload()
        _print_json(payload)
        return 0 if payload["healthy"] else 1
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
