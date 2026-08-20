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
from neural_search.runtime import (
    PROFILES,
    artifact_status,
    build_reproducibility_manifest,
    list_artifacts,
    list_profiles,
    profile_status,
)
from neural_search.search import search_datasets


def _package_version() -> str:
    """Return the installed package version, with a source-checkout fallback."""

    try:
        return version("neural-search")
    except PackageNotFoundError:
        return "0.1.0"


def _doctor_payload(profile_name: str = "demo") -> dict[str, Any]:
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
    selected_profile = profile_status(profile_name)
    healthy = (
        python_supported
        and all(core.values())
        and all(assets.values())
        and selected_profile["ready"]
    )
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
        "profile": selected_profile,
    }


def _add_runtime_commands(subparsers: Any) -> None:
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check the local environment for a supported execution profile.",
    )
    doctor_parser.add_argument(
        "--profile",
        choices=list(PROFILES),
        default="demo",
        help="Execution profile to validate (default: demo).",
    )

    profile_parser = subparsers.add_parser(
        "profile",
        help="Inspect execution profiles and create reproducibility manifests.",
    )
    profile_subparsers = profile_parser.add_subparsers(
        dest="profile_command",
        required=True,
    )
    profile_subparsers.add_parser("list", help="List supported execution profiles.")

    show_parser = profile_subparsers.add_parser("show", help="Show one profile contract.")
    show_parser.add_argument("name", choices=list(PROFILES))

    check_parser = profile_subparsers.add_parser(
        "check",
        help="Validate dependencies and required artifacts for a profile.",
    )
    check_parser.add_argument("name", choices=list(PROFILES))

    manifest_parser = profile_subparsers.add_parser(
        "manifest",
        help="Build a reproducibility manifest for a profile.",
    )
    manifest_parser.add_argument("name", choices=list(PROFILES))
    manifest_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON manifest; stdout is always available.",
    )

    artifacts_parser = subparsers.add_parser(
        "artifacts",
        help="Inspect the first-class artifact registry.",
    )
    artifacts_subparsers = artifacts_parser.add_subparsers(
        dest="artifacts_command",
        required=True,
    )
    artifacts_subparsers.add_parser("list", help="List registered artifact contracts.")
    status_parser = artifacts_subparsers.add_parser(
        "status",
        help="Inspect artifact availability.",
    )
    status_parser.add_argument(
        "artifact_ids",
        nargs="*",
        help="Artifact IDs to inspect; omit to inspect all registered artifacts.",
    )


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

    _add_runtime_commands(subparsers)
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
        payload = _doctor_payload(args.profile)
        _print_json(payload)
        return 0 if payload["healthy"] else 1
    if args.command == "profile":
        if args.profile_command == "list":
            _print_json({"profiles": list_profiles()})
            return 0
        if args.profile_command == "show":
            _print_json(profile_status(args.name))
            return 0
        if args.profile_command == "check":
            payload = profile_status(args.name)
            _print_json(payload)
            return 0 if payload["ready"] else 1
        if args.profile_command == "manifest":
            payload = build_reproducibility_manifest(args.name)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8",
                )
            _print_json(payload)
            return 0 if payload["profile_ready"] else 1
    if args.command == "artifacts":
        if args.artifacts_command == "list":
            _print_json({"artifacts": list_artifacts()})
            return 0
        if args.artifacts_command == "status":
            artifact_ids = args.artifact_ids or [item["id"] for item in list_artifacts()]
            payload = [artifact_status(artifact_id) for artifact_id in artifact_ids]
            _print_json({"artifacts": payload})
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
