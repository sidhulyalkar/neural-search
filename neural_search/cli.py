"""Top-level command-line interface for Neural Search."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import urllib.parse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from neural_search.evaluation.adoption import evaluate_adoption_file
from neural_search.evaluation.run_benchmark import main as benchmark_main
from neural_search.ingestion.demo_seed import DEFAULT_DATABASE_URL, seed_demo_database
from neural_search.ingestion.services import ingest_source
from neural_search.reports.dataset_compilation import main as report_main
from neural_search.runtime import (
    ARTIFACTS,
    PROFILES,
    artifact_status,
    build_reproducibility_manifest,
    fetch_bundle_ref,
    list_artifacts,
    list_profiles,
    load_release_index,
    lock_snapshot,
    profile_status,
    read_lineage,
    verify_locked_artifacts,
    write_lineage,
)
from neural_search.runtime.bundles import DEFAULT_INDEX_PATH
from neural_search.runtime.publishing import build_bundle_manifest, write_bundle_manifest
from neural_search.search import search_datasets
from neural_search.services import ReanalysisPlanningService


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


def _source_value(value: str) -> str | Path:
    """Keep remote URLs as strings while making local manifest paths explicit."""

    return value if urllib.parse.urlparse(value).scheme else Path(value)


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
        help="Inspect, fetch, verify, stamp, and publish artifact contracts.",
    )
    artifacts_subparsers = artifacts_parser.add_subparsers(
        dest="artifacts_command",
        required=True,
    )
    artifacts_subparsers.add_parser("list", help="List registered artifact contracts.")
    status_parser = artifacts_subparsers.add_parser(
        "status",
        help="Inspect artifact availability and lineage.",
    )
    status_parser.add_argument(
        "artifact_ids",
        nargs="*",
        help="Artifact IDs to inspect; omit to inspect all registered artifacts.",
    )

    releases_parser = artifacts_subparsers.add_parser(
        "releases",
        help="List immutable artifact bundles from a release index.",
    )
    releases_parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))

    fetch_parser = artifacts_subparsers.add_parser(
        "fetch",
        help="Fetch and checksum-verify an immutable bundle into the local cache.",
    )
    fetch_parser.add_argument("bundle_ref", help="Bundle ref such as name@version")
    fetch_parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    fetch_parser.add_argument("--manifest", default=None)
    fetch_parser.add_argument("--cache-dir", type=Path, default=None)
    fetch_parser.add_argument("--lock", type=Path, default=None)
    fetch_parser.add_argument(
        "--allow-local-files",
        action="store_true",
        help="Allow local manifest/artifact sources for testing or controlled lab mirrors.",
    )

    verify_parser = artifacts_subparsers.add_parser(
        "verify",
        help="Re-hash locally locked artifacts and compare them with bundle checksums.",
    )
    verify_parser.add_argument("--lock", type=Path, default=None)

    lock_parser = artifacts_subparsers.add_parser(
        "lock",
        help="Show locally pinned artifact bundles and lineage IDs.",
    )
    lock_parser.add_argument("--lock", type=Path, default=None)

    stamp_parser = artifacts_subparsers.add_parser(
        "stamp",
        help="Write a content-addressed lineage sidecar for a generated artifact.",
    )
    stamp_parser.add_argument("artifact_id", choices=list(ARTIFACTS))
    stamp_parser.add_argument("--version", default=None)
    stamp_parser.add_argument("--producer", default=None)
    stamp_parser.add_argument(
        "--allow-portable",
        action="store_true",
        help="Allow stamping committed/frozen fixtures (normally unnecessary).",
    )

    bundle_parser = artifacts_subparsers.add_parser(
        "bundle-build",
        help="Build an immutable bundle manifest from stamped local artifacts.",
    )
    bundle_parser.add_argument("name")
    bundle_parser.add_argument("version")
    bundle_parser.add_argument("artifact_ids", nargs="+")
    bundle_parser.add_argument("--compatibility-group", required=True)
    bundle_parser.add_argument("--source-base-url", required=True)
    bundle_parser.add_argument("--source-commit", default=None)
    bundle_parser.add_argument("--description", default=None)
    bundle_parser.add_argument("--output", type=Path, required=True)
    bundle_parser.add_argument("--allow-untracked", action="store_true")


def _add_workflow_commands(subparsers: Any) -> None:
    reanalysis_parser = subparsers.add_parser(
        "reanalysis",
        help="Build an evidence-aware reanalysis plan for one dataset.",
    )
    reanalysis_parser.add_argument("dataset_id")
    reanalysis_parser.add_argument("--limit", type=int, default=12)

    adoption_parser = subparsers.add_parser(
        "adoption",
        help="Evaluate external-user workflow events.",
    )
    adoption_subparsers = adoption_parser.add_subparsers(
        dest="adoption_command",
        required=True,
    )
    report_parser = adoption_subparsers.add_parser(
        "report",
        help="Compute usability/downstream workflow metrics from JSONL events.",
    )
    report_parser.add_argument(
        "--events",
        type=Path,
        default=Path("artifacts/frontend/adoption_events.jsonl"),
    )
    report_parser.add_argument("--output", type=Path, default=None)


def _stamp_artifact(args: Any) -> dict[str, Any]:
    spec = ARTIFACTS[args.artifact_id]
    if spec.kind in {"committed_fixture", "frozen_evaluation"} and not args.allow_portable:
        raise ValueError(
            "Portable Git-tracked inputs already have Git identity; pass --allow-portable "
            "only if you intentionally want an additional lineage sidecar."
        )
    status = artifact_status(args.artifact_id)
    if not status["usable"]:
        raise ValueError(f"Artifact is not usable: {args.artifact_id}")

    parents: dict[str, str] = {}
    for parent_id in spec.derived_from:
        parent_status = artifact_status(parent_id)
        if not parent_status["usable"]:
            raise ValueError(
                f"Cannot stamp {args.artifact_id}: parent artifact is unavailable: {parent_id}"
            )
        parent_lineage = read_lineage(Path(parent_status["absolute_path"]))
        if parent_lineage is None:
            raise ValueError(
                f"Cannot stamp {args.artifact_id}: stamp parent {parent_id} first so the "
                "derivation can be content-addressed."
            )
        parents[parent_id] = parent_lineage.lineage_id

    artifact_version = args.version or spec.version
    if not artifact_version:
        raise ValueError("Artifact version is required because the registry has no default")
    lineage = write_lineage(
        Path(status["absolute_path"]),
        artifact_id=args.artifact_id,
        artifact_version=artifact_version,
        compatibility_group=spec.compatibility_group,
        producer=args.producer or spec.producer,
        derived_from=parents,
        metadata={"registry_path": spec.path, "capability": spec.capability},
    )
    return lineage.to_dict()


def _write_json_output(payload: Any, output: Path | None) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
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
    _add_workflow_commands(subparsers)
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

    try:
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
                _write_json_output(payload, args.output)
                _print_json(payload)
                return 0 if payload["profile_ready"] else 1
        if args.command == "artifacts":
            if args.artifacts_command == "list":
                _print_json({"artifacts": list_artifacts()})
                return 0
            if args.artifacts_command == "status":
                artifact_ids = args.artifact_ids or [item["id"] for item in list_artifacts()]
                _print_json({"artifacts": [artifact_status(item) for item in artifact_ids]})
                return 0
            if args.artifacts_command == "releases":
                index = load_release_index(_source_value(args.index), allow_local_files=True)
                _print_json(
                    {
                        "bundles": [
                            {
                                "name": item.name,
                                "version": item.version,
                                "ref": item.ref,
                                "manifest_url": item.manifest_url,
                                "compatibility_group": item.compatibility_group,
                                "deprecated": item.deprecated,
                            }
                            for item in index.bundles
                        ]
                    }
                )
                return 0
            if args.artifacts_command == "fetch":
                payload = fetch_bundle_ref(
                    args.bundle_ref,
                    index_source=_source_value(args.index),
                    manifest_source=(
                        _source_value(args.manifest) if args.manifest is not None else None
                    ),
                    cache_dir=args.cache_dir,
                    lock_path=args.lock,
                    allow_local_files=args.allow_local_files,
                )
                _print_json(payload)
                return 0
            if args.artifacts_command == "verify":
                payload = verify_locked_artifacts(lock_path=args.lock)
                _print_json(payload)
                return 0 if payload["valid"] else 1
            if args.artifacts_command == "lock":
                _print_json(lock_snapshot(lock_path=args.lock))
                return 0
            if args.artifacts_command == "stamp":
                _print_json(_stamp_artifact(args))
                return 0
            if args.artifacts_command == "bundle-build":
                bundle = build_bundle_manifest(
                    name=args.name,
                    version=args.version,
                    compatibility_group=args.compatibility_group,
                    artifact_ids=args.artifact_ids,
                    source_base_url=args.source_base_url,
                    source_commit=args.source_commit,
                    description=args.description,
                    allow_untracked=args.allow_untracked,
                )
                write_bundle_manifest(args.output, bundle)
                _print_json({"output": str(args.output), "bundle": bundle.to_dict()})
                return 0
        if args.command == "reanalysis":
            if args.limit < 1 or args.limit > 30:
                raise ValueError("--limit must be between 1 and 30")
            _print_json(ReanalysisPlanningService().plan(args.dataset_id, limit=args.limit))
            return 0
        if args.command == "adoption" and args.adoption_command == "report":
            payload = evaluate_adoption_file(args.events)
            _write_json_output(payload, args.output)
            _print_json(payload)
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
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
