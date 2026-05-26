#!/usr/bin/env python3
"""Corpus expansion script for Neural Search.

This script orchestrates ingestion of real datasets from DANDI and OpenNeuro
to expand the corpus from demo-only to production-ready validation.

Usage:
    # Expand with all sources
    python scripts/expand_real_corpus.py --all

    # Expand from DANDI only
    python scripts/expand_real_corpus.py --dandi --limit 20

    # Expand from specific dandisets
    python scripts/expand_real_corpus.py --dandisets 000003,000004,000005

    # Dry run to see what would be added
    python scripts/expand_real_corpus.py --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Priority datasets for expansion
PRIORITY_DANDISETS = [
    # Neuropixels - high-quality ephys
    "000003",  # Allen Institute Visual Coding Neuropixels
    "000004",  # Steinmetz et al. 2019
    "000005",  # Allen Brain Observatory
    "000026",  # Steinmetz neuropixels (already have?)
    "000034",  # Neuropixels Ultra
    # Calcium imaging
    "000037",  # Tolias lab calcium imaging
    "000049",  # Svoboda lab calcium imaging
    "000016",  # Two-photon imaging
    # iEEG/ECoG
    "000055",  # Human iEEG
    "000060",  # ECoG motor cortex
    # Behavior-rich
    "000045",  # IBL behavior + ephys
    "000067",  # Decision making neuropixels
    "000017",  # Behavioral recordings
    # Diverse modalities
    "000006",  # Multi-modal
    "000008",  # Patch-seq
    "000010",  # Optogenetics
]

PRIORITY_OPENNEURO = [
    # fMRI
    "ds000001",  # Balloon Analog Risk Task
    "ds000002",  # Classification learning
    "ds000003",  # Rhyme judgment
    "ds000030",  # UCLA Consortium
    "ds000228",  # MPI-Leipzig Mind-Brain-Body
    # EEG
    "ds002778",  # ERP Core
    "ds003061",  # LEMON EEG
    "ds001787",  # EEG Motor imagery
    # MEG
    "ds000117",  # MEG multimodal
    "ds000248",  # MEG resting state
    # Multi-modal
    "ds000224",  # HCP-like
]


def ingest_allen_datasets(
    output_dir: Path,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Ingest datasets from Allen Brain Atlas.

    Args:
        output_dir: Directory to write normalized records
        dry_run: If True, don't actually write files
        limit: Max datasets to ingest

    Returns:
        Summary of ingestion results
    """
    from neural_search.ingestion.allen_brain import (
        get_curated_allen_datasets,
        normalize_allen_record,
    )

    results = {
        "source": "allen",
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "errors": [],
        "records": [],
    }

    existing = load_existing_corpus(output_dir)
    datasets = get_curated_allen_datasets()
    if limit:
        datasets = datasets[:limit]

    results["attempted"] = len(datasets)

    for dataset in datasets:
        dataset_id = dataset["id"]

        # Check if already exists
        if f"allen_{dataset_id}" in existing or dataset_id in existing:
            print(f"  [SKIP] {dataset_id} - already in corpus")
            results["skipped"] += 1
            continue

        try:
            print(f"  [PROCESS] {dataset_id}...")
            record = normalize_allen_record(dataset)
            record_dict = record.model_dump(mode="json")
            results["records"].append(record_dict)
            results["succeeded"] += 1
            print(f"  [OK] {dataset_id} - {record.title[:50]}")

        except Exception as e:
            print(f"  [ERROR] {dataset_id} - {e}")
            results["failed"] += 1
            results["errors"].append(f"{dataset_id}: {str(e)}")

    # Write records if not dry run
    if not dry_run and results["records"]:
        output_file = output_dir / "real_allen.jsonl"
        mode = "a" if output_file.exists() else "w"
        with open(output_file, mode, encoding="utf-8") as f:
            for record in results["records"]:
                f.write(json.dumps(record) + "\n")
        print(f"\nWrote {len(results['records'])} records to {output_file}")

    return results


def ingest_nemo_datasets(
    output_dir: Path,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Ingest datasets from NeMO Archive.

    Args:
        output_dir: Directory to write normalized records
        dry_run: If True, don't actually write files
        limit: Max datasets to ingest

    Returns:
        Summary of ingestion results
    """
    from neural_search.ingestion.nemo_archive import (
        get_curated_nemo_datasets,
        normalize_nemo_record,
    )

    results = {
        "source": "nemo",
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "errors": [],
        "records": [],
    }

    existing = load_existing_corpus(output_dir)
    datasets = get_curated_nemo_datasets()
    if limit:
        datasets = datasets[:limit]

    results["attempted"] = len(datasets)

    for dataset in datasets:
        dataset_id = dataset["id"]

        # Check if already exists
        if f"nemo_{dataset_id}" in existing or dataset_id in existing:
            print(f"  [SKIP] {dataset_id} - already in corpus")
            results["skipped"] += 1
            continue

        try:
            print(f"  [PROCESS] {dataset_id}...")
            record = normalize_nemo_record(dataset)
            record_dict = record.model_dump(mode="json")
            results["records"].append(record_dict)
            results["succeeded"] += 1
            print(f"  [OK] {dataset_id} - {record.title[:50]}")

        except Exception as e:
            print(f"  [ERROR] {dataset_id} - {e}")
            results["failed"] += 1
            results["errors"].append(f"{dataset_id}: {str(e)}")

    # Write records if not dry run
    if not dry_run and results["records"]:
        output_file = output_dir / "real_nemo.jsonl"
        mode = "a" if output_file.exists() else "w"
        with open(output_file, mode, encoding="utf-8") as f:
            for record in results["records"]:
                f.write(json.dumps(record) + "\n")
        print(f"\nWrote {len(results['records'])} records to {output_file}")

    return results


def load_existing_corpus(corpus_dir: Path) -> set[str]:
    """Load IDs of existing datasets in corpus."""
    existing = set()

    for jsonl_file in corpus_dir.glob("*.jsonl"):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if "dataset_id" in data:
                        existing.add(data["dataset_id"])

    return existing


def fetch_single_dandiset(dandiset_id: str) -> dict[str, Any] | None:
    """Fetch full metadata for a single dandiset by ID.

    Fetches both the dandiset info and the version-specific metadata
    to get complete information including description, contributors, etc.
    """
    import httpx

    base_url = f"https://api.dandiarchive.org/api/dandisets/{dandiset_id}/"
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            # Fetch base dandiset info
            response = client.get(base_url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()

            # Get the version tag
            version_info = data.get("most_recent_published_version") or data.get("draft_version")
            if version_info:
                version_tag = version_info.get("version", "draft")
                # Fetch version-specific metadata which has description, etc.
                version_url = f"{base_url}versions/{version_tag}/"
                try:
                    version_response = client.get(version_url)
                    if version_response.status_code == 200:
                        version_data = version_response.json()
                        # Merge version metadata into the response
                        data["metadata"] = version_data
                        data["description"] = version_data.get("description")
                        data["name"] = version_data.get("name") or data.get("name")
                except Exception:
                    pass  # Continue with basic data if version fetch fails

            return data
    except Exception:
        return None


def ingest_dandi_datasets(
    dandiset_ids: list[str],
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ingest datasets from DANDI Archive.

    Args:
        dandiset_ids: List of dandiset IDs to ingest
        output_dir: Directory to write normalized records
        dry_run: If True, don't actually write files

    Returns:
        Summary of ingestion results
    """
    from neural_search.ingestion.dandi import normalize_dandiset_record

    results = {
        "source": "dandi",
        "attempted": len(dandiset_ids),
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "errors": [],
        "records": [],
    }

    existing = load_existing_corpus(output_dir)

    for dandiset_id in dandiset_ids:
        # Check if already exists
        if f"DANDI_{dandiset_id}" in existing or dandiset_id in existing:
            print(f"  [SKIP] {dandiset_id} - already in corpus")
            results["skipped"] += 1
            continue

        try:
            print(f"  [FETCH] {dandiset_id}...")
            metadata = fetch_single_dandiset(dandiset_id)

            if metadata is None:
                print(f"  [FAIL] {dandiset_id} - could not fetch metadata")
                results["failed"] += 1
                results["errors"].append(f"{dandiset_id}: fetch failed")
                continue

            record = normalize_dandiset_record(metadata)
            # Convert Pydantic model to dict for JSON serialization
            record_dict = record.model_dump(mode="json")
            results["records"].append(record_dict)
            results["succeeded"] += 1
            print(f"  [OK] {dandiset_id} - {record.title[:50]}")

        except Exception as e:
            print(f"  [ERROR] {dandiset_id} - {e}")
            results["failed"] += 1
            results["errors"].append(f"{dandiset_id}: {str(e)}")

    # Write records if not dry run
    if not dry_run and results["records"]:
        output_file = output_dir / "real_dandi.jsonl"
        mode = "a" if output_file.exists() else "w"
        with open(output_file, mode, encoding="utf-8") as f:
            for record in results["records"]:
                f.write(json.dumps(record) + "\n")
        print(f"\nWrote {len(results['records'])} records to {output_file}")

    return results


def fetch_single_openneuro(dataset_id: str) -> dict[str, Any] | None:
    """Fetch metadata for a single OpenNeuro dataset by ID."""
    import httpx

    # Use same fields as the working search query in openneuro.py
    query = """
    query GetDataset($id: ID!) {
        dataset(id: $id) {
            id
            name
            created
            public
            latestSnapshot {
                tag
                created
                size
                readme
                summary {
                    subjects
                    tasks
                    modalities
                }
            }
        }
    }
    """
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.post(
                "https://openneuro.org/crn/graphql",
                json={"query": query, "variables": {"id": dataset_id}},
            )
            response.raise_for_status()
            data = response.json()
            if data.get("errors"):
                return None
            dataset = data.get("data", {}).get("dataset")
            # Add description from readme if available
            if dataset and dataset.get("latestSnapshot"):
                dataset["description"] = dataset["latestSnapshot"].get("readme")
            return dataset
    except Exception:
        return None


def ingest_openneuro_datasets(
    dataset_ids: list[str],
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ingest datasets from OpenNeuro.

    Args:
        dataset_ids: List of OpenNeuro dataset IDs
        output_dir: Directory to write normalized records
        dry_run: If True, don't actually write files

    Returns:
        Summary of ingestion results
    """
    from neural_search.ingestion.openneuro import normalize_openneuro_record

    results = {
        "source": "openneuro",
        "attempted": len(dataset_ids),
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "errors": [],
        "records": [],
    }

    existing = load_existing_corpus(output_dir)

    for dataset_id in dataset_ids:
        # Check if already exists
        if f"ON_{dataset_id}" in existing or dataset_id in existing:
            print(f"  [SKIP] {dataset_id} - already in corpus")
            results["skipped"] += 1
            continue

        try:
            print(f"  [FETCH] {dataset_id}...")
            metadata = fetch_single_openneuro(dataset_id)

            if metadata is None:
                print(f"  [FAIL] {dataset_id} - could not fetch metadata")
                results["failed"] += 1
                results["errors"].append(f"{dataset_id}: fetch failed")
                continue

            record = normalize_openneuro_record(metadata)
            # Convert Pydantic model to dict for JSON serialization
            record_dict = record.model_dump(mode="json")
            results["records"].append(record_dict)
            results["succeeded"] += 1
            print(f"  [OK] {dataset_id} - {record.title[:50]}")

        except Exception as e:
            print(f"  [ERROR] {dataset_id} - {e}")
            results["failed"] += 1
            results["errors"].append(f"{dataset_id}: {str(e)}")

    # Write records if not dry run
    if not dry_run and results["records"]:
        output_file = output_dir / "real_openneuro.jsonl"
        mode = "a" if output_file.exists() else "w"
        with open(output_file, mode, encoding="utf-8") as f:
            for record in results["records"]:
                f.write(json.dumps(record) + "\n")
        print(f"\nWrote {len(results['records'])} records to {output_file}")

    return results


def generate_expansion_report(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    """Generate expansion report.

    Args:
        results: List of ingestion results
        output_dir: Output directory

    Returns:
        Report as markdown string
    """
    lines = [
        "# Corpus Expansion Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        "| Source | Attempted | Succeeded | Failed | Skipped |",
        "| --- | --- | --- | --- | --- |",
    ]

    total_succeeded = 0
    total_failed = 0

    for result in results:
        lines.append(
            f"| {result['source']} | {result['attempted']} | "
            f"{result['succeeded']} | {result['failed']} | {result['skipped']} |"
        )
        total_succeeded += result["succeeded"]
        total_failed += result["failed"]

    lines.extend([
        "",
        f"**Total new records:** {total_succeeded}",
        f"**Total failures:** {total_failed}",
        "",
    ])

    # Errors section
    all_errors = []
    for result in results:
        all_errors.extend(result.get("errors", []))

    if all_errors:
        lines.extend([
            "## Errors",
            "",
        ])
        for error in all_errors[:20]:  # Limit to 20
            lines.append(f"- {error}")
        lines.append("")

    # Coverage analysis
    existing = load_existing_corpus(output_dir)
    lines.extend([
        "## Current Corpus Coverage",
        "",
        f"**Total datasets:** {len(existing)}",
        "",
    ])

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Expand Neural Search corpus with real datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest from all sources (DANDI + OpenNeuro + Allen + NeMO)",
    )
    parser.add_argument(
        "--dandi",
        action="store_true",
        help="Ingest from DANDI Archive",
    )
    parser.add_argument(
        "--openneuro",
        action="store_true",
        help="Ingest from OpenNeuro",
    )
    parser.add_argument(
        "--allen",
        action="store_true",
        help="Ingest from Allen Brain Atlas/Map",
    )
    parser.add_argument(
        "--nemo",
        action="store_true",
        help="Ingest from NeMO Archive (BRAIN Initiative)",
    )
    parser.add_argument(
        "--genomics",
        action="store_true",
        help="Ingest genomic sources only (Allen + NeMO)",
    )
    parser.add_argument(
        "--dandisets",
        type=str,
        help="Comma-separated list of specific dandiset IDs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of datasets to ingest per source",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "normalized",
        help="Output directory for normalized records",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.all, args.dandi, args.openneuro, args.allen, args.nemo,
                args.genomics, args.dandisets]):
        print("Error: Specify --all, --dandi, --openneuro, --allen, --nemo, --genomics, or --dandisets")
        return 1

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print("=" * 60)
    print("NEURAL SEARCH CORPUS EXPANSION")
    print("=" * 60)
    print()

    if args.dry_run:
        print("[DRY RUN] No files will be written\n")

    # DANDI ingestion
    if args.all or args.dandi or args.dandisets:
        print("Ingesting from DANDI Archive...")
        print("-" * 40)

        if args.dandisets:
            dandiset_ids = [d.strip() for d in args.dandisets.split(",")]
        else:
            dandiset_ids = PRIORITY_DANDISETS

        if args.limit:
            dandiset_ids = dandiset_ids[:args.limit]

        dandi_results = ingest_dandi_datasets(
            dandiset_ids,
            args.output_dir,
            dry_run=args.dry_run,
        )
        results.append(dandi_results)
        print()

    # OpenNeuro ingestion
    if args.all or args.openneuro:
        print("Ingesting from OpenNeuro...")
        print("-" * 40)

        openneuro_ids = PRIORITY_OPENNEURO
        if args.limit:
            openneuro_ids = openneuro_ids[:args.limit]

        openneuro_results = ingest_openneuro_datasets(
            openneuro_ids,
            args.output_dir,
            dry_run=args.dry_run,
        )
        results.append(openneuro_results)
        print()

    # Allen Brain Atlas ingestion
    if args.all or args.allen or args.genomics:
        print("Ingesting from Allen Brain Atlas/Map...")
        print("-" * 40)

        allen_results = ingest_allen_datasets(
            args.output_dir,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        results.append(allen_results)
        print()

    # NeMO Archive ingestion
    if args.all or args.nemo or args.genomics:
        print("Ingesting from NeMO Archive (BRAIN Initiative)...")
        print("-" * 40)

        nemo_results = ingest_nemo_datasets(
            args.output_dir,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        results.append(nemo_results)
        print()

    # Generate report
    print("=" * 60)
    print("EXPANSION COMPLETE")
    print("=" * 60)

    report = generate_expansion_report(results, args.output_dir)
    print()
    print(report)

    # Save report
    if not args.dry_run:
        report_path = PROJECT_ROOT / "data" / "reports" / "corpus_expansion_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
