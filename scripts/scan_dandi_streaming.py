#!/usr/bin/env python3
"""Scan DANDI datasets via streaming to extract metadata and signatures.

This script uses remfile to stream NWB file headers without downloading
entire files, making it efficient to scan many datasets.

Usage:
    # Scan a single dandiset
    python scripts/scan_dandi_streaming.py --dandiset 000003

    # Scan multiple dandisets
    python scripts/scan_dandi_streaming.py --dandisets 000003,000005,000020

    # Scan diverse set for corpus enrichment
    python scripts/scan_dandi_streaming.py --diverse --max-per-dandiset 5

    # Output signatures for search corpus
    python scripts/scan_dandi_streaming.py --diverse --output data/signatures/streaming/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Diverse set of dandisets covering different modalities and tasks
DIVERSE_DANDISETS = [
    # Decision-making / behavioral
    "000003",  # Steinmetz - Neuropixels, decision-making
    "000005",  # IBL - Neuropixels, decision-making
    "000017",  # Svoboda - whisker, motor cortex

    # Visual system
    "000020",  # Allen - Neuropixels, visual cortex
    "000021",  # Stringer - visual cortex

    # Motor / BMI
    "000128",  # Sabes - motor cortex, BMI
    "000129",  # Shenoy - motor cortex

    # Calcium imaging
    "000034",  # Tank - calcium imaging, navigation
    "000037",  # Bhagat - calcium imaging

    # Hippocampus / memory
    "000044",  # Frank - hippocampus
    "000053",  # Buzsaki - hippocampus

    # Multi-region
    "000067",  # International Brain Lab
    "000114",  # Allen Institute - Neuropixels survey

    # Human data
    "000019",  # Human epilepsy ECoG
    "000023",  # Human intracranial
]


def main():
    parser = argparse.ArgumentParser(
        description="Scan DANDI datasets via streaming"
    )
    parser.add_argument(
        "--dandiset", "-d",
        help="Single dandiset ID to scan",
    )
    parser.add_argument(
        "--dandisets",
        help="Comma-separated list of dandiset IDs",
    )
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="Scan diverse set of dandisets",
    )
    parser.add_argument(
        "--max-per-dandiset", "-n",
        type=int,
        default=5,
        help="Max assets to scan per dandiset",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/signatures/streaming"),
        help="Output directory",
    )
    parser.add_argument(
        "--signatures",
        action="store_true",
        help="Also extract NeuralSignatureV1 objects",
    )

    args = parser.parse_args()

    # Determine which dandisets to scan
    if args.dandiset:
        dandiset_ids = [args.dandiset]
    elif args.dandisets:
        dandiset_ids = [d.strip() for d in args.dandisets.split(",")]
    elif args.diverse:
        dandiset_ids = DIVERSE_DANDISETS
    else:
        print("Specify --dandiset, --dandisets, or --diverse")
        sys.exit(1)

    # Import here to defer dependency check
    try:
        from neural_search.data.dandi_streaming import (
            extract_signature_streaming,
            list_dandiset_assets,
            scan_dandiset_for_affordances,
        )
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install: pip install dandi remfile pynwb h5py")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Results
    all_metadata = {}
    all_signatures = []
    summary = {
        "dandisets_scanned": 0,
        "assets_scanned": 0,
        "assets_with_units": 0,
        "assets_with_trials": 0,
        "assets_with_imaging": 0,
        "errors": [],
    }

    print(f"\n{'='*60}")
    print("DANDI STREAMING SCAN")
    print(f"{'='*60}")
    print(f"Dandisets to scan: {len(dandiset_ids)}")
    print(f"Max assets per dandiset: {args.max_per_dandiset}")
    print(f"Output: {args.output}")
    print()

    for dandiset_id in dandiset_ids:
        print(f"\n--- Dandiset {dandiset_id} ---")

        try:
            # Scan metadata
            metadata_list = scan_dandiset_for_affordances(
                dandiset_id,
                max_assets=args.max_per_dandiset,
            )

            all_metadata[dandiset_id] = metadata_list
            summary["dandisets_scanned"] += 1
            summary["assets_scanned"] += len(metadata_list)

            # Aggregate stats
            for meta in metadata_list:
                if meta.get("has_units"):
                    summary["assets_with_units"] += 1
                if meta.get("has_trials"):
                    summary["assets_with_trials"] += 1
                if meta.get("has_imaging"):
                    summary["assets_with_imaging"] += 1

            # Extract signatures if requested
            if args.signatures:
                assets = list_dandiset_assets(dandiset_id, max_assets=args.max_per_dandiset)
                for asset in assets[:3]:  # Limit to 3 for signatures
                    try:
                        sig = extract_signature_streaming(dandiset_id, asset.asset_id)
                        all_signatures.append(sig.model_dump())
                        logger.info(f"  Signature: {sig.n_units or 0} units")
                    except Exception as e:
                        logger.warning(f"  Signature extraction failed: {e}")

        except Exception as e:
            logger.error(f"Failed to scan {dandiset_id}: {e}")
            summary["errors"].append({"dandiset_id": dandiset_id, "error": str(e)})

    # Save results
    metadata_file = args.output / "streaming_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(all_metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved to {metadata_file}")

    if all_signatures:
        signatures_file = args.output / "streaming_signatures.jsonl"
        with open(signatures_file, "w") as f:
            for sig in all_signatures:
                f.write(json.dumps(sig, default=str) + "\n")
        logger.info(f"Signatures saved to {signatures_file}")

    summary_file = args.output / "scan_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SCAN COMPLETE")
    print(f"{'='*60}")
    print(f"Dandisets scanned: {summary['dandisets_scanned']}")
    print(f"Assets scanned: {summary['assets_scanned']}")
    print(f"  With units: {summary['assets_with_units']}")
    print(f"  With trials: {summary['assets_with_trials']}")
    print(f"  With imaging: {summary['assets_with_imaging']}")
    if summary["errors"]:
        print(f"  Errors: {len(summary['errors'])}")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
