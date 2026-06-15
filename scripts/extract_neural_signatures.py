#!/usr/bin/env python3
"""Extract neural signatures from NWB files.

This script extracts NeuralSignatureV1 features from downloaded NWB files
and saves them for content-based retrieval.

Usage:
    # Extract from a single file
    python scripts/extract_neural_signatures.py path/to/file.nwb

    # Extract from a directory of NWB files
    python scripts/extract_neural_signatures.py data/nwb/ --recursive

    # Extract from DANDI download directory
    python scripts/extract_neural_signatures.py DANDI/ --recursive --output data/signatures/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_search.core.neural_signature import (
    NeuralSignatureV1,
    SignatureQuality,
    extract_signature_from_nwb,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_nwb_files(path: Path, recursive: bool = False) -> list[Path]:
    """Find all NWB files in a path."""
    if path.is_file() and path.suffix == ".nwb":
        return [path]

    if path.is_dir():
        pattern = "**/*.nwb" if recursive else "*.nwb"
        return list(path.glob(pattern))

    return []


def extract_signatures(
    input_path: Path,
    output_dir: Path,
    recursive: bool = False,
    max_files: int | None = None,
) -> list[NeuralSignatureV1]:
    """Extract signatures from NWB files."""
    nwb_files = find_nwb_files(input_path, recursive)

    if max_files:
        nwb_files = nwb_files[:max_files]

    logger.info(f"Found {len(nwb_files)} NWB files")

    signatures: list[NeuralSignatureV1] = []
    failed: list[tuple[Path, str]] = []

    for i, nwb_path in enumerate(nwb_files, 1):
        logger.info(f"[{i}/{len(nwb_files)}] Processing {nwb_path.name}")

        try:
            # Infer dataset_id from path (DANDI format: DANDI/000XXX/sub-YYY/...)
            parts = nwb_path.parts
            dataset_id = None
            for _j, part in enumerate(parts):
                if part.startswith("000") and len(part) == 6:
                    dataset_id = f"dandi:{part}"
                    break

            if not dataset_id:
                dataset_id = nwb_path.stem

            signature = extract_signature_from_nwb(str(nwb_path), dataset_id)
            signatures.append(signature)

            logger.info(
                f"  Extracted: {signature.n_units or 0} units, "
                f"{signature.n_rois or 0} ROIs, "
                f"{len(signature.brain_regions)} regions"
            )

        except Exception as e:
            logger.error(f"  Failed: {e}")
            failed.append((nwb_path, str(e)))

    # Save signatures
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    output_file = output_dir / "neural_signatures.jsonl"
    with open(output_file, "w") as f:
        for sig in signatures:
            f.write(sig.model_dump_json() + "\n")

    logger.info(f"Saved {len(signatures)} signatures to {output_file}")

    # Save summary
    summary = {
        "total_files": len(nwb_files),
        "successful": len(signatures),
        "failed": len(failed),
        "quality_breakdown": {
            "high": sum(1 for s in signatures if s.quality == SignatureQuality.HIGH),
            "medium": sum(1 for s in signatures if s.quality == SignatureQuality.MEDIUM),
            "low": sum(1 for s in signatures if s.quality == SignatureQuality.LOW),
        },
        "modality_breakdown": {},
        "failed_files": [{"path": str(p), "error": e} for p, e in failed],
    }

    for sig in signatures:
        mod = str(sig.modality)
        summary["modality_breakdown"][mod] = summary["modality_breakdown"].get(mod, 0) + 1

    summary_file = output_dir / "extraction_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {summary_file}")

    return signatures


def main():
    parser = argparse.ArgumentParser(description="Extract neural signatures from NWB files")
    parser.add_argument("input", type=Path, help="NWB file or directory")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/signatures"),
                        help="Output directory")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Search directories recursively")
    parser.add_argument("--max-files", "-n", type=int, default=None,
                        help="Maximum number of files to process")

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)

    try:
        signatures = extract_signatures(
            args.input,
            args.output,
            recursive=args.recursive,
            max_files=args.max_files,
        )

        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Signatures extracted: {len(signatures)}")
        print(f"Output: {args.output}")

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install pynwb: pip install pynwb")
        sys.exit(1)


if __name__ == "__main__":
    main()
